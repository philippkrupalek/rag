"""
RAG System für österreichisches Umsatzsteuerrecht
Version 13 - MIT ALLEN PARSER FIXES

FIXES ANGEWENDET:
1. UStG Parser: Absatz 1 wird jetzt korrekt erfasst (war vorher systematisch fehlend)
2. Anhang Parser: Absatz 1 wird jetzt korrekt erfasst
3. UStR Parser: Deduplizierung der mehrfachen XML-Versionen (70-80% Duplikate entfernt)

Basierend auf deinem Original-Code mit folgenden Änderungen:
- parse_ustg_hierarchical() -> NEU mit Absatz 1 Capture
- parse_anhang_hierarchical() -> NEU mit Absatz 1 Capture  
- parse_ustr() -> NEU mit Deduplizierung
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import faiss

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Document class import with fallback
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from typing import Any, Dict
        class Document:
            def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

from openai import OpenAI

# Ragas imports (optional)
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
USTG_RTF_PATH = SCRIPT_DIR / "UStG1994.rtf"
ANHANG_RTF_PATH = SCRIPT_DIR / "anhang_ustg.rtf"
USTR_XML_PATH = SCRIPT_DIR / "UStR2000_html.xml"
EVAL_DATA_PATH = SCRIPT_DIR / "evaluation_data.json"

EMBEDDING_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Retrieval parameters
TOP_K_USTG = 15
TOP_K_USTR = 4
TOP_K_FINAL = 15
MIN_CROSS_ENCODER_SCORE = 0.1

# Anhang detection keywords
ANHANG_KEYWORDS = [
    'innergemeinschaftlich', 'binnenmarkt', 'erwerbsschwelle',
    'umsatzsteuer-identifikationsnummer', 'uid', 'versandhandel',
    'gemeinschaftsgebiet', 'mitgliedstaat', 'artikel'
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Chunk:
    """Enhanced chunk with source tracking"""
    text: str
    full_context: str
    metadata: Dict
    source: str  # 'ustg', 'anhang', 'ustr'
    
    def get_citation(self) -> str:
        """Get citation string for this chunk"""
        if self.source == 'ustg':
            para = self.metadata.get('paragraph', '')
            absatz = self.metadata.get('absatz', '')
            if para and absatz:
                return f"§ {para} Abs. {absatz} UStG 1994"
            elif para:
                return f"§ {para} UStG 1994"
            return "UStG 1994"
        elif self.source == 'anhang':
            art = self.metadata.get('artikel', '')
            absatz = self.metadata.get('absatz', '')
            if art and absatz:
                return f"Art. {art} Abs. {absatz} UStG 1994"
            elif art:
                return f"Art. {art} UStG 1994"
            return "Anhang UStG 1994"
        elif self.source == 'ustr':
            rz = self.metadata.get('randzahl', '')
            return f"UStR 2000 Rz {rz}" if rz else "UStR 2000"
        return "Unknown"


# =============================================================================
# PARSER: UStG 1994 (RTF) - MIT ABSATZ 1 FIX
# =============================================================================

def parse_ustg_hierarchical(rtf_path: Path) -> List[Chunk]:
    """
    Parse UStG 1994 RTF mit korrekter Erfassung von Absatz 1.
    
    FIX: Der alte Parser übersprang Absatz 1, weil dieser nicht mit "(1)" beginnt,
    sondern direkt nach dem §-Header folgt. Dieser Parser erfasst alles.
    
    Vorher: ~292 Chunks (Absatz 1 fehlte überall)
    Nachher: ~370 Chunks (alle Absätze inkl. Absatz 1)
    """
    print(f"   📂 Reading UStG from: {rtf_path}")
    
    if not rtf_path.exists():
        print(f"   ❌ ERROR: File not found: {rtf_path}")
        return []
    
    with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
        rtf_content = f.read()
    
    text = rtf_to_text(rtf_content)
    lines = text.split('\n')
    
    chunks = []
    current_paragraph = None
    current_paragraph_title = ""
    current_absatz = None
    current_absatz_text = ""
    
    # NEU: Buffer für Absatz 1 (Text zwischen § und erstem (2))
    absatz_1_buffer = ""
    in_absatz_1 = False
    
    # Regex patterns
    para_pattern = r'^§\s*(\d+[a-z]?)\.?\s*(.*?)$'
    abs_pattern = r'^\((\d+[a-z]?)\)\s*(.*)'
    
    def save_current_chunk():
        """Speichert den aktuellen numerierten Absatz"""
        nonlocal current_absatz_text, current_absatz, current_paragraph
        
        if current_absatz and current_absatz_text.strip() and current_paragraph:
            chunk_text = current_absatz_text.strip()
            full_context = f"§ {current_paragraph} Abs. {current_absatz} UStG 1994\n{chunk_text}"
            
            chunks.append(Chunk(
                text=chunk_text,
                full_context=full_context,
                metadata={
                    'paragraph': current_paragraph,
                    'absatz': current_absatz,
                    'hierarchy_path': f"§ {current_paragraph} → Abs. {current_absatz}",
                    'title': current_paragraph_title
                },
                source='ustg'
            ))
    
    def save_absatz_1():
        """NEU: Speichert Absatz 1 (Text vor erstem (2))"""
        nonlocal absatz_1_buffer, current_paragraph, current_paragraph_title
        
        if absatz_1_buffer.strip() and current_paragraph:
            chunk_text = absatz_1_buffer.strip()
            
            # Prüfe ob es substantieller Inhalt ist (nicht nur Überschrift)
            if len(chunk_text) > 20:
                full_context = f"§ {current_paragraph} Abs. 1 UStG 1994\n{chunk_text}"
                
                chunks.append(Chunk(
                    text=chunk_text,
                    full_context=full_context,
                    metadata={
                        'paragraph': current_paragraph,
                        'absatz': '1',
                        'hierarchy_path': f"§ {current_paragraph} → Abs. 1",
                        'title': current_paragraph_title
                    },
                    source='ustg'
                ))
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Neuer Paragraph gefunden (z.B. "§ 1." oder "§ 6.")
        para_match = re.match(para_pattern, line)
        if para_match:
            # Vorherigen Absatz speichern
            save_current_chunk()
            
            # NEU: Absatz 1 des vorherigen Paragraphen speichern
            if in_absatz_1 and absatz_1_buffer.strip():
                save_absatz_1()
            
            # Reset für neuen Paragraphen
            current_paragraph = para_match.group(1)
            current_paragraph_title = para_match.group(2).strip() if para_match.group(2) else ""
            current_absatz = None
            current_absatz_text = ""
            absatz_1_buffer = ""
            in_absatz_1 = True  # NEU: Ab jetzt sammeln wir Absatz 1
            continue
        
        # Absatz (2), (3), etc. gefunden
        abs_match = re.match(abs_pattern, line)
        if abs_match and current_paragraph:
            absatz_num = abs_match.group(1)
            
            # NEU: Wenn wir (2) oder höher finden und noch in Absatz 1 sind
            if in_absatz_1 and absatz_1_buffer.strip():
                save_absatz_1()
                in_absatz_1 = False
            
            # Vorherigen numerierten Absatz speichern
            save_current_chunk()
            
            # Neuen Absatz starten
            current_absatz = absatz_num
            current_absatz_text = line
            in_absatz_1 = False
        else:
            # Textzeile zum aktuellen Kontext hinzufügen
            if current_absatz:
                # Zu numeriertem Absatz hinzufügen
                current_absatz_text += " " + line
            elif in_absatz_1 and current_paragraph:
                # NEU: Zu Absatz 1 hinzufügen
                absatz_1_buffer += " " + line
    
    # Letzten Chunk speichern
    save_current_chunk()
    if in_absatz_1 and absatz_1_buffer.strip():
        save_absatz_1()
    
    # Sortiere nach Paragraph und Absatz
    def sort_key(chunk):
        para = chunk.metadata.get('paragraph', '0')
        absatz = chunk.metadata.get('absatz', '0')
        
        # Numerische Sortierung mit Buchstaben-Support
        para_match = re.match(r'(\d+)', para)
        para_num = int(para_match.group(1)) if para_match else 0
        para_suffix = re.search(r'[a-z]+', para)
        para_suffix = para_suffix.group() if para_suffix else ''
        
        absatz_match = re.match(r'(\d+)', absatz)
        absatz_num = int(absatz_match.group(1)) if absatz_match else 0
        absatz_suffix = re.search(r'[a-z]+', absatz)
        absatz_suffix = absatz_suffix.group() if absatz_suffix else ''
        
        return (para_num, para_suffix, absatz_num, absatz_suffix)
    
    chunks.sort(key=sort_key)
    
    # Statistik ausgeben
    abs1_count = sum(1 for c in chunks if c.metadata.get('absatz') == '1')
    print(f"   ✅ Parsed {len(chunks)} UStG chunks (davon {abs1_count} Absatz 1 Sektionen)")
    
    return chunks


# =============================================================================
# PARSER: ANHANG (RTF) - MIT ABSATZ 1 FIX
# =============================================================================

def parse_anhang_hierarchical(rtf_path: Path) -> List[Chunk]:
    """
    Parse Anhang UStG (Binnenmarktregelung) mit korrekter Erfassung von Absatz 1.
    
    FIX: Identisches Problem wie UStG - Absatz 1 wurde übersprungen.
    
    Vorher: ~71 Chunks (Absatz 1 fehlte bei 17 von 18 Artikeln)
    Nachher: ~88 Chunks (alle Absätze inkl. Absatz 1)
    """
    print(f"   📂 Reading Anhang from: {rtf_path}")
    
    if not rtf_path.exists():
        print(f"   ❌ ERROR: File not found: {rtf_path}")
        return []
    
    # RTF einlesen (verschiedene Encodings versuchen)
    rtf_content = None
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(rtf_path, 'r', encoding=encoding, errors='ignore') as f:
                rtf_content = f.read()
            break
        except:
            continue
    
    if not rtf_content:
        print(f"   ❌ ERROR: Could not read file")
        return []
    
    text = rtf_to_text(rtf_content)
    lines = text.split('\n')
    
    chunks = []
    current_artikel = None
    current_artikel_title = ""
    current_absatz = None
    current_absatz_text = ""
    
    # NEU: Buffer für Absatz 1
    absatz_1_buffer = ""
    in_absatz_1 = False
    
    # Patterns für Artikel
    art_patterns = [
        r'^Art\.?\s*(\d+[a-z]?)\.?\s*(.*?)$',
        r'^Artikel\s+(\d+[a-z]?)\.?\s*(.*?)$',
    ]
    abs_pattern = r'^\((\d+[a-z]?)\)\s*(.*)'
    
    def save_current_chunk():
        nonlocal current_absatz_text, current_absatz, current_artikel
        
        if current_absatz and current_absatz_text.strip() and current_artikel:
            chunk_text = current_absatz_text.strip()
            full_context = f"Art. {current_artikel} Abs. {current_absatz} UStG 1994 (Anhang - Binnenmarkt)\n{chunk_text}"
            
            chunks.append(Chunk(
                text=chunk_text,
                full_context=full_context,
                metadata={
                    'artikel': current_artikel,
                    'absatz': current_absatz,
                    'hierarchy_path': f"Art. {current_artikel} → Abs. {current_absatz}",
                    'title': current_artikel_title
                },
                source='anhang'
            ))
    
    def save_absatz_1():
        nonlocal absatz_1_buffer, current_artikel, current_artikel_title
        
        if absatz_1_buffer.strip() and current_artikel:
            chunk_text = absatz_1_buffer.strip()
            
            if len(chunk_text) > 20:
                full_context = f"Art. {current_artikel} Abs. 1 UStG 1994 (Anhang - Binnenmarkt)\n{chunk_text}"
                
                chunks.append(Chunk(
                    text=chunk_text,
                    full_context=full_context,
                    metadata={
                        'artikel': current_artikel,
                        'absatz': '1',
                        'hierarchy_path': f"Art. {current_artikel} → Abs. 1",
                        'title': current_artikel_title
                    },
                    source='anhang'
                ))
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Neuer Artikel gefunden
        art_match = None
        for pattern in art_patterns:
            art_match = re.match(pattern, line, re.IGNORECASE)
            if art_match:
                break
        
        if art_match:
            # Vorherige speichern
            save_current_chunk()
            if in_absatz_1 and absatz_1_buffer.strip():
                save_absatz_1()
            
            # Reset
            current_artikel = art_match.group(1)
            current_artikel_title = art_match.group(2).strip() if len(art_match.groups()) > 1 and art_match.group(2) else ""
            current_absatz = None
            current_absatz_text = ""
            absatz_1_buffer = ""
            in_absatz_1 = True
            continue
        
        # Absatz gefunden
        abs_match = re.match(abs_pattern, line)
        if abs_match and current_artikel:
            absatz_num = abs_match.group(1)
            
            # Absatz 1 speichern wenn wir höheren Absatz finden
            if in_absatz_1 and absatz_1_buffer.strip():
                save_absatz_1()
                in_absatz_1 = False
            
            save_current_chunk()
            
            current_absatz = absatz_num
            current_absatz_text = line
            in_absatz_1 = False
        else:
            if current_absatz:
                current_absatz_text += " " + line
            elif in_absatz_1 and current_artikel:
                absatz_1_buffer += " " + line
    
    # Letzte speichern
    save_current_chunk()
    if in_absatz_1 and absatz_1_buffer.strip():
        save_absatz_1()
    
    # Sortieren
    def sort_key(chunk):
        art = chunk.metadata.get('artikel', '0')
        absatz = chunk.metadata.get('absatz', '0')
        
        art_match = re.match(r'(\d+)', art)
        art_num = int(art_match.group(1)) if art_match else 0
        art_suffix = re.search(r'[a-z]+', art)
        art_suffix = art_suffix.group() if art_suffix else ''
        
        absatz_match = re.match(r'(\d+)', absatz)
        absatz_num = int(absatz_match.group(1)) if absatz_match else 0
        absatz_suffix = re.search(r'[a-z]+', absatz)
        absatz_suffix = absatz_suffix.group() if absatz_suffix else ''
        
        return (art_num, art_suffix, absatz_num, absatz_suffix)
    
    chunks.sort(key=sort_key)
    
    # Statistik ausgeben
    abs1_count = sum(1 for c in chunks if c.metadata.get('absatz') == '1')
    print(f"   ✅ Parsed {len(chunks)} Anhang chunks (davon {abs1_count} Absatz 1 Sektionen)")
    
    return chunks


# =============================================================================
# PARSER: UStR 2000 (XML) - MIT DEDUPLIZIERUNG
# =============================================================================

def parse_ustr(xml_path: Path) -> List[Chunk]:
    """
    Parse UStR 2000 XML mit Deduplizierung der Versionen.
    
    FIX: Das XML enthält mehrere Versionen/Fassungen jeder Randzahl.
    Der alte Parser erzeugte 4-7 Duplikate pro Randzahl (70-80% redundant).
    Dieser Parser behält nur die aktuellste/vollständigste Version.
    
    Vorher: ~16.274 Chunks (massive Duplikation)
    Nachher: ~2.500-3.500 Chunks (dedupliziert)
    """
    print(f"   📂 Reading UStR from: {xml_path}")
    
    if not xml_path.exists():
        print(f"   ❌ ERROR: File not found: {xml_path}")
        return []
    
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    soup = BeautifulSoup(xml_content, 'xml')
    segments = soup.find_all('Segment')
    
    # Alle Einträge sammeln (mit Duplikaten)
    all_entries = []
    
    for segment in segments:
        segbez = segment.find('segbez')
        txt = segment.find('txt')
        
        if not txt:
            continue
        
        title = segbez.text.strip() if segbez else "Ohne Titel"
        html_content = txt.text if txt.text else ""
        
        # Parse HTML content
        html_soup = BeautifulSoup(html_content, 'html.parser')
        
        # Randzahlen im HTML finden
        for rz_tag in html_soup.find_all('a', {'name': re.compile(r'RZ_\d+')}):
            rz_id = rz_tag.get('name', '').replace('RZ_', '')
            
            if not rz_id:
                continue
            
            # Text nach dieser Randzahl bis zur nächsten sammeln
            text_parts = []
            for sibling in rz_tag.find_next_siblings():
                if sibling.name == 'a' and sibling.get('name', '').startswith('RZ_'):
                    break
                if hasattr(sibling, 'get_text'):
                    text_parts.append(sibling.get_text(separator=' ', strip=True))
            
            chunk_text = ' '.join(text_parts).strip()
            
            if not chunk_text or len(chunk_text) < 10:
                continue
            
            # Extrahiere neuestes Jahr für Versionserkennung
            latest_year = _extract_latest_year(chunk_text)
            
            all_entries.append({
                'randzahl': rz_id,
                'titel': title,
                'text': chunk_text,
                'segment': title,
                'text_length': len(chunk_text),
                'latest_year': latest_year
            })
    
    print(f"   📊 Gefunden: {len(all_entries)} Einträge (vor Deduplizierung)")
    
    # DEDUPLIZIERUNG
    chunks = _deduplicate_ustr_entries(all_entries)
    
    print(f"   ✅ Parsed {len(chunks)} UStR chunks (nach Deduplizierung)")
    
    return chunks


def _extract_latest_year(text: str) -> int:
    """
    Extrahiere das neueste Jahr aus dem Text.
    Sucht nach Jahreszahlen in Zitaten wie "EuGH 9.2.2023" oder "BGBl. I Nr. 91/2024"
    """
    year_patterns = [
        r'\b(20[0-2]\d)\b',  # 2000-2029
        r'\b(199\d)\b',      # 1990-1999
    ]
    
    years = []
    for pattern in year_patterns:
        matches = re.findall(pattern, text)
        years.extend([int(y) for y in matches])
    
    return max(years) if years else 1994


def _deduplicate_ustr_entries(entries: List[Dict]) -> List[Chunk]:
    """
    Dedupliziere UStR Einträge - behalte nur die beste Version pro Randzahl.
    
    Kriterien für "beste Version":
    1. Neueste Rechtsprechung (höchstes Jahr in Zitaten)
    2. Längster Text (vollständigste Version)
    """
    # Gruppiere nach Randzahl
    grouped = defaultdict(list)
    for entry in entries:
        rz = entry['randzahl']
        grouped[rz].append(entry)
    
    chunks = []
    duplicates_removed = 0
    
    for rz, versions in grouped.items():
        if len(versions) > 1:
            duplicates_removed += len(versions) - 1
            
            # Beste Version auswählen: neuestes Jahr, dann längster Text
            best = max(versions, key=lambda v: (
                v['latest_year'],
                v['text_length'],
            ))
        else:
            best = versions[0]
        
        # Chunk erstellen
        chunk_text = best['text']
        full_context = f"UStR 2000 Rz {rz}\n{best['titel']}\n{chunk_text}"
        
        chunks.append(Chunk(
            text=chunk_text,
            full_context=full_context,
            metadata={
                'randzahl': rz,
                'titel': best['titel'],
                'segment': best['segment']
            },
            source='ustr'
        ))
    
    # Nach Randzahl sortieren
    def sort_key(chunk):
        rz = chunk.metadata.get('randzahl', '0')
        match = re.match(r'(\d+)', rz)
        return int(match.group(1)) if match else 0
    
    chunks.sort(key=sort_key)
    
    print(f"   🗑️ Entfernt: {duplicates_removed} Duplikate")
    
    return chunks


# =============================================================================
# RETRIEVAL SYSTEM (unverändert aus deinem Code)
# =============================================================================

class UltimateUStGRetriever:
    """Production-grade retriever with score threshold"""
    
    def __init__(self, ustg_chunks: List[Chunk], anhang_chunks: List[Chunk], 
                 ustr_chunks: List[Chunk]):
        self.ustg_chunks = ustg_chunks
        self.anhang_chunks = anhang_chunks
        self.ustr_chunks = ustr_chunks
        
        print("   🔨 Building retrieval system...")
        
        # Load models
        print("   📦 Loading BGE-M3...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("   ✅ BGE-M3 loaded!")
        
        print("   📦 Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        print("   ✅ Cross-Encoder loaded!")
        
        # Compute Anhang reference embedding
        print("   🧮 Computing Anhang reference embeddings...")
        anhang_ref_text = " ".join(ANHANG_KEYWORDS)
        self.anhang_ref_embedding = self.embeddings.embed_query(anhang_ref_text)
        
        # Build indices
        self._build_indices()
        
        print("   ✅ Retrieval system ready!")
    
    def _build_indices(self):
        """Build FAISS and BM25 indices"""
        # UStG
        if self.ustg_chunks:
            print(f"   🔍 Building UStG indices ({len(self.ustg_chunks)} chunks)...")
            ustg_docs = [Document(page_content=c.full_context, metadata=c.metadata) 
                         for c in self.ustg_chunks]
            self.ustg_vectorstore = FAISS.from_documents(ustg_docs, self.embeddings)
            self.ustg_bm25 = BM25Okapi([c.full_context.split() for c in self.ustg_chunks])
        
        # Anhang
        if self.anhang_chunks:
            print(f"   🔍 Building Anhang indices ({len(self.anhang_chunks)} chunks)...")
            anhang_docs = [Document(page_content=c.full_context, metadata=c.metadata)
                           for c in self.anhang_chunks]
            self.anhang_vectorstore = FAISS.from_documents(anhang_docs, self.embeddings)
            self.anhang_bm25 = BM25Okapi([c.full_context.split() for c in self.anhang_chunks])
        else:
            self.anhang_vectorstore = None
            self.anhang_bm25 = None
        
        # UStR
        if self.ustr_chunks:
            print(f"   🔍 Building UStR indices ({len(self.ustr_chunks)} chunks)...")
            ustr_docs = [Document(page_content=c.full_context, metadata=c.metadata)
                         for c in self.ustr_chunks]
            self.ustr_vectorstore = FAISS.from_documents(ustr_docs, self.embeddings)
            self.ustr_bm25 = BM25Okapi([c.full_context.split() for c in self.ustr_chunks])
    
    def retrieve(self, query: str, verbose: bool = True) -> List[Chunk]:
        """Retrieve with cross-encoder threshold"""
        if verbose:
            print(f"\n🔎 Retrieval: '{query[:60]}...'")
        
        # Detect Anhang need
        query_lower = query.lower()
        anhang_keywords_found = [kw for kw in ANHANG_KEYWORDS if kw in query_lower]
        
        query_emb = self.embeddings.embed_query(query)
        anhang_similarity = float(np.dot(query_emb, self.anhang_ref_embedding))
        
        use_anhang = len(anhang_keywords_found) > 0 and anhang_similarity > 0.65
        
        if verbose:
            print(f"   Anhang: {use_anhang} ({'Keywords: ' + str(anhang_keywords_found) if use_anhang else 'No keywords'})")
        
        # Retrieve candidates
        candidates = []
        
        # UStG (primary source)
        ustg_candidates = self._hybrid_retrieve(
            query, self.ustg_chunks, self.ustg_vectorstore, 
            self.ustg_bm25, TOP_K_USTG
        )
        candidates.extend(ustg_candidates)
        if verbose:
            print(f"   UStG candidates: {len(ustg_candidates)}")
        
        # Anhang (if needed)
        if use_anhang and self.anhang_vectorstore:
            anhang_candidates = self._hybrid_retrieve(
                query, self.anhang_chunks, self.anhang_vectorstore,
                self.anhang_bm25, TOP_K_USTG
            )
            candidates.extend(anhang_candidates)
            if verbose:
                print(f"   Anhang candidates: {len(anhang_candidates)}")
        
        # UStR (support only)
        ustr_candidates = self._hybrid_retrieve(
            query, self.ustr_chunks, self.ustr_vectorstore,
            self.ustr_bm25, TOP_K_USTR
        )
        candidates.extend(ustr_candidates)
        if verbose:
            print(f"   UStR candidates: {len(ustr_candidates)}")
        
        # Rerank with threshold
        if verbose:
            print(f"   Reranking {len(candidates)} candidates...")
        
        reranked = self._rerank_with_threshold(query, candidates, TOP_K_FINAL)
        
        if verbose:
            print(f"   Final: {len(reranked)} chunks (score >= {MIN_CROSS_ENCODER_SCORE})")
            
            source_counts = {}
            for c in reranked:
                source_counts[c.source] = source_counts.get(c.source, 0) + 1
            
            print(f"\n   Distribution:")
            for source, count in source_counts.items():
                pct = count / len(reranked) * 100 if reranked else 0
                print(f"      {source.upper()}: {count} ({pct:.0f}%)")
        
        return reranked
    
    def _hybrid_retrieve(self, query: str, chunks: List[Chunk], 
                         vectorstore, bm25, top_k: int) -> List[Chunk]:
        """Hybrid retrieval (FAISS + BM25)"""
        if not chunks:
            return []
        
        # FAISS
        docs = vectorstore.similarity_search(query, k=top_k)
        faiss_texts = {doc.page_content for doc in docs}
        
        # BM25
        query_tokens = query.split()
        bm25_scores = bm25.get_scores(query_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[-top_k:]
        bm25_texts = {chunks[i].full_context for i in top_bm25_idx}
        
        # Combine
        combined_texts = faiss_texts | bm25_texts
        return [c for c in chunks if c.full_context in combined_texts]
    
    def _rerank_with_threshold(self, query: str, candidates: List[Chunk], 
                               top_k: int) -> List[Chunk]:
        """Rerank with minimum score threshold"""
        if not candidates:
            return []
        
        pairs = [(query, c.full_context) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        scored = [(c, float(s)) for c, s in zip(candidates, scores) 
                  if s >= MIN_CROSS_ENCODER_SCORE]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = [c for c, s in scored[:top_k]]
        
        # Safety: If threshold filtered everything, take top 5 anyway
        if len(result) == 0 and len(candidates) > 0:
            print(f"   ⚠️ WARNING: Threshold filtered all! Taking top 5.")
            all_scored = [(c, float(s)) for c, s in zip(candidates, scores)]
            all_scored.sort(key=lambda x: x[1], reverse=True)
            result = [c for c, s in all_scored[:5]]
        
        return result


# =============================================================================
# LLM INTERFACE (unverändert)
# =============================================================================

class DeepSeekLLM:
    """DeepSeek API client"""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        key = api_key or os.getenv('DEEPSEEK_API_KEY', 'sk-9aa1b071f49145b5a6b9a257847a6964')
        
        self.client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com"
        )
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer with optimized prompt"""
        
        prompt = f"""Du bist ein auf österreichisches Umsatzsteuerrecht spezialisierter Jurist.

AUFGABE:
Beantworte die folgende Frage AUSSCHLIESSLICH mit den unten bereitgestellten Rechtsquellen.

FRAGE:
{question}

BEREITGESTELLTE RECHTSQUELLEN:
{context}

QUELLEN-HIERARCHIE:
1. PRIMÄR: § ... UStG 1994 / Art. ... UStG 1994 (Gesetzestext)
2. SEKUNDÄR: UStR 2000 Rz ... (Richtlinien als Auslegungshilfe)

STRIKTE REGELN:
1. Beantworte die Frage VORRANGIG mit UStG 1994 (§ oder Art.)
2. Verwende UStR 2000 NUR als ergänzende Hilfe zur Auslegung
3. Jede Aussage MUSS durch Quellen gedeckt sein
4. Zitiere EXAKT: "§ 16 Abs. 3 Z 1 UStG 1994" oder "Art. 12 Abs. 4 UStG 1994"
5. Wenn Quellen nicht ausreichen: "Die Quellen reichen nicht aus."
6. NIEMALS Paragraphen erfinden

ANTWORTFORMAT:

GESETZLICHE GRUNDLAGEN:
[Liste ALLE verwendeten §§ und Artikel UStG 1994]

RECHTLICHE WÜRDIGUNG:
[Subsumtion des Sachverhalts unter die Gesetzestexte]
[UStR 2000 nur zur Erklärung, NICHT als Rechtsgrundlage]

FAZIT:
JA/NEIN
[2-3 Sätze Begründung basierend auf UStG 1994]

ANTWORTE NUR AUF DEUTSCH."""

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Du bist ein Experte für österreichisches Umsatzsteuerrecht."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content


# =============================================================================
# LLM-AS-JUDGE EVALUATION
# =============================================================================

class LLMJudge:
    """LLM-as-Judge for qualitative RAG evaluation"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    def extract_score(self, text: str) -> Optional[float]:
        match = re.search(r'([01](?:\.\d+)?)', text)
        return float(match.group(1)) if match else None
    
    def score_document_relevance(self, question: str, contexts: List[str]) -> Optional[float]:
        prompt = f"""You are grading *document relevance* in a RAG system for Austrian tax law.

Question:
{question}

Retrieved context chunks:
{"-----".join(contexts[:5])}

On a scale from 0 to 1, where:
- 1.0 means contexts are highly relevant and sufficient
- 0.0 means contexts are completely irrelevant

Respond with ONLY a single number between 0 and 1."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            return self.extract_score(response.choices[0].message.content)
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    def score_answer_relevance(self, question: str, answer: str) -> Optional[float]:
        prompt = f"""You are grading *answer relevance*.

Question:
{question}

Model answer:
{answer}

On a scale from 0 to 1, where:
- 1.0 means the answer fully addresses the question
- 0.0 means the answer is completely off-topic

Respond with ONLY a single number between 0 and 1."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            return self.extract_score(response.choices[0].message.content)
        except Exception as e:
            return None
    
    def score_groundedness(self, question: str, answer: str, contexts: List[str]) -> Optional[float]:
        prompt = f"""You are grading *groundedness/faithfulness*.

Question:
{question}

Retrieved context:
{chr(10).join(contexts[:5])}

Model answer:
{answer}

On a scale from 0 to 1, where:
- 1.0 means every claim is clearly supported by context
- 0.0 means answer invents information

Respond with ONLY a single number between 0 and 1."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            return self.extract_score(response.choices[0].message.content)
        except Exception as e:
            return None
    
    def score_answer_correctness(self, question: str, answer: str, gold_answer: str) -> Optional[float]:
        prompt = f"""You are grading *answer correctness* for Austrian tax law.

Question:
{question}

Model answer:
{answer}

Reference (gold) answer:
{gold_answer}

On a scale from 0 to 1, where:
- 1.0 means model answer is fully consistent with reference
- 0.0 means model answer contradicts reference

Respond with ONLY a single number between 0 and 1."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            return self.extract_score(response.choices[0].message.content)
        except Exception as e:
            return None


# =============================================================================
# EVALUATION MODULE
# =============================================================================

@dataclass
class EvaluationResult:
    """Results for one case"""
    caseid: str
    predicted_answer: str
    gold_answer: str
    gold_paragraphen: List[str]
    retrieved_chunks: List[Chunk]
    citation_precision: float
    citation_recall: float
    citation_f1: float
    exact_match: bool
    partial_match_rate: float
    mean_reciprocal_rank: float
    doc_relevance: Optional[float]
    answer_relevance: Optional[float]
    groundedness: Optional[float]
    answer_correctness: Optional[float]
    answered: bool


class Evaluator:
    """Evaluation against gold dataset"""
    
    def __init__(self, eval_data_path: Path, use_llm_judge: bool = False, api_key: Optional[str] = None):
        print(f"\n📊 Loading evaluation dataset from: {eval_data_path}")
        
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.cases = data['cases']
        print(f"   Loaded {len(self.cases)} cases")
        
        self.use_llm_judge = use_llm_judge
        self.llm_judge = None
        if use_llm_judge and api_key:
            print(f"   Initializing LLM-as-Judge...")
            self.llm_judge = LLMJudge(api_key)
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from answer text"""
        patterns = [
            r'§\s*\d+[a-z]?\s+Abs\.\s*\d+[a-z]?(?:\s+Z\s*\d+[a-z]?)?(?:\s+lit\.\s*[a-z])?',
            r'Art\.\s*\d+[a-z]?\s+Abs\.\s*\d+[a-z]?(?:\s+Z\s*\d+[a-z]?)?(?:\s+lit\.\s*[a-z])?',
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        normalized = []
        for cit in citations:
            cit = re.sub(r'\s+', ' ', cit.strip())
            normalized.append(cit)
        
        return list(set(normalized))
    
    def normalize_citation(self, citation: str) -> str:
        citation = re.sub(r'\s+UStG\s+1994', '', citation, flags=re.IGNORECASE)
        citation = re.sub(r'\s+', ' ', citation.strip())
        return citation.lower()
    
    def compute_citation_metrics(self, predicted: List[str], 
                                  gold: List[str]) -> Tuple[float, float, float]:
        if not gold:
            return 0.0, 0.0, 0.0
        
        pred_norm = {self.normalize_citation(c) for c in predicted}
        gold_norm = {self.normalize_citation(c) for c in gold}
        
        true_positives = len(pred_norm & gold_norm)
        
        precision = true_positives / len(pred_norm) if pred_norm else 0.0
        recall = true_positives / len(gold_norm) if gold_norm else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def compute_exact_match(self, predicted: List[str], gold: List[str]) -> bool:
        if not gold:
            return len(predicted) == 0
        
        pred_norm = {self.normalize_citation(c) for c in predicted}
        gold_norm = {self.normalize_citation(c) for c in gold}
        
        return pred_norm == gold_norm
    
    def compute_partial_match_rate(self, predicted: List[str], gold: List[str]) -> float:
        if not gold:
            return 1.0 if not predicted else 0.0
        
        pred_norm = {self.normalize_citation(c) for c in predicted}
        gold_norm = {self.normalize_citation(c) for c in gold}
        
        matches = len(pred_norm & gold_norm)
        return matches / len(gold_norm)
    
    def compute_mean_reciprocal_rank(self, predicted: List[str], gold: List[str]) -> float:
        if not gold:
            return 0.0
        
        gold_norm = {self.normalize_citation(c) for c in gold}
        
        for rank, pred_cite in enumerate(predicted, 1):
            pred_norm = self.normalize_citation(pred_cite)
            if pred_norm in gold_norm:
                return 1.0 / rank
        
        return 0.0
    
    def evaluate_case(self, case: Dict, rag_system, retriever) -> EvaluationResult:
        full_question = f"{case['case']}\n\nFrage: {case['frage']}"
        
        retrieved_chunks = retriever.retrieve(full_question, verbose=False)
        
        context = "\n\n".join([
            f"[{i+1}] {c.get_citation()}\n{c.full_context}"
            for i, c in enumerate(retrieved_chunks)
        ])
        
        answer = rag_system.generate_answer(full_question, context)
        
        predicted_cites = self.extract_citations(answer)
        gold_cites = case['paragraphen']
        
        precision, recall, f1 = self.compute_citation_metrics(predicted_cites, gold_cites)
        exact_match = self.compute_exact_match(predicted_cites, gold_cites)
        partial_match = self.compute_partial_match_rate(predicted_cites, gold_cites)
        mrr = self.compute_mean_reciprocal_rank(predicted_cites, gold_cites)
        
        doc_rel, ans_rel, grounded, ans_corr = None, None, None, None
        if self.use_llm_judge and self.llm_judge:
            contexts = [c.full_context for c in retrieved_chunks]
            doc_rel = self.llm_judge.score_document_relevance(full_question, contexts)
            ans_rel = self.llm_judge.score_answer_relevance(full_question, answer)
            grounded = self.llm_judge.score_groundedness(full_question, answer, contexts)
            ans_corr = self.llm_judge.score_answer_correctness(full_question, answer, case['antwort'])
        
        return EvaluationResult(
            caseid=case['caseid'],
            predicted_answer=answer,
            gold_answer=case['antwort'],
            gold_paragraphen=gold_cites,
            retrieved_chunks=retrieved_chunks,
            citation_precision=precision,
            citation_recall=recall,
            citation_f1=f1,
            exact_match=exact_match,
            partial_match_rate=partial_match,
            mean_reciprocal_rank=mrr,
            doc_relevance=doc_rel,
            answer_relevance=ans_rel,
            groundedness=grounded,
            answer_correctness=ans_corr,
            answered=len(answer.strip()) > 0
        )
    
    def evaluate_all(self, rag_system, retriever) -> Dict:
        print(f"\n{'='*70}")
        print(f"STARTING EVALUATION")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, case in enumerate(self.cases, 1):
            print(f"📝 Case {case['caseid']} ({i}/{len(self.cases)})...")
            
            result = self.evaluate_case(case, rag_system, retriever)
            results.append(result)
            
            print(f"   ✅ F1: {result.citation_f1:.2f}")
        
        # Aggregate
        avg_precision = np.mean([r.citation_precision for r in results])
        avg_recall = np.mean([r.citation_recall for r in results])
        avg_f1 = np.mean([r.citation_f1 for r in results])
        avg_partial_match = np.mean([r.partial_match_rate for r in results])
        avg_mrr = np.mean([r.mean_reciprocal_rank for r in results])
        exact_match_count = sum(1 for r in results if r.exact_match)
        
        llm_metrics = {}
        if self.use_llm_judge:
            doc_rels = [r.doc_relevance for r in results if r.doc_relevance is not None]
            ans_rels = [r.answer_relevance for r in results if r.answer_relevance is not None]
            groundeds = [r.groundedness for r in results if r.groundedness is not None]
            ans_corrs = [r.answer_correctness for r in results if r.answer_correctness is not None]
            
            llm_metrics = {
                'avg_doc_relevance': np.mean(doc_rels) if doc_rels else None,
                'avg_answer_relevance': np.mean(ans_rels) if ans_rels else None,
                'avg_groundedness': np.mean(groundeds) if groundeds else None,
                'avg_answer_correctness': np.mean(ans_corrs) if ans_corrs else None,
            }
        
        summary = {
            'total_cases': len(results),
            'answered': sum(1 for r in results if r.answered),
            'avg_citation_precision': avg_precision,
            'avg_citation_recall': avg_recall,
            'avg_citation_f1': avg_f1,
            'exact_match_count': exact_match_count,
            'exact_match_rate': exact_match_count / len(results),
            'avg_partial_match_rate': avg_partial_match,
            'avg_mrr': avg_mrr,
            **llm_metrics,
            'results': results
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Cases: {summary['total_cases']}")
        print(f"\n📊 Citation Metrics:")
        print(f"   Precision: {avg_precision:.3f}")
        print(f"   Recall:    {avg_recall:.3f}")
        print(f"   F1 Score:  {avg_f1:.3f}")
        print(f"\n📊 Additional:")
        print(f"   Exact Match: {exact_match_count}/{len(results)}")
        print(f"   MRR:         {avg_mrr:.3f}")
        
        if self.use_llm_judge:
            print(f"\n📊 LLM-as-Judge:")
            for k, v in llm_metrics.items():
                if v is not None:
                    print(f"   {k}: {v:.3f}")
        
        return summary


# =============================================================================
# MAIN RAG SYSTEM
# =============================================================================

class UStGRAGSystem:
    """Complete RAG system with all fixes applied"""
    
    def __init__(self):
        print(f"\n{'='*70}")
        print(f"🚀 UStG RAG System v13 - MIT PARSER FIXES")
        print(f"{'='*70}")
        print(f"   ✅ UStG: Absatz 1 Fix aktiv")
        print(f"   ✅ Anhang: Absatz 1 Fix aktiv")
        print(f"   ✅ UStR: Deduplizierung aktiv")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n   🖥️ GPU: {gpu_name}")
        
        # Load data with FIXED parsers
        print(f"\n{'='*70}")
        print(f"LOADING DOCUMENTS (FIXED PARSERS)")
        print(f"{'='*70}\n")
        
        print(f"📖 Parsing UStG (mit Absatz 1 Fix)...")
        self.ustg_chunks = parse_ustg_hierarchical(USTG_RTF_PATH)
        
        print(f"\n📖 Parsing Anhang (mit Absatz 1 Fix)...")
        self.anhang_chunks = parse_anhang_hierarchical(ANHANG_RTF_PATH)
        
        print(f"\n📖 Parsing UStR (mit Deduplizierung)...")
        self.ustr_chunks = parse_ustr(USTR_XML_PATH)
        
        total = len(self.ustg_chunks) + len(self.anhang_chunks) + len(self.ustr_chunks)
        
        print(f"\n{'='*70}")
        print(f"✅ LOADED: {total} chunks total")
        print(f"   UStG:   {len(self.ustg_chunks)}")
        print(f"   Anhang: {len(self.anhang_chunks)}")
        print(f"   UStR:   {len(self.ustr_chunks)}")
        print(f"{'='*70}\n")
        
        # Build retriever
        print(f"🔨 Building retrieval system...")
        self.retriever = UltimateUStGRetriever(
            self.ustg_chunks, self.anhang_chunks, self.ustr_chunks
        )
        
        # Initialize LLM
        self.llm = DeepSeekLLM()
        
        print(f"\n{'='*70}")
        print(f"✅ SYSTEM READY!")
        print(f"{'='*70}\n")
    
    def answer_question(self, question: str) -> Tuple[str, List[Chunk]]:
        """Answer a question"""
        print(f"\n❓ Frage: {question}")
        
        chunks = self.retriever.retrieve(question, verbose=True)
        
        context = "\n\n".join([
            f"[{i+1}] {c.get_citation()}\n{c.full_context}"
            for i, c in enumerate(chunks)
        ])
        
        answer = self.llm.generate_answer(question, context)
        
        return answer, chunks


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_evaluation(system, use_llm_judge: bool = False):
    """Run evaluation mode"""
    evaluator = Evaluator(
        EVAL_DATA_PATH, 
        use_llm_judge=use_llm_judge,
        api_key="sk-9aa1b071f49145b5a6b9a257847a6964" if use_llm_judge else None
    )
    summary = evaluator.evaluate_all(system.llm, system.retriever)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_type = "llm_judge" if use_llm_judge else "citation"
    results_file = SCRIPT_DIR / f"eval_results_{eval_type}_{timestamp}.json"
    
    results_data = {
        'timestamp': timestamp,
        'evaluation_type': eval_type,
        'parser_version': 'v13_fixed',
        'fixes_applied': ['ustg_absatz1', 'anhang_absatz1', 'ustr_dedup'],
        'chunk_counts': {
            'ustg': len(system.ustg_chunks),
            'anhang': len(system.anhang_chunks),
            'ustr': len(system.ustr_chunks)
        },
        'total_cases': summary['total_cases'],
        'metrics': {
            'citation_precision': summary['avg_citation_precision'],
            'citation_recall': summary['avg_citation_recall'],
            'citation_f1': summary['avg_citation_f1'],
            'exact_match_rate': summary['exact_match_rate'],
            'mrr': summary['avg_mrr']
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_file}")


def show_parser_stats(system):
    """Show detailed parser statistics"""
    print(f"\n{'='*70}")
    print(f"PARSER STATISTICS (FIXED VERSION)")
    print(f"{'='*70}\n")
    
    # UStG Stats
    print(f"📊 UStG:")
    ustg_paras = defaultdict(list)
    for chunk in system.ustg_chunks:
        ustg_paras[chunk.metadata.get('paragraph')].append(chunk.metadata.get('absatz'))
    
    abs1_count = sum(1 for p, a in ustg_paras.items() if '1' in a)
    print(f"   Paragraphen: {len(ustg_paras)}")
    print(f"   Mit Absatz 1: {abs1_count}")
    print(f"   Total Chunks: {len(system.ustg_chunks)}")
    
    # Check critical paragraphs
    critical = ['1', '6', '10', '11', '12']
    print(f"\n   Kritische §§:")
    for para in critical:
        has = '1' in ustg_paras.get(para, [])
        status = "✅" if has else "❌"
        print(f"     {status} § {para} Abs. 1")
    
    # Anhang Stats
    print(f"\n📊 Anhang:")
    anhang_arts = defaultdict(list)
    for chunk in system.anhang_chunks:
        anhang_arts[chunk.metadata.get('artikel')].append(chunk.metadata.get('absatz'))
    
    abs1_count = sum(1 for a, ab in anhang_arts.items() if '1' in ab)
    print(f"   Artikel: {len(anhang_arts)}")
    print(f"   Mit Absatz 1: {abs1_count}")
    print(f"   Total Chunks: {len(system.anhang_chunks)}")
    
    # UStR Stats
    print(f"\n📊 UStR:")
    rz_set = set()
    for chunk in system.ustr_chunks:
        rz_set.add(chunk.metadata.get('randzahl'))
    
    print(f"   Unique Randzahlen: {len(rz_set)}")
    print(f"   Total Chunks: {len(system.ustr_chunks)}")
    print(f"   Duplikate: 0 (dedupliziert)")


def ask_question_mode(system):
    """Interactive Q&A mode"""
    print(f"\n💬 Fragen stellen (oder 'back' für Menü):\n")
    
    while True:
        question = input("❓ Frage: ").strip()
        
        if question.lower() in ['back', 'menu', 'm']:
            break
        
        if question.lower() in ['exit', 'quit', 'q']:
            return True
        
        if not question:
            continue
        
        answer, chunks = system.answer_question(question)
        
        print(f"\n{'─'*70}")
        print(f"📝 ANTWORT:")
        print(f"{'─'*70}")
        print(answer)
        
        print(f"\n{'─'*70}")
        print(f"📚 QUELLEN ({len(chunks)}):")
        print(f"{'─'*70}")
        
        for i, chunk in enumerate(chunks[:5], 1):
            cite = chunk.get_citation()
            preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            print(f"[{i}] {cite}")
            print(f"    {preview}\n")
        
        print(f"{'='*70}\n")
    
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    system = UStGRAGSystem()
    
    # Show parser stats
    show_parser_stats(system)
    
    # Interactive menu
    while True:
        print(f"\n{'='*70}")
        print(f"🎯 UStG RAG SYSTEM v13 - HAUPTMENÜ")
        print(f"{'='*70}\n")
        print(f"  (1) 💬 Fragen stellen")
        print(f"  (2) 📊 Evaluation - Citation Metrics")
        print(f"  (3) 🤖 Evaluation - LLM-as-Judge")
        print(f"  (4) 📈 Parser-Statistiken anzeigen")
        print(f"  (5) 🚪 Beenden")
        print(f"\n{'='*70}\n")
        
        choice = input("Auswahl (1-5): ").strip()
        
        if choice == '1':
            if ask_question_mode(system):
                break
        
        elif choice == '2':
            print(f"\n🎯 Running Citation-Based Evaluation...")
            run_evaluation(system, use_llm_judge=False)
            input("\n📌 Enter drücken für Menü...")
        
        elif choice == '3':
            print(f"\n🤖 Running LLM-as-Judge Evaluation...")
            confirm = input("   Fortfahren? (y/n): ").strip().lower()
            if confirm == 'y':
                run_evaluation(system, use_llm_judge=True)
            input("\n📌 Enter drücken für Menü...")
        
        elif choice == '4':
            show_parser_stats(system)
            input("\n📌 Enter drücken für Menü...")
        
        elif choice == '5':
            print("\n👋 Auf Wiedersehen!")
            break
        
        else:
            print(f"⚠️ Ungültige Auswahl: '{choice}'")


if __name__ == "__main__":
    main()
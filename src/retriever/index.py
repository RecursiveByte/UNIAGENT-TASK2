import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s

FULL_DOC_CHARS = 6000
CHUNK_CHARS = 1500

PARAGRAPH_SPLIT = re.compile(r"\n{2,}")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

@dataclass
class IndexedChunk:
    doc_id: str
    chunk_id: int
    title: str
    url: str
    text: str


@dataclass
class CorpusIndex:
    bm25: bm25s.BM25
    chunks: list[IndexedChunk]


def hard_split(text: str, size: int) -> list[str]:
    pieces = []
    current = ""
    for sentence in SENTENCE_SPLIT.split(text):
        while len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= FULL_DOC_CHARS:
        return [text]

    chunks = []
    current = ""
    for paragraph in PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = [paragraph] if len(paragraph) <= CHUNK_CHARS else hard_split(paragraph, CHUNK_CHARS)
        for part in parts:
            if current and len(current) + len(part) + 2 > CHUNK_CHARS:
                chunks.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
    if current:
        chunks.append(current)
    return chunks


def clean(text: str) -> str:
    return text.replace("\u00ad", "")


def load_chunks(corpus_root: Path, corpus_name: str) -> list[IndexedChunk]:
    path = corpus_root / corpus_name / "documents.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    chunks = []
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = clean(record.get("text") or record.get("content", ""))
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            pieces = [piece for piece in split_text(text) if piece]
            for chunk_id, piece in enumerate(pieces):
                chunks.append(
                    IndexedChunk(
                        doc_id=record["doc_id"],
                        chunk_id=chunk_id,
                        title=clean(record.get("title", "")),
                        url=record.get("url", ""),
                        text=piece,
                    )
                )
    return chunks


def index_text(chunk: IndexedChunk) -> str:
    if chunk.chunk_id == 0:
        return chunk.text
    return f"{chunk.title}\n{chunk.text}"


def build_index(corpus_root: Path, corpus_name: str) -> CorpusIndex:
    chunks = load_chunks(corpus_root, corpus_name)
    tokenized = bm25s.tokenize([index_text(chunk) for chunk in chunks], show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokenized, show_progress=False)
    return CorpusIndex(bm25=bm25, chunks=chunks)
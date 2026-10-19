from dataclasses import dataclass
from pathlib import Path

import bm25s

from .index import CorpusIndex, build_index

_index_cache: dict[tuple[str, str], CorpusIndex] = {}


@dataclass
class SearchResult:
    doc_id: str
    title: str
    url: str
    text: str
    score: float
    position: int


def get_index(corpus_root: Path, corpus_name: str) -> CorpusIndex:
    key = (str(corpus_root), corpus_name)
    if key not in _index_cache:
        _index_cache[key] = build_index(corpus_root, corpus_name)
    return _index_cache[key]


def search_corpus(corpus_root: Path, corpus_name: str, query: str, top_k: int = 3) -> list[SearchResult]:
    index = get_index(corpus_root, corpus_name)
    query_tokens = bm25s.tokenize([query], show_progress=False)
    k = min(top_k, len(index.chunks))
    positions, scores = index.bm25.retrieve(query_tokens, k=k, show_progress=False)

    results = []
    for position, score in zip(positions[0], scores[0]):
        if score <= 0:
            continue
        chunk = index.chunks[int(position)]
        results.append(
            SearchResult(
                doc_id=chunk.doc_id,
                title=chunk.title,
                url=chunk.url,
                text=chunk.text,
                score=float(score),
                position=int(position),
            )
        )
    return results


def get_context(corpus_root: Path, corpus_name: str, result: SearchResult, radius: int = 1) -> str:
    chunks = get_index(corpus_root, corpus_name).chunks
    start = end = result.position
    while start > result.position - radius and start > 0 and chunks[start - 1].doc_id == result.doc_id:
        start -= 1
    while end < result.position + radius and end + 1 < len(chunks) and chunks[end + 1].doc_id == result.doc_id:
        end += 1
    return "\n\n".join(chunk.text for chunk in chunks[start:end + 1])
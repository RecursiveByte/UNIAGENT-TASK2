import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from retriever import SearchResult, get_context, search_corpus

QUERIES_PER_ROUND = 3
MAX_ROUNDS = 2
TOP_K = 3
MAX_CANDIDATES = 3
RETURN_DOCS = 2
CONTEXT_RADIUS = 1
RELEVANCE_THRESHOLD = 95
REQUIRE_QUOTE = True

GRADE_PROMPT = """You grade retrieved policy text.

NEED: {need}

Rate each document from 0 to 100 for how completely it answers the NEED. 100 means it states the exact rule, limit or deadline that is needed. For each document copy one exact sentence that supports the rating into "quote". If no sentence answers the NEED, use score 0 and an empty quote.

If no document scores {threshold} or higher, also write {count} new search queries for the NEED: short German keyword phrases with the same meaning but different words than the queries already tried: {used_queries}. If a document scores {threshold} or higher, use an empty list.

Reply with JSON only:
{{"scores": {{"D1": {{"score": 0, "quote": ""}}}}, "new_queries": []}}

DOCUMENTS
{documents}"""


@dataclass
class GradedDoc:
    doc_id: str
    title: str
    context: str
    score: int
    quote: str


@dataclass
class SearchOutcome:
    docs: list[GradedDoc]
    rounds: int
    best_score: int
    last_event_id: str | None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def quote_found(quote: str, text: str) -> bool:
    return bool(quote.strip()) and normalize(quote) in normalize(text)


def to_score(value) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 0


def parse_json(content: str) -> dict | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def merge_results(per_query: list[list[SearchResult]]) -> list[SearchResult]:
    merged = []
    depth = max((len(results) for results in per_query), default=0)
    for rank in range(depth):
        for results in per_query:
            if rank >= len(results):
                continue
            candidate = results[rank]
            overlaps = any(
                kept.doc_id == candidate.doc_id and abs(kept.position - candidate.position) <= 2 * CONTEXT_RADIUS
                for kept in merged
            )
            if not overlaps:
                merged.append(candidate)
    return merged


def grade_round(llm, need: str, used_queries: list[str], candidates: list[SearchResult], contexts: list[str]):
    labels = {f"D{i}": (doc, contexts[i - 1]) for i, doc in enumerate(candidates, 1)}
    documents = "\n\n".join(f"{label}: {doc.title}\n{context}" for label, (doc, context) in labels.items()) or "(none)"
    prompt = GRADE_PROMPT.format(
        need=need,
        threshold=RELEVANCE_THRESHOLD,
        count=QUERIES_PER_ROUND,
        used_queries=json.dumps(used_queries, ensure_ascii=False),
        documents=documents,
    )
    parsed = parse_json(llm.invoke([HumanMessage(content=prompt)]).content or "")
    if parsed is None:
        return None

    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    graded = []
    for label, (doc, context) in labels.items():
        entry = raw_scores.get(label)
        if not isinstance(entry, dict):
            entry = {"score": entry, "quote": ""}
        quote = str(entry.get("quote") or "")
        score = to_score(entry.get("score"))
        if REQUIRE_QUOTE and not quote_found(quote, context):
            score = 0
        graded.append(GradedDoc(doc.doc_id, doc.title, context, score, quote))

    raw_queries = parsed.get("new_queries")
    if not isinstance(raw_queries, list):
        raw_queries = []
    new_queries = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()][:QUERIES_PER_ROUND]
    return graded, new_queries


def corrective_search(llm, corpus_root, corpus: str, need: str, queries: list[str], logger, case_id: str, parent_event_id):
    best: dict[str, GradedDoc] = {}
    fallback: list[GradedDoc] = []
    current = list(queries)
    used = list(queries)
    parent = parent_event_id
    best_score = 0
    rounds = 0

    for round_number in range(1, MAX_ROUNDS + 1):
        rounds = round_number
        per_query = [search_corpus(corpus_root, corpus, query, top_k=TOP_K) for query in current]
        candidates = merge_results(per_query)[:MAX_CANDIDATES]

        parent = logger.log_tool_call(
            case_id, parent, "bm25_search",
            {"queries": current, "corpus": corpus, "round": round_number},
            {"doc_ids": [c.doc_id for c in candidates]},
        )

        if not candidates:
            break

        contexts = [get_context(corpus_root, corpus, c, CONTEXT_RADIUS) for c in candidates]

        error_message = None
        try:
            graded_round = grade_round(llm, need, used, candidates, contexts)
            if graded_round is None:
                error_message = "grading response was not valid JSON"
        except Exception as error:
            graded_round = None
            error_message = f"grading call failed: {error}"

        if graded_round is None:
            fallback = [GradedDoc(c.doc_id, c.title, ctx, -1, "") for c, ctx in list(zip(candidates, contexts))[:RETURN_DOCS]]
            parent = logger.log_error(case_id, parent, error_message, event_type="model_call")
            break

        graded, new_queries = graded_round
        parent = logger.log_model_call(
            case_id, parent,
            {"task": "relevance_grading", "round": round_number, "documents": len(graded)},
            {"scores": {d.doc_id: d.score for d in graded}, "new_queries": new_queries},
        )

        for doc in graded:
            previous = best.get(doc.doc_id)
            if previous is None or doc.score > previous.score:
                best[doc.doc_id] = doc

        best_score = max((d.score for d in best.values()), default=0)
        if best_score >= RELEVANCE_THRESHOLD or not new_queries:
            break

        current = new_queries
        used.extend(new_queries)

    ranked = sorted(best.values(), key=lambda d: d.score, reverse=True)
    docs = ranked[:RETURN_DOCS] or fallback
    return SearchOutcome(docs, rounds, best_score, parent)


def format_outcome(outcome: SearchOutcome) -> str:
    if not outcome.docs:
        return "No relevant policy found for this need."

    lines = []
    if outcome.docs[0].score >= 0 and outcome.best_score < RELEVANCE_THRESHOLD:
        lines.append(f"Note: no passage fully answered the need after {outcome.rounds} rounds. Best relevance {outcome.best_score}/100.")

    for doc in outcome.docs:
        label = f" (relevance {doc.score})" if doc.score >= 0 else ""
        lines.append(f"[{doc.doc_id}] {doc.title}{label}\n{doc.context}")

    return "\n\n".join(lines)
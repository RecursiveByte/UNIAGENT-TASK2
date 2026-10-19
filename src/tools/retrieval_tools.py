from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .policy_search import QUERIES_PER_ROUND, corrective_search, format_outcome

VALID_CORPORA = ("university-kassel-public-de", "university-kassel-public-en", "hessian-law-de")


def make_search_policy_tool(corpus_root: Path, llm):
    @tool
    def search_policy(
        need: str,
        queries: list[str],
        corpus: str = "university-kassel-public-de",
        state: Annotated[dict, InjectedState] = None,
    ) -> str:
        """Find the policy text that answers one specific need.
        need: one sentence naming the exact rule, limit or deadline required.
        queries: exactly 3 short German keyword phrases with the same meaning but different words.
        corpus: university-kassel-public-de, university-kassel-public-en or hessian-law-de."""

        if corpus not in VALID_CORPORA:
            return f"Invalid corpus '{corpus}'. Use one of: {', '.join(VALID_CORPORA)}"

        if isinstance(queries, str):
            queries = [queries]
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:QUERIES_PER_ROUND]
        if not cleaned:
            return "Provide 3 non-empty search queries."

        logger = state["logger"]
        case_id = state["case_id"]

        outcome = corrective_search(llm, corpus_root, corpus, need, cleaned, logger, case_id, state.get("last_event_id"))

        logger.log_tool_call(
            case_id, outcome.last_event_id, "search_policy",
            {"need": need, "queries": cleaned, "corpus": corpus},
            {"rounds": outcome.rounds, "best_score": outcome.best_score, "doc_ids": [d.doc_id for d in outcome.docs]},
        )

        return format_outcome(outcome)

    return search_policy
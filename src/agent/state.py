from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from app_logging import EventLogger


class AgentState(TypedDict):
    case_id: str
    case_type: str
    documents: dict[str, str]
    messages: Annotated[list, add_messages]
    decision: str | None
    begruendung: str | None
    logger: EventLogger
    last_event_id: str | None
    tool_call_count: int
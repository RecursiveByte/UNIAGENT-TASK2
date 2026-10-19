import re
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from .prompts import SYSTEM_PROMPT, build_case_prompt
from .state import AgentState

DECISION_PATTERN = re.compile(r"DECISION:\s*(angenommen|abgelehnt)", re.IGNORECASE)

def prepare_messages(state: AgentState) -> dict:
    system = SystemMessage(content=SYSTEM_PROMPT.format(case_type=state["case_type"]))
    human = HumanMessage(content=build_case_prompt(state["case_type"], state["documents"]))

    event_id = state["logger"].log(
        state["case_id"], "input", parent_event_id=None,
        input={"case_type": state["case_type"], "documents": list(state["documents"].keys())},
    )

    return {"messages": [system, human], "last_event_id": event_id, "tool_call_count": 0}


def make_call_model(llm_with_tools):
    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])

        event_id = state["logger"].log_model_call(
            state["case_id"], state["last_event_id"],
            prompt={"message_count": len(state["messages"])},
            response={"content": response.content, "tool_calls": getattr(response, "tool_calls", None)},
        )

        return {"messages": [response], "last_event_id": event_id}

    return call_model        
        

def extract_decision(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content or ""
    match = DECISION_PATTERN.search(content)
    decision = match.group(1).lower() if match else "abgelehnt"

    state["logger"].log_decision(
        state["case_id"], state["last_event_id"],
        decision=decision, answer=content, evidence=[],
    )

    return {"decision": decision, "begruendung": content.strip()}

def make_tool_node(tools: list) -> ToolNode:
    base_node = ToolNode(tools)

    def tool_node(state: AgentState) -> dict:
        result = base_node.invoke(state)
        return {**result, "tool_call_count": state["tool_call_count"] + 1}

    return tool_node


def make_final_call_model(llm):
    def final_call_model(state: AgentState) -> dict:
        forced = HumanMessage(content="You have used all available searches. Based on everything gathered so far, write your reasoning and end with the DECISION line now.")
        response = llm.invoke(state["messages"] + [forced])

        event_id = state["logger"].log_model_call(
            state["case_id"], state["last_event_id"],
            prompt={"message_count": len(state["messages"]) + 1, "forced_final": True},
            response={"content": response.content},
        )

        return {"messages": [forced, response], "last_event_id": event_id}

    return final_call_model
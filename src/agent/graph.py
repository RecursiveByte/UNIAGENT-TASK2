from langgraph.graph import END, StateGraph

from .nodes import extract_decision, make_call_model, make_final_call_model, make_tool_node, prepare_messages
from .state import AgentState

MAX_TOOL_CALLS = 2

def has_tool_calls(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        if state["tool_call_count"] < MAX_TOOL_CALLS:
            return "tools"
        return "final_call"
    return "decide"


def build_graph(llm, tools: list):
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_messages)
    graph.add_node("call_model", make_call_model(llm_with_tools))
    graph.add_node("tools", make_tool_node(tools))
    graph.add_node("final_call", make_final_call_model(llm))
    graph.add_node("decide", extract_decision)

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "call_model")
    graph.add_conditional_edges("call_model", has_tool_calls, {"tools": "tools", "final_call": "final_call", "decide": "decide"})
    graph.add_edge("tools", "call_model")
    graph.add_edge("final_call", "decide")
    graph.add_edge("decide", END)

    return graph.compile()
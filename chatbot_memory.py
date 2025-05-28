from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human"""
    human_response = interrupt({"query": query})
    return human_response["data"]


search_tool = TavilySearch(max_results=2)
tools = [search_tool, human_assistance]

llm = init_chat_model("google_genai:gemini-2.0-flash")

llm_with_tool = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tool.invoke(state["messages"])
    print("message caslls", len(message.tool_calls))
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()


def call_llm():
    user_input = input("User: ")
    stream_graph_updates(user_input)


def human_response(response, config):
    human_command = Command(resume={"data": response})
    for event in graph.stream(human_command, config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    call_llm()
    print("intrupption")
    human_response(
        "Hey there, I am here. What kind of assistance do you want.", config)

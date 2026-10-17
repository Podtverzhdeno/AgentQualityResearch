from __future__ import annotations
import os
import random
import threading
import time
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openai import RateLimitError
from .environment import require_openrouter_key
from .models import AgentConfig
from .tools import ToolRuntime


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int


class ModelRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            if delay:
                time.sleep(delay)
            self.next_allowed = time.monotonic() + self.interval


_LIMITERS: dict[tuple[str, int], ModelRateLimiter] = {}


def model_limiter(model: str, requests_per_minute: int) -> ModelRateLimiter:
    key = (model, requests_per_minute)
    if key not in _LIMITERS:
        _LIMITERS[key] = ModelRateLimiter(requests_per_minute)
    return _LIMITERS[key]


def build_agent(config: AgentConfig, runtime: ToolRuntime):
    tools = runtime.build()
    model = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=require_openrouter_key(),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    ).bind_tools(tools)
    limiter = model_limiter(config.model, config.requests_per_minute)

    def invoke_model(messages):
        for attempt in range(config.max_rate_limit_retries + 1):
            limiter.wait()
            try:
                return model.invoke(messages)
            except RateLimitError:
                if attempt == config.max_rate_limit_retries:
                    raise
                delay = min(60.0, 2 ** min(attempt + 1, 6)) + random.uniform(0, 1)
                print(f"  OpenRouter 429; retry {attempt + 1}/{config.max_rate_limit_retries} in {delay:.1f}s", flush=True)
                time.sleep(delay)

    def agent_node(state: AgentState):
        response = invoke_model([SystemMessage(config.system_prompt), *state["messages"]])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def route(state: AgentState):
        message = state["messages"][-1]
        if getattr(message, "tool_calls", None) and state["iterations"] < config.max_iterations:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def invoke_agent(graph, user_id: str, message: str, callbacks: list | None = None):
    prompt = f"Authenticated user_id: {user_id}\nUser request: {message}"
    return graph.invoke(
        {"messages": [HumanMessage(prompt)], "iterations": 0},
        config={"callbacks": callbacks or [], "recursion_limit": 30},
    )


def final_text(result: dict) -> str:
    message = result["messages"][-1]
    return str(message.content if isinstance(message, AIMessage) else message)

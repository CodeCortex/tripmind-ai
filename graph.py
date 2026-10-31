
from contextlib import ExitStack

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent, final_response_agent, flight_agent, hotel_agent,
    human_approval_agent, itinerary_agent, supervisor_agent, weather_agent,
)
from config import DATABASE_URL
from state import TravelState
from logging_config import logger

import logging
log = logging.getLogger(__name__)

AGENT_ORDER = [
    "flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent",
]
ROUTE_MAP = {agent: agent for agent in AGENT_ORDER}
ROUTE_MAP["end"] = END


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents") or []
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("request_allowed", False):
        log.warning("Graph stopping: supervisor rejected request")
        return "end"
    selected = _selected_agents(state)
    target = selected[0] if selected else "end"
    log.info("Graph routing after supervisor | next_node=%s", target)
    return target


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent in selected:
                log.info("Graph routing | current_node=%s | next_node=%s", current_agent, next_agent)
                return next_agent
        target = "itinerary_agent" if state.get("request_allowed", False) else "end"
        log.info("Graph routing | current_node=%s | next_node=%s", current_agent, target)
        return target
    return route


def build_graph():
    log.info("Building Tripmind-AI LangGraph workflow")
    graph = StateGraph(TravelState)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
    for agent in AGENT_ORDER[:-1]:
        graph.add_conditional_edges(agent, route_after_agent(agent), ROUTE_MAP)
    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_response")
    graph.add_edge("final_response", END)

    if DATABASE_URL:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        conn = psycopg.connect(DATABASE_URL, autocommit=True, prepare_threshold=0)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
    else:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


app = build_graph()

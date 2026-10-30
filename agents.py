import asyncio
import json
import re
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import current_weather, forecast, tavily_search
from state import TravelState
from logging_config import logger


log = logging.getLogger(__name__)
llm = get_llm()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _llm_text(system: str, prompt: str, *, max_tokens: int = 600) -> str:
    """Invoke the configured model with a bounded response length."""
    log.debug("LLM invoke started | max_tokens=%s | prompt_chars=%s", max_tokens, len(prompt))
    try:
        response = llm.bind(max_tokens=max_tokens).invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ]
        )
    except Exception:
        log.exception("LLM invoke failed | max_tokens=%s | prompt_chars=%s", max_tokens, len(prompt))
        raise
    content = response.content
    log.debug("LLM invoke completed | response_chars=%s", len(content) if isinstance(content, str) else -1)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _compact(value: Any, limit: int = 1200) -> str:
    """Serialize and truncate context before passing it to another agent."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _json_from_llm(text: str) -> dict:
    """Parse JSON returned by an LLM, including JSON wrapped in code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON object.")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the model.")
    return parsed


def _run_async(coro):
    """Run MCP async calls from synchronous LangGraph nodes."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A sync node should normally not run inside an active event loop.
    # Fail with a useful message rather than silently creating nested loops.
    coro.close()
    raise RuntimeError(
        "An MCP call was invoked inside an active event loop. "
        "Use async LangGraph nodes or call this graph from a synchronous context."
    )


def _count_calls(state: TravelState, increment: int = 1) -> int:
    return int(state.get("llm_calls", 0)) + increment


def _destination_city(constraints: dict) -> str:
    """Choose a city for weather APIs when the destination is a country."""
    explicit_city = constraints.get("weather_city") or constraints.get("destination_city")
    if explicit_city:
        return str(explicit_city)

    destination = str(constraints.get("destination") or "Tokyo").strip()
    # A small fallback for common country-level destination requests.
    # The supervisor should prefer a city when the user has specified one.
    country_defaults = {
        "japan": "Tokyo",
        "india": "Delhi",
        "united states": "New York",
        "usa": "New York",
        "united kingdom": "London",
        "uk": "London",
        "france": "Paris",
        "italy": "Rome",
        "spain": "Madrid",
        "thailand": "Bangkok",
        "singapore": "Singapore",
        "uae": "Dubai",
        "united arab emirates": "Dubai",
        "australia": "Sydney",
        "germany": "Berlin",
        "south korea": "Seoul",
    }
    return country_defaults.get(destination.casefold(), destination)


# ---------------------------------------------------------------------------
# Supervisor and input guardrail
# ---------------------------------------------------------------------------

def _looks_like_travel_request(query: str) -> bool:
    """Deterministic first-pass guardrail; avoids rejecting valid requests on malformed LLM JSON."""
    text = query.casefold()
    travel_terms = (
        "travel", "trip", "itinerary", "vacation", "holiday", "visit",
        "flight", "airfare", "hotel", "accommodation", "tour", "tourism",
        "backpack", "destination", "traveling", "travelling", "honeymoon",
    )
    return any(term in text for term in travel_terms)


def _extract_constraints(query: str) -> dict:
    """Extract obvious constraints deterministically, then let the LLM enrich them."""
    text = query.strip()
    lower = text.casefold()
    constraints = {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
        "weather_city": "",
    }

    duration_match = re.search(
        r"\b(\d{1,2})\s*[- ]?\s*(day|days|week|weeks|night|nights)\b", lower
    )
    if duration_match:
        constraints["duration"] = duration_match.group(0)

    budget_match = re.search(
        r"(?:under|below|within|budget(?:\s+of)?|less than|maximum|max\.?)\s*"
        r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|k)?",
        lower,
    )
    if budget_match:
        amount = budget_match.group(1).replace(",", "")
        unit = (budget_match.group(2) or "").lower()
        if unit in {"lakh", "lakhs", "lac", "lacs"}:
            constraints["budget"] = f"INR {float(amount) * 100000:g} (under/budget limit)"
        elif unit == "k":
            constraints["budget"] = f"INR {float(amount) * 1000:g} (under/budget limit)"
        else:
            constraints["budget"] = f"INR {amount} (under/budget limit)"

    # Common country/city names; the LLM can add other destinations.
    destinations = [
        "Japan", "Tokyo", "Kyoto", "Osaka", "India", "Delhi", "Mumbai",
        "Thailand", "Bangkok", "Singapore", "France", "Paris", "Italy",
        "Rome", "Spain", "Madrid", "United Kingdom", "London", "Germany",
        "Berlin", "Australia", "Sydney", "United States", "New York", "Dubai",
        "South Korea", "Seoul", "Indonesia", "Bali", "Malaysia", "Kuala Lumpur",
    ]
    for destination in sorted(destinations, key=len, reverse=True):
        if re.search(rf"\b{re.escape(destination.casefold())}\b", lower):
            constraints["destination"] = destination
            if destination.casefold() not in {
                "tokyo", "kyoto", "osaka", "delhi", "mumbai", "bangkok",
                "singapore", "paris", "rome", "madrid", "london", "berlin",
                "sydney", "new york", "dubai", "seoul", "bali", "kuala lumpur",
            }:
                constraints["weather_city"] = ""
            else:
                constraints["weather_city"] = destination
            break

    if any(word in lower for word in ("budget", "cheap", "affordable", "backpack")):
        constraints["travel_style"] = "budget-conscious"
    elif any(word in lower for word in ("luxury", "five-star", "5-star")):
        constraints["travel_style"] = "luxury"

    return constraints


def supervisor_agent(state: TravelState):
    query = str(state.get("user_query", "")).strip()
    if not query:
        log.warning("Supervisor rejected empty request")
        message = "Please enter a travel request with a destination or travel goal."
        return {
            "request_allowed": False,
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": message,
            "final_response": message,
            "messages": [AIMessage(content=message)],
            "llm_calls": _count_calls(state, 0),
        }

    # Fast deterministic guardrail avoids false rejection when a model returns
    # malformed JSON. Clearly unrelated requests stop before any specialist runs.
    if not _looks_like_travel_request(query):
        log.info("Supervisor rejected non-travel request | query_chars=%s", len(query))
        message = (
            "I can help with travel planning, destinations, flights, hotels, "
            "weather for a trip, itineraries, and travel budgets. Please rephrase "
            "your request as a travel-planning question."
        )
        return {
            "request_allowed": False,
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": message,
            "final_response": message,
            "messages": [AIMessage(content=message)],
            "llm_calls": _count_calls(state, 0),
        }

    constraints = _extract_constraints(query)
    log.info(
        "Supervisor accepted travel intent | extracted_destination=%s | duration=%s | budget=%s",
        constraints.get("destination") or "unspecified",
        constraints.get("duration") or "unspecified",
        constraints.get("budget") or "unspecified",
    )
    supervisor_prompt = f"""
Route this valid travel-planning request to the necessary specialist agents.
Available agents: flight_agent, hotel_agent, weather_agent, budget_agent, itinerary_agent.
Return strict JSON only with keys selected_agents, trip_constraints, reasoning.
trip_constraints keys: destination, origin, duration, budget, travel_style,
special_preferences, weather_city.
Preserve all user constraints. Do not replace a non-empty extracted value with an empty value.
Use empty strings for unknown values and do not invent details.
Deterministic extraction: {json.dumps(constraints, ensure_ascii=False)}
Request: {_compact(query, 900)}
"""
    try:
        parsed = _json_from_llm(
            _llm_text(
                "You are a concise multi-agent travel supervisor. Return strict JSON only.",
                supervisor_prompt,
                max_tokens=400,
            )
        )
        allowed_agents = {
            "flight_agent", "hotel_agent", "weather_agent",
            "budget_agent", "itinerary_agent",
        }
        selected = [
            agent for agent in parsed.get("selected_agents", [])
            if agent in allowed_agents
        ]
        # Always run the budget assessment and itinerary for a normal trip plan.
        for required in ("flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"):
            if required not in selected:
                selected.append(required)

        model_constraints = parsed.get("trip_constraints", {})
        if isinstance(model_constraints, dict):
            for key, value in model_constraints.items():
                if key in constraints and value not in ("", None, []):
                    constraints[key] = value
        reasoning = str(parsed.get("reasoning") or "Travel request accepted; specialist agents selected.")
    except Exception as exc:
        # Keep a valid travel request usable even if routing JSON is malformed or the LLM fails.
        log.warning("Supervisor LLM routing failed; using fallback | error_type=%s", type(exc).__name__, exc_info=True)
        selected = [
            "flight_agent", "hotel_agent", "weather_agent",
            "budget_agent", "itinerary_agent",
        ]
        reasoning = "Travel request accepted using deterministic constraint extraction and the default agent workflow."

    # Preserve obvious duration/budget/destination extracted from the original user text.
    extracted = _extract_constraints(query)
    for key, value in extracted.items():
        if value and (key not in constraints or not constraints[key]):
            constraints[key] = value

    log.info("Supervisor routing complete | selected_agents=%s", selected)
    return {
        "request_allowed": True,
        "selected_agents": selected,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor accepted the travel request and created the agent plan.")],
        "llm_calls": _count_calls(state, 1),
    }


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

def flight_agent(state: TravelState):
    """Generate cautious flight guidance without calling restricted AviationStack endpoints."""
    log.info("flight_agent started")
    query = str(state.get("user_query", ""))
    constraints = state.get("trip_constraints", {})

    prompt = f"""
Give concise flight-planning guidance for this trip.
Request: {_compact(query, 350)}
Constraints: {_compact(constraints, 450)}

Include likely route/airport choices, typical duration, broad fare ranges only
if reasonably known, seasonality, and booking tips. Clearly label prices as
estimates and say live fares are not verified. Do not claim a flight is
available or booked. Do not invent exact schedules or live prices.
Keep the response under 350 words.
"""
    result = _llm_text(
        "You are a cautious flight-planning specialist. Distinguish estimates from live data.",
        prompt,
        max_tokens=550,
    )
    log.info("flight_agent completed | result_chars=%s", len(result))
    log.info("flight agent result= %", result);
    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight guidance completed; live fares were not verified.")],
        "llm_calls": _count_calls(state),
    }


def hotel_agent(state: TravelState):
    log.info("hotel_agent started")
    query = str(state.get("user_query", ""))
    constraints = state.get("trip_constraints", {})
    search_query = (
        f"Budget accommodation areas and hotel options for "
        f"{_compact(constraints, 350)}. User request: {_compact(query, 350)}"
    )

    try:
        result = _run_async(tavily_search(search_query))
        result_text = _compact(result, 2200)
    except Exception as exc:
        log.warning("hotel_agent search failed | error_type=%s", type(exc).__name__, exc_info=True)
        result_text = (
            "Live accommodation search unavailable. "
            f"Reason: {type(exc).__name__}. Recommend comparing current rates "
            "on reputable booking sites; do not treat any hotel price as confirmed."
        )

    log.info("hotel_agent completed | result_chars=%s", len(result_text))
    return {
        "hotel_results": result_text,
        "messages": [AIMessage(content="Accommodation research completed.")],
    }


def weather_agent(state: TravelState):
    log.info("weather_agent started")
    constraints = state.get("trip_constraints", {})
    city = _destination_city(constraints)

    try:
        current = _run_async(current_weather(city))
        outlook = _run_async(forecast(city))
        result = (
            f"Weather location: {city}\n"
            f"Current conditions: {_compact(current, 700)}\n"
            f"Forecast: {_compact(outlook, 1000)}\n"
            "Note: forecast availability and dates depend on the weather provider."
        )
    except Exception as exc:
        log.warning("weather_agent lookup failed | city=%s | error_type=%s", city, type(exc).__name__, exc_info=True)
        result = (
            f"Weather lookup for {city} failed ({type(exc).__name__}). "
            "Do not infer current weather. Check a live weather source before travel."
        )

    log.info("weather_agent completed | city=%s | result_chars=%s", city, len(result))
    return {
        "weather_results": result,
        "messages": [AIMessage(content=f"Weather lookup attempted for {city}.")],
    }


def budget_agent(state: TravelState):
    log.info("budget_agent started")
    constraints = state.get("trip_constraints", {})
    prompt = f"""
Assess whether this trip is plausible within the stated budget.
Request: {_compact(state.get('user_query', ''), 350)}
Constraints: {_compact(constraints, 450)}
Flight guidance: {_compact(state.get('flight_results', ''), 850)}
Accommodation research: {_compact(state.get('hotel_results', ''), 850)}
Weather notes: {_compact(state.get('weather_results', ''), 250)}

Return a concise estimate in INR when applicable:
- flight, accommodation, local transport, food, attractions, visa/insurance,
- estimated total and remaining budget,
- major uncertainties and cost-saving options,
- feasible / not feasible / uncertain verdict.
Use ranges for unknown costs. Avoid duplicating the research. Do not present
estimates as live quotes or confirmed prices. Keep the answer under 350 words.
"""
    result = _llm_text(
        "You are a careful travel budget analyst. Never imply unverified prices are confirmed.",
        prompt,
        max_tokens=550,
    )
    log.info("budget_agent completed | result_chars=%s", len(result))
    return {
        "budget_results": result,
        "messages": [AIMessage(content="Budget assessment completed.")],
        "llm_calls": _count_calls(state),
    }


def itinerary_agent(state: TravelState):
    log.info("itinerary_agent started")
    constraints = state.get("trip_constraints", {})
    prompt = f"""
Create a practical draft itinerary for human review.
Original request: {_compact(state.get('user_query', ''), 350)}
Trip constraints: {_compact(constraints, 450)}
Flight guidance: {_compact(state.get('flight_results', ''), 850)}
Accommodation research: {_compact(state.get('hotel_results', ''), 1000)}
Weather: {_compact(state.get('weather_results', ''), 350)}
Budget assessment: {_compact(state.get('budget_results', ''), 800)}

Requirements:
- Organize by day, with sensible geographic grouping and realistic pacing.
- Respect duration and budget; point out if the budget appears insufficient.
- Do not invent confirmed prices, flight schedules, hotel availability, or bookings.
- Mark estimates and missing live data clearly.
- Include transport suggestions, a rough daily cost where useful, and practical tips.
- Keep the complete draft concise (around 600 words).
"""
    result = _llm_text(
        "You are an expert itinerary planner. Be practical, accurate, and concise.",
        prompt,
        max_tokens=900,
    )
    approval_request = (
        "Please review this draft itinerary. Approve it or provide feedback.\n\n"
        + result
    )

    log.info("itinerary_agent completed | result_chars=%s", len(result))
    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": _count_calls(state),
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------

def human_approval_agent(state: TravelState):
    log.info("human_approval_agent waiting for user decision")
    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    if not isinstance(feedback, dict):
        feedback = {"approved": False, "feedback": str(feedback)}

    log.info("human_approval_agent resumed | approved=%s", bool(feedback.get("approved", False)))
    return {
        "approved": bool(feedback.get("approved", False)),
        "human_feedback": str(feedback.get("feedback", "")),
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# ---------------------------------------------------------------------------
# Final response
# ---------------------------------------------------------------------------

def final_response_agent(state: TravelState):
    log.info("final_response_agent started | approved=%s", bool(state.get("approved", False)))
    approved = bool(state.get("approved", False))
    itinerary = _compact(state.get("itinerary", ""), 2400)
    budget = _compact(state.get("budget_results", ""), 700)

    if approved:
        prompt = f"""
Polish the human-approved travel itinerary for the user.
Preserve its facts and caveats; do not add unverified live prices or bookings.
Itinerary:
{itinerary}
Budget notes:
{budget}
Return the final plan with clear headings and concise practical advice.
"""
    else:
        prompt = f"""
The draft was not approved. Revise it using the user's feedback.
Original request: {_compact(state.get('user_query', ''), 350)}
Draft itinerary:
{itinerary}
Human feedback:
{_compact(state.get('human_feedback', ''), 700)}
Budget notes:
{budget}
Preserve verified facts, flag uncertainty, and return a revised draft for review.
"""
    result = _llm_text(
        "You produce clear, honest, user-ready travel plans.",
        prompt,
        max_tokens=1000,
    )
    log.info("final_response_agent completed | result_chars=%s", len(result))
    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": _count_calls(state),
    }


from __future__ import annotations

import logging
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app
from logging_config import logger

log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Tripmind AI | Travel Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- Design system -------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --tm-bg:#0b1020; --tm-panel:#111a2e; --tm-line:rgba(148,163,184,.18); --tm-muted:#9aa9c4; --tm-text:#edf3ff; --tm-accent:#8b9cff; --tm-green:#4adea3; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, #18264a 0%, #0b1020 44%, #080d18 100%); color:var(--tm-text); }
      [data-testid="stHeader"] { background:rgba(8,13,24,.65); }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#10182b,#0c1323); border-right:1px solid var(--tm-line); }
      [data-testid="stSidebar"] > div { padding-top:1.1rem; }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
      h1 { font-weight:800 !important; }
      .hero { padding:1.45rem 1.6rem; border:1px solid var(--tm-line); border-radius:22px; background:linear-gradient(120deg,rgba(45,62,111,.72),rgba(17,26,46,.78)); margin-bottom:1.2rem; box-shadow:0 20px 60px rgba(0,0,0,.18); }
      .eyebrow { color:#aebaff; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
      .hero-title { font-family:'Manrope',sans-serif; font-size:2.1rem; line-height:1.15; font-weight:800; margin:.45rem 0 .5rem; color:#f4f7ff; }
      .hero-sub { color:#b6c3dc; font-size:.96rem; max-width:850px; }
      .section-kicker { color:#aab8d3; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin-bottom:.35rem; }
      .panel { background:rgba(17,26,46,.78); border:1px solid var(--tm-line); border-radius:17px; padding:1rem 1.1rem; height:100%; }
      .panel-title { font-family:'Manrope',sans-serif; font-weight:700; font-size:1rem; margin-bottom:.45rem; color:#f1f5ff; }
      .muted { color:var(--tm-muted); font-size:.86rem; }
      .agent-card { background:rgba(17,26,46,.72); border:1px solid var(--tm-line); border-radius:13px; padding:.8rem .85rem; margin:.35rem 0; }
      .agent-name { font-weight:700; color:#eef3ff; font-size:.91rem; }
      .agent-desc { color:#9baaca; font-size:.76rem; margin-top:.15rem; }
      .status-pill { display:inline-block; border-radius:999px; padding:.18rem .55rem; font-size:.68rem; font-weight:700; background:rgba(148,163,184,.14); color:#b9c6dc; }
      .status-ok { background:rgba(74,222,163,.13); color:#6ee7b7; }
      .status-wait { background:rgba(251,191,36,.14); color:#fcd34d; }
      .status-error { background:rgba(248,113,113,.14); color:#fca5a5; }
      .small-label { color:#91a1bf; font-size:.73rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
      div[data-testid="stMetric"] { background:rgba(17,26,46,.75); border:1px solid var(--tm-line); border-radius:14px; padding:.9rem 1rem; }
      div[data-testid="stMetricLabel"] { color:#a9b8d2; }
      div[data-testid="stMetricValue"] { color:#f3f6ff; font-family:'Manrope',sans-serif; }
      .stButton > button[kind="primary"] { border:0; background:linear-gradient(100deg,#8b9cff,#6c7df2); color:#081126; font-weight:800; border-radius:10px; min-height:2.8rem; box-shadow:0 8px 24px rgba(108,125,242,.18); }
      .stButton > button { border-radius:10px; }
      .stTextArea textarea, .stTextInput input { background:#0b1325; border-color:var(--tm-line); border-radius:10px; }
      div[data-testid="stTabs"] button { font-weight:700; }
      .footer { color:#71819f; font-size:.75rem; text-align:center; padding:1.2rem 0 .3rem; }
      hr { border-color:var(--tm-line); }
    </style>
    """,
    unsafe_allow_html=True,
)

AGENTS = [
    ("supervisor", "Supervisor", "Intent validation, constraint extraction & orchestration", "🧭"),
    ("flight_agent", "Flight Agent", "Routes, airport guidance & airfare estimates", "✈️"),
    ("hotel_agent", "Hotel Agent", "Accommodation research via web search", "🏨"),
    ("weather_agent", "Weather Agent", "Current conditions & forecast lookup", "🌦️"),
    ("budget_agent", "Budget Agent", "Cost breakdown, affordability & trade-offs", "💰"),
    ("itinerary_agent", "Itinerary Agent", "Day-by-day plan with realistic pacing", "🗺️"),
    ("human_approval", "Human Approval", "Human-in-the-loop review checkpoint", "🧑‍💻"),
    ("final_response", "Final Response", "Polished, user-ready travel plan", "✨"),
]

def _has_content(value) -> bool:
    return bool(value and str(value).strip())

def _agent_status(key: str, result: dict) -> tuple[str, str]:
    if key == "supervisor":
        if result.get("request_allowed") is False:
            return "Rejected", "status-error"
        return ("Accepted", "status-ok") if result.get("request_allowed") else ("Pending", "")
    if key == "flight_agent":
        return ("Complete", "status-ok") if _has_content(result.get("flight_results")) else ("Pending", "")
    if key == "hotel_agent":
        return ("Complete", "status-ok") if _has_content(result.get("hotel_results")) else ("Pending", "")
    if key == "weather_agent":
        return ("Complete", "status-ok") if _has_content(result.get("weather_results")) else ("Pending", "")
    if key == "budget_agent":
        return ("Complete", "status-ok") if _has_content(result.get("budget_results")) else ("Pending", "")
    if key == "itinerary_agent":
        return ("Complete", "status-ok") if _has_content(result.get("itinerary")) else ("Pending", "")
    if key == "human_approval":
        if "__interrupt__" in result:
            return "Awaiting review", "status-wait"
        if "approved" in result:
            return ("Approved", "status-ok") if result.get("approved") else ("Revision requested", "status-wait")
        return "Pending", ""
    if key == "final_response":
        return ("Complete", "status-ok") if _has_content(result.get("final_response")) else ("Pending", "")
    return "Pending", ""

def _render_agent_card(key: str, title: str, description: str, emoji: str, result: dict):
    status, css_class = _agent_status(key, result)
    st.markdown(
        f"""
        <div class="agent-card">
          <div style="display:flex;align-items:center;gap:.55rem">
            <span style="font-size:1.25rem">{emoji}</span>
            <div style="flex:1">
              <div class="agent-name">{title}</div>
              <div class="agent-desc">{description}</div>
            </div>
            <span class="status-pill {css_class}">{status}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _safe_text(value, fallback="Not available yet"):
    if not _has_content(value):
        return fallback
    return str(value)

# ------------------------------- Session state -------------------------------
with st.sidebar:
    st.markdown("## ✈️ Tripmind **AI**")
    st.caption("MULTI-AGENT TRAVEL INTELLIGENCE")
    st.divider()
    st.markdown('<div class="section-kicker">Workspace</div>', unsafe_allow_html=True)
    user_id = st.text_input("User ID", value="demo_user", help="Used as the session owner label.")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
    if st.button("＋  New planning session", use_container_width=True):
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        st.session_state.pop("latest_result", None)
        st.session_state.waiting_for_approval = False
        st.session_state.pop("submitted_query", None)
        log.info("New UI thread created")
        st.rerun()
    st.markdown('<div class="small-label">Active thread</div>', unsafe_allow_html=True)
    st.code(st.session_state.thread_id, language=None)
    st.divider()
    st.markdown('<div class="section-kicker">Agent control room</div>', unsafe_allow_html=True)
    sidebar_result = st.session_state.get("latest_result", {}) or {}
    for key, title, description, emoji in AGENTS:
        _render_agent_card(key, title, description, emoji, sidebar_result)
    st.divider()
    st.caption("🔒 Secrets are not displayed in this dashboard.")
    st.caption("Logs: `logs/tripmind.log`")

# ---------------------------------- Hero --------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered trip orchestration</div>
      <div class="hero-title">Plan less. Explore more.</div>
      <div class="hero-sub">A travel-planning workspace powered by specialist agents, live research tools, budget reasoning and human approval.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown('<div class="section-kicker">01 / Your trip brief</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Describe the trip you want",
        placeholder="Plan a 7-day Japan trip under ₹2 lakh from Mumbai. I prefer budget hotels, public transport and no overnight flights.",
        height=125,
        label_visibility="collapsed",
    )
    quick1, quick2, quick3 = st.columns(3)
    if quick1.button("🇯🇵 Japan · 7 days", use_container_width=True):
        st.session_state["trip_prompt"] = "Plan a 7-day Japan trip under ₹2 lakh from Mumbai. I prefer budget hotels and public transport."
        st.rerun()
    if quick2.button("🇹🇭 Thailand · 5 days", use_container_width=True):
        st.session_state["trip_prompt"] = "Plan a 5-day Thailand trip under ₹80,000 from Mumbai with budget hotels."
        st.rerun()
    if quick3.button("🇸🇬 Singapore · 4 days", use_container_width=True):
        st.session_state["trip_prompt"] = "Plan a 4-day Singapore trip under ₹90,000 from Mumbai, focused on public transport and affordable food."
        st.rerun()
    if "trip_prompt" in st.session_state and not query:
        query = st.session_state["trip_prompt"]
        st.info(f"Selected example: {query}")
    create = st.button("✦  Build my travel plan", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="section-kicker">02 / Workflow overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">From request to reviewed plan</div>
          <p class="muted">Every trip follows a traceable workflow. Agents contribute specialist results, the budget agent estimates feasibility, and a human reviews the draft before the final response.</p>
          <div style="display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.8rem">
            <span class="status-pill">Intent guardrail</span><span class="status-pill">Parallel-style specialists</span><span class="status-pill">Budget check</span><span class="status-pill">Human approval</span>
          </div>
          <p class="muted" style="margin-top:1rem;margin-bottom:0">Provider prices and availability are only confirmed when the connected provider actually returns them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if create:
    if not query.strip():
        st.warning("Enter a travel request first.")
    else:
        log.info("New travel request submitted | query_chars=%s", len(query.strip()))
        st.session_state.pop("latest_result", None)
        st.session_state.waiting_for_approval = False
        st.session_state.submitted_query = query.strip()
        with st.spinner("Supervisor and specialist agents are working…"):
            try:
                result = app.invoke(
                    {
                        "messages": [HumanMessage(content=query.strip())],
                        "user_id": user_id,
                        "user_query": query.strip(),
                        "flight_results": "",
                        "hotel_results": "",
                        "weather_results": "",
                        "budget_results": "",
                        "itinerary": "",
                        "final_response": "",
                        "llm_calls": 0,
                    },
                    config=config,
                )
            except Exception:
                log.exception("Travel graph invocation failed")
                st.error("Trip planning failed. Check `logs/tripmind.log` for the error details.")
                st.stop()
        st.session_state.latest_result = result
        st.session_state.waiting_for_approval = "__interrupt__" in result
        log.info(
            "Travel graph invocation finished | request_allowed=%s | selected_agents=%s",
            result.get("request_allowed"), result.get("selected_agents", []),
        )
        st.rerun()

result = st.session_state.get("latest_result") or {}

# ---------------------------------- Results -----------------------------------
if result:
    st.divider()
    st.markdown('<div class="section-kicker">03 / Trip intelligence dashboard</div>', unsafe_allow_html=True)
    if not result.get("request_allowed", False):
        st.error(_safe_text(result.get("final_response") or result.get("supervisor_reasoning"), "Request could not be validated."))
        st.stop()

    constraints = result.get("trip_constraints") or {}
    total_agents = len(AGENTS)
    completed_agents = sum(_agent_status(item[0], result)[0] in {"Complete", "Accepted", "Approved"} for item in AGENTS)
    awaiting = st.session_state.get("waiting_for_approval", False)
    metrics = st.columns(4)
    metrics[0].metric("Agents complete", f"{completed_agents}/{total_agents}", help="Includes supervisor, specialist agents, approval and final response.")
    metrics[1].metric("LLM calls", result.get("llm_calls", 0))
    metrics[2].metric("Approval status", "Review needed" if awaiting else ("Final ready" if result.get("final_response") else "In progress"))
    metrics[3].metric("Active thread", st.session_state.thread_id[-8:])

    with st.expander("🧾 Extracted trip constraints", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**Destination**")
        c1.write(constraints.get("destination") or "Not specified")
        c2.markdown("**Duration**")
        c2.write(constraints.get("duration") or "Not specified")
        c3.markdown("**Budget**")
        c3.write(constraints.get("budget") or "Not specified")
        c4.markdown("**Travel style**")
        c4.write(constraints.get("travel_style") or "Not specified")
        c5, c6, c7 = st.columns(3)
        c5.markdown("**Origin**")
        c5.write(constraints.get("origin") or "Not specified")
        c6.markdown("**Weather city**")
        c6.write(constraints.get("weather_city") or "Auto-resolved")
        c7.markdown("**Preferences**")
        prefs = constraints.get("special_preferences") or []
        c7.write(", ".join(map(str, prefs)) if prefs else "No extra preferences")
        st.caption("Supervisor reasoning")
        st.write(result.get("supervisor_reasoning") or "No reasoning returned.")

    # Agent observability view
    with st.expander("🧠 Agent execution & status", expanded=True):
        for start in range(0, len(AGENTS), 2):
            row = st.columns(2)
            for col, item in zip(row, AGENTS[start:start+2]):
                with col:
                    _render_agent_card(*item, result)

    tabs = st.tabs(["🧭 Overview", "✈️ Flights", "🏨 Hotels", "🌦️ Weather", "💰 Budget", "🗺️ Itinerary", "🪵 Execution"])
    with tabs[0]:
        o1, o2 = st.columns([1.25, 1], gap="large")
        with o1:
            st.markdown("### Supervisor summary")
            st.write(result.get("supervisor_reasoning") or "No summary available.")
            st.markdown("### Recommended next step")
            if awaiting:
                st.info("Review the draft itinerary below, then approve it or request changes.")
            elif result.get("final_response"):
                st.success("The final travel response is ready.")
            else:
                st.info("Agent workflow completed its current step. Check the tabs for details.")
        with o2:
            st.markdown("### Research coverage")
            for label, key in [
                ("Flight guidance", "flight_results"),
                ("Accommodation research", "hotel_results"),
                ("Weather lookup", "weather_results"),
                ("Budget assessment", "budget_results"),
                ("Draft itinerary", "itinerary"),
            ]:
                present = _has_content(result.get(key))
                st.write(("✅" if present else "◯") + "  " + label)

    with tabs[1]:
        st.markdown("### Flight intelligence")
        st.caption("Live fares and schedules are not guaranteed unless a connected provider verifies them.")
        st.markdown(_safe_text(result.get("flight_results")))
    with tabs[2]:
        st.markdown("### Accommodation research")
        st.markdown(_safe_text(result.get("hotel_results")))
    with tabs[3]:
        st.markdown("### Weather & packing")
        st.markdown(_safe_text(result.get("weather_results")))
    with tabs[4]:
        st.markdown("### Budget & feasibility")
        st.markdown(_safe_text(result.get("budget_results")))
        st.caption("Treat generated cost ranges as estimates, not confirmed quotes.")
    with tabs[5]:
        st.markdown("### Draft itinerary")
        interrupt_data = result.get("__interrupt__", [])
        if interrupt_data:
            try:
                draft = interrupt_data[0].value.get("draft_itinerary", "")
            except (AttributeError, IndexError, TypeError):
                draft = result.get("itinerary", "")
        else:
            draft = result.get("itinerary", "")
        st.markdown(_safe_text(draft))
        if result.get("final_response"):
            st.divider()
            st.markdown("### ✨ Final travel plan")
            st.markdown(result["final_response"])
    with tabs[6]:
        st.markdown("### Execution diagnostics")
        st.write("**Workflow thread:**", st.session_state.thread_id)
        st.write("**Selected agents:**", result.get("selected_agents", []))
        st.write("**LLM call counter:**", result.get("llm_calls", 0))
        st.write("**Last UI update:**", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
        st.caption("Detailed stack traces and MCP/LLM diagnostics are written to `logs/tripmind.log`.")
        with st.expander("Show returned state keys"):
            st.code("\n".join(sorted(result.keys())), language="text")

    # Human-in-the-loop approval
    if st.session_state.get("waiting_for_approval"):
        st.divider()
        st.markdown('<div class="section-kicker">04 / Human-in-the-loop checkpoint</div>', unsafe_allow_html=True)
        st.markdown("## Review before finalizing")
        st.info("The itinerary is a draft. Approve it to generate the final response, or request a revision.")
        approved = st.radio("Your decision", ["Approve draft", "Request changes"], horizontal=True)
        feedback = st.text_area(
            "Revision feedback",
            placeholder="For example: reduce hotel costs, add Kyoto, remove expensive attractions…",
            disabled=approved == "Approve draft",
            height=90,
        )
        if st.button("Submit review & continue", type="primary", use_container_width=True):
            with st.spinner("Resuming the workflow…"):
                try:
                    final_result = app.invoke(
                        Command(resume={
                            "approved": approved == "Approve draft",
                            "feedback": feedback,
                        }),
                        config=config,
                    )
                except Exception:
                    log.exception("Approval/resume invocation failed")
                    st.error("Could not resume the approval workflow. Check `logs/tripmind.log`.")
                    st.stop()
            log.info("Approval submitted | approved=%s", approved == "Approve draft")
            st.session_state.latest_result = final_result
            st.session_state.waiting_for_approval = "__interrupt__" in final_result
            st.rerun()

    if result.get("final_response"):
        st.divider()
        st.markdown('<div class="section-kicker">05 / Final deliverable</div>', unsafe_allow_html=True)
        st.markdown("## ✨ Your travel plan")
        st.markdown(result["final_response"])
        st.download_button(
            "⬇️ Download final plan (.md)",
            data=result["final_response"],
            file_name="tripmind-travel-plan.md",
            mime="text/markdown",
        )
else:
    st.markdown("---")
    st.markdown("### What your travel copilot coordinates")
    feature_cols = st.columns(4)
    feature_data = [
        ("🧭", "Intent & routing", "Validate requests and extract trip constraints."),
        ("🔎", "Real-world research", "Connect search, weather and travel data tools."),
        ("💰", "Budget reasoning", "Estimate costs and surface affordability risks."),
        ("🧑‍💻", "Human oversight", "Review the draft before the final answer."),
    ]
    for col, (emoji, title, description) in zip(feature_cols, feature_data):
        with col:
            st.markdown(f"""<div class="panel"><div style="font-size:1.5rem">{emoji}</div><div class="panel-title">{title}</div><div class="muted">{description}</div></div>""", unsafe_allow_html=True)

st.markdown('<div class="footer">TRIPMIND AI · MULTI-AGENT TRAVEL PLANNING · VERIFY LIVE PRICES BEFORE BOOKING</div>', unsafe_allow_html=True)

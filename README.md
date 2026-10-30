# ✈️ Tripmind AI

**An Agentic AI travel planner powered by LangGraph, Groq, MCP, Streamlit, and PostgreSQL.**

Tripmind AI turns a natural-language travel request into a researched trip plan. Specialist agents gather travel information, estimate costs, build an itinerary, and pause for human approval before generating the final response.

![Tripmind AI Architecture](docs/architecture.png)

## ✨ Highlights

- **Multi-agent orchestration:** LangGraph supervisor validates requests, extracts constraints, and coordinates specialist agents.
- **Specialist agents:** Flight, Hotel, Weather, Budget, and Itinerary.
- **MCP tool integration:** Connects external research and weather services through the Model Context Protocol.
- **Budget reasoning:** Estimates trip costs and highlights potential budget issues.
- **Human-in-the-loop:** Approve a draft itinerary or request changes before finalization.
- **Persistent state:** PostgreSQL checkpointer with an in-memory fallback for local development.
- **Observability:** Centralized Python logging with rotating log files.
- **Interactive dashboard:** Streamlit interface for research results, agent status, approval, and final plans.

## 🧠 Agent Workflow

| Agent | Role |
|---|---|
| Supervisor | Validates travel intent, extracts constraints, and routes the request |
| Flight | Provides flight guidance and estimates |
| Hotel | Researches accommodation options |
| Weather | Retrieves current conditions and forecasts |
| Budget | Estimates costs and checks budget feasibility |
| Itinerary | Creates a day-by-day draft |
| Human Approval | Pauses for approval or revision feedback |
| Final Response | Produces the final travel plan |

## 🔌 MCP Integrations

- **Tavily MCP** — web research
- **Weather MCP** — weather and forecast data via OpenWeather
- **AviationStack MCP** — aviation tools, subject to API-plan access and workflow integration

Clone the AviationStack MCP repository:

```bash
git clone https://github.com/Pradumnasaraf/aviationstack-mcp.git
```

Follow its README for installation and configuration. Tool availability depends on your AviationStack subscription.

## 🛠️ Tech Stack

`Python` · `Streamlit` · `LangGraph` · `LangChain` · `Groq` · `MCP` · `Tavily` · `OpenWeather` · `AviationStack` · `PostgreSQL`

## 🚀 Run Locally

```bash
git clone https://github.com/CodeCortex/tripmind-ai.git
cd tripmind-ai

python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env` (use the variable names expected by `config.py`):

```dotenv
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
AVIATION_STACK_API_KEY=your_aviationstack_api_key

# Optional: PostgreSQL checkpoint persistence
DATABASE_URL=postgresql://username:password@localhost:5432/tripmind
```

Start the application:

```bash
streamlit run frontend.py
```

Open [http://localhost:8501](http://localhost:8501).


## 📸 Application Preview

| Travel dashboard | Agent workflow |
|---|---|
| ![Dashboard](docs/01.png) | ![Agent workflow](docs/02.png) |

|  Human approval | Research results |
|---|---|
| ![Human approval](docs/03.png) | ![Research results](docs/04.png) |

## 🔍 Engineering Concepts

**LangGraph state management · Prompt engineering · MCP tool integration · Human-in-the-loop workflows · PostgreSQL checkpointing · LLM orchestration · Logging and error handling**

> Travel prices, availability, and generated budgets are estimates unless verified by a live provider. Keep API keys in `.env` and never commit secrets.

## 👨‍💻 Author

**Roshan Jaiswal**  
GenAI Software Engineer · Agentic AI · Python · LangGraph · MCP · LLM Applications

---

If you find this project interesting, consider giving the repository a ⭐.

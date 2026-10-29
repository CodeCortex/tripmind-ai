import os 
from dotenv import load_dotenv
from functools import lru_cache
from langchain_groq import ChatGroq

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# def get_llm():
#     return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=api_key,
        temperature=0,
    )
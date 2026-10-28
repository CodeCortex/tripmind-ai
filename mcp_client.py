import os 
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (AVIATION_STACK_API_KEY, OPENWEATHER_API_KEY, TAVILY_API_KEY)




client= MultiServerMCPClient(
    {
        "tavily":{},
        "aviationstack":{},
        "weather":{}
    }
)
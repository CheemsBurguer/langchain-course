from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    
    param nuim: a number to triple
    returns: the triple of the input number

    """

    return float(num) * 3

tools = [triple, TavilySearch(max_results=1)]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).bind_tools(tools)



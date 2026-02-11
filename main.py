from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()



llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    results = agent.invoke({"messages": [HumanMessage(content="Busca 3 curiosidades de la historia de Halo (lore expandido) que probablemente un fan con conocimiento medio no conozca")]})
    print(results)


if __name__ == "__main__":
    main()

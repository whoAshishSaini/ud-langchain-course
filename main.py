import os

from langchain import hub
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain.agents import Tool
from langchain_groq import ChatGroq
from langchain.agents.react.agent import create_react_agent

load_dotenv()


def firecrawl_search(query):
    firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    results = firecrawl.search(
        query=query,
        limit=10,
    )
    return results


def main():
    print("Hello from ud-langchain-course!")
    firecrawl_search_tool = Tool(
        name="firecrawl Search",
        func=firecrawl_search,
        description="Tool for searching the web and retrieving the full content of the top results. Use this for general knowledge, up-to-date information, or current events.",
    )

    tools = [firecrawl_search_tool]
    llm = ChatGroq(model="llama-3.1-8b-instant",
    temperature=0.0,)
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm,prompt=react_prompt,tools=tools)
    agent_executor = AgentExecutor(agent=agent,verbose=True, tools=tools)

    chain = agent_executor

    result = chain.invoke(input={"input":"Search for 3 job postings for MDM using AI in India and list their details."})
    print(result)

if __name__ == "__main__":
    main()

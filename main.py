import os

from langchain import hub
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain.agents import Tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents.react.agent import create_react_agent
from langchain_core.runnables import RunnableLambda
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

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
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.0,reasoning_format="hidden")
    react_prompt = hub.pull("hwchase17/react")
    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    react_prompt_with_format_instructions = PromptTemplate(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["input", "tools", "tool_names", "agent_Scratchpad"],
    ).partial(format_instructions=output_parser.get_format_instructions())

    agent = create_react_agent(
        llm=llm, prompt=react_prompt_with_format_instructions, tools=tools
    )
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools,handle_parsing_errors=True)
    extract_output = RunnableLambda(lambda x: x["output"])
    parse_output = RunnableLambda(lambda x: output_parser.parse(x))
    chain = agent_executor | extract_output | parse_output

    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for MDM using AI in India and list their details."
        }
    )
    print(result)


if __name__ == "__main__":
    main()

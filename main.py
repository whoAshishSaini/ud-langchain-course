from typing import List

from dotenv import load_dotenv
from langchain_core.messages.tool import tool_call
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from callbacks import AgentCallbackHandler

load_dotenv()


def main():
    print("Hello from ud-langchain-course!")


@tool
def get_text_length(text: str) -> int:
    """
    It returns the length of the given text
    """
    text = text.get("text")
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    main()
    tools = [get_text_length]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        callbacks=[AgentCallbackHandler()],
    )

    llm_with_tools = llm.bind_tools(tools)
    message = [HumanMessage(content="What is the length of the word: DOG")]
    while True:
        ai_message = llm_with_tools.invoke(message)
        print(ai_message)

        tools_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tools_calls) > 0:
            message.append(ai_message)
            for tool_call in tools_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.func(tool_args)
                print(f"observation={observation}")
                message.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id
                ))
            continue
        print(ai_message.content)
        break

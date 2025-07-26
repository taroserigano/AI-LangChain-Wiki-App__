from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.agents import AgentFinish
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist. Instead, use the 'describe_tables' function"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [
    run_query_tool,
    describe_tables_tool,
    write_report_tool
]

# Bind tools to the model
llm_with_tools = chat.bind_tools(tools)

# Create the LCEL agent chain using pipe operator
agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_tool_messages(
            x.get("intermediate_steps", [])
        )
    )
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Tool executor function
def execute_tools(output):
    if isinstance(output, list):
        output = output[0]
    
    if hasattr(output, 'tool'):
        tool_name = output.tool
        tool_input = output.tool_input
        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                return tool.invoke(tool_input)
    return output

# Create tool executor as a runnable
tool_executor = RunnableLambda(execute_tools)

# Set up chat history
chat_history = ChatMessageHistory()

# Function to run the agent loop
def run_agent(inputs):
    intermediate_steps = []
    current_input = inputs.copy()
    
    while True:
        # Add intermediate steps to input
        current_input["intermediate_steps"] = intermediate_steps
        
        # Run the agent
        output = agent.invoke(current_input)
        
        # If it's a final answer, return it
        if isinstance(output, AgentFinish):
            return {"output": output.return_values["output"]}
        
        # Otherwise, execute the tool
        if isinstance(output, list):
            output = output[0]
        
        observation = tool_executor.invoke(output)
        intermediate_steps.append((output, str(observation)))

# Create the agent executor as a runnable
agent_executor = RunnableLambda(run_agent)

# Wrap with message history using LCEL
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Use the LCEL chain
print("Initial query:")
initial_query_result = agent_with_history.invoke(
    {"input": "How many orders are there? Write the result to an html report."},
    config={"configurable": {"session_id": "default"}}
)
print(f"Response: {initial_query_result['output']}")

print("\nFollowup query:")
followup_query_result = agent_with_history.invoke(
    {"input": "Repeat the exact same process for users."},
    config={"configurable": {"session_id": "default"}}
)
print(f"Response: {followup_query_result['output']}")

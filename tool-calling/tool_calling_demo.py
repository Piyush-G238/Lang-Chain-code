from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def multiply(a: float, b:float):
    """Given two numbers a and b; this tool return their product"""
    return a * b

chatmodel = ChatOllama(model='llama3.2:3b',temperature=0.1)

chatmodel_with_tools = chatmodel.bind_tools([multiply])

# chatmodel_with_tools.invoke('Hi how are you')
query=HumanMessage("if a = 67 and b = 8.59, then what is a * b = ")
messages = [query]

response = chatmodel_with_tools.invoke(messages)
messages.append(response)

tool_result = multiply.invoke(response.tool_calls[0])
messages.append(tool_result)

final_output = chatmodel_with_tools.invoke(messages).content
print(final_output)
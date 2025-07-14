from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

chatmodel = ChatOllama(model='llama3.2:3b', temperature=0)

messages = [
    SystemMessage(content='You are experienced maths professor'),
    HumanMessage(content='Tell me, what are the use cases of integration?')
]

response = chatmodel.invoke(messages)
content = response.content
messages.append(AIMessage(content))

print(content)

# chat_history = []

# while True:
#     userinput = input("You: ")
#     chat_history.append(userinput)
#     if userinput == 'exit':
#         break
#     result = chatmodel.invoke(chat_history)
#     chat_history.append(result.content)
#     print(f"AI: {result.content}")
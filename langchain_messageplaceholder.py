from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

template = ChatPromptTemplate([
    ('system', 'You are a customer support agent'),
    '{chat_history}',
    ('human', '{query}')
])

chat_history = []

with open('chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())

print(chat_history)

prompt = template.invoke({
    'chat_history':chat_history, 
    'query':'Where is my refund now'})

model = ChatOllama(model='llama3.2:3b', temperature=0)

response = model.invoke(prompt)
print(response.content)
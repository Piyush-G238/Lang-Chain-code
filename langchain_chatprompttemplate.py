# usage of chat prompt template

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain me in 100 wods, what is {topic}')
    # SystemMessage(content='You are a helpful {domain} expert'),
    # HumanMessage(content='')
])

prompt = chat_template.format(domain='astronomy',topic='solar eclipse')
# print(prompt)

chatmodel = ChatOllama(model='llama3.2:3b', temperature=0.5)
response = chatmodel.invoke(prompt)
print(response.content)
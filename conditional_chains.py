from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model='llama3.2:3b',temperature=0.2)

parser = StrOutputParser()

template1 = PromptTemplate(
    template=''
)
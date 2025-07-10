from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='llama3.2:3b')



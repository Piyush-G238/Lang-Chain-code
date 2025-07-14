from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='llama3.2:3b', temperature=0.2)

template = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | chatmodel | parser
response = chain.invoke({'topic':'Marco polo'})
print(response)
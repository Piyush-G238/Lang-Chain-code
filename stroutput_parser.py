from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='llama3.2:3b',temperature=0)

template1 = PromptTemplate(
    template='write a detailed report on following topic: {topic}',
    input_variables=['topic']
)

# template1.invoke({'topic':'black hole'})

template2 = PromptTemplate(
    template='write a 5 line summary report on the following text: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# template2.invoke({'topic':'black hole'})

chain = template1 | chatmodel | parser | template2 | chatmodel | parser

result = chain.invoke({'topic':'black hole'})

print(result)
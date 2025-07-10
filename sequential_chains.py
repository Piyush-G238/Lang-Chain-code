from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='llama3.2:3b')

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic} within 250 words",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

sequential_chain = prompt1 | chatmodel | parser | prompt2 | chatmodel | parser

response = sequential_chain.invoke({'topic':'Big Bang theory'})
print(response)
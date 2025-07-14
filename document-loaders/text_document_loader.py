from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='gemma3:latest',temperature=0.3)

parser = StrOutputParser()

text_loader = TextLoader(file_path='content.txt', encoding='utf-8')
documents = text_loader.load()

prompt = PromptTemplate(
    template='Write a 5 line summary for the following text:\n{text}',
    input_variables=['text']
)

main_chain = prompt | chatmodel | parser
response = main_chain.invoke({'text':documents[0].page_content})
print(response)
# print(documents[0].metadata)
# print(documents[0].page_content)
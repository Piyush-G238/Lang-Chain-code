from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

loader = WebBaseLoader(web_path='https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421')

docs = loader.load()

chatmodel = ChatOllama(model='gemma3:latest',temperature=0.2)

prompt = PromptTemplate(template='Find the technical specification from following text:\n{text}', input_variables=['text'])

parser = StrOutputParser

chain = RunnableSequence(prompt, chatmodel, parser)
response = chain.invoke({'text':docs[0].page_content})
print(response)

# for doc in docs:
#     print(doc.metadata)
# parser = S
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='gemma3:latest',temperature=0.3)

parser = StrOutputParser()

# text_loader = TextLoader(file_path='llm_context.txt',encoding='utf-8')
# documents = text_loader.lazy_load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=10)
# textdata = text_splitter.split_documents(documents=documents)

embedding_func = OllamaEmbeddings(model='nomic-embed-text:v1.5')
vector_store = Chroma(
    persist_directory='rag_outline',
    collection_name='llm_facts',
    embedding_function=embedding_func)

# vector_store.add_documents(documents=textdata)
vectorstore_retriever = vector_store.as_retriever(kwargs={'k':5})

query = 'important points on transformers\n'
retrieved_data = vectorstore_retriever.invoke(query)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer only from provided context only.
If context is insufficient, just say you don't know.

{context}

Question:{question}
""",
input_variables=['context','question'])

context_data = ''
for data in retrieved_data:
    context_data = context_data.join(data.page_content)
print('context data','\n',context_data)

simple_chain = prompt | chatmodel | parser
question = 'What'
response = simple_chain.invoke({'context':context_data,'question':query})
print(response)
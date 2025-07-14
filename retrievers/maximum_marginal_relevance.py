# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

embedding_func = OllamaEmbeddings(model='llama3.2:3b')

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vectorstore = Chroma.from_documents(
    embedding=embedding_func,
    documents=docs,
    collection_name='llm_facts'
)

retriever = vectorstore.as_retriever(kwargs={'k':2, 'lambda_mult':0})

responses = retriever.invoke('what is langchain used for? Also Which LLM we can use for RAG')

for i, res in enumerate(responses):
    print(i + 1, res.page_content)
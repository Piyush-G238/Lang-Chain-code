from langchain.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

documents = [
    Document(
        page_content="LangChain helps developers build LLM applications easily.",
        metadata={'topic':'LangChain'}),
    Document(
        page_content="Chroma is a vector database optimized for LLM-based search.",
        metadata={'topic':'ChromaDB'}),
    Document(
        page_content="Embeddings convert text into high-dimensional vectors.",
        metadata={'topic':'Embedding'}),
    Document(
        page_content="OpenAI provides powerful embedding models.",
        metadata={'topic':'OpenAI'}),
]

embedding = OllamaEmbeddings(model='llama3.2:3b')

vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embedding, 
    collection_name='my_collection')

retriever = vectorstore.as_retriever(kwargs={'k':2})

result = retriever.invoke('Which vendor has powerful embedding models')
print(result)

# Chroma(embedding_function=embedding, collection_name='ai_facts',persist_directory='m')
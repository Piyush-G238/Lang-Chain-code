from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever

health_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_func = OllamaEmbeddings(model='llama3.2:3b')

vectorstore = Chroma.from_documents(
    embedding=embedding_func,
    documents=health_docs,
    collection_name='health_coll'
)

# simple_retriever = vectorstore.as_retriever(kwargs={'k':4,'lambda_mult':0.5})
# query = 'How to improve health?'
# result_docs = simple_retriever.invoke(query)
# print(result_docs)

chatmodel = ChatOllama(model='llama3.2:3b',temperature='0.8')

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(kwargs={'k':3}),
    llm=chatmodel
)

query = 'How to improve energy level and stamina?'

multiquery_retriever.invoke()

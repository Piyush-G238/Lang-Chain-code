from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(lang='en',top_k_results='2')

query = 'the geopolitical history of india and pakistan from the perspective of a chinese'

documents = retriever.invoke(query)

print(documents)
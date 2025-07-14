from langchain.schema import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

document_1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
)
document_2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
)
document_3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
)
document_4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
)
document_5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
)

embedding_model = OllamaEmbeddings(model='llama3.2:3b')

docs = [document_1, document_2, document_3, document_4, document_5]

chroma_client = Chroma(
    embedding_function=embedding_model, 
    persist_directory='my_db',
    collection_name='sample'
)

# ids = chroma_client.add_documents(docs)
# print(ids)

# result = chroma_client.similarity_search_with_score(query='Which player is known for its fitness', k=2)
# print(result)

# result = chroma_client.similarity_search_with_score(query='',k=1,filter={'team':'Royal Challengers Bangalore'})
# print(result)

new_doc = Document(
    page_content='Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli\'s passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.',
    metadata={'team':'Royal Challengers Bangalore'}
)
chroma_client.update_document('85f766ba-2962-4537-b680-748e76f092f7', document=new_doc)

chroma_client.delete('85f766ba-2962-4537-b680-748e76f092f7')

result = chroma_client.get(ids=['85f766ba-2962-4537-b680-748e76f092f7'], include=['embeddings','documents','metadatas'])
print(result)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path='dl-curriculum.pdf')

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator='',)

docs = pdf_loader.load()
result = splitter.split_documents(docs)

print(result[0].page_content)
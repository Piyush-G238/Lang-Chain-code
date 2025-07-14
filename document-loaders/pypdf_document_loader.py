from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path='dl-curriculum.pdf')

docs = pdf_loader.load()

print(len(docs))
from langchain_ollama import ChatOllama

chatmodel = ChatOllama(model='llama3.2:3b', temperature=0.2)

response = chatmodel.invoke('Who is known as father of nation in india in one line?')

print(response.content)
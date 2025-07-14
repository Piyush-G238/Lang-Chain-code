from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
# from langchain_core.prompts import 

# class Person(TypedDict):
#     name: str
#     age: int

# new_person:Person = {'name':'piyush', 'age':27}
# print(new_person) 

class Review(TypedDict):
    summary: Annotated[str, 'A brief summary of product review']
    sentiment: Annotated[str, 'A sentiment of product review either negative, positive or neutral']

model = ChatOllama(model='llama3.2:3b', temperature=0.3)
schema = model.with_structured_output(Review)
result = schema.invoke('The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can\'t remove. Also, the UI looks outdated compared to other brands. Hoping for a software updates to fix this.')

print(result)
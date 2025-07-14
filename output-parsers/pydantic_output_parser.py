from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

chatmodel = ChatOllama(model='gemma3:latest',temperature=0.2)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a fictional person with a name, age, and city.\nRespond ONLY with the following format:\n{format}.\nDo not include any explanation or schema.",
    partial_variables={'format':parser.get_format_instructions()}
)

prompt = template.format()
# print(prompt)

# response = chatmodel.invoke(prompt)
# print(response.content)
# print(template.format())

chain = template | chatmodel | parser

response = chain.invoke({})
print(response)
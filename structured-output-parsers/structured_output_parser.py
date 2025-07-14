from langchain_ollama import ChatOllama
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

chatmodel = ChatOllama(model='llama3.2:3b',temperature=0)

schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]

parser = StructuredOutputParser(response_schemas=schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke()
# print(prompt)

chain = template | chatmodel | parser
response = chain.invoke({'topic':'Albert Einstein'})
print(response)

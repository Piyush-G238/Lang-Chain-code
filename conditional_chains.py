from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

class Sentiment(BaseModel):
    value: Literal['Positive', 'Negative'] = Field(description='Give the sentiment feedback of the review')

model = ChatOllama(model='gemma3:latest',temperature=0)

str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Sentiment)
# StrOutputParser()

template1 = PromptTemplate(
    template='Classify the sentiment of the following feedback:\n{feedback}.\n\nRespond ONLY with the following format:\n{format_instruction}.\nDo not include any explanation or schema.',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_parser.get_format_instructions()}
)

# print(template1.format(feedback='This phone is very beautiful'))
# response = model.invoke(template1.invoke({'feedback':'this phone is very beautiful'}))
# print(pydantic_parser.parse(response.content))

chain_1 = template1 | model | pydantic_parser
# response = chain_1.invoke({'feedback':'This is a terrible smartphone'})
# sentiment = response.value
# print(response)

template2 = PromptTemplate(
    template='Write only one response to this positive feedback \n {feedback}.',
    input_variables=['feedback']
)

template3 = PromptTemplate(
    template='Write only one response to this negative feedback \n {feedback}.',
    input_variables=['feedback']
)

conditional_chain = RunnableBranch(
    (lambda x: x.value == 'Positive', template2 | model | str_parser), # tuple 1
    (lambda x: x.value == 'Negative', template3 | model | str_parser),  # tupe 2
    RunnableLambda(lambda x: 'could not find the sentiment')
)

chain = chain_1 | conditional_chain
response = chain.invoke({'feedback': 'This phone is very awesome and It\'s performance is very awesome'})
print(response)
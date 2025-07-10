from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

chatmodel = ChatOllama(model='llama3.2:3b', temperature=0)
parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional female person in {format}',
    # input_variables=['format']
    partial_variables={'format':parser.get_format_instructions()}
)

chain = template | chatmodel | parser
print(chain.invoke({}))
# template.invoke({'format':'json'})
# prompt = template.format()
# response = chatmodel.invoke(prompt)

# print(parser.parse(response.content))
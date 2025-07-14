from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='gemma3:latest', temperature=0.8)

prompt1 = PromptTemplate(
    template='Tell me the joke on following topic - {topic}',
    input_Variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain me the following joke - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(
    prompt1, chatmodel, parser, prompt2, chatmodel, parser
)

response = chain.invoke({'topic':'Butterfly'})
print(response)
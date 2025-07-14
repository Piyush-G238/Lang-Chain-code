from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnableSequence

chatmodel = ChatOllama(model='gemma3:latest')

prompt1 = PromptTemplate(
    template='Generate a linkedin post about - {topic} within 100 words',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a tweet on this topic - {topic} within 100 words',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'linkedin': RunnableSequence(prompt1, chatmodel, parser),
    'tweet': RunnableSequence(prompt2, chatmodel, parser)
})

response = chain.invoke({'topic':'ChatGPT'})
print(response)

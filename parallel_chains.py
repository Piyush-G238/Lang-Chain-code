from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

chatmodel = ChatOllama(model='llama3.2:3b',temperature=0.5)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short questions answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | chatmodel | parser,
    'quiz': prompt2 | chatmodel | parser
})

merged_chain = prompt3 | chatmodel | parser

chain = parallel_chain | merged_chain

text = """
Machine learning is a branch of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to perform tasks without explicit instructions, instead relying on patterns and inference from data. 
It allows systems to learn and improve from experience, making it a powerful tool for solving complex problems across various domains, such as healthcare, finance, marketing, and autonomous systems. 
Machine learning is typically categorized into supervised learning, where models are trained on labeled datasets to make predictions or classifications; unsupervised learning, which seeks to find hidden structures in unlabeled data; semi-supervised learning, combining both labeled and unlabeled data for more efficient learning; and reinforcement learning, where agents learn to make decisions by receiving rewards or penalties in an interactive environment. 
The machine learning process involves several stages, including problem definition, data collection, preprocessing (such as cleaning, normalization, and feature selection), model selection, training, evaluation, and deployment. 
Common algorithms include decision trees, support vector machines, neural networks, k-nearest neighbors, and ensemble methods like random forests and gradient boosting. 
As data continues to grow in volume and complexity, machine learning plays a vital role in extracting insights, automating processes, and driving innovation, though it also raises challenges in terms of interpretability, data privacy, fairness, and the need for large, high-quality datasets.
"""

response = chain.invoke({'text': text})

print(response)
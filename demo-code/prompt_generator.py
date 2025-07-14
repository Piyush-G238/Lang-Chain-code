from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
You are a helpful and knowledgeable assistant. Use the provided context to answer the question clearly and accurately.

context = {context}
question = {question}

answer =
""")

prompt_template.save('prompt.json')
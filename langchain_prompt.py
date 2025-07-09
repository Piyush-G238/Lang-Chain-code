from langchain_ollama import ChatOllama
from langchain_core.prompts import load_prompt

chatmodel = ChatOllama(model='llama3.2:3b', temperature=0)

prompt_template = load_prompt('prompt.json')

context = "ChatGPT is a powerful GenAI tool, prepared by OpenAI.  It can be used for reasoning, summarization, coding, and more."
question = "What are some common use cases of ChatGPT?"

actual_prompt = prompt_template.format(context=context, question=question)
result = chatmodel.invoke(actual_prompt)
print(result.content)
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
loader = WebBaseLoader(web_path='https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421')

# parser = S
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import requests
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
# import json

@tool
def get_weather_data(city_value: str):
    """Returns the current weather for a given location in a readable format."""
    language='EN'
    base_url = f'https://open-weather13.p.rapidapi.com/city?city={city_value}&lang={language}'
    response = requests.get(url=base_url,headers={
            'x-rapidapi-host': 'open-weather13.p.rapidapi.com',
            'x-rapidapi-key':'b9d38f1055msh697ebb25aaf29cdp1148efjsn5239d90f5f7b'
    })
    if response.status_code != 200:
        raise Exception(response.json()['message'])
    return response.json()

chatmodel = ChatOllama(model='llama3.2:3b',temperature=0)

prompt = hub.pull('hwchase17/react',include_model=True)

ai_agent = create_react_agent(
    llm=chatmodel,
    prompt=prompt,
    tools=[get_weather_data]
)

agent_executor = AgentExecutor(
    agent=ai_agent,
    tools=[get_weather_data],
    verbose=True,
    max_iterations=1,
    handle_parsing_errors=True
    # callbacks=
)

try:
    output = agent_executor.invoke({'input':"What is min and max temperature in Kolkata"})
except Exception as e:
    output = str(e)
print(output)    
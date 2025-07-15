from langchain_core.tools import tool, InjectedToolArg
from langchain_ollama import ChatOllama
from typing import Annotated
import requests
from datetime import datetime
from langchain_core.messages import HumanMessage
import json

@tool
def get_currency_conversion_factor(base_curr: str, target_curr: str):
    """This tool will find out the conversion rate between the base currency and target currency"""
    date_str = datetime.now().strftime('%Y-%m-%d')
    base_url = f"https://currency-conversion-and-exchange-rates.p.rapidapi.com/timeseries?start_date={date_str}&end_date={date_str}&base={base_curr}&symbols={target_curr}"
    response = requests.get(
        url=base_url,
        headers={
            'X-RapidAPI-Key':'b9d38f1055msh697ebb25aaf29cdp1148efjsn5239d90f5f7b',
            'x-rapidapi-host':'currency-conversion-and-exchange-rates.p.rapidapi.com'
        })
    if response.status_code != 200:
        return 0.0
    return response.json()

@tool
def convert(base_value: float, conversion_rate: Annotated[float, InjectedToolArg]):
    """given a currency conversion rate; this function calculates the target currency value from a given base currency value"""
    return base_value * conversion_rate

# response = get_currency_conversion_factor.invoke({'base_curr':'USD','target_curr':'INR'})
# print(response)

chatmodel = ChatOllama(model='llama3.2:3b')

tool_model = chatmodel.bind_tools([get_currency_conversion_factor, convert])

messages = [
    HumanMessage('What is current conversion rate between USD and INR and on the basis of that, what is 15 USD in INR?')
]

ai_response = tool_model.invoke(messages)
# print(ai_response)
messages.append(ai_response)

actual_tool_calls = ai_response.tool_calls
today_str = datetime.now().strftime('%Y-%m-%d')
conversion_rate = 0.0
for tool_call in actual_tool_calls:
    if tool_call['name'] == 'get_currency_conversion_factor':
        response1 = get_currency_conversion_factor.invoke(tool_call)
        target_curr = tool_call['args']['target_curr']
        # print(response1.content)
        conversion_rate = json.loads(response1.content)['rates'][today_str][target_curr]
        # print(conversion_rate)
        messages.append(response1)
    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        respone2 = convert(tool_call)
        messages.append(tool_call)

final_response = tool_model.invoke(messages)
print(final_response.content)
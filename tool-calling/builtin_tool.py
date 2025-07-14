from langchain_community.tools import DuckDuckGoSearchResults, ShellTool

search = DuckDuckGoSearchResults(output_format='json')
search_response = search.invoke('indian stock market news')
print(search_response)

shell = ShellTool()
shell_response = shell.invoke('whoami')
print(shell_response)
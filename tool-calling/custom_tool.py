from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
@tool
def multiply(a, b):
    """Performs multiplication between two numbers a and b"""
    return a * b

# response = multiply.invoke({'a':3, 'b':5})
# print(response)

class InputSchema(BaseModel):
    a: int = Field(required=True, description='First number for add')
    b: int = Field(required=True, description='Second number for add')

add = lambda a,b: (a + b)

add_tool = StructuredTool.from_function(
    func=add,
    name='add function',
    description='add two numbers',
    args_schema=InputSchema
)

# response = add_tool.invoke({'a':34, 'b':56})
# print(response)
class SubtractTool(BaseTool):
    name: str = 'Subtract tool'
    description: str = 'tool to perform subtraction between two numbers a and b'
    args_schema: Type[BaseModel] = InputSchema

    def _run(self, a, b):
        return a - b
    
subtract_tool = SubtractTool()
response = subtract_tool.invoke({'a':55, 'b':42})
print(response)
from langchain.agents import Tool, AgentOutputParser, AgentType, initialize_agent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union, Callable, Type, Optional, Dict
from langchain.prompts import StringPromptTemplate
import re
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from prompts import prompt_start_template, prompt_mid_template, prompt_final_template
from langchain.chains import LLMChain
import json

class CustomLLMChain(LLMChain):
    def run(self, query: str, *args, **kwargs) -> Dict[str, Any]:
        raw_output = super().run(query, *args, **kwargs)
        print(raw_output)
        return self.parse_output(raw_output)
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        return parse_llm_output(output)

def parse_llm_output(output: str) -> list:
    entities_match = re.search(r"Entities: (\[.*?\])", output)
    if entities_match:
        entities = json.loads(entities_match.group(1))
        return entities
    else:
        return []
    
class CustomLLMChain(LLMChain):
    def run(self, query: str, *args, **kwargs) -> Dict[str, Any]:
        raw_output = super().run(query, *args, **kwargs)
        print(raw_output)
        return self.parse_output(raw_output)
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        return parse_llm_output(output)

def parse_llm_output(output: str) -> list:
    entities_match = re.search(r"Entities: (\[.*?\])", output)
    if entities_match:
        entities = json.loads(entities_match.group(1))
        return entities
    else:
        return []
    
# Set up a prompt template
class PubmedAgentPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class FinalAgentOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Return the entire LLM output without parsing
        return AgentFinish(
            return_values={"output": llm_output},
            log=llm_output,
        )

class PubmedAgentOuputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Findings:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Findings:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


valid_answers = ['Action', 'Final Answer']
valid_tools = ['Retrieve articles from Pubmed']

class CustomAgentGuidance:
    def __init__(self, guidance, tools, context, num_iter=5):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter
        self.context = context

    def do_tool(self, tool_name, actInput):
        return self.tools[tool_name](actInput)
    
    def __call__(self, query):
        prompt_start = self.guidance(prompt_start_template)
        result_start = prompt_start(question=query, valid_answers=valid_answers, context=self.context)
        print(result_start)

        result_mid = result_start
        
        for _ in range(self.num_iter - 1):
            if result_mid['answer'] == 'Final Answer':
                break
            history = result_mid.__str__()
            prompt_mid = self.guidance(prompt_mid_template)
            result_mid = prompt_mid(history=history, do_tool=self.do_tool, valid_answers=valid_answers, valid_tools=valid_tools)
            print(result_mid)
        if result_mid['answer'] != 'Final Answer':
            history = result_mid.__str__()
            prompt_mid = self.guidance(prompt_final_template)
            result_final = prompt_mid(history=history, do_tool=self.do_tool, valid_answers=['Final Answer'], valid_tools=valid_tools)
        else:
            history = result_mid.__str__()
            prompt_mid = self.guidance(history + "{{gen 'fn' stop='\\n'}}")
            result_final = prompt_mid()
            print(result_final)
        return result_final
    
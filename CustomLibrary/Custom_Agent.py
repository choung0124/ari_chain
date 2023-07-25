from langchain.agents import Tool, AgentOutputParser, AgentType, initialize_agent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union, Callable, Type, Optional, Dict
from langchain.prompts import StringPromptTemplate
import re
from langchain.schema import AgentAction, AgentFinish, OutputParserException
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
    entities_match = re.search(r"Entities: \[(.*?)\]", output)
    if entities_match:
        entities = entities_match.group(1)
        # Split the entities by comma and strip the surrounding spaces and quotes
        entities = [e.strip().strip('"') for e in entities.split(',')]
        return entities
    else:
        return []
    
class CustomLLMChainAdditionalEntities(LLMChain):
    def run(self, *args, query: str = "", **kwargs) -> Dict[str, Any]:
        raw_output = super().run(query=query, *args, **kwargs)
        print(raw_output)
        return self.parse_output_additional(raw_output)
    
    def parse_output_additional(self, output: str) -> Dict[str, Any]:
        return parse_llm_output_additional(output)

def parse_llm_output_additional(output: str) -> list:
    entities_match = re.search(r"Additional Entities: \[(.*?)\]", output)
    if entities_match:
        entities = entities_match.group(1)
        # Split the entities by comma and strip the surrounding spaces and quotes
        entities = [e.strip().strip('"') for e in entities.split(',')]
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


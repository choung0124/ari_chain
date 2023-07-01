from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, AgentType, initialize_agent
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Tuple, Any, Union, Callable, Type, Optional, Dict
from langchain.prompts import BaseChatPromptTemplate, StringPromptTemplate
import re
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import RWKV, CTransformers
from auto_gptq import AutoGPTQForCausalLM
import requests
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from prompts import prompt_start_template, prompt_mid_template, prompt_final_template
from langchain.chains import LLMChain
import json
from transformers import pipeline, AutoModelForCausalLM
import gc
import psycopg2
from langchain.embeddings import HuggingFaceEmbeddings

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
    
def get_similar_compounds(drug_name, top_n):
    # Get CID of the drug from PubChem
    pubchem_cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
    response = requests.get(pubchem_cid_url)
    cid = response.json()['IdentifierList']['CID'][0]  # assuming the first CID is the correct one

    # Get canonical SMILES of the drug from PubChem
    pubchem_smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    response = requests.get(pubchem_smiles_url)
    smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    print(smiles)
    # Use the ChEMBL API to find similar compounds
    chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/40?format=json"
    response = requests.get(chembl_url)
    print(len(response.json()['molecules']))
    similar_compounds = [molecule['pref_name'] for molecule in response.json()['molecules'] if 'pref_name' in molecule and molecule['pref_name'] is not None]
    # If there are less compounds than top_n, return all compounds
    if len(similar_compounds) < top_n:
        return similar_compounds

    # Otherwise, return the top_n similar compounds
    return similar_compounds[:top_n]

def get_umls_id(search_string: str) -> list:
    api_key = "7cc294c9-98ed-486b-add8-a60bd53de1c6"
    base_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    query = f"?string={search_string}&inputType=atom&returnIdType=concept&apiKey={api_key}"
    url = f"{base_url}{query}"
    
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        results = data["result"]["results"]
        if results:
            filtered_results = [result for result in results if search_string.lower() in result['name'].lower()]
            if filtered_results:
                top_result = filtered_results[0]
                result_string = f"Name: {top_result['name']} UMLS_CUI: {top_result['ui']}"
                return [result_string]
            else:
                return ["No results found."]
        else:
            return ["No results found."]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

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

def create_gptq_pipeline(model_name_or_path, model_basename, tokenizer):
    # Check if the model exists in memory and delete it
    if 'model' in globals():
        del model
        gc.collect()
        print("The existing model has been deleted from memory.")

    # Load the new model
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=True,
            quantize_config=None)

    return pipeline(
       "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=8192,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15
        )


def create_8bit_pipeline(model_name_or_path, tokenizer):
    # Check if the model exists in memory and delete it
    if 'model' in globals():
        del model
        gc.collect()
        print("The existing model has been deleted from memory.")

    # Load the new model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True)
    
    return pipeline(
       "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15
        )

def create_ggml_model(model_name_or_path, model_filename):
    config ={"temperature": 0.2,
             "repetition_penalty": 1.15,
             "max_new_tokens": 2048,
             "context_length": 8192,
             "gpu_layers": 10000000}
    
    llm = CTransformers(model=model_name_or_path, 
                            model_file=model_filename,
                            config=config)
    
    return llm

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
    

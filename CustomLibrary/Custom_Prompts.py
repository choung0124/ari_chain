Schema_simplify_template_no_vicuna = """You are an artificial intelligence assistant that generates cypher statements for the User.

USER: Generate Cypher statement to query a Neo4j graph database for all paths between the nodes. Use the names of the nodes provided below to help you generate a cypher statement. Only include the generated Cypher statement in your response. Strictly follow the syntax of the cypher query language, for Neo4j version 4.

Use the template below, make no changes except replacing x and y with the node names provided:
MATCH p = shortestPath((n)-[*]-(m))
WHERE n:Test AND m:Test
AND toLower(n.name) CONTAINS toLower(x)
AND toLower(m.name) CONTAINS toLower(y)
RETURN reduce(s = "", node in nodes(p) | s + node.name + " ") AS sentence, reduce(s = "", rel in relationships(p) | s + type(rel) + " ") AS relations

Use double quotations for the node names.
Node names: {question}

ASSISTANT:"""


Entity_type_Template1 = """### Instruction:
Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, and strictly follow this format:
Entities: [(Entity1_name, Entity1_ype), (Entity2_name, Entity2_type)]
Response example:
Entities: [("Alzheimer's Disease", "Disease"), ("Autophagy", "Biological_process")]

### Input: {input}

### Response:"""

Entity_type_Template = """### Human: Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, use double quotes instead of single quotes.
Your response should strictly follow this format:
Entities: [(Entity1_name, Entity1_ype), (Entity2_name, Entity2_type)]
Input: {input}
### Assistant:"""

Entity_type_Template_add = """### Human: Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, use double quotes instead of single quotes.
Your response should strictly follow this format:
Entities: [(Entity_name, Entity_type)
Input: {input}
### Assistant:"""

Entity_type_Template_airo = """USER: Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, and strictly follow this format:
Entities: [(Entity1_name, Entity1_ype), Entity2_name, Entity2_type)]
Response example:
Entities: [("Alzheimer's Disease", "Disease"), ("Autophagy", "Biological_process")]
Input: {input}
ASSISTANT:"""

Entity_Extraction_Template_airo = """You are an artificial intelligence assistant that extracts biomedical entities from the question asked by the user.
USER: Extract two entities from the question, do not try to answer the question. Include only the extracted entities as a python list of strings in your response. Your response should strictly follow this format:
Entities: [extracted entity1, extracted entity2]
Question: {input}
ASSISTANT:"""

Entity_Extraction_Template1 = """You are an artificial intelligence assistant that extracts biomedical entities from any given question.

### Instruction: 
Extract two entities from the question, do not try to answer the question. Include only the extracted entities as a python list of strings in your response. Your response should strictly follow this format, use speach marks around the entity names:
Entities: [extracted entity1, extracted entity2]
Question: {input}

### Response:
"""

Entity_Extraction_Template = """### Human: Extract two entities from the question, do not try to answer the question. Include only the extracted entities as a python list of strings in your response. Your response should strictly follow this format, use speach marks around the entity names:
Entities: [extracted entity1, extracted entity2]
Question: {input}
### Assistant:
"""

Additional_Entity_Extraction_Template = """### Human: Below is a list of entities and the question they were extracted from, do not try to answer the question. If there are any other entities in the question, extract those additional entities from the question. If there are no other entities, do nothing. Include only the extracted entities as a python list of strings in your response. Your response should strictly follow this format, use speach marks around the entity names:
Response(If there are no other entities):
Additional Entities: []
Response(If there are other entities):
Additional Entities: ["entity"]
Entities: {entities}
Question: {input}
### Assistant:
"""


Entity_Extraction_Template_alpaca = """You are an artificial intelligence assistant that extracts biomedical entities from any given question.

### Instruction: 
Extract two entities from the question, do not try to answer the question. Include only the extracted entities as a list in your response. Your response should strictly follow this format:
Entities: [extracted entity1, extracted entity2]
Question: {input}

### Response:
"""

Coherent_sentences_template1 = """USER: BEGININPUT
BEGINCONTEXT
ENDCONTEXT
{input}
ENDINPUT
BEGININSTRUCTION
Turn these into coherent sentences
ENDINSTRUCTION

ASSISTANT:
"""
Coherent_sentences_template2 = """USER: 
Turn these into coherent sentences
{input}

ASSISTANT:
"""
Coherent_sentences_template3 = """### Instruction: 
Turn these into coherent sentences
{input}

### Response:
"""

Coherent_sentences_template = """### Instruction: 
Summarize this information, include only the generated summary in your response:
{input}

### Response:
"""

Graph_Answer_Gen_Template_alpaca = """### Instruction: Give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Context: 
{input}

### Response:
"""

Graph_Answer_Gen_Template = """### Human: Give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Context: 
{input}
### Assistant: """

Graph_Answer_Gen_Template_airo = """USER: BEGININPUT
BEGINCONTEXT
Indirect relations between {source} and {target}
ENDCONTEXT
{multihop_relations}
ENDINPUT
BEGININPUT
BEGINCONTEXT
Direct relations from {source}
ENDCONTEXT
{direct_relations_source}
ENDINPUT
BEGININPUT
BEGINCONTEXT
Direct relations from {target}
ENDCONTEXT
{direct_relations_target}
ENDINPUT
BEGININPUT
BEGINCONTEXT
Relations between the targets of {source} and {target}
ENDCONTEXT
{inter_relations}
ENDINPUT
BEGININSTRUCTION
{question}
ENDINSTRUCTION
 ASSISTANT:
"""

Final_chain_template = """### Instruction: 
Answer the question, using ithe information below:
Context: {input}
Relevant abstracts: {relevant_pubmed_abstracts}
Question: {question}

### Response:
"""

CYPHER_GENERATION_TEMPLATE = """You are an artifical intelligence assistant that generates Cypher statements to answer questions from a user.

USER: Generate Cypher statement to query a Neo4j graph database for all information that is needed to answer the question. Use only the provided relationship types and properties in the schema. Make sure you find indirect relations too. Use the UMLS_CUIs provided below to help you generate a cypher statement. Only include the generated Cypher statement in your response. Strictly follow the syntax of the cypher query language, for Neo4j version 4.
Here is the Graph database schema: {schema}
Here are the UMLS IDs you should use in your cypher statement: {UMLS_context}
The question is: {question}

ASSISTANT:"""

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

Vicuna_QA_PROMPT_TEMPLATE = """You are a biomedical research assistant gives detailed answers to a user's questions, using the information given to you.
USER: give a detailed response to the question using the context below, the context is sourced from a curated biomedical knowledge graph.
Context:
{context}
The question is: {input}

ASSISTANT:"""

Med_alpaca_QA_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: Give a detailed answer to the question using the context below, try to infer a biological pathway from the context:
Context: {context}
The question is: {input}

### Response:"""

PUBMED_AGENT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: Perform further research based on the context given to you. You have access to the following tools:{tools}. Strictly use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Findings: The results of your research

Context: {input}

### Response:
{agent_scratchpad}"""

PUBMED_AGENT_TEMPLATE_VICUNA = """USER: Perform further research based on the context given to you. You have access to the following tools: {tools}
Strictly use the following format:

Thought: you should always think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(this Thought/Action/Action Input/Observation can repeat many times, repeat until you have enough information to infer a biological pathway from the context)
Thought: I now know the final answer
Findings: The results of your research

Context: {input}

ASSISTANT:
{agent_scratchpad}"""

prompt_start_template2 = """USER: Using the context and tools given to you, give an answer to the question, explaining your thought process step by step. You have access to the following tools:

Retrieve articles from Pubmed: Useful for when you need to retrieve information from Pubmed.

Use the information from this context in your answer: {{context}}

Strictly use the following format, repeat until you have enough information to give a definitive answer:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Retrieve articles from Pubmed]
Action Input: the input to the action, should be a question.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation repeats at least 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, with reasoning

The question is: {{question}}

ASSISTANT:
Question: {{question}}
Thought: {{gen 't1' stop='\\n'}}
{{select 'answer' logprobs='logprobs' options=valid_answers}}: """

prompt_start_template = """USER: Based on the context below, perform further research until you can explain the underlying mechanism in the context. You have access to the following tools:

Retrieve articles from Pubmed: Useful for when you need to retrieve information from Pubmed.

Perform further research based on this context until you can give an answer to the question, with reasoning: {{context}}

Strictly use the following format, repeat until you have enough information to answer the question:

Context: the context you have been given
Question: the input question you must answer
Thought: I should understand the context, then plan how I will gather more information related to the context
Action: the action to take, should be one of [Retrieve articles from Pubmed]
Action Input: the input to the action, should be a single python string
Observation: the result of the action
... (this Thought/Action/Action Input/Observation repeats at least 5 times)
Thought: I now have enough information to explain the underlying mechanisms in the context
Final Answer: the final answer to the original input question, with reasoning

The question is: {{question}}

ASSISTANT:
Context: {{context}}
Question: {{question}}
Thought: {{gen 't1' stop='\\n'}}
{{select 'answer' logprobs='logprobs' options=valid_answers}}: """

prompt_mid_template = """{{history}}{{select 'tool_name' options=valid_tools}}
Action Input: {{gen 'actInput' stop='\\n'}}
Observation: {{do_tool tool_name actInput}}
Thought: {{gen 'thought' stop='\\n'}}
{{select 'answer' logprobs='logprobs' options=valid_answers}}: """

prompt_final_template = """{{history}}{{select 'tool_name' options=valid_tools}}
Action Input: {{gen 'actInput' stop='\\n'}}
Observation: {{do_tool tool_name actInput}}
Thought: {{gen 'thought' stop='\\n'}}
{{select 'answer' options=valid_answers}}: {{gen 'fn' stop='\\n'}}"""


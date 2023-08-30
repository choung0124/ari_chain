Final_Answer_Template = """### Human: Give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Your Previous Answers:
{CKG_Answer}
{Pharos_Answer}
{OpenTargets_Answer}
{Predicted_Answer}
Question: {question}
### Assistant: """

Final_Answer_Template_Alpaca = """### Instruction:
Give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Your Previous Answers:
{input}

### Input:
Quesiton: {question}

### Response: """

Entity_type_Template_Alpaca = """### Instruction:
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

Entity_type_Template_Upstage = """### System: Give the entity types of all the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Your response should be a list of tuples, use double quotes instead of single quotes. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should strictly follow this format:
Entities: [("Entity1_name", "Entity1_ype"), ("Entity2_name", "Entity2_type")]

### User:
Input: {input}

### Assistant:"""

OPC_Entity_type_Template = """### Instruction:

Give the entity types of all the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Your response should be a list of tuples, use double quotes instead of single quotes. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should strictly follow this format:
Entities: [("Entity1_name", "Entity1_ype"), ("Entity2_name", "Entity2_type")]
Input: {input}

### Response:"""

Entity_type_Template_add = """### Human: Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, use double quotes instead of single quotes.
Your response should strictly follow this format:
Entities: [(Entity_name, Entity_type)]
Input: {input}
### Assistant:"""

Entity_type_Template_add_Upstage = """### System: Give the entity types of the bimedical entity/entities provided in the input, choose the entity types that accurately represent each/the entity provided in the input. Your response should be a list of tuples, use double quotes instead of single quotes. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units
Your response should strictly follow this format:
Entities: [(Entity_name, Entity_type)]

### User:
Input: {input}

### Assistant:"""

Entity_type_Template_add_Alpaca = """### Instruction:
Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, use double quotes instead of single quotes.
Your response should strictly follow this format:
Entities: [(Entity_name, Entity_type)]

### Input: {input}

### Response:
"""
Entity_type_Template_add_Vicuna= """USER: Tell me the entity types of the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should be a list of tuples, use double quotes instead of single quotes.
Your response should strictly follow this format:
Entities: [(Entity_name, Entity_type)]
Input: {input}
ASSISTANT:"""

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

Entity_Extraction_Template = """### Human: Extract two entities from the question, do not try to answer the question. Include only the extracted entities as a python list of strings in your response. Your response should strictly follow this format:
Entities: ["extracted entity1", "extracted entity2"]
Question: {input}
### Assistant:
"""

Entity_Extraction_Template_Upstage = """### System: Extract only two entities from the question, do not try to answer the question. Include only the two extracted entities as a python list of strings in your response. Your response should strictly follow this format, use speach marks around the entity names:
Entities: ["extracted entity1", "extracted entity2"]

### User:
Question: {input}

### Assistant:
"""

Entity_Extraction_Template_alpaca = """You are an artificial intelligence assistant that extracts biomedical entities from any given question.

### Instruction: 
Extract two entities from the question, do not try to answer the question. Include only the extracted entities as a list in your response. Your response should strictly follow this format:
Entities: [entity1, entity2]
Do not use speachmarks or quotation marks around the entity names.
Question: {input}

### Response:
"""

Additional_Entity_Extraction_Template = """### Human: Do not answer the question. Below is a list of entities and the question they were extracted from, if there are any other entities in the question, extract those additional entities from the question. If there are no other entities, do nothing. Do not use speachmarks or quotation marks around the entity names.
Your response should strictly follow this format if there is only one other biomedical entity in the question:
Additional Entities: ["entity"]
Your response should strictly follow this format if there are no other biomedical entities in the question:
Additional Entities: []
Your response should strictly follow this format if there are multiple other biomedical entities in the question:
Additional Entities: ["entity1", "entity2", "entity3"]
Entities: {entities}
Question: {input}
### Assistant:
"""

Additional_Entity_Extraction_Template_Upstage = """### System:
Do not answer the question. Below is a list of entities and the question they were extracted from. If there are any other entities that may help answer the question, extract those additional entities from the question. If there are no other entities, do nothing.
Your response should strictly follow this format if there is only one other biomedical entity in the question:
Additional Entities: ["entity"]
Your response should strictly follow this format if there are no other biomedical entities in the question:
Additional Entities: []
Your response should strictly follow this format if there are multiple other biomedical entities in the question:
Additional Entities: ["entity1", "entity2", "entity3"]
### User:
Entities: {entities}
Question: {input}

### Assistant:
"""

OPC_Entity_Extraction_Template_Upstage = """### System:
Do not answer the question. Extract all biomedical entities from the question.
Your response should strictly follow this format if there is only one biomedical entity in the question:
Additional Entities: ["entity"]
Your response should strictly follow this format if there are two biomedical entities in the question:
Additional Entities: ["entity1", "entity2"]
Your response should strictly follow this format if there are three biomedical entities in the question:
Additional Entities: ["entity1", "entity2", "entity3"]
Your response should strictly follow this format if there are four biomedical entities in the question:
Additional Entities: ["entity1", "entity2", "entity3", "entity4"]
Your response should strictly follow this format if there are no biomedical entities in the question:
Additional Entities: []
### User:
Question: {input}

### Assistant:
"""

OPC_Entity_Extraction_Template_Guanaco = """### Instruction:

Do not answer the question. Extract all entities from the question below. Do not put square brackets around each extracted entity.

Response template examples:

Your response should look like this if there is only one entity in the question:
Entities: ["entity"]

Your response should look like this if there are two entities in the question:
Entities: ["entity1", "entity2"]

Your response should look like this if there are three entities in the question:
Entities: ["entity1", "entity2", "entity3"]

Your response should look like this if there are four entities in the question:
Entities: ["entity1", "entity2", "entity3", "entity4"]

Your response should strictly follow this format if there are no entities in the question:
Entities: []

PLEASE DO NOT DO SOMETHING LIKE THIS:
Entities: [["entity1"], ["entity2"]]

Question: {input}

### Response:"""


Additional_Entity_Extraction_Template_Alpaca = """### Instruction:
Do not answer the question. Below is a list of entities and the question they were extracted from, if there are any other entities that may help answer the question, extract those additional entities from the question. If there are no other entities, do nothing.:

Your response should strictly follow this format if there are other biomedical entities in the question:
Additional Entities: ["entity"]

Your response should strictly follow this format if there are no other biomedical entities in the question:
Additional Entities: []

For Examples:
Entities: ["Donepezil", "Mirodenafil"]
Question: "Could there be a synergistic drug-drug interaction between Donepezil and Mirodneafil for Alzheimer's Disease?"
Response: ["Alzheimer's Disease"]

Entities ["Alzheimer's Disease", "Autophagy"]
Question: What is the relationship between Alzheimer's Disease and Autophagy?
Response: []

### Input:
Entities: {entities}
Question: {input}

### Response:
"""

Additional_Entity_Extraction_Template_Vicuna = """USER: Do not answer the question. Below is a list of entities and the question they were extracted from, if there are any other entities in the question, extract those additional entities from the question. If there are no other entities, do nothing.:

Your response should strictly follow this format if there are other biomedical entities in the question:
Additional Entities: ["entity"]

Your response should strictly follow this format if there are no other biomedical entities in the question:
Additional Entities: []

Entities: {entities}
Question: {input}
ASSISTANT:
"""

Coherent_sentences_template = """### Instruction: 
Summarize this information, include only the generated summary in your response:
{input}

### Response:
"""

Graph_Answer_Gen_Template_alpaca = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: As if you were a healthcare professional/biomedical researcher, give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
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

Graph_Answer_Gen_Template_Upstage = """### System: Give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.

### User:
Question: {question}
Context:
{input}

### Assistant: """

Graph_Answer_Gen_Template2 = """### Human: Below is a question, your previous answer to the question, and some additional context. Using the additional context, improve your answer. Use the additional context as if it were the results of further research to answer the question. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Previous Answer: {previous_answer}
Context: 
{input}
### Assistant: """

Graph_Answer_Gen_Template2_alpaca = """### Instruction: Below is an instruction that describes a task. Write a response that appropriately completes the request.

Below is a question, your previous answer to the question, and some additional context. As if you were a healthcare professional/biomedical researcher, use the additional context to improve your answer. Use the additional context as if it were the results of further research to answer the question. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Previous Answer: {previous_answer}
Context: 
{input}

### Response: """

Graph_Answer_Gen_Template3 = """### Human: Below is a question, your previous answer to the question, and some additional context. The additional context contains paths and relations related to similar entities to the entities in the question. Think of the paths and relations of the similar entities as predicted paths and relations of the enties in the question. This is based on the premise that similar drugs have similar mechanisms and similar diseases have similar pathologies, etc. Using the additional context, improve your answer. Use the additional context as if it were the results of further research to answer the question. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Previous Answer: {previous_answer}
Context: 
{input}
### Assistant: """

Graph_Answer_Gen_Template3_alpaca = """### Instruction: Below is an instruction that describes a task. Write a response that appropriately completes the request.

Below is a question, your previous answer to the question, and some additional context. The additional context contains paths and relations related to similar entities to the entities in the question. Think of the paths and relations of the similar entities as predicted paths and relations of the enties in the question. This is based on the premise that similar drugs have similar mechanisms and similar diseases have similar pathologies, etc. Using the additional context, improve your answer. Use the additional context as if it were the results of further research to answer the question. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Previous Answer: {previous_answer}
Context: 
{input}

### Response: """

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

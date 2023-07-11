import streamlit as st
from transformers import logging
from langchain.llms import TextGen
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import streamlit as st
import streamlit as st
from pyvis.network import Network
from CustomLibrary.Custom_Agent import CustomLLMChain, CustomLLMChainAdditionalEntities
from CustomLibrary.Custom_Prompts import Entity_type_Template_add, Entity_Extraction_Template_alpaca, Entity_type_Template_airo, Entity_Extraction_Template_airo, Entity_Extraction_Template,  Entity_type_Template, Additional_Entity_Extraction_Template
from CustomLibrary.App_Utils import(
    get_umls_info, 
    extract_entities, 
    get_names_list, 
    get_names_list, 
    get_entity_types, 
    get_additional_entity_umls_dict,
    create_and_display_network
)
from CustomLibrary.Graph_Visualize import parse_relationships_pyvis
from CustomLibrary.Graph_Class import KnowledgeGraphRetrieval
from CustomLibrary.Pharos_Graph_QA import PharosGraphQA
#Could there be a synergistic drug-drug interaction between lamotrigine and rivastigmine for lewy body dementia?
# Set logging verbosity
logging.set_verbosity(logging.CRITICAL)
@st.cache_data()
def initialize_models():
    model_url = "https://summit-area-your-parker.trycloudflare.com/"
    llm = TextGen(model_url=model_url, max_new_tokens=2048)
    #Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_alpaca, input_variables=["input"])
    Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template, input_variables=["input"])
    #Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_airo, input_variables=["input"])
    entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
    return llm, entity_extraction_chain

@st.cache_data()
def initialize_knowledge_graph():
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "NeO4J"
    return uri, username, password

st.set_page_config(layout="wide")
st.title("Multi-Hop Question Answering")

# Define the progress bar
progress_bar = st.empty()

# Define the callback function to update the progress bar
def progress_callback(progress):
    progress_bar.progress(progress)

question = st.text_input("Enter your question")
if question:
    additional_entity_extraction_prompt = PromptTemplate(template=Additional_Entity_Extraction_Template, input_variables=["input", "entities"])
    llm, entity_extraction_chain = initialize_models()
    uri, username, password = initialize_knowledge_graph()
    additional_entity_extraction_chain = CustomLLMChainAdditionalEntities(prompt=additional_entity_extraction_prompt, llm=llm, output_key="output",)
    
    Entity_type_prompt = PromptTemplate(template=Entity_type_Template, input_variables=["input"])
    Entity_type_prompt_add = PromptTemplate(template=Entity_type_Template_add, input_variables=["input"])
    #Entity_type_prompt = PromptTemplate(template=Entity_type_Template_airo, input_variables=["input"])
    Entity_type_chain = LLMChain(prompt=Entity_type_prompt, llm=llm)
    Entity_type_chain_add = LLMChain(prompt=Entity_type_prompt_add, llm=llm)
    
    if st.button("Check Interaction"):
        with st.spinner("Checking drug interaction..."):
            # Entity extraction
            entities, additional_entities = extract_entities(question, entity_extraction_chain, additional_entity_extraction_chain)

            entities_umls_ids = get_umls_info(entities)

            names_list = get_names_list(entities_umls_ids)

            entity_types = get_entity_types(Entity_type_chain, names_list)
            print(entity_types)

            if additional_entities:
                additional_entity_umls_dict = get_additional_entity_umls_dict(additional_entities, Entity_type_chain_add)
                print(additional_entity_umls_dict)
                knowledge_graph = KnowledgeGraphRetrieval(uri, username, password, llm, entity_types, additional_entity_types=additional_entity_umls_dict)
            else:
                knowledge_graph = KnowledgeGraphRetrieval(uri, username, password, llm, entity_types)

            # Query the knowledge graph
            graph_query = knowledge_graph._call(names_list, 
                                                question, 
                                                generate_an_answer=True, 
                                                related_interactions=True,
                                                progress_callback=progress_callback)
            
            context = graph_query["result"]
            all_rels = graph_query['all_rels']

            #rint(all_rels)
            print(len(all_rels))
            nodes = set()

            nodes, edges = parse_relationships_pyvis(all_rels)
            col1, col2 = st.columns([3, 2], gap="small")

            # Display the graph in the left column
            with col1:
                st.subheader("Network:")
                create_and_display_network(nodes, edges)
            # Display the answer in the right column

            col2.subheader("Answer:")
            col2.write(context)
            # Display the relationships below the columns

            st.divider()
            st.header("Pharos Graph QA")

            if additional_entities:
                additional_entity_umls_dict = get_additional_entity_umls_dict(additional_entities, Entity_type_chain_add)
                print(additional_entity_umls_dict)
                Pharos = PharosGraphQA(llm, entity_types, additional_entity_types=additional_entity_umls_dict)
            else:
                Pharos = PharosGraphQA(llm, entity_types)

            # Query the knowledge graph
            graph_query = knowledge_graph._call(names_list, 
                                                question, 
                                                generate_an_answer=True, 
                                                related_interactions=True,
                                                progress_callback=progress_callback)
            

            Pharos_Context = graph_query["result"]
            all_rels = graph_query['all_rels']
            pharos_nodes = set()

            pharos_nodes, pharos_edges = parse_relationships_pyvis(all_rels)
            col1, col2 = st.columns([3, 2], gap="small")

            # Display the graph in the left column
            with col1:
                st.subheader("Network:")
                create_and_display_network(pharos_nodes, pharos_edges)

            col2.subheader("Answer:")
            col2.write(Pharos_Context)
            # Display the relationships below the columns

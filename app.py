import re
import streamlit as st
from transformers import logging
from langchain.llms import TextGen
from Custom_Agent import get_umls_id, CustomLLMChain
from customGraphCypherQA import KnowledgeGraphRetrieval
from prompts import Entity_Extraction_Template_alpaca, Entity_type_Template_airo, Entity_Extraction_Template_airo, Entity_type_Template
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import ast
from graph_visualizer import parse_relationships
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit.components.v1 as components

def display_relationships(relationships):
    html = ""
    for relationship in relationships:
        nodes_and_relationships = relationship.split("->")
        for i in range(0, len(nodes_and_relationships), 2):
            html += f'<div class="node">{nodes_and_relationships[i]}</div>'
            if i+1 < len(nodes_and_relationships):
                html += f'<div class="relationship">{nodes_and_relationships[i+1]}</div>'
    html = f'<div class="relationship-graph">{html}</div>'
    components.html(html, height=600)

#Could there be a synergistic drug-drug interaction between lamotrigine and rivastigmine for lewy body dementia?
# Set logging verbosity
logging.set_verbosity(logging.CRITICAL)
@st.cache_data()
def initialize_models():
    model_url = "https://mongolia-steering-genealogy-manchester.trycloudflare.com/"
    llm = TextGen(model_url=model_url, max_new_tokens=512)
    Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_alpaca, input_variables=["input"])
    #Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_airo, input_variables=["input"])
    entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
    
    return llm, entity_extraction_chain

@st.cache_data()
def initialize_knowledge_graph():
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "NeO4J"
    return uri, username, password

st.title("Multi-Hop Question Answering")

# Define the progress bar
progress_bar = st.empty()

# Define the callback function to update the progress bar
def progress_callback(progress):
    progress_bar.progress(progress)

# CSS for the component
st.markdown("""
    <style>
    .relationship-graph {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 10px;
    }
    .node {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .relationship {
        color: #F44336;
    }
    </style>
""", unsafe_allow_html=True)


question = st.text_input("Enter your question")
if question:
    llm, entity_extraction_chain = initialize_models()
    uri, username, password = initialize_knowledge_graph()
    Entity_type_prompt = PromptTemplate(template=Entity_type_Template, input_variables=["input"])
    #Entity_type_prompt = PromptTemplate(template=Entity_type_Template_airo, input_variables=["input"])
    Entity_type_chain = LLMChain(prompt=Entity_type_prompt, llm=llm)
    if st.button("Check Interaction"):
        with st.spinner("Checking drug interaction..."):
            # Entity extraction
            result = entity_extraction_chain.run(question)
            entities = result
 
            entities_umls_ids = {}
            for entity in entities:
                umls_id = get_umls_id(entity)
                entities_umls_ids[entity] = umls_id

            names_list = []
            for entity, umls_info_list in entities_umls_ids.items():
                if umls_info_list:
                    umls_info = umls_info_list[0]
                    match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
                    if match:
                        umls_name = match.group(1)
                        umls_cui = match.group(2)
                        names_list.append(umls_name)
                    else:
                        names_list.append(entity)
                else:
                    names_list.append(entity)
            print(names_list)
            Entity_type_chain_result = Entity_type_chain.run(names_list)
            print(Entity_type_chain_result)
            start = Entity_type_chain_result.index("[")
            end = Entity_type_chain_result.index("]") + 1
            list_str = Entity_type_chain_result[start:end]
            extracted_types = ast.literal_eval(list_str)

            entity_types = {entity_info[0]: entity_info[1] for entity_info in extracted_types}
            print(entity_types)
            # Query the knowledge graph
            knowledge_graph = KnowledgeGraphRetrieval(uri, username, password, llm, entity_types)
            graph_query = knowledge_graph._call(names_list, 
                                                question, 
                                                generate_an_answer=True, 
                                                related_interactions=True,
                                                progress_callback=progress_callback)
            
            context = graph_query["result"]
            multi_hop_relationships = graph_query["multi_hop_relationships"]
            source_relationships = graph_query["source_relationships"]
            target_relationships = graph_query["target_relationships"]
            inter_direct_relationships = graph_query["inter_direct_relationships"]
            all_nodes = graph_query["all_nodes"]
            all_rels = graph_query['all_rels']

            associated_genes_string = graph_query.get("associated_genes_string")
            def enclose_in_quotes(relationships):
                quoted_relationships = []
                for relationship in relationships:
                    elements = relationship.split(' -> ')
                    quoted_elements = ['"{}"'.format(e) for e in elements]
                    quoted_relationship = ' -> '.join(quoted_elements)
                    quoted_relationships.append(quoted_relationship)
                return quoted_relationships
            print(all_rels)
            nodes = set()

            st.subheader("Result:")
            st.write("Answer:")
            st.write(context)
            # Assuming paths is a list of your paths
            nodes, edges = parse_relationships(all_rels)
            print(nodes)
            print(edges)
            node_objects = [Node(id=node, label=node, size=10) for node in nodes]
            edge_objects = [Edge(source=edge[0], target=edge[1]) for edge in edges]

            config = Config(
                height=500,
                width=700,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                directed=True,
                collapsible=True
            )
            agraph(nodes=node_objects, edges=edge_objects, config=config)

            st.write("Multi-Hop Relationships")
            st.write(multi_hop_relationships)
            formatted_multi_hop = [" → ".join(relationship.split("->")) for relationship in multi_hop_relationships]
            for relationship in formatted_multi_hop:
                st.markdown(relationship)
            display_relationships(multi_hop_relationships)
            st.write(f"Direct Relationships from {names_list[0]}")
            st.write(source_relationships)
            st.write(f"Direct Relationships from {names_list[1]}")
            st.write(target_relationships)
            st.write(f"Direct Relationships from targets of {names_list[0]} and {names_list[1]}")
            st.write(inter_direct_relationships)
            if associated_genes_string:
                st.write("Associated Genes:")
                st.write(associated_genes_string)



from CustomLibrary.Utils import get_umls_id
import re
import ast
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import webbrowser
import os
import colour
import json

def extract_entities(question, entity_extraction_chain, additional_entity_extraction_chain):
    result = entity_extraction_chain.run(question)
    entities = result

    additional_entities = additional_entity_extraction_chain.run(input=question, entities=entities)

    return entities, additional_entities


def get_umls_info(entities):
    entities_umls_ids = {}
    for entity in entities:
        umls_id = get_umls_id(entity)
        entities_umls_ids[entity] = umls_id

    return entities_umls_ids


def get_names_list(entities_umls_ids):
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

    return names_list


def get_entity_types(Entity_type_chain, names_list):
    Entity_type_chain_result = Entity_type_chain.run(names_list)

    start = Entity_type_chain_result.index("[")
    end = Entity_type_chain_result.index("]") + 1
    list_str = Entity_type_chain_result[start:end]
    extracted_types = ast.literal_eval(list_str)

    entity_types = {entity_info[0]: entity_info[1] for entity_info in extracted_types}

    return entity_types


def get_additional_entity_umls_dict(additional_entities, Entity_type_chain_add):
    entities_umls_ids = get_umls_info(additional_entities)

    additional_entity_umls_dict = {}

    for entity, umls_info_list in entities_umls_ids.items():
        if umls_info_list:
            umls_info = umls_info_list[0]
            match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
            if match:
                umls_name = match.group(1)
                umls_cui = match.group(2)
                additional_entity_umls_dict[entity] = umls_cui
            else:
                additional_entity_umls_dict[entity] = None
        else:
            additional_entity_umls_dict[entity] = None

    for entity, umls_cui in additional_entity_umls_dict.items():
        if umls_cui:
            entity_type_result = Entity_type_chain_add.run(entity)
            start = entity_type_result.index("[")
            end = entity_type_result.index("]") + 1
            list_str = entity_type_result[start:end]
            extracted_types = ast.literal_eval(list_str)
            entity_type = extracted_types[0] if extracted_types else None
            additional_entity_umls_dict[entity] = {"umls_cui": umls_cui, "entity_type": entity_type}
        else:
            additional_entity_umls_dict[entity] = {"umls_cui": None, "entity_type": None}

    return additional_entity_umls_dict

def create_and_display_network(nodes, edges, back_color, name):
    back_color = colour.Color(back_color)
    bg_color = back_color.get_hex()

    # darken the color by reducing the luminance
    # darken the color by reducing the luminance
    back_color.luminance *= 0.8  # reduce luminance by 20%
    border_color = back_color.get_hex()

    # darken the color even more for the nodes
    back_color.luminance *= 0.5  # reduce luminance by additional 50%
    node_color = back_color.get_hex()

    # Initialize Network with the hexadecimal color string
    net = Network(height='750px', width='100%', bgcolor=bg_color, font_color='black', directed=True)

    for node in nodes:
        net.add_node(node, label=node, title=node, color=node_color, url="http://example.com/{}".format(node))

    # add edges
    for edge in edges:
        net.add_edge(edge[0], edge[1], title=edge[2])
    net.toggle_physics(True)

    # save to HTML file
    net.save_graph(f'{name}network.html')

    # Create a border around the network using custom CSS
    st.markdown(
        f"""
        <style>
        .network {{
            border: 4px solid {border_color};
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    net.save_graph(f'{name}network.html')

    # Display in Streamlit within a div with the class "network"
    with st.spinner("Rendering network..."):
        html_string = open(f'{name}network.html', 'r').read()
        components.html(
            f"""
            <div class="network" style="display: flex; justify-content: center;">
                {html_string}
            </div>
            """, 
            width=1050, 
            height=750
        )

    # Add a button to open the network in full size in a new tab
    st.markdown(
        f'<a href="file://{os.path.realpath(f"{name}network.html")}" target="_blank">Open Network in Full Size</a>', 
        unsafe_allow_html=True
    )

from CustomLibrary.Utils import get_umls_id
import re
import ast
import streamlit as st
from pyvis.network import Network

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

def create_and_display_network(nodes, edges):
    net = Network(height='750px', 
                  width='100%', 
                  bgcolor='#dcfaf3', 
                  font_color='black',
                  directed=True,
                  )

    # add nodes
    for node in nodes:
        net.add_node(node, label=node, title=node, url="http://example.com/{}".format(node))

    # add edges
    for edge in edges:
        net.add_edge(edge[0], edge[1], title=edge[2])
    net.toggle_physics(True)

    # save to HTML file
    net.save_graph('network.html')

    # display in streamlit
    with st.spinner("Rendering network..."):
        st.components.v1.html(open('network.html', 'r').read(), height=750)

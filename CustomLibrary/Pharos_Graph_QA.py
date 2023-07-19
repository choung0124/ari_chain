from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from langchain.prompts import PromptTemplate

from CustomLibrary.Graph_Queries import (
    query_inter_relationships_direct1, 
    query_inter_relationships_between_direct,
    get_node_label)
from CustomLibrary.Graph_Utils import (
    select_paths, 
    select_paths2, 
)
from CustomLibrary.Custom_Prompts import Graph_Answer_Gen_Template2

from CustomLibrary.Pharos_Queries import (
    ligand_query,
    target_query,
    disease_query
)

def generate_answer(llm, source_list, target_list, inter_direct_list, inter_direct_inter, question, previous_answer, source, target, additional_list:Optional[List[str]]=None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template2, input_variables=["input", "question", "previous_answer"])
    #prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    source_rels = ', '.join(source_list)
    target_rels = ','.join(target_list)
    multi_hop_rels = inter_direct_list + inter_direct_inter
    multi_hop_sentences = ','.join(multi_hop_rels)
    sep_1 = f"Direct paths from {source}:"
    sep2 = f"Direct paths from {target}:"
    sep3 = f"Paths between the nodes in the direct paths of {source} and {target}:"
    if additional_list:
        additional_sentences = ','.join(additional_list)
        sep4 = f"Additional relations related to the question"
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences, sep4, additional_sentences])
    else:
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences])
    answer = gen_chain.run(input=sentences, question=question, previous_answer=previous_answer)
    print(answer)
    return answer

class PharosGraphQA:
    def __init__(self, uri, username, password, llm, entity_types, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary

    def _call(self, names_list, question, previous_answer, generate_an_answer, progress_callback=None):
        
        entities_list = list(self.entity_types.items())

        # The first entity is the source entity
        source_entity_name, source_entity_type = entities_list[0]

        # The second entity is the target entity
        target_entity_name, target_entity_type = entities_list[1]

        if source_entity_type == "Disease":
            formatted_source_rels, source_nodes = disease_query(source_entity_name, question)
        if source_entity_type == "Drug":
            formatted_source_rels, source_nodes= ligand_query(source_entity_name, question)
        if source_entity_type == "Gene":
            formatted_source_rels, source_nodes = target_query(source_entity_name, question)

        if target_entity_type == "Disease":
            formatted_target_rels, target_nodes = disease_query(target_entity_name, question)
        if target_entity_type == "Drug":
            formatted_target_rels, target_nodes= ligand_query(target_entity_name, question)
        if target_entity_type == "Gene":
            formatted_target_rels, target_nodes = target_query(target_entity_name, question)

        query_nodes = source_nodes + target_nodes
        query_nodes = set(query_nodes)
        graph_rels = formatted_source_rels + formatted_target_rels
        graph_rels = set(graph_rels)

        additional_entity_direct_graph_rels = set()
        additional_entity_nodes = set()
        
        if self.additional_entity_types is not None:
            additional_entity_rels = []
            additional_entity_paths = []
            additional_entity_direct_graph_rels = []
            for entityname, entity_info in self.additional_entity_types.items():
                if entity_info is not None and entity_info['entity_type'] is not None:  
                    entity_type = entity_info['entity_type'][1]  # Extract entity type from the nested dictionary
                    if entity_type == "Disease":
                        formatted_additional_entity_rels, additional_entity_nodes = disease_query(entityname, question)
                    elif entity_type == "Drug":
                        formatted_additional_entity_rels, additional_entity_nodes = ligand_query(entityname, question)
                    elif entity_type == "Gene":
                        formatted_additional_entity_rels, additional_entity_nodes = target_query(entityname, question)
                    query_nodes.update(additional_entity_nodes)
                    additional_entity_direct_graph_rels.extend(formatted_additional_entity_rels)
                    graph_rels.update(formatted_additional_entity_rels)

        
        names_set = set(names_list)
        #query_nodes.update(final_path_nodes)
        query_nodes = [name for name in query_nodes if name.lower() not in names_set]
        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        og_target_direct_relations = set()
        selected_inter_direct_nodes = set()
        inter_direct_unique_graph_rels = set()
        final_selected_target_direct_paths = []

        for node in query_nodes:
            target_direct_relations, inter_direct_graph_rels, source_and_target_nodes1, direct_nodes = query_inter_relationships_direct1(self.graph, node)
            if target_direct_relations:
                inter_direct_relationships, selected_nodes, inter_direct_unique_rels, selected_target_direct_paths = select_paths(target_direct_relations, question, 15, 3, progress_callback)
                og_target_direct_relations.update(inter_direct_relationships)
                selected_inter_direct_nodes.update(selected_nodes)
                inter_direct_unique_graph_rels.update(inter_direct_unique_rels)
                final_selected_target_direct_paths.append(selected_target_direct_paths)
                print("success")
                print(len(inter_direct_relationships))
                print(inter_direct_relationships)
            else:
                print("skipping")
                continue

        print("nodes before clustering and embedding")
        print(len(selected_inter_direct_nodes))
        
        final_inter_direct_relationships = list(og_target_direct_relations)
        final_selected_inter_direct_nodes = list(set(selected_inter_direct_nodes))
        final_inter_direct_unique_graph_rels = list(set(inter_direct_unique_graph_rels))
        print("number of unique inter_direct_relationships:")
        print(len(final_inter_direct_relationships))

        if final_inter_direct_relationships:
            target_inter_relations, inter_direct_inter_unique_graph_rels, source_and_target_nodes2 = query_inter_relationships_between_direct(self.graph, final_selected_inter_direct_nodes, query_nodes)
            final_inter_direct_inter_relationships, selected_inter_direct_inter_nodes, inter_direct_inter_unique_rels = select_paths2(target_inter_relations, question, 15, 30, progress_callback)
        else:
            final_inter_direct_relationships = []
            selected_inter_direct_nodes = []

            target_inter_relations, inter_direct_inter_unique_graph_rels, source_and_target_nodes2 = query_inter_relationships_between_direct(self.graph, query_nodes, query_nodes)
            if target_inter_relations:
                final_inter_direct_inter_relationships, selected_inter_direct_inter_nodes, inter_direct_inter_unique_rels = select_paths2(target_inter_relations, question, len(target_inter_relations), 10, progress_callback)
            else:
                final_inter_direct_inter_relationships = []
                selected_inter_direct_inter_nodes = []

        print("final_inter_direct_inter_relationships")
        print(len(final_inter_direct_inter_relationships))
        all_nodes = set()
        if selected_inter_direct_nodes:
            all_nodes.update(selected_inter_direct_nodes)
        if selected_inter_direct_inter_nodes:
            all_nodes.update(selected_inter_direct_inter_nodes)
        all_nodes.update(query_nodes)
        print("all nodes:")
        print(len(all_nodes))
        #print(all_nodes)

        all_unique_graph_rels = set()
        all_unique_graph_rels.update(graph_rels)
        all_unique_graph_rels.update(final_inter_direct_unique_graph_rels)
        all_unique_graph_rels.update(inter_direct_inter_unique_rels)
        all_unique_graph_rels.update(additional_entity_direct_graph_rels)


########################################################################################################
        
        if generate_an_answer == True and self.additional_entity_types is not None:
            #final_context = generate_answer_airo(llm=self.llm,
            final_context = generate_answer(llm=self.llm, 
                                            question=question,
                                            previous_answer=previous_answer,
                                            source_list=formatted_source_rels, 
                                            target_list=formatted_target_rels,
                                            inter_direct_list=final_inter_direct_relationships,
                                            inter_direct_inter = final_inter_direct_inter_relationships,
                                            source=names_list[0],
                                            target=names_list[1], 
                                            additional_list=additional_entity_direct_graph_rels
                                            )
        else:
            final_context = generate_answer(llm=self.llm, 
                                question=question,
                                previous_answer=previous_answer,
                                source_list=formatted_source_rels, 
                                target_list=formatted_target_rels,
                                inter_direct_list=final_inter_direct_relationships,
                                inter_direct_inter = final_inter_direct_inter_relationships,
                                source=names_list[0],
                                target=names_list[1]
                                )
            
                                            
        answer = final_context

        response = {"result": answer, 
                    "multi_hop_relationships": final_inter_direct_inter_relationships,
                    "source_relationships": formatted_source_rels,
                    "target_relationships": formatted_target_rels,
                    "inter_direct_relationships": final_inter_direct_relationships,
                    "inter_direct_inter_relationships": final_inter_direct_inter_relationships,
                    "all_nodes": all_nodes,
                    "all_rels": all_unique_graph_rels}
        return response

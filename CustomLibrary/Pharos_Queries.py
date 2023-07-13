import requests
import json
from typing import Optional
import numpy as np
from CustomLibrary.Graph_Utils import select_paths_pharos

class GraphQLClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}

    def execute_query(self, query, variables=None):
        data = {"query": query}
        if variables:
            data["variables"] = variables

        response = requests.post(self.endpoint, headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Query failed with status code: {} and message: {}".format(response.status_code, response.text))

def send_query(query, variables=None):
    client = GraphQLClient('https://pharos-api.ncats.io/graphql')
    return client.execute_query(query, variables)

      
def query_id(string, get_target: Optional[bool], get_disease:Optional[bool], get_drug:Optional[bool]):
    query = """
    query search($name: String!){
    autocomplete(name: $name) {
        value
        categories {
        category
        }
    }
    }
    """
    variables = {"name": string}
    response = send_query(query, variables)

    desired_category = None
    if get_target:
        desired_category = "Genes"
    elif get_disease:
        desired_category = "Diseases"
    elif get_drug:
        desired_category = "Drugs"

    if desired_category:
        for item in response['data']['autocomplete']:
            for category in item['categories']:
                if category['category'] == desired_category:
                    return item['value']
                
    return None


def query_disease_associated_targets(string):
    query = """
    query associatedTargets($name: String!){
    targets(filter: { associatedDisease: $name }) {
    targets(top: 5){
      name
      sym
      diseaseAssociationDetails {
        name
        dataType
        evidence
        }
        }
    }
    }
    """
    id = query_id(string, False, True, False)
    variables = {"name": id}
    response = send_query(query, variables)
    print(response)

    target_list = []
    formatted_targets = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            target_list.append(target['sym'])
            formatted_target = "{} -> ASSOCIATED_WITH -> {}".format(string, target['sym'])
            formatted_targets.append(formatted_target)

    paths = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            # Create a path for each target
            path = {
                'nodes': [string, target['sym']],
                'relationships': ['ASSOCIATED_WITH']
            }
            paths.append(path)  # Wrap the path dictionary in a list before appending


    return response, formatted_targets, target_list, paths

def query_target_associated_diseases(string):
    query = """
    query associatedDiseases($name: String!){
    diseases(filter: { associatedTarget: $name }){
        diseases(top: 5){
        name
        }
    }
    }
    """
    print
    id = query_id(string, True, False, False)
    variables = {"name": id}
    response = send_query(query, variables)
    print(response)
    disease_list = []
    if response.get('data') and response['data'].get('diseases') and response['data']['diseases'].get('diseases'):
        for disease in response['data']['diseases']['diseases']:
            disease_list.append(disease['name'])

    formatted_diseases = []
    if response.get('data') and response['data'].get('diseases') and response['data']['diseases'].get('diseases'):
        for disease in response['data']['diseases']['diseases']:
            formatted_disease = "{} -> ASSOCIATED_WITH -> {}".format(string, disease['name'])
            formatted_diseases.append(formatted_disease)

    paths = []
    if response.get('data') and response['data'].get('diseases') and response['data']['diseases'].get('diseases'):
        for disease in response['data']['diseases']['diseases']:
            # Create a path for each disease
            path = {
                'nodes': [string, disease['name']],
                'relationships': ['ASSOCIATED_WITH']
            }
            paths.append(path)  # Wrap the path dictionary in a list before appending

    return response, formatted_diseases, disease_list, paths


def query_protein_protein_interactions(string):
    query = """
    query proteinProteinInteractions($name: String!){
    targets(filter: { associatedTarget: $name }) {
        targets(top:5) {
        name
        sym
        }
    }
    }
    """
    print(query)
    id = query_id(string, True, False, False)
    variables = {"name": id}
    response = send_query(query, variables)
    print(response)
    target_list = []
    formatted_targets = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            target_list.append(target['sym'])
            formatted_target = "{} -> ASSOCIATED_WITH -> {}".format(string, target['sym'])
            formatted_targets.append(formatted_target)

    paths = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            # Create a path for each target
            path = {
                'nodes': [string, target['sym']],
                'relationships': ['ASSOCIATED_WITH']
            }
            paths.append(path)  # Wrap the path dictionary in a list before appending

    return response, formatted_targets, target_list, paths


def query_ligand_targets(string):
    query = """
    query ligandDetails($name: String!){
    ligand(ligid: $name) {
        name
        description
        activities {
        target {
            sym
        }
        type
        value
        }
    }
    }
    """
    print(query)
    name = query_id(string, False, False, True)
    if not name:
        raise Exception("No match found in the desired category for string: {}".format(string))
    variables = {"name": name}
    response = send_query(query, variables)
    print(response)
    
    activity_data = {}
    formatted_targets = []
    paths = []
    
    if response.get('data') and response['data'].get('ligand') and response['data']['ligand'].get('activities'):
        for activity in response['data']['ligand']['activities']:
            target_sym = activity['target']['sym']
            activity_type = activity['type']
            activity_value = activity['value']

            # Store activity data for later averaging
            if target_sym not in activity_data:
                activity_data[target_sym] = {}
            if activity_type not in activity_data[target_sym]:
                activity_data[target_sym][activity_type] = []
            activity_data[target_sym][activity_type].append(activity_value)
    
    # Average the values for each target and activity type
    for target_sym in activity_data:
        for activity_type in activity_data[target_sym]:
            avg_value = np.mean(activity_data[target_sym][activity_type])
            activity_data[target_sym][activity_type] = avg_value

            # Format the target and path
            if activity_type == '-':
                activity_type = "interacts with"
                formatted_target = "{} -> {} -> {}".format(string, activity_type, target_sym)
            else:
                activity_type = "{}={}".format(activity_type, avg_value)
                formatted_target = "{} -> {} (interacts with) -> {}".format(string, activity_type, target_sym)
            formatted_targets.append(formatted_target)
            
            # Create a path for each activity
            path = {
                'nodes': [string, target_sym],
                'relationships': [activity_type]
            }
            paths.append(path)  # Wrap the path dictionary in a list before appending

    # Convert the averaged data into a list of unique targets
    target_list = list(activity_data.keys())
    
    return response, formatted_targets, target_list, paths

### if Drug

def ligand_query(string, question, progress_callback=None):
    response, formatted_ligand_targets, ligand_target_list, paths = query_ligand_targets(string)
    first_ligand_target_target_paths = []
    first_ligand_target_disease_paths = []

    for target in ligand_target_list:
        response, ligand_target_formatted_ppi, ligand_target_ppi_target, ligand_target_target_paths = query_protein_protein_interactions(target)
        first_ligand_target_target_paths. append(ligand_target_target_paths)
        response, ligand_disease_formatted, ligand_disease_disease, ligand_target_disease_paths = query_target_associated_diseases(target)
        first_ligand_target_disease_paths.append(ligand_target_disease_paths)
    selected_ligand_target_target_paths, selected_ligand_target_target_paths_nodes, unique_ligand_target_target_paths_rels_list, ligand_target_target_paths_selected_paths_stage2 = select_paths_pharos(first_ligand_target_target_paths, 
                                                                                                                                                                                                 question, 
                                                                                                                                                                                                 15,
                                                                                                                                                                                                 5,
                                                                                                                                                                                                 progress_callback)
    selected_ligand_target_disease_paths, selected_ligand_target_disease_paths_nodes, unique_ligand_target_disease_paths_rels_list, ligand_target_disease_paths_selected_paths_stage2 = select_paths_pharos(first_ligand_target_disease_paths,
                                                                                                                                                                                                     question,
                                                                                                                                                                                                     15,
                                                                                                                                                                                                     5,
                                                                                                                                                                                                     progress_callback)
    
    final_formatted_list = unique_ligand_target_disease_paths_rels_list + unique_ligand_target_target_paths_rels_list + formatted_ligand_targets
    final_nodes_list = selected_ligand_target_disease_paths_nodes + selected_ligand_target_target_paths_nodes + ligand_target_list

    return final_formatted_list, final_nodes_list

def disease_query(string, question, progress_callback=None):
    response, formatted_disease_targets, disease_target_list, paths = query_disease_associated_targets(string)
    first_disease_target_target_paths = []
    first_disease_target_ppi_paths = []

    for target in disease_target_list:
        response, disease_target_formatted_ppi, disease_target_ppi_target, disease_target_target_paths = query_protein_protein_interactions(target)
        first_disease_target_target_paths.append(disease_target_target_paths)
        response, disease_target_ppi_formatted, disease_target_ppi_target, disease_target_ppi_paths = query_target_associated_diseases(target)
        first_disease_target_ppi_paths.append(disease_target_ppi_paths)

    selected_disease_target_target_paths, selected_disease_target_target_paths_nodes, unique_disease_target_target_paths_rels_list, disease_target_target_paths_selected_paths_stage2 = select_paths_pharos(first_disease_target_target_paths, 
                                                                                                                                                                                                 question, 
                                                                                                                                                                                                 15,
                                                                                                                                                                                                 5,
                                                                                                                                                                                                 progress_callback)
    selected_disease_target_ppi_paths, selected_disease_target_ppi_paths_nodes, unique_disease_target_ppi_paths_rels_list, disease_target_ppi_paths_selected_paths_stage2 = select_paths_pharos(first_disease_target_ppi_paths,
                                                                                                                                                                                                     question,
                                                                                                                                                                                                     15,
                                                                                                                                                                                                     5,
                                                                                                                                                                                                     progress_callback)
    
    final_formatted_list = unique_disease_target_ppi_paths_rels_list + unique_disease_target_target_paths_rels_list + formatted_disease_targets
    final_nodes_list = selected_disease_target_ppi_paths_nodes + selected_disease_target_target_paths_nodes + disease_target_list

    return final_formatted_list, final_nodes_list


def target_query(string, question, progress_callback=None):
    response, formatted_target_diseases, target_disease_list, paths = query_target_associated_diseases(string)
    first_target_disease_target_paths = []
    first_target_disease_ppi_paths = []

    for disease in target_disease_list:
        response, formatted_target_ppi_target_list, target_disease_ppi_target, target_disease_target_paths = query_disease_associated_targets(disease)
        first_target_disease_target_paths.append(target_disease_target_paths)
        response, target_ppi_formatted, target_ppi_target, target_disease_ppi_paths = query_protein_protein_interactions(disease)
        first_target_disease_ppi_paths.append(target_disease_ppi_paths)

    selected_target_disease_target_paths, selected_target_disease_target_paths_nodes, unique_target_disease_target_paths_rels_list, target_disease_target_paths_selected_paths_stage2 = select_paths_pharos(first_target_disease_target_paths, 
                                                                                                                                                                                                 question, 
                                                                                                                                                                                                 15,
                                                                                                                                                                                                 5,
                                                                                                                                                                                                 progress_callback)
    final_formatted_list = unique_target_disease_target_paths_rels_list + formatted_target_diseases
    final_nodes_list =  selected_target_disease_target_paths_nodes + target_disease_list

    return final_formatted_list, final_nodes_list

import requests
import json
from typing import Optional

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
    targets(top: 5) {
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

    formatted_targets = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            formatted_target = "{} -> ASSOCIATED_WITH -> {}".format(string, target['sym'])
            formatted_targets.append(formatted_target)

    target_list = []
    formatted_targets = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            target_list.append(target['sym'])
            formatted_target = "{} -> ASSOCIATED_WITH -> {}".format(string, target['sym'])
            formatted_targets.append(formatted_target)

    return response, target_list, formatted_targets

def query_target_associated_diseases(string):
    query = """
    query associatedDiseases($name: String!){
    diseases(filter: { associatedTarget: $name }){
        diseases{
        name
        }
    }
    }
    """
    id = query_id(string, True, False, False)
    variables = {"name": id}
    response = send_query(query, variables)

    disease_list = []
    if response.get('data') and response['data'].get('diseases') and response['data']['diseases'].get('diseases'):
        for disease in response['data']['diseases']['diseases']:
            disease_list.append(disease['name'])

    formatted_diseases = []
    if response.get('data') and response['data'].get('diseases') and response['data']['diseases'].get('diseases'):
        for disease in response['data']['diseases']['diseases']:
            formatted_disease = "{} -> ASSOCIATED_WITH -> {}".format(string, disease['name'])
            formatted_diseases.append(formatted_disease)

    return response, disease_list, formatted_diseases


def query_protein_protein_interactions(string):
    query = """
    query proteinProteinInteractions($name: String!){
    targets(filter: { associatedTarget: $name }) {
        targets {
        name
        sym
        }
    }
    }
    """
    id = query_id(string, True, False, False)
    variables = {"name": id}
    response = send_query(query, variables)

    formatted_targets = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            formatted_target = "{} -> ASSOCIATED_WITH -> {}".format(string, target['sym'])
            formatted_targets.append(formatted_target)

    target_list = []
    formatted_targets = []
    if response.get('data') and response['data'].get('targets') and response['data']['targets'].get('targets'):
        for target in response['data']['targets']['targets']:
            target_list.append(target['sym'])
            formatted_target = "{} -> ASSOCIATED_WITH -> {}".format(string, target['sym'])
            formatted_targets.append(formatted_target)

    return response, formatted_targets, target_list

def get_disease_targets_predictions(string):
    query = """
    query getTARGETSPredictions($name: String) {
    model: target(q: {sym: $name, uniprot: $name, stringid: $name}) {
        name
    }
    }
    """
    name = query_id(string, False, True, False)
    variables = {"name": name}
    response = send_query(query, variables)
    return response

print(get_disease_targets_predictions("lewy body dementia"))

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
    name = query_id(string, False, False, True)
    variables = {"name": name}
    response = send_query(query, variables)
    target_list = []
    formatted_targets = []
    if response.get('data') and response['data'].get('ligand') and response['data']['ligand'].get('activities'):
        for activity in response['data']['ligand']['activities']:
            target_sym = activity['target']['sym']
            activity_type = activity['type']
            activity_value = activity['value']
            target_list.append(target_sym)
            if activity_type == '-':
                activity_type = "interacts with"
                formatted_target = "{} -> {} -> {}".format(string, activity_type, target_sym)
            else:
                formatted_target = "{} -> {}={} (interacts with) -> {}".format(string, activity_type, activity_value, target_sym)
            formatted_targets.append(formatted_target)
    
    return target_list, formatted_targets



### if Drug


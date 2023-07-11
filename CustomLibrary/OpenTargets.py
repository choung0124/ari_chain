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


def query_open_targets(query, variables=None):
    client = GraphQLClient('https://api.platform.opentargets.org/api/v4/graphql')
    return client.execute_query(query, variables)

def query_open_targets_id(string, get_target: Optional[bool], get_disease:Optional[bool], get_drug:Optional[bool]):
    open_targets_search_query = """
    query search($queryString: String!) {
    search(queryString: $queryString){
        total
        hits {
        id
        entity
        description
        }
    }
    }
    """
    variables = {"queryString": string}
    response = query_open_targets(open_targets_search_query, variables)
    if get_target == True:
        for hit in response['data']['search']['hits']:
            if hit['id'].startswith('ENSG'):
                print(hit['id'])
                break
    if get_disease == True:
        for hit in response['data']['search']['hits']:
            if hit['id'].startswith('EFO') or hit['id'].startswith('MONDO'):
                print(hit['id'])
                break
    if get_drug == True:
        for hit in response['data']['search']['hits']:
            if hit['id'].startswith('CHEMBL'):
                print(hit['id'])
                break
    return hit['id']

def query_open_targets_disease_associated_targets(string):
    query = """
    query associatedTargets($efoId: String!) {
    disease(efoId: $efoId) {
        id
        name
        associatedTargets {
        count
        rows {
            target {
            id
            approvedSymbol
            }
            score
        }
        }
    }
    }
    """
    id = query_open_targets_id(string, False, True, False)
    variables = {"efoId": id}
    response = query_open_targets(query, variables)
    return response

def query_open_targets_target_associated_diseases(string):
    query = """query associatedDiseases($ensgId: String!) {
    target(ensemblId: $ensgId) {
        id
        approvedSymbol
        associatedDiseases {
        count
        rows {
            disease {
            id
            name
            }
            datasourceScores {
            id
            score
            }
        }
        }
    }
    }
    """
    id = query_open_targets_id(string, True, False, False)
    variables = {"ensgId": id}
    response = query_open_targets(query, variables)
    return response

def query_similar_diseases(string):
    query = """
    query similarEntities($efoId: String!) {
      disease(efoId: $efoId) {
        similarEntities {
          score
          id
          object {
            ... on Disease {
              id
              name
            }
          }
          __typename
        }
      }
    }
    """
    id = query_open_targets_id(string, False, True, False)
    variables = {"efoId": id}
    response = query_open_targets(query, variables)
    return response



def query_similar_drugs(string):
    query = """
    query similarDrugMOA($chemblId: String!) {
      drug(chemblId: $chemblId) {
        similarEntities {
          score
          id
          object {
            ... on Drug {
              id
              name
            }
          }
          __typename
        }
      }
    }
    """
    id = query_open_targets_id(string, False, False, True)
    variables = {"chemblId": id}
    response = query_open_targets(query, variables)
    return response


print(query_similar_drugs("mirodenafil"))


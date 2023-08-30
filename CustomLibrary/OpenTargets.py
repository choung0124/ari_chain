import requests
import json
from typing import Optional
from CustomLibrary.Graph_Utils import select_paths

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



def query_target_info(string, question, progress_callback=None):
    query = """query associatedDiseases($ensgId: String!) {
    target(ensemblId: $ensgId) {
        id
        approvedSymbol
        pathways {
        pathway
        }
        interactions {
        rows {
            targetB{
            approvedSymbol
            }
        }
        }
        associatedDiseases {
        rows {
            disease {
            id
            name
            }
            }
        }
        }
      }
    """
    id = query_open_targets_id(string, True, False, False)
    variables = {"ensgId": id}
    response = query_open_targets(query, variables)

    if response.get('data') and response['data'].get('target') and response['data']['target'].get('associatedDiseases'):
        associated_diseases = response['data']['target']['associatedDiseases']['rows']
        target_disease_paths = []
        target_disease_pathway_paths = []
        target_interactions = []

        for associated_disease in associated_diseases:
            disease_name = associated_disease['disease']['name']
            target_disease_path = {
                'nodes': [string, disease_name],
                'relationships': ['ASSOCIATED_WITH']
            }
            target_disease_paths.append(target_disease_path)

        target = response['data']['target']
        if target['pathways']:
            pathways = target['pathways']
            for pathway in pathways:
                pathway_name = pathway['pathway']
                target_disease_pathway_path = {
                    'nodes': [string, pathway_name],
                    'relationships': ['ASSOCIATED_WITH']
                }
                target_disease_pathway_paths.append(target_disease_pathway_path)

        if target['interactions']:
            interactions = target['interactions']['rows']
            for interaction in interactions:
                interaction_name = interaction['targetB']['approvedSymbol']
                target_disease_path = {
                    'nodes': [string, interaction_name],
                    'relationships': ['INTERACTS_WITH']
                }
                target_interactions.append(target_disease_path)


    selected_target_disease_paths, selected_target_disease_nodes, selected_target_disease_rels, selected_stage2 = select_paths(target_disease_paths, 
                                                                                                                               question,
                                                                                                                              len(target_disease_paths),
                                                                                                                              5,
                                                                                                                              progress_callback)
    selected_target_disease_pathway_paths, selected_target_disease_pathway_nodes, selected_target_disease_pathway_rels, selected_stage2 = select_paths(target_disease_pathway_paths,
                                                                                                                                                      question,
                                                                                                                                                      len(target_disease_pathway_paths),
                                                                                                                                                      5,
                                                                                                                                                      progress_callback)
    selected_target_interactions, selected_target_interaction_nodes, selected_target_interaction_rels, selected_stage2 = select_paths(target_interactions,
                                                                                                                                    question,
                                                                                                                                    len(target_interactions),
                                                                                                                                    5,
                                                                                                                                    progress_callback)
    
    final_paths = selected_target_disease_paths + selected_target_disease_pathway_paths + selected_target_interactions
    final_nodes = selected_target_disease_nodes + selected_target_disease_pathway_nodes + selected_target_interaction_nodes
    final_rels = selected_target_disease_rels + selected_target_disease_pathway_rels + selected_target_interaction_rels
                                                                                                                            
    return final_paths, final_nodes, final_rels

def query_disease_info(string, question, progress_callback=None):
    query = """
    query diseaseAnnotation($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets {
          rows {
            target {
              approvedSymbol
              pathways {
                pathway
              }
              interactions {
              rows {
                  targetB{
                  approvedSymbol
                  }
              }
              }
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

    if response.get('data') and response['data'].get('disease') and response['data']['disease'].get('associatedTargets'):
        associated_targets = response['data']['disease']['associatedTargets']['rows']
        disease_target_paths = []
        disease_target_pathway_paths = []
        disease_target_target_paths = []

        for associated_target in associated_targets:
            target_name = associated_target['target']['approvedSymbol']
            disease_target_path = {
                'nodes': [string, target_name],
                'relationships': ['ASSOCIATED_WITH']
            }
            disease_target_paths.append(disease_target_path)

            if associated_target['target']['pathways']:
                pathways = associated_target['target']['pathways']
                for pathway in pathways:
                    pathway_name = pathway['pathway']
                    disease_target_pathway_path = {
                        'nodes': [string, target_name, pathway_name],
                        'relationships': ['ASSOCIATED_WITH', 'ASSOCIATED_WITH']
                    }
                    disease_target_pathway_paths.append(disease_target_pathway_path)

            if associated_target['target']['interactions']:
                interactions = associated_target['target']['interactions']['rows']
                for interaction in interactions:
                    if interaction['targetB'] is not None:
                      interaction_name = interaction['targetB']['approvedSymbol']
                      disease_target_target_path = {
                          'nodes': [string, target_name, interaction_name],
                          'relationships': ['ASSOCIATED_WITH', 'INTERACTS_WITH']
                      }
                      disease_target_target_paths.append(disease_target_target_path)

    selected_disease_paths, selected_disease_nodes, unique_disease_rels, selected_stage2 = select_paths(disease_target_pathway_paths,
                                                                                                        question,
                                                                                                        len(disease_target_pathway_paths)//15,
                                                                                                        5,
                                                                                                        progress_callback)
    
    selected_disease_target_target_paths, selected_disease_target_target_nodes, unique_disease_target_target_rels, selected_stage2 = select_paths(disease_target_target_paths,
                                                                                                                                      question,
                                                                                                                                      len(disease_target_target_paths)//15,
                                                                                                                                      5,
                                                                                                                                      progress_callback)
    
    final_disease_target_target_paths = selected_disease_paths + selected_disease_target_target_paths
    final_disease_target_target_nodes = selected_disease_nodes + selected_disease_target_target_nodes
    final_unique_disease_target_target_rels = unique_disease_rels + unique_disease_target_target_rels
    

    return final_disease_target_target_paths, final_disease_target_target_nodes, final_unique_disease_target_target_rels


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

    chembl_id_list = []
    formatted_paths = []
    paths = []
    nodes = []
    if response.get('data') and response['data'].get('drug') and response['data']['drug'].get('similarEntities'):
        limit = 2
        count = 0
        for similar_entity in response['data']['drug']['similarEntities']:
            if count >= limit:
                break
            chembl_id = similar_entity['id']
            score = round(similar_entity['score'], 3)
            name = similar_entity['object'].get('name')
            if name:
                name = name.capitalize()  # Change this line
                nodes.append(name)
                chembl_id_list.append(name)
                similarity_string = "has a similarity score of"
                similarity_string = "{}={} to".format(similarity_string, score)

                formatted_path = "{} -> has a similarity score of {} to -> {}".format(string, score, name)
                formatted_paths.append(formatted_path)
                path = {
                    'nodes': [string, name],
                    'relationships': [similarity_string]
                }
                paths.append(path)
                count += 1
                
    return paths, formatted_paths, chembl_id_list, nodes


def query_predicted_target_info(string, question, progress_callback=None):
    similar_targets_query = """
      target(ensemblId: $ensgId) {
			similarEntities{
          score
          id
          object {
          ... on Target {
            id
            approvedSymbol
          }
        }
        __typename
      }
    }
  }
  """
    target_info_query = """
    query associatedDiseases($ensgId: String!) {
    target(ensemblId: $ensgId) {
        id
        approvedSymbol
        pathways {
        pathway
        }
        interactions {
        rows {
            targetB{
            approvedSymbol
            }
        }
        }
        associatedDiseases {
        rows {
            disease {
            id
            name
            }
            }
        }
        }
      }
    """
    id = query_open_targets_id(string, True, False, False)
    variables = {"ensgId": id}
    similar_targets_response = query_open_targets(similar_targets_query, variables)

    ensg_id_list = []
    formatted_paths = []
    paths = []
    nodes = []

    result_dict = {}

    if similar_targets_response.get('data') and similar_targets_response['data'].get('target') and similar_targets_response['data']['target'].get('similarEntities'):
        count = 0
        for similar_entity in similar_targets_response['data']['target']['similarEntities'][1:]:
            if count >= 4:
                break
            ensg_id = similar_entity['id']
            score = round(similar_entity['score'], 3)

            
            name = similar_entity['object'].get('approvedSymbol')
            if name:
                ensg_id_list.append(name)
                nodes.append(name)
                similarity_string = "has a similarity score of"
                similarity_string = "{}={} to".format(similarity_string, score)

                formatted_path = "{} -> has a similarity score of {} to -> {}".format(string, score, name)
                formatted_paths.append(formatted_path)
                path = {
                    'nodes': [string, name],
                    'relationships': [similarity_string]
                }
                paths.append(path)
                target_info_variables = {"ensgId": ensg_id}
                target_info_response = query_open_targets(target_info_query, target_info_variables)

                if target_info_response.get('data') and target_info_response['data'].get('target') and target_info_response['data']['target'].get('associatedDiseases'):
                    associated_diseases = target_info_response['data']['target']['associatedDiseases']['rows']
                    target_disease_paths = []
                    target_disease_pathway_paths = []
                    target_interactions = []

                    for associated_disease in associated_diseases:
                        disease_name = associated_disease['disease']['name']
                        target_disease_path = {
                            'nodes': [string, name, disease_name],
                            'relationships': [similarity_string, 'ASSOCIATED_WITH']
                        }
                        target_disease_paths.append(target_disease_path)

                    target = target_info_response['data']['target']
                    if target['pathways']:
                        pathways = target['pathways']
                        for pathway in pathways:
                            pathway_name = pathway['pathway']
                            target_disease_pathway_path = {
                                'nodes': [string, name, pathway_name],
                                'relationships': [similarity_string, 'ASSOCIATED_WITH']
                            }
                            target_disease_pathway_paths.append(target_disease_pathway_path)

                    if target['interactions']:
                        interactions = target['interactions']['rows']
                        for interaction in interactions:
                            interaction_name = interaction['targetB']['approvedSymbol']
                            target_disease_path = {
                                'nodes': [string, name, interaction_name],
                                'relationships': [similarity_string, 'INTERACTS_WITH']
                            }
                            target_interactions.append(target_disease_path)


                    selected_target_disease_paths, selected_target_disease_nodes, selected_target_disease_rels, selected_stage2 = select_paths(target_disease_paths, 
                                                                                                                                              question,
                                                                                                                                              len(target_disease_paths),
                                                                                                                                              3,
                                                                                                                                              progress_callback)
                    selected_target_disease_pathway_paths, selected_target_disease_pathway_nodes, selected_target_disease_pathway_rels, selected_stage2 = select_paths(target_disease_pathway_paths,
                                                                                                                                                                      question,
                                                                                                                                                                      len(target_disease_pathway_paths),
                                                                                                                                                                      3,
                                                                                                                                                                      progress_callback)
                    selected_target_interactions, selected_target_interaction_nodes, selected_target_interaction_rels, selected_stage2 = select_paths(target_interactions,
                                                                                                                                                    question,
                                                                                                                                                    len(target_interactions),
                                                                                                                                                    3,
                                                                                                                                                    progress_callback)
                    
                    mid_paths = selected_target_disease_paths + selected_target_disease_pathway_paths + selected_target_interactions
                    mid_nodes = selected_target_disease_nodes + selected_target_disease_pathway_nodes + selected_target_interaction_nodes
                    mid_rels = selected_target_disease_rels + selected_target_disease_pathway_rels + selected_target_interaction_rels

                    result_dict[name] = {"paths": mid_paths,
                                        "nodes": mid_nodes,
                                        "rels": mid_rels,
                                        "score": score
                                        }
                    count += 1
    return result_dict
    

def query_predicted_disease_info(string, question, progress_callback=None):
  similar_diseases_query = """
  query drugAnnotation($efoId: String!) {
    disease(efoId: $efoId) {
      similarEntities{
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
  disease_info_query = """query diseaseAnnotation($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets {
          rows {
            target {
              approvedSymbol
              pathways {
                pathway
              }
              interactions {
              rows {
                  targetB{
                  approvedSymbol
                  }
              }
              }
            }
            score
          }
        }
      }
    }
    """
  id = query_open_targets_id(string, False, True, False)
  variables = {"efoId": id}
  similar_disease_response = query_open_targets(similar_diseases_query, variables)
  
  result_dict = {}
  efo_id_list = []
  formatted_paths = []
  paths = []
  nodes = []
  final_disease_target_target_paths = []
  final_disease_target_target_nodes = []
  final_unique_disease_target_target_rels = []

  if similar_disease_response.get('data') and similar_disease_response['data'].get('disease') and similar_disease_response['data']['disease'].get('similarEntities'):
      count = 0
      for similar_entity in similar_disease_response['data']['disease']['similarEntities'][1:]:
          if count >= 4:
              break
          efo_id = similar_entity['id']
          score = round(similar_entity['score'], 3)
          
          name = similar_entity['object'].get('name')
          if name:
              name = name.capitalize()  # Change this line
              nodes.append(name)
              efo_id_list.append(name)
              similarity_string = "has a similarity score of"
              similarity_string = "{}={} to".format(similarity_string, score)

              formatted_path = "{} -> has a similarity score of {} to -> {}".format(string, score, name)
              formatted_paths.append(formatted_path)
              path = {
                  'nodes': [string, name],
                  'relationships': [similarity_string]
              }
              paths.append(path)
              disease_info_variables = {"efoId": efo_id}
              diseasse_info_response = query_open_targets(disease_info_query, disease_info_variables)

              if diseasse_info_response.get('data') and diseasse_info_response['data'].get('disease') and diseasse_info_response['data']['disease'].get('associatedTargets'):
                  associated_targets = diseasse_info_response['data']['disease']['associatedTargets']['rows']
                  disease_target_paths = []
                  disease_target_pathway_paths = []
                  disease_target_target_paths = []

                  for associated_target in associated_targets:
                      target_name = associated_target['target']['approvedSymbol']
                      disease_target_path = {
                          'nodes': [string, name, target_name],
                          'relationships': [similarity_string, 'ASSOCIATED_WITH']
                      }
                      disease_target_paths.append(disease_target_path)

                      if associated_target['target']['pathways']:
                          pathways = associated_target['target']['pathways']
                          for pathway in pathways:
                              pathway_name = pathway['pathway']
                              disease_target_pathway_path = {
                                  'nodes': [string, name, target_name, pathway_name],
                                  'relationships': [similarity_string, 'ASSOCIATED_WITH', 'ASSOCIATED_WITH']
                              }
                              disease_target_pathway_paths.append(disease_target_pathway_path)

                      if associated_target['target']['interactions']:
                          interactions = associated_target['target']['interactions']['rows']
                          for interaction in interactions:
                              if interaction['targetB'] is not None:
                                interaction_name = interaction['targetB']['approvedSymbol']
                                disease_target_target_path = {
                                    'nodes': [string, name, target_name, interaction_name],
                                    'relationships': [similarity_string, 'ASSOCIATED_WITH', 'INTERACTS_WITH']
                                }
                                disease_target_target_paths.append(disease_target_target_path)

                  selected_disease_paths, selected_disease_nodes, unique_disease_rels, selected_stage2 = select_paths(disease_target_pathway_paths,
                                                                                                                      question,
                                                                                                                      max(1, len(disease_target_pathway_paths)//15),
                                                                                                                      3,
                                                                                                                      progress_callback)
                  selected_disease_target_target_paths, selected_disease_target_target_nodes, unique_disease_target_target_rels, selected_stage2 = select_paths(disease_target_target_paths,
                                                                                                                                                    question,
                                                                                                                                                    max(1,len(disease_target_target_paths)//15),
                                                                                                                                                    3,
                                                                                                                                                    progress_callback)
                  mid_disease_target_target_paths = selected_disease_paths + selected_disease_target_target_paths
                  final_disease_target_target_paths.extend(mid_disease_target_target_paths)
                  mid_disease_target_target_nodes = selected_disease_nodes + selected_disease_target_target_nodes
                  final_disease_target_target_nodes.extend(mid_disease_target_target_nodes)
                  mid_unique_disease_target_target_rels = unique_disease_rels + unique_disease_target_target_rels
                  final_unique_disease_target_target_rels.extend(mid_unique_disease_target_target_rels)
            
              result_dict[name] = {"paths": mid_disease_target_target_paths,
                                    "nodes": mid_disease_target_target_nodes,
                                    "rels": mid_unique_disease_target_target_rels,
                                    "score": score
                                    }
                        
              count += 1
      return result_dict
    
def query_predicted_drug_info(string, question, progress_callback=None):
    similar_drugs_query = """
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
    drug_annotation_query = """
    query drugAnnotation($chemblId: String!) {
      drug(chemblId: $chemblId) {
        name
        id
        mechanismsOfAction{
          uniqueActionTypes
          uniqueTargetTypes
          rows{
            mechanismOfAction
            actionType
            targetName
          }
        }
        linkedTargets{
          rows{
            approvedSymbol
            pathways{
              pathway
            }
            interactions{
              rows{
                targetB{
                  approvedSymbol
                }
              }
            }
          }
        }
        linkedDiseases{
          rows{
            name
          }
        }
      }
    }
    """
  

    id = query_open_targets_id(string, False, False, True)
    similar_drugs_variables = {"chemblId": id}
    similar_drugs_response = query_open_targets(similar_drugs_query, similar_drugs_variables)

    chembl_id_list = []
    formatted_paths = []
    paths = []
    nodes = []
    result_dict = {}
    if similar_drugs_response.get('data') and similar_drugs_response['data'].get('drug') and similar_drugs_response['data']['drug'].get('similarEntities'):
        count = 0
        for similar_entity in similar_drugs_response['data']['drug']['similarEntities'][1:]:
            
            if count >= 4:
                break
            chembl_id = similar_entity['id']
            score = round(similar_entity['score'], 3)
            
            name = similar_entity['object'].get('name')
            if name:
                name = name.capitalize()  # Change this line
                nodes.append(name)
                chembl_id_list.append(name)
                similarity_string = "has a similarity score of"
                similarity_string = "{}={} to".format(similarity_string, score)

                formatted_path = "{} -> has a similarity score of {} to -> {}".format(string, score, name)
                formatted_paths.append(formatted_path)
                path = {
                    'nodes': [string, name],
                    'relationships': [similarity_string]
                }
                paths.append(path)
                drug_annotation_variables = {"chemblId": chembl_id}
                drug_annotation_response = query_open_targets(drug_annotation_query, drug_annotation_variables)

                drug_target_paths = []
                drug_target_pathway_paths = []
                drug_disease_paths = []
                drug_target_target_paths = []

                if drug_annotation_response.get('data') and drug_annotation_response['data'].get('drug') and drug_annotation_response['data']['drug']['linkedTargets'].get('rows'):
                    linked_targets = []
                    linked_target_pathways = []
                    for target in drug_annotation_response['data']['drug']['linkedTargets']['rows']:
                        linked_target_sym = target['approvedSymbol']
                        linked_target_pathways.append(target['pathways'])
                        linked_targets.append(linked_target_sym)
                        path = {
                            'nodes': [string, name, linked_target_sym],
                            'relationships': [similarity_string, 'ASSOCIATED_WITH']
                        }
                        drug_target_paths.append(path)
                        if target['pathways']:
                            for pathway in target['pathways']:
                                pathway_name = pathway['pathway']
                                path = {
                                    'nodes': [string, name, linked_target_sym, pathway_name],
                                    'relationships': [similarity_string, 'ASSOCIATED_WITH', 'ASSOCIATED_WITH']
                                }
                                drug_target_pathway_paths.append(path)

                        if target['interactions']:
                            for interaction in target['interactions']['rows']:
                                if interaction['targetB']:
                                  interaction_target_sym = interaction['targetB']['approvedSymbol']
                                  path = {
                                      'nodes': [string, name, interaction_target_sym, linked_target_sym],
                                      'relationships': [similarity_string, 'ASSOCIATED_WITH', 'INTERACTS_WITH']
                                  }
                                  drug_target_target_paths.append(path)
                    selected_drug_target_pathway_paths, selected_drug_target_pathway_nodes, unique_drug_target_pathway_rels, selected_stage2 = select_paths(drug_target_pathway_paths,
                                                                                                                                                      question,
                                                                                                                                                      len(drug_target_pathway_paths),
                                                                                                                                                      3,
                                                                                                                                                      progress_callback)

                    selected_drug_target_target_paths, selected_drug_target_target_nodes, unique_drug_target_target_rels, selected_stage2 = select_paths(drug_target_target_paths,
                                                                                                                                                      question,
                                                                                                                                                      len(drug_target_target_paths),
                                                                                                                                                      3,
                                                                                                                                                      progress_callback)

                if drug_annotation_response.get('data') and drug_annotation_response['data'].get('drug') and drug_annotation_response['data']['drug']['linkedDiseases'].get('rows'):
                    linked_diseases = []
                    for disease in drug_annotation_response['data']['drug']['linkedDiseases']['rows']:
                        linked_disease_name = disease['name']
                        linked_diseases.append(linked_disease_name)
                        path = {
                            'nodes': [string, name, linked_disease_name],
                            'relationships': [similarity_string, 'ASSOCIATED_WITH']
                        }
                        drug_disease_paths.append(path)
                    selected_drug_disease_paths, selected_drug_disease_nodes, unique_drug_disease_rels, selected_stage2 = select_paths(drug_disease_paths,
                                                                                                                                      question,
                                                                                                                                      len(drug_disease_paths),
                                                                                                                                      3,
                                                                                                                                      progress_callback)

                result_dict[name] = {"paths": selected_drug_disease_paths + selected_drug_target_pathway_paths + selected_drug_target_target_paths,
                                     "nodes": selected_drug_disease_nodes + selected_drug_target_pathway_nodes + selected_drug_target_target_nodes,
                                     "rels": unique_drug_disease_rels + unique_drug_target_pathway_rels + unique_drug_target_target_rels,
                                     "score": score
                }
                count += 1

    return result_dict

def query_drug_annotation(string):
    query = """
    query drugAnnotation($chemblId: String!) {
      drug(chemblId: $chemblId) {
        name
        id
        mechanismsOfAction{
          uniqueActionTypes
          uniqueTargetTypes
          rows{
            mechanismOfAction
            actionType
            targetName
          }
        }
        linkedTargets{
          rows{
            approvedSymbol
            pathways{
              pathway
            }
            interactions{
              rows{
                targetB{
                  approvedSymbol
                }
              }
            }
          }
        }
        linkedDiseases{
          rows{
            name
          }
        }
      }
    }
    """
  
    id = query_open_targets_id(string, False, False, True)
    variables = {"chemblId": id}
    response = query_open_targets(query, variables)

    drug_target_paths = []
    drug_target_pathway_paths = []
    drug_disease_paths = []
    drug_target_target_paths = []

    if response.get('data') and response['data'].get('drug') and response['data']['drug']['linkedTargets'].get('rows'):
        linked_targets = []
        linked_target_pathways = []
        for target in response['data']['drug']['linkedTargets']['rows']:
            linked_target_sym = target['approvedSymbol']
            linked_target_pathways.append(target['pathways'])
            linked_targets.append(linked_target_sym)
            path = {
                'nodes': [string, linked_target_sym],
                'relationships': ['ASSOCIATED_WITH']
            }
            drug_target_paths.append(path)
            if target['pathways']:
                for pathway in target['pathways']:
                    pathway_name = pathway['pathway']
                    path = {
                        'nodes': [string, linked_target_sym, pathway_name],
                        'relationships': ['ASSOCIATED_WITH', 'ASSOCIATED_WITH']
                    }
                    drug_target_pathway_paths.append(path)

            if target['interactions']:
                for interaction in target['interactions']['rows']:
                    if interaction['targetB']:
                      interaction_target_sym = interaction['targetB']['approvedSymbol']
                      path = {
                          'nodes': [string, interaction_target_sym, linked_target_sym],
                          'relationships': ['ASSOCIATED_WITH', 'INTERACTS_WITH']
                      }
                      drug_target_target_paths.append(path)

    if response.get('data') and response['data'].get('drug') and response['data']['drug']['linkedDiseases'].get('rows'):
        linked_diseases = []
        for disease in response['data']['drug']['linkedDiseases']['rows']:
            linked_disease_name = disease['name']
            linked_diseases.append(linked_disease_name)
            path = {
                'nodes': [string, linked_disease_name],
                'relationships': ['ASSOCIATED_WITH']
            }
            drug_disease_paths.append(path)
    print(drug_target_paths)
    print(drug_target_pathway_paths)
    return drug_target_paths, drug_target_pathway_paths, drug_disease_paths, drug_target_target_paths

def query_drug_info(string, question, progress_callback=None):
    final_paths = []
    final_drug_target_nodes = []
    final_drug_pathway_nodes = []
    final_drug_disease_nodes = []
    final_drug_target_target_nodes = []
    final_drug_target_rels = []
    final_drug_pathway_rels = []
    final_drug_disease_rels = []
    final_drug_target_target_rels = []

    similar_drug_paths, similar_drugs_formatted, chembl_id_list, nodes = query_similar_drugs(string)

    for id in chembl_id_list:
        print(f"Processing chembl_id: {id}")  # Add this line
        drug_target_paths, drug_target_pathway_paths, drug_disease_paths, drug_target_target_paths = query_drug_annotation(id)
        if drug_target_paths:
          selected_drug_target_paths, selected_drug_target_nodes, unique_drug_target_rels, selected_stage2 = select_paths(drug_target_paths,
                                                                                                                                question,
                                                                                                                                15,
                                                                                                                                3,
                                                                                                                                progress_callback)
          final_paths.extend(selected_drug_target_paths)
          final_drug_target_nodes.extend(selected_drug_target_nodes)
          final_drug_target_rels.extend(unique_drug_target_rels)

        if drug_target_pathway_paths:                                                                                                                          
          selected_drug_pathway_paths, selected_drug_pathway_nodes, unique_drug_pathway_rels, selected_stage2 = select_paths(drug_target_pathway_paths,
                                                                                                                                    question,
                                                                                                                                    15,
                                                                                                                                    3,
                                                                                                                                    progress_callback)    
          final_paths.extend(selected_drug_pathway_paths)
          final_drug_pathway_nodes.extend(selected_drug_pathway_nodes)
          final_drug_pathway_rels.extend(unique_drug_pathway_rels)

        if drug_disease_paths:                                                                                                                                               
          selected_drug_disease_paths, selected_drug_disease_nodes, unique_drug_disease_rels, selected_stage2 = select_paths(drug_disease_paths,
                                                                                                                                    question,
                                                                                                                                    15,
                                                                                                                                    3,
                                                                                                                                    progress_callback)
          final_paths.extend(selected_drug_disease_paths)
          final_drug_disease_nodes.extend(selected_drug_disease_nodes)
          final_drug_disease_nodes.extend(unique_drug_disease_rels)

        if drug_target_target_paths:
          selected_drug_target_target_paths, selected_drug_target_target_nodes, unique_drug_target_target_rels, selected_stage2 = select_paths(drug_target_target_paths,
                                                                                                                                    question,
                                                                                                                                    15,
                                                                                                                                    3,
                                                                                                                                    progress_callback)
          final_paths.extend(selected_drug_target_target_paths)
          final_drug_target_target_nodes.extend(selected_drug_target_target_nodes)
          final_drug_target_target_rels.extend(unique_drug_target_target_rels)


    final_nodes = final_drug_target_nodes + final_drug_pathway_nodes + final_drug_disease_nodes
    final_rels = final_drug_target_rels + final_drug_pathway_rels + final_drug_disease_rels + final_drug_target_target_rels + similar_drugs_formatted
    final_nodes = list(set(final_nodes))
    print(final_rels)
    final_rels = list(set(final_rels))
    final_paths.extend(similar_drugs_formatted)
    print("final_paths")
    print(final_paths)
    return final_paths, final_nodes, final_rels


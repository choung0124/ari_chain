import requests

def get_similar_compounds(drug_name, top_n):
    # Get CID of the drug from PubChem
    pubchem_cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
    response = requests.get(pubchem_cid_url)
    cid = response.json()['IdentifierList']['CID'][0]  # assuming the first CID is the correct one

    # Get canonical SMILES of the drug from PubChem
    pubchem_smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    response = requests.get(pubchem_smiles_url)
    smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    print(smiles)
    # Use the ChEMBL API to find similar compounds
    chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/40?format=json"
    response = requests.get(chembl_url)
    print(len(response.json()['molecules']))
    similar_compounds = [molecule['pref_name'] for molecule in response.json()['molecules'] if 'pref_name' in molecule and molecule['pref_name'] is not None]
    # If there are less compounds than top_n, return all compounds
    if len(similar_compounds) < top_n:
        return similar_compounds

    # Otherwise, return the top_n similar compounds
    return similar_compounds[:top_n]

def get_umls_id(search_string: str) -> list:
    api_key = "7cc294c9-98ed-486b-add8-a60bd53de1c6"
    base_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    query = f"?string={search_string}&inputType=atom&returnIdType=concept&apiKey={api_key}"
    url = f"{base_url}{query}"
    
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        results = data["result"]["results"]
        if results:
            filtered_results = [result for result in results if search_string.lower() in result['name'].lower()]
            if filtered_results:
                top_result = filtered_results[0]
                result_string = f"Name: {top_result['name']} UMLS_CUI: {top_result['ui']}"
                return [result_string]
            else:
                return ["No results found."]
        else:
            return ["No results found."]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
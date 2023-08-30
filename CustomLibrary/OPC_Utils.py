import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
import glob
import os
from CustomLibrary.Graph_Utils import select_paths2

base_url = "https://www.google.com/search?q="

def get_url(drug_name):
    query = f"{drug_name} pubchem"
    response = requests.get(base_url + query)
    soup = BeautifulSoup(response.content, 'html.parser')

    div = soup.find('div', class_='kCrYT')

    if div is None:
        print(f"No div found for {drug_name}")
        return None

    a = div.find('a', href=True)

    if a is None:
        print(f"No link found for {drug_name}")
        return None

    href = a.get('href')

    if href is None:
        print(f"No href found for {drug_name}")
        return None

    url = href.split('/url?q=')[1].split('&')[0]
    final_url = f"{url}#section=Chemical-Target-Interactions"

    return final_url


def get_similar_compounds(drug_name, top_n):
    # Get CID of the drug from PubChem
    pubchem_cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
    response = requests.get(pubchem_cid_url)
    cid = response.json()['IdentifierList']['CID'][0]

    # Get canonical SMILES of the drug from PubChem
    pubchem_smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    response = requests.get(pubchem_smiles_url)
    smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']

    # Use the ChEMBL API to find similar compounds
    chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/40?format=json"
    response = requests.get(chembl_url)
    similar_compounds = [molecule['pref_name'] for molecule in response.json()['molecules'] if 'pref_name' in molecule and molecule['pref_name'] is not None]

    # If there are less compounds than top_n + 1 (including the top most similar one which we are going to ignore), return all compounds except the first one
    if len(similar_compounds) < top_n + 1:
        return similar_compounds[1:]

    # Otherwise, return the top_n similar compounds after the first one
    return similar_compounds[1:top_n + 1]


def download_wait(directory, timeout, nfiles=None):
    seconds = 0
    dl_wait = True
    while dl_wait and seconds < timeout:
        time.sleep(1)
        dl_wait = False
        files = os.listdir(directory)
        if nfiles and len(files) != nfiles:
            dl_wait = True

        for fname in files:
            if fname.endswith('.crdownload'):
                dl_wait = True

        seconds += 1
    return seconds

def download_pubchem(drug_name):
    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory' : '/mnt/c/ari_chain'}
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=chrome_options)

    url = get_url(drug_name)
    if url is None:
        print(f"No URL found for {drug_name}")
        return None

    try:
        driver.set_page_load_timeout(60)  # set timeout to 60 seconds
        driver.get(url)
    except TimeoutException:
        print(f"Timeout while trying to access {url}")
        return None

    try:
        wait = WebDriverWait(driver, 30)
        download_btn = wait.until(EC.element_to_be_clickable((By.ID, 'download-consolidatedcompoundtarget')))

        download_btn.click()

        csv_download_link = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[download]')))

        csv_download_link.click()

        # Add delay to allow download to finish
        time.sleep(10)

        # Assuming the downloaded file has a .csv extension and its name starts with 'pubchem_cid'
        # Change the file name pattern based on your actual downloaded file name
        downloaded_file = glob.glob('/mnt/c/ari_chain/pubchem_cid*.csv')[0]
        
        # Read the CSV file into a pandas DataFrame
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(downloaded_file)

        # Keep only the columns 'dsn', 'cmpdname', 'action', 'genename', 'pmids'
        df = df.loc[:, ['dsn', 'cmpdname', 'action', 'genename', 'pmids']]

        # Split the 'pmids' column at '|' and explode into separate rows
        # Convert 'pmids' column to string type
        df['pmids'] = df['pmids'].astype(str)
        df['pmids'] = df['pmids'].str.split('|')
        df = df.explode('pmids')
        df.dropna(subset=['genename', 'cmpdname', 'dsn'], inplace=True)
        # Delete the CSV file
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)

        return df

    except TimeoutException:
        print(f"Timeout while trying to download data for {drug_name}. The table may not exist on the page.")
        return None

    except NoSuchElementException:
        print("Download button or CSV download link not found")

    finally:
        driver.quit()

def pubchem_query(entity, string, question, progress_callback=None):
    df = download_pubchem(string)
    if df is None:
        print(f"No data found for {string}")
        return None

    path_list = []

    for _, row in df.iterrows():
        path_dict = {}
        path_dict['nodes'] = [entity, row['cmpdname'], row['genename']]
        path_dict['relationships'] = ["contains constituent", row['action']]
        path_list.append(path_dict)

    selected_paths, selected_nodes, selected_rels = select_paths2(path_list, question, max(len(path_list)//3, 1), max(len(path_list)//3, 1), progress_callback)

    return selected_paths, selected_nodes, selected_rels

def similar_pubchem_query(entity, string, question, progress_callback=None):
    similar_compounds = get_similar_compounds(string, 100)
    paths = []
    nodes = []
    rels = []
    counter = 0
    if counter < 2:
        for compound in similar_compounds:
            df = download_pubchem(compound)
            if df is None:
                print(f"No data found for {compound}")
                continue

            path_list = []

            for _, row in df.iterrows():
                path_dict = {}
                path_dict['nodes'] = [entity, compound, row['cmpdname'], row['genename']]
                path_dict['relationships'] = ["contains constituent", "is similar to", row['action']]
                path_list.append(path_dict)
            selected_paths, selected_nodes, selected_rels = select_paths2(path_list, question, max(len(path_list)//3, 1), max(len(path_list)//3, 1), progress_callback)
            paths.extend(selected_paths)
            nodes.extend(selected_nodes)
            rels.extend(selected_rels)
            counter += 1

    return paths, nodes, rels




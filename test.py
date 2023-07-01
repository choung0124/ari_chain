from Custom_Agent import get_similar_compounds

drug_name = "Mirodenafil"
top_3_similar_compounds = get_similar_compounds(drug_name, top_n=3)
print(f"Top 3 similar compounds to {drug_name} (ChEMBL IDs): {top_3_similar_compounds}")


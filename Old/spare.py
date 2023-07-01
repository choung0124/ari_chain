#def search_abstracts(question: str):
#    query_embedding_model = HuggingFaceEmbeddings(
#        model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
#        model_kwargs={'device': 'cuda'},
#        encode_kwargs={'normalize_embeddings': True}
#    )
#    query_vector = query_embedding_model.embed_query(question)
#
#    with psycopg2.connect(
#        dbname='pubmed',
#        user="hschoung",
#        password="Reeds0124",
#        host="localhost",
#        port="5432"
#    ) as conn:
#        with conn.cursor() as c:
#            c.execute("""
#                SELECT id, title, doi, abstract
#                FROM abstracts
#                ORDER BY pubmed_bert_embeddings <-> %s::vector LIMIT 5;
#            """, (query_vector,))
#            similar_abstracts = c.fetchall()
#
#            for abstract in similar_abstracts:
#                abstract_id, title, doi, abstract_text = abstract
#                print(f"Abstract ID: {abstract_id}")
#                print(f"Title: {title}")
#                print(f"DOI: {doi}")
#                print(f"Abstract: {abstract_text}")
#                print()
#
#    return similar_abstracts


#graph = Neo4jGraph(
#    url="bolt://localhost:7687", username="neo4j", password="pleaseletmein")
#graph.refresh_schema()

#chain = GraphCypherQAChain.from_llm(
#    llm(temperature=0), graph=graph, verbose=True, top_k=2
#)


CYPHER_QA_TEMPLATE = """Below is an instruction that describes a task, Write a response that appropriately completes the request.

USER:
Instructions:
The Information part contains the provided information that you must use to construct an answer.
The provided information is authorative, you must never doubt it or try to use your internal knowledge to correct it.
If the provided information is empty, say that you don't know the answer.
Information:
{context}
The question is: 
{question}

ASSISTANT:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

CYPHER_GENERATION_TEMPLATE_2 = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.

USER:
Generate Cypher statement to query a Neo4j graph database for all information that is needed to answer the question. Use only the provided relationship types and properties in the schema. Only include the generated Cypher statement in your response. Make sure the syntax of the Cypher statement is correct for Neo4j Version 4.

Schema: {schema}
The question is: {question}

ASSISTANT:"""

CYPHER_GENERATION_PROMPT_2 = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

    intermediate_steps = result['intermediate_steps']
    if not intermediate_steps[1]['context']:
        graph_chain_2 = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True, cypher_prompt=CYPHER_GENERATION_PROMPT_2, return_intermediate_steps=True)
    else:
        exit


An example of a node and its labels in the neo4j graph database:
"identity": 0,
"labels": ["Disease"],
"properties": 
    "name": "angiosarcoma",
    "description": "A vascular cancer that derives_from the cells that line the walls of blood vessels or lymphatic vessels. [url:http\://en.wikipedia.org/wiki/Hemangiosarcoma, url:https\://en.wikipedia.org/wiki/Angiosarcoma, url:https\://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C3088, url:https\://www.ncbi.nlm.nih.gov/pubmed/23327728]",
    "id": "DOID:0001816",
    "type": "-26",
    "synonyms": [
        "angiosarcoma",
        "hemangiosarcoma",
        "ICDO:9120/3",
        "MESH:D006394",
        "NCI:C3088",
        "NCI:C9275",
        "SNOMEDCT_US_2020_09_01:39000009",
        "UMLS_CUI:C0018923",
        "UMLS_CUI:C0854893"
        ]

Strictly use the following format: 

Question: the input question you must find UMLS IDs for
Thought: you should always think about waht to do
Action: the action to take, should be one of [{tools}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Final Answer: The list of UMLS_CUIs and the words they represent

output_parser = UMLS_Agent_OutputParser()
UMLS_chain = LLMChain(llm=llm, prompt=prompt)
UMLS_agent = LLMSingleActionAgent(
    llm_chain=UMLS_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=[umls_tool]
)

umls_tool = Tool.from_function(
    func=lambda query: get_umls_id(query),
    name="UMLS_CUI_Finder",
    description="Finds the UMLS_CUI for a given search string"
)

type: LOCATED_IN, properties: property: end, type: STRING, property: strand, type: STRING, property: start, type: STRING
type: HAS_SEQUENCE, properties: property: source, type: STRING
type: BELONGS_TO_PROTEIN, properties: property: source, type: STRING
type: ASSOCIATED_WITH, properties: property: evidence_type, type: STRING, property: source, type: STRING, property: score, type: FLOAT, property: number_publications, type: STRING
type: HAS_MODIFICATION, properties: property: source, type: STRING
type: IS_SUBSTRATE_OF, properties: property: regulation, type: STRING, property: evidence_type, type: STRING, property: source, type: STRING, property: score, type: FLOAT
type: IS_SUBUNIT_OF, properties: property: source, type: STRING, property: cell_lines, type: LIST, property: publication, type: STRING, property: evidences, type: LIST
type: CURATED_INTERACTS_WITH, properties: property: score, type: FLOAT, property: source, type: LIST, property: method, type: LIST, property: evidence, type: LIST, property: interaction_type, type: STRING
type: COMPILED_INTERACTS_WITH, properties: property: score, type: FLOAT, property: source, type: LIST, property: evidence, type: LIST, property: interaction_type, type: STRING, property: scores, type: LIST
type: ACTS_ON, properties: property: directionality, type: BOOLEAN, property: action, type: STRING, property: source, type: STRING, property: score, type: FLOAT
type: DETECTED_IN_PATHOLOGY_SAMPLE, properties: property: not_detected, type: STRING, property: linkout, type: STRING, property: negative_prognosis_logrank_pvalue, type: STRING, property: source, type: STRING, property: positive_prognosis_logrank_pvalue, type: STRING, property: expression_low, type: STRING, property: expression_medium, type: STRING, property: expression_high, type: STRING
type: MENTIONED_IN_PUBLICATION, properties: property: source, type: STRING
type: FOUND_IN_PROTEIN, properties: property: end, type: STRING, property: source, type: STRING, property: alignment, type: STRING, property: start, type: STRING
type: HAS_STRUCTURE, properties: property: source, type: STRING
type: STUDIES_TRAIT, properties: property: source, type: STRING
type: ANNOTATED_IN_PATHWAY, properties: property: cellular_component, type: STRING, property: organism, type: STRING, property: evidence, type: STRING, property: source, type: STRING
type: HAS_QUANTIFIED_PROTEIN, properties: property: score, type: FLOAT, property: proteinGroup, type: STRING, property: is_razor, type: STRING, property: value, type: FLOAT, property: qvalue, type: FLOAT, property: intensity, type: FLOAT
type: HAS_QUANTIFIED_MODIFIED_PROTEIN, properties: property: sequenceWindow, type: STRING, property: value, type: FLOAT, property: score, type: STRING, property: deltaScore, type: STRING, property: localizationProb, type: STRING, property: scoreLocalization, type: STRING, property: is_razor, type: STRING

### Instruction: You are an artificial intelligence assistant that removes unecessary information from a graph database schema. Keep any parts of the schema may be required to generate cypher statements that can answer question. Include only the filtered schema in your response, do not attempt to answer the question.
Schema:
:Disease:HAS_PARENT:Disease
:Disease:MENTIONED_IN_PUBLICATION:Publication
:Tissue:MENTIONED_IN_PUBLICATION:Publication
:Tissue:HAS_PARENT:Tissue
:Biological_process:HAS_PARENT:Biological_process
:Molecular_function:HAS_PARENT:Molecular_function
:Cellular_component:HAS_PARENT:Cellular_component
:Cellular_component:MENTIONED_IN_PUBLICATION:Publication
:Modification:HAS_PARENT:Modification
:Phenotype:HAS_PARENT:Phenotype
:Experiment:HAS_PARENT:Experiment
:Experimental_factor:HAS_PARENT:Experimental_factor
:Experimental_factor:MAPS_TO:Disease
:Units:HAS_PARENT:Units
:Gene:TRANSCRIBED_INTO:Transcript
:Gene:TRANSLATED_INTO:Protein
:Transcript:TRANSLATED_INTO:Protein
:Transcript:LOCATED_IN:Chromosome
:Protein:DETECTED_IN_PATHOLOGY_SAMPLE:Disease
:Protein:HAS_SEQUENCE:Amino_acid_sequence
:Protein:ANNOTATED_IN_PATHWAY:Pathway
:Protein:ASSOCIATED_WITH:Cellular_component
:Protein:CURATED_INTERACTS_WITH:Protein
:Protein:COMPILED_INTERACTS_WITH:Protein
:Protein:ACTS_ON:Protein
:Protein:MENTIONED_IN_PUBLICATION:Publication
:Protein:HAS_STRUCTURE:Protein_structure
:Protein:HAS_MODIFIED_SITE:Modified_protein
:Protein:IS_SUBUNIT_OF:Complex
:Protein:IS_QCMARKER_IN_TISSUE:Tissue
:Peptide:BELONGS_TO_PROTEIN:Protein
:Modified_protein:HAS_MODIFICATION:Modification
:Modified_protein:IS_SUBSTRATE_OF:Protein
:Modified_protein:MENTIONED_IN_PUBLICATION:Publication
:Complex:ASSOCIATED_WITH:Biological_process
:Known_variant:VARIANT_FOUND_IN_PROTEIN:Protein
:Known_variant:VARIANT_FOUND_IN_CHROMOSOME:Chromosome
:Known_variant:VARIANT_FOUND_IN_GENE:Gene
:Clinically_relevant_variant:ASSOCIATED_WITH:Disease
:Functional_region:MENTIONED_IN_PUBLICATION:Publication
:Functional_region:FOUND_IN_PROTEIN:Protein
:Metabolite:ANNOTATED_IN_PATHWAY:Pathway
:Metabolite:ASSOCIATED_WITH:Protein
:GWAS_study:PUBLISHED_IN:Publication
:GWAS_study:STUDIES_TRAIT:Experimental_factor
:User:PARTICIPATES_IN:Project
:User:IS_RESPONSIBLE:Project
:Project:STUDIES_DISEASE:Disease
:Project:HAS_ENROLLED:Subject
:Project:STUDIES_TISSUE:Tissue
:Biological_sample:SPLITTED_INTO:Analytical_sample
:Biological_sample:BELONGS_TO_SUBJECT:Subject
:Analytical_sample:HAS_QUANTIFIED_PROTEIN:Protein
:Analytical_sample:HAS_QUANTIFIED_MODIFIED_PROTEIN:Modified_protein


#UMLS_IDs = output_string

#Entity_type_prompt = PromptTemplate(template=Entity_type_Template, input_variables=["input"])
#Entity_type_chain = LLMChain(prompt=Entity_type_prompt, llm=llm)
#Entity_type_result = Entity_type_chain.run(UMLS_IDs)
#print(Entity_type_result)
#UMLS_context = Entity_type_result

#model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"
#model_basename = "gptq_model-4bit-128g"
#use_triton=False

#tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
#model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
#       model_basename=model_basename,
#        use_safetensors=True,
#        trust_remote_code=True,
#        device="cuda:0",
#        use_triton=use_triton,
#        quantize_config=None)

#pipe = pipeline(
#   "text-generation",
#    model=model,
#   tokenizer=tokenizer,
#    max_new_tokens=1024,
#    temperature=0.2,
#   top_p=0.95,
#    repetition_penalty=1.15
#    )


#Schema_simplify_prompt = PromptTemplate(template=Schema_simplify_template, input_variables=["question", "UMLS_context"])
#Schema_simplify_chain = LLMChain(prompt=Schema_simplify_prompt, llm=llm)
#result = Schema_simplify_chain.predict(question="""What is the relationship between alzheimer's and autophagy?""", UMLS_context=UMLS_context)
#print(result)


class CustomGraphCypherQAChain(Chain):
    
    #graph: Neo4jGraph = Field(exclude=True)
    qa_chain: LLMChain
    input_key: str = "question"
    output_key: str = "result"
    top_k: int = 10
    return_intermediate_steps: bool = False
    return_direct: bool = False
    graph: Optional[Neo4jGraph] = None

    @classmethod
    def with_driver(cls, driver, **kwargs):
        graph = Neo4jGraph(driver=driver)
        return cls(graph=graph, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        _output_keys = [self.output_key]
        return _output_keys

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        driver: Optional[Any] = None,
        **kwargs: Any,
    ) -> CustomGraphCypherQAChain:
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        return cls.with_driver(driver=driver, qa_chain=qa_chain, **kwargs)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        question = inputs["question"]
        names_list = inputs["names_list"]  # Get the names_list from inputs

        intermediate_steps: List = []
        final_result = None

        while not final_result:
            # Extract names from the question

            context = find_shortest_path(self.graph, "Test", names_list)
            print(context)
            hf = HuggingFaceEmbeddings(model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
                            model_kwargs= {'device': 'cpu'},
                            encode_kwargs = {'normalize_embeddings': True})
            


            embedding_size = 768
            index = faiss.IndexFlatL2(embedding_size)  # Create a Faiss index

            embeddings = []
            for sentence in tqdm(sentences, desc="Embedding sentences"):
                embedding = hf.embed_documents(sentence)
                embeddings.append(embedding)

            index.add(embeddings)  # Add the embeddings to the index
            num_clusters = 10  # Number of clusters
            clustering = faiss.Clustering(embedding_size, num_clusters)  # Create a Faiss clustering object
            clustering.train(embeddings, index)  # Perform clustering on the embeddings
            labels = clustering.assign(embeddings)  # Assign each embedding to a cluster
            
            cluster_documents = {}
            for i, label in tqdm(enumerate(labels), desc="Performing clustering"):
                document = sentences[i]  # Get the document corresponding to the embedding
                if label in cluster_documents:
                    continue  # Skip if we already have a document from this cluster
                cluster_documents[label] = document  #

            #pubmed_retriever = PubMedRetriever()
            #pubmed_context = pubmed_retriever.get_relevant_documents(question)
            if self.return_direct:
                final_result = context
            else:
                _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
                _run_manager.on_text(
                    str(context), color="green", end="\n", verbose=self.verbose
                )

                intermediate_steps.append({"context": context})
                intermediate_steps.append({"compressed KG context": document})
                #intermediate_steps.append({"pubmed context": pubmed_context})
                result = self.qa_chain(
                    {"question": question, "context": document},
                    callbacks=callbacks,
                )
                final_result = result[self.qa_chain.output_key]

            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

            return chain_result
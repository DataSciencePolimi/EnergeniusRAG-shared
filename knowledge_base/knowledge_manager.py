"""Knowledge Manager Module."""
import ast
import os
import pandas as pd
from langchain_core.documents import Document
from langchain.embeddings import init_embeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph

from llm import LLMHandler

from .utils.graph_helpers import (process_name_or_relationship, normalize_l2,
                                  sparql_query, dataframe_to_text)
from .utils.graph_parameter import GRAPH_PARAMETER
from .utils.graph_prompt import graph_prompt, wrong_answer_prompt


class KnowledgeManager:
    """A class to manage knowledge base operations."""
    embedding: object
    language: str
    rdf_graph: Graph
    llm_graph_transformer: LLMGraphTransformer

    def __init__(self, provider: str, model: str, embedding: str, language: str):
        """Initialize the KnowledgeManager.

        Args:
            provider (str): _Description of the model provider._
            model (str): _Description of the model name._
            embedding (str): _Description of the embedding model name._
            language (str): _Description of the language for the model._
        """
        # Initialize the LLMHandler and embedding model.
        llm_handler = LLMHandler(
            provider=provider,
            model=model,
            temperature=0.0,
            language=None
        )
        self.embeddings = init_embeddings(
            model=embedding,
            provider=provider
        )
        self.language = language

        # Load the RDF graph from the official.ttl file.
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.rdf_graph = Graph()
        self.rdf_graph.parse(os.path.join(
            dir_path, "files" , "rdf_graph_new.ttl"), format="turtle")

        self.llm_graph_transformer = LLMGraphTransformer(
            llm=llm_handler.get_model(),
            node_properties=True,
            relationship_properties=True
        )

    def __get_chunks_by_entities(self, entity_ids: list, limit_chunks=3) -> str:
        """
        Generate a SPARQL query to get chunks related to the given entity IDs.
        Args:
            entity_ids (list): List of entity IDs to filter the chunks.
            limit_chunks (int): Maximum number of chunks to return.
        Returns:
            str: SPARQL query string.
        """

        values_clause = " ".join(
            [f"(<{row['entity']}>)" for index, row in entity_ids.iterrows()])

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX onto: <http://example.org/ontology#>
        PREFIX ex: <http://example.org/data#>

        SELECT ?chunk ?uri ?content (COUNT(?entity) AS ?freq)
        WHERE {{      
            VALUES (?entity) {{
                {values_clause}
            }}

            ?entity rdf:type onto:Entity .
            ?entity onto:hasName ?name .

            ?chunk rdf:type onto:Chunk ;
                    onto:hasContent ?content ;
                    onto:hasEntity ?entity ;
                    onto:belongsToDocument ?document.
            ?document rdf:type onto:Document ;
                    onto:hasUri ?uri.
        }}
        GROUP BY ?chunk ?content
        ORDER BY DESC(?freq)
        LIMIT {limit_chunks}
        """

        return query

    def __get_outgoing_relationships(self, entity_ids, limit_outgoing_relationships=10) -> str:
        """
        Generate a SPARQL query to get outgoing relationships for the given entity IDs.
        Args:
            entity_ids (list): List of entity IDs to filter the relationships.
            limit_outgoing_relationships (int): Maximum number of outgoing relationships to return.
        Returns:
            str: SPARQL query string.
        """

        values_clause = " ".join(
            [f"(<{row['entity']}> {row['cosine_similarity']})" for index, row in entity_ids.iterrows()])

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX onto: <http://example.org/ontology#>
        PREFIX ex: <http://example.org/data#>

        SELECT ?source_name ?name ?target_name
        WHERE {{
            ?relationship rdf:type onto:Relationship ;
                        onto:hasName ?name ;
                        onto:hasSource ?source_entity ;
                        onto:hasTarget ?target_entity .  
            ?source_entity onto:relatesTarget ?target_entity .
            ?target_entity onto:relatesSource ?source_entity .
            ?source_entity onto:isSourceOf ?relationship .
            ?target_entity onto:isTargetOf ?relationship .
            ?source_entity onto:hasName ?source_name .
            ?target_entity onto:hasName ?target_name .

            VALUES (?source_entity ?cosine_similarity) {{
                {values_clause}
            }}
            
            {{ SELECT * WHERE {{ ?reltriple onto:hasRelationship ?relationship ; rdf:type onto:RelTriple ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity . }} }}
            
        }}
        #GROUP BY ?source_entity ?relationship ?target_entity
        ORDER BY DESC(?cosine_similarity)
        LIMIT {limit_outgoing_relationships}
        """

        return query

    def __get_incoming_relationships(self, entity_ids, limit_incoming_relationships: int = 10) -> str:
        """
        Generate a SPARQL query to get incoming relationships for the given entity IDs.
        Args:
            entity_ids (list): List of entity IDs to filter the relationships.
            limit_incoming_relationships (int): Maximum number of incoming relationships to return.
        Returns:
            str: SPARQL query string.
        """

        values_clause = " ".join(
            [f"(<{row['entity']}> {row['cosine_similarity']})" for index, row in entity_ids.iterrows()])

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX onto: <http://example.org/ontology#>
        PREFIX ex: <http://example.org/data#>

        SELECT ?source_name ?name ?target_name
        WHERE {{
            ?relationship rdf:type onto:Relationship ;
                        onto:hasName ?name ;
                        onto:hasSource ?source_entity ;
                        onto:hasTarget ?target_entity .
            ?source_entity onto:relatesTarget ?target_entity .
            ?target_entity onto:relatesSource ?source_entity .
            ?source_entity onto:isSourceOf ?relationship .
            ?target_entity onto:isTargetOf ?relationship .
            ?source_entity onto:hasName ?source_name .
            ?target_entity onto:hasName ?target_name .

            VALUES (?target_entity ?cosine_similarity) {{
                {values_clause}
            }}
            
            {{ SELECT * WHERE {{ ?reltriple onto:hasRelationship ?relationship ; rdf:type onto:RelTriple ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity . }} }}
            
        }}
        #GROUP BY ?source_entity ?relationship ?target_entity
        ORDER BY DESC(?cosine_similarity)
        LIMIT {limit_incoming_relationships}
        """

        return query

    def user_message(self, message: str, user_type: str, house_type: str, region: str) -> str:
        """
        Process a user message and return a response.
        Args:
            message (str): The user message to process.
            user_type (str): _type of user (e.g., "working class", etc.)._
            house_type (str): _type of house (e.g., "apartment", "villa", etc.)._
        Returns:
            str: The generated prompt to setup the answer.
        """
        # Compute nodes and relationships for the user message.
        graph_question = self.llm_graph_transformer.convert_to_graph_documents(
            [Document(page_content=message)])

        # Embed the question and its ndoes/relationships
        query_embedding_vectors = []
        query_embedding_vectors.append(normalize_l2(self.embeddings.embed_query(message)[:512]))
        # query_embedding_vectors.append(self.embeddings.embed_query(message))
        for _, node in enumerate(graph_question[0].nodes):
            query_embedding_vectors.append(normalize_l2(self.embeddings.embed_query(process_name_or_relationship(node.id))[:512]))
            # query_embedding_vectors.append(self.embeddings.embed_query(process_name_or_relationship(node.id)))

        #####
        # Search similar entities
        #####

        # Get all entities
        query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX onto: <http://example.org/ontology#>
            
            SELECT ?entity ?name ?name_embedding
            WHERE {{
                ?entity rdf:type onto:Entity .
                ?entity onto:hasName ?name .
                ?entity onto:hasNameEmbedding ?name_embedding .
            }}
            GROUP BY ?entity #?name
        """
        entities = sparql_query(query, self.rdf_graph)

        # For each vector in the question
        entities_vector = []
        for query_embedding_vector in query_embedding_vectors:

            temp = entities.copy()
            temp['cosine_similarity'] = temp['name_embedding'].apply(
                lambda x: cosine_similarity([ast.literal_eval(x)], [query_embedding_vector])[0][0])
            temp = temp.sort_values(
                by='cosine_similarity', ascending=False, ignore_index=True)
            temp = temp.drop(
                temp[temp.cosine_similarity < GRAPH_PARAMETER["SIMILARITY_THRESHOLD"]].index)
            temp = temp.drop(columns=['name_embedding'])
            temp = temp[:int(GRAPH_PARAMETER["TOP_ENTITIES"] /
                             len(query_embedding_vectors))]
            entities_vector.append(temp)

        entity_list = pd.concat(entities_vector)
        chunks_df = sparql_query(self.__get_chunks_by_entities(
            entity_list, GRAPH_PARAMETER["TOP_CHUNKS"]), self.rdf_graph)

        if not chunks_df.empty:
            chunks_df['content'] = chunks_df['content'].apply(
                lambda x: x.replace("\n", " ").replace("  ", " ").lstrip(".").strip())
            chunks_df = chunks_df.drop(columns=['chunk', 'freq'])

        chunk_text = dataframe_to_text(chunks_df, context_name='Resources')
        outgoing_relationships_df = sparql_query(
            self.__get_outgoing_relationships(entity_list, GRAPH_PARAMETER["TOP_OUTGOING_RELATIONSHIPS"]), self.rdf_graph)
        incoming_relationships_df = sparql_query(
            self.__get_incoming_relationships(entity_list, GRAPH_PARAMETER["TOP_INCOMING_RELATIONSHIPS"]), self.rdf_graph)

        # Mix relationships
        outgoing_relationships_text = dataframe_to_text(
            outgoing_relationships_df, context_name='Relationships')
        incoming_relationships_text = dataframe_to_text(
            incoming_relationships_df)
        relationships_text = outgoing_relationships_text + \
            "\n" + incoming_relationships_text

        # Final prompt
        context_data = "\n" + chunk_text + "\n" + relationships_text

        if len(context_data.strip()) == 0:
            return wrong_answer_prompt(self.language)

        return graph_prompt(self.language, user_type, house_type, region, context_data)

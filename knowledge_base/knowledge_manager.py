"""Knowledge Manager Module."""
import ast
import os
import pandas as pd
from langchain_core.documents import Document
from langchain.embeddings import init_embeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph
import time

from llm import LLMHandler

from .utils.graph_helpers import (process_name_or_relationship, normalize_l2,
                                  sparql_query, dataframe_to_text)
from .utils.graph_parameter import GRAPH_PARAMETER
from .utils.graph_prompt import graph_prompt, wrong_answer_prompt

import chromadb
from chromadb.config import Settings
from itertools import permutations


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
            dir_path, "files" , "rdf_graph.ttl"), format="turtle")

        self.llm_graph_transformer = LLMGraphTransformer(
            llm=llm_handler.get_model(),
            #node_properties=True,
            #relationship_properties=True
            ignore_tool_usage=True,
            additional_instructions="Nodes and relationships should be in English."
        )
    
    def __merge_strings(self, a, b):
        """Merge two strings with maximum overlap."""
        max_overlap = 0
        overlap_index = 0
        min_len = min(len(a), len(b))
        # Check suffix of a vs prefix of b
        for i in range(1, min_len + 1):
            if a[-i:] == b[:i]:
                max_overlap = i
        return a + b[max_overlap:]

    def __merge_three_overlapping_strings(self, s1, s2, s3):
        best = None
        max_len = float('inf')
        if s1 is None:
            s1 = ""
        if s2 is None:
            s2 = ""
        if s3 is None:
            s3 = ""
        for perm in permutations([s1, s2, s3]):
            merged = self.__merge_strings(self.__merge_strings(perm[0], perm[1]), perm[2])
            if len(merged) < max_len:
                best = merged
                max_len = len(merged)
        return best

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
            
            {{ SELECT * WHERE {{ ?triple onto:hasRelationship ?relationship ; rdf:type onto:triple ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity . }} }}
            
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
            
            {{ SELECT * WHERE {{ ?triple onto:hasRelationship ?relationship ; rdf:type onto:triple ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity . }} }}
            
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

        print(f"\n\n-----User message-----\n{message}")

        # Compute nodes and relationships for the user message.
        start = time.time()
        graph_question = self.llm_graph_transformer.convert_to_graph_documents(
            [Document(page_content=message)])
        end = time.time()
        print(f"'Compute nodes and relationships for the user message' execution time: {end - start:.6f} seconds")

        print(f"-----GraphDocument from the question-----\n{graph_question}")

        # Embed the question and its ndoes/relationships
        start = time.time()
        graph_question[0].source.metadata["embedding"] = self.embeddings.embed_query(message)
        for i, relationship in enumerate(graph_question[0].relationships):
            source = process_name_or_relationship(relationship.source.id)
            rel = process_name_or_relationship(relationship.type)
            target = process_name_or_relationship(relationship.target.id)
            relationship.source.properties["embeddings"] = self.embeddings.embed_query(source)
            relationship.properties["embeddings"] = self.embeddings.embed_query(rel)
            relationship.target.properties["embeddings"] = self.embeddings.embed_query(target)
            relationship.properties["triple_embeddings"] = self.embeddings.embed_query(source + " " + rel + " " + target)
        end = time.time()
        print(f"'Embed the question and its ndoes/relationships' execution time: {end - start:.6f} seconds")

        #####
        # Search similar entities
        #####

        start = time.time()
        # Load chromadb
        client = chromadb.PersistentClient(
            path="knowledge_base/files/chroma_db",
            settings=Settings(
                anonymized_telemetry=False
            ),
        )
        collection_entity_embeddings = client.get_or_create_collection(name="graph_entity_embeddings")
        collection_relationship_embeddings = client.get_or_create_collection(name="graph_relationship_embeddings")
        collection_triple_embeddings = client.get_or_create_collection(name="graph_triple_embeddings")

        if graph_question[0].relationships:

            filtered_triple_ids = []
            for rel in graph_question[0].relationships:
                
                entity_query_results = collection_entity_embeddings.query(
                    query_embeddings=rel.source.properties["embeddings"],
                    n_results=GRAPH_PARAMETER["TOP_ENTITIES"],
                )
                print(f"-----Entities for {relationship.source.id}-----\n{entity_query_results}")
                filtered_source_entity_ids = [(id, dist) for id, dist in zip(entity_query_results['ids'][0], entity_query_results['distances'][0]) if dist <= GRAPH_PARAMETER["ENTITIES_DISTANCE_THRESHOLD"]]
                print(f"-----Filtered entities for {relationship.source.id}-----\n{filtered_source_entity_ids}")
                
                entity_query_results = collection_entity_embeddings.query(
                    query_embeddings=rel.target.properties["embeddings"],
                    n_results=GRAPH_PARAMETER["TOP_ENTITIES"],
                )
                print(f"-----Entities for {relationship.target.id}-----\n{entity_query_results}")
                filtered_target_entity_ids = [(id, dist) for id, dist in zip(entity_query_results['ids'][0], entity_query_results['distances'][0]) if dist <= GRAPH_PARAMETER["ENTITIES_DISTANCE_THRESHOLD"]]
                print(f"-----Filtered entities for {relationship.target.id}-----\n{filtered_target_entity_ids}")

                filtered_entity_ids = filtered_source_entity_ids + filtered_target_entity_ids

                # No entity found: return wrong answer prompt
                if not filtered_entity_ids:
                    return wrong_answer_prompt(self.language)
                
                filtered_sorted_entity_ids = sorted(filtered_entity_ids, key=lambda x: x[1])
                filtered_unique_entity_ids = [tup for i, tup in enumerate(filtered_sorted_entity_ids) if tup[0] not in {x[0] for x in filtered_sorted_entity_ids[:i]}]
                filtered_unique_entity_ids = filtered_unique_entity_ids[:GRAPH_PARAMETER["TOP_MERGED_ENTITIES"]]
                print(f"-----Filtered unique entities for {relationship.source.id} and {relationship.target.id}-----\n{filtered_unique_entity_ids}")
            
                ids_list = list(map(lambda x: x[0], filtered_unique_entity_ids))
                triple_query_results = collection_triple_embeddings.query(
                    query_embeddings=rel.properties["triple_embeddings"],
                    where={
                        "source": {"$in": ids_list}
                        #"$and": [
                        #    {"source": {"$in": ids_list}},
                        #    {"target": {"$in": ids_list}}
                        #]
                    },
                    n_results=GRAPH_PARAMETER["TOP_TRIPLES"],
                )
                print(f"-----Triples for {relationship.source.id + " " + relationship.type + " " + relationship.target.id} having {list(map(lambda x: x[0], filtered_unique_entity_ids))}-----\n{triple_query_results}")
                filtered_triple_ids.extend([(id, dist, metadata) for id, dist, metadata in zip(triple_query_results['ids'][0], triple_query_results['distances'][0], triple_query_results['metadatas'][0]) if dist <= GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]])
                print(f"-----Filtered triples for {relationship.source.id + " " + relationship.type + " " + relationship.target.id} having {list(map(lambda x: x[0], filtered_unique_entity_ids))}-----\n{filtered_triple_ids}")

            # No triple found: return wrong answer prompt
            if not filtered_triple_ids:
                return wrong_answer_prompt(self.language)
            
            filtered_sorted_triple_ids = sorted(filtered_triple_ids, key=lambda x: x[1])
            filtered_unique_triple_ids = [tup for i, tup in enumerate(filtered_sorted_triple_ids) if tup[0] not in {x[0] for x in filtered_sorted_triple_ids[:i]}]
            filtered_unique_triple_ids = filtered_unique_triple_ids[:GRAPH_PARAMETER["TOP_MERGED_TRIPLES"]]
            print(f"-----Filtered unique triples-----\n{filtered_unique_triple_ids}")
        
            # Get entity and triple names
            values_clause = " ".join(f"(<{metadata["source"]}> <{metadata["rel"]}> <{metadata["target"]}> {dist})" for id, dist, metadata in filtered_unique_triple_ids)
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX onto: <http://example.org/ontology#>
                PREFIX ex: <http://example.org/data#>
                
                SELECT ?triple ?document_url ?prev_chunk_content ?chunk_content ?next_chunk_content ?description ?source_entity ?source_entity_name ?relationship ?relationship_name ?target_entity ?target_entity_name
                WHERE {{
                    ?triple rdf:type onto:Triple ; onto:hasRelationship ?relationship ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity ; onto:belongsToChunk ?chunk ; onto:hasDescription ?description .
                    ?source_entity onto:hasName ?source_entity_name .
                    ?relationship onto:hasName ?relationship_name .
                    ?target_entity onto:hasName ?target_entity_name .
                    ?chunk onto:hasContent ?chunk_content .
                    ?chunk onto:belongsToDocument ?document .
                    ?document onto:hasUri ?document_url .

                    OPTIONAL {{ ?chunk onto:hasNext ?next_chunk . ?next_chunk onto:hasContent ?next_chunk_content . }}
                    OPTIONAL {{ ?chunk onto:hasPrevious ?prev_chunk . ?prev_chunk onto:hasContent ?prev_chunk_content . }}
        
                    VALUES (?source_entity ?relationship ?target_entity ?dist) {{ {values_clause} }}
                }}
                ORDER BY ?dist
                LIMIT {GRAPH_PARAMETER["TOP_MERGED_TRIPLES"]}
            """
            triples = sparql_query(query, self.rdf_graph)

            # No triple found: return wrong answer prompt
            if triples.empty:
                return wrong_answer_prompt(self.language)
            
            triples["prev_chunk_content"] = triples["prev_chunk_content"].str.replace("\n", " ", regex=False)
            triples["chunk_content"] = triples["chunk_content"].str.replace("\n", " ", regex=False)
            triples["next_chunk_content"] = triples["next_chunk_content"].str.replace("\n", " ", regex=False)
            triples["content"] = triples.apply(lambda row: self.__merge_three_overlapping_strings(row["prev_chunk_content"], row["chunk_content"], row["next_chunk_content"]), axis=1)
            triples.loc[triples["content"].duplicated(), "content"] = " "

            context_data = dataframe_to_text(triples[["source_entity_name", "relationship_name", "target_entity_name", "description", "document_url", "content"]], context_name='')

            end = time.time()
            print(f"'Searching in the db' execution time: {end - start:.6f} seconds")
            
            prompt = graph_prompt(self.language, user_type, house_type, region, context_data)
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            return prompt
                
        # No entities extracted from the question: fallback to only the question embedding
        else:
            entity_query_results = collection_entity_embeddings.query(
                query_embeddings=graph_question[0].source.metadata["embedding"],
                n_results=GRAPH_PARAMETER["TOP_ENTITIES_FALLBACK"],
            )
            filtered_entity_ids = [(id, dist) for id, dist in zip(entity_query_results['ids'][0], entity_query_results['distances'][0]) if dist <= GRAPH_PARAMETER["ENTITIES_DISTANCE_THRESHOLD_FALLBACK"]]

            # No entity found: return wrong answer prompt
            if not filtered_entity_ids:
                return wrong_answer_prompt(self.language)
            
            ids_list = list(map(lambda x: x[0], filtered_entity_ids))
            triple_query_results = collection_triple_embeddings.query(
                query_embeddings=graph_question[0].source.metadata["embedding"],
                where={
                    "source": {"$in": ids_list}
                    #"$and": [
                    #    {"source": {"$in": ids_list}},
                    #    {"target": {"$in": ids_list}}
                    #]
                },
                n_results=GRAPH_PARAMETER["TOP_TRIPLES_FALLBACK"],
            )
            filtered_triple_ids = [(id, dist, metadata) for id, dist, metadata in zip(triple_query_results['ids'][0], triple_query_results['distances'][0], triple_query_results['metadatas'][0]) if dist <= GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD_FALLBACK"]]

            # No triple found: return wrong answer prompt
            if not filtered_triple_ids:
                return wrong_answer_prompt(self.language)
            
            # Get entity and triple names
            values_clause = " ".join(f"(<{metadata["source"]}> <{metadata["rel"]}> <{metadata["target"]}> {dist})" for id, dist, metadata in filtered_triple_ids)
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX onto: <http://example.org/ontology#>
                PREFIX ex: <http://example.org/data#>
                
                SELECT ?triple ?document_url ?prev_chunk_content ?chunk_content ?next_chunk_content ?description ?source_entity ?source_entity_name ?relationship ?relationship_name ?target_entity ?target_entity_name
                WHERE {{
                    ?triple rdf:type onto:Triple ; onto:hasRelationship ?relationship ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity ; onto:belongsToChunk ?chunk ; onto:hasDescription ?description .
                    ?source_entity onto:hasName ?source_entity_name .
                    ?relationship onto:hasName ?relationship_name .
                    ?target_entity onto:hasName ?target_entity_name .
                    ?chunk onto:hasContent ?chunk_content .
                    ?chunk onto:belongsToDocument ?document .
                    ?document onto:hasUri ?document_url .

                    OPTIONAL {{ ?chunk onto:hasNext ?next_chunk . ?next_chunk onto:hasContent ?next_chunk_content . }}
                    OPTIONAL {{ ?chunk onto:hasPrevious ?prev_chunk . ?prev_chunk onto:hasContent ?prev_chunk_content . }}
        
                    VALUES (?source_entity ?relationship ?target_entity ?dist) {{ {values_clause} }}
                }}
                ORDER BY ?dist
                LIMIT {GRAPH_PARAMETER["TOP_MERGED_TRIPLES_FALLBACK"]}
            """
            triples = sparql_query(query, self.rdf_graph)

            # No triple found: return wrong answer prompt
            if triples.empty:
                return wrong_answer_prompt(self.language)
            
            triples["prev_chunk_content"] = triples["prev_chunk_content"].str.replace("\n", " ", regex=False)
            triples["chunk_content"] = triples["chunk_content"].str.replace("\n", " ", regex=False)
            triples["next_chunk_content"] = triples["next_chunk_content"].str.replace("\n", " ", regex=False)
            triples["content"] = triples.apply(lambda row: self.__merge_three_overlapping_strings(row["prev_chunk_content"], row["chunk_content"], row["next_chunk_content"]), axis=1)
            triples.loc[triples["content"].duplicated(), "content"] = " "

            context_data = dataframe_to_text(triples[["source_entity_name", "relationship_name", "target_entity_name", "document_url", "content"]], context_name='')

            end = time.time()
            print(f"'Searching in the db' execution time: {end - start:.6f} seconds")
            
            prompt = graph_prompt(self.language, user_type, house_type, region, context_data)
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            return prompt
        


        # Find similar entities
        entities_vector = []
        for source_entity_ids, triple_ids, source_entity_distances, triple_distances, triple_metadatas in zip(entity_query_results["ids"], triple_query_results['ids'], entity_query_results['distances'], triple_query_results['distances'], triple_query_results['metadata']):
            for source_entity_id, triple_id, source_entity_distance, triple_distance, triple_metadata in zip(source_entity_ids, triple_ids, source_entity_distances, triple_distances, triple_metadatas):
                print("\t", distance, id)
                
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

            # Perform similarity search in ChromaDB for each query embedding
            query_results = collection.query(
                query_embeddings=query_embedding_vector,
                n_results=GRAPH_PARAMETER["TOP_ENTITIES"],
            )

            # Extract the results: 'ids' will give us the entity IDs and 'distances' will give us the similarity scores
            result_df = pd.DataFrame({
                'entity': query_results['ids', 0],
                'distances': query_results['distances'],
            })
            print(result_df)

            # Merge with original entity names (from metadata)
            result_df['name'] = result_df['entity'].map({str(row['entity']): row['name'] for _, row in entities.iterrows()})

            # Filter based on similarity threshold
            result_df = result_df[result_df['cosine_similarity'] >= GRAPH_PARAMETER["SIMILARITY_THRESHOLD"]]

            # Add to the list of similar entities
            entities_vector.append(result_df)

        #start = time.time()
        # Get all entities
        #query = """
        #    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        #    PREFIX onto: <http://example.org/ontology#>
        #    
        #    SELECT ?entity ?name ?name_embedding
        #    WHERE {{
        #        ?entity rdf:type onto:Entity .
        #        ?entity onto:hasName ?name .
        #        ?entity onto:hasNameEmbedding ?name_embedding .
        #    }}
        #    GROUP BY ?entity #?name
        #"""
        #entities = sparql_query(query, self.rdf_graph)
        #end = time.time()
        #print(f"'Get all entities' execution time: {end - start:.6f} seconds")

        # For each vector in the question
        #start = time.time()
        #entities_vector = []
        #for query_embedding_vector in query_embedding_vectors:
        #
        #    temp = entities.copy()
        #    temp['cosine_similarity'] = temp['name_embedding'].apply(
        #        lambda x: cosine_similarity([ast.literal_eval(x)], [query_embedding_vector])[0][0])
        #    temp = temp.sort_values(
        #        by='cosine_similarity', ascending=False, ignore_index=True)
        #    temp = temp.drop(
        #        temp[temp.cosine_similarity < GRAPH_PARAMETER["SIMILARITY_THRESHOLD"]].index)
        #    temp = temp.drop(columns=['name_embedding'])
        #    temp = temp[:int(GRAPH_PARAMETER["TOP_ENTITIES"] /
        #                     len(query_embedding_vectors))]
        #    entities_vector.append(temp)
        #end = time.time()
        #print(f"'Get similar vectors' execution time: {end - start:.6f} seconds")

        # Get chunks for the found entities
        start = time.time()
        entity_list = pd.concat(entities_vector)
        chunks_df = sparql_query(self.__get_chunks_by_entities(
            entity_list, GRAPH_PARAMETER["TOP_CHUNKS"]), self.rdf_graph)

        if not chunks_df.empty:
            chunks_df['content'] = chunks_df['content'].apply(
                lambda x: x.replace("\n", " ").replace("  ", " ").lstrip(".").strip())
            chunks_df = chunks_df.drop(columns=['chunk', 'freq'])
        end = time.time()
        print(f"'Get chunks' execution time: {end - start:.6f} seconds")

        # Get relationships for the found entities
        start = time.time()
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
        end = time.time()
        print(f"'Get relationships' execution time: {end - start:.6f} seconds")

        # Final prompt
        context_data = "\n" + chunk_text + "\n" + relationships_text

        if len(context_data.strip()) == 0:
            return wrong_answer_prompt(self.language)

        return graph_prompt(self.language, user_type, house_type, region, context_data)

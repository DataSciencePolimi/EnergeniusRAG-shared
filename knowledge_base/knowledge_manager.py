"""Knowledge Manager Module."""
import ast
import os
import pandas as pd
from langchain_core.documents import Document
from langchain.embeddings import init_embeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph
from chromadb.api import ClientAPI
import time

from llm import LLMHandler

from .utils.graph_helpers import (process_name_or_relationship, normalize_l2,
                                  sparql_query, dataframe_to_text)
from .utils.graph_parameter import GRAPH_PARAMETER
from .utils.graph_prompt import graph_prompt, translate_chunk, wrong_answer_prompt
from .utils.energenius_graph import EnergeniusGraph

import chromadb
from chromadb.config import Settings
from itertools import permutations


class KnowledgeManager:
    """A class to manage knowledge base operations."""
    embedding: object
    language: str
    graph: EnergeniusGraph
    llm_graph_transformer: LLMGraphTransformer
    chromadbClient: ClientAPI

    knowledge_base_path: str
    path: str

    def __init__(self, provider: str, model: str, embedding: str, language: str, knowledge_base_path: str = "files"):
        """Initialize the KnowledgeManager.

        Args:
            provider (str): _Description of the model provider._
            model (str): _Description of the model name._
            embedding (str): _Description of the embedding model name._
            language (str): _Description of the language for the model._
        """
        # Initialize the LLMHandler and embedding model.
        self.llm_handler = LLMHandler(
            provider=provider,
            model=model,
            temperature=0.0,
            language=None,
            keep_history=False
        )
        self.embeddings = init_embeddings(
            model=embedding,
            provider=provider
        )
        self.language = language

        self.knowledge_base_path = knowledge_base_path

    def _update_entries(self, existing, new_entries):
        # Convert list to dict for fast access
        entry_map = {item["id"]: item for item in existing}

        # Handle batch updates
        for entry in new_entries:
            eid = entry["id"]
            dist = entry["distance"]
            if eid in entry_map:
                # Keep the nearest distance
                if dist < entry_map[eid]["distance"]:
                    entry_map[eid]["distance"] = dist
            else:
                entry_map[eid] = {"id": eid, "distance": dist}

        # Convert back to sorted list
        updated_list = sorted(entry_map.values(), key=lambda x: x["distance"])
        return updated_list


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

        # --- Init ---

        # Initialize the variables
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.knowledge_base_path)

        # Load the RDF graph
        self.graph = EnergeniusGraph()
        self.graph.load_from_file(os.path.join(self.path , "rdf_graph.ttl"))

        # Langchain package
        self.llm_graph_transformer = LLMGraphTransformer(
            llm=self.llm_handler.get_model(),
            #node_properties=["type", "description", "timestamp", "location", "alias"],
            #relationship_properties=True,
            ignore_tool_usage=True,
            additional_instructions="""Ensure that:
- No detail is omitted, even if implicit or inferred
- Compound or nested relationships are captured
- Temporal, causal, or hierarchical links are included
- Synonyms or aliases are noted if present
- Prefer short and concise node and relationship names
- Do not merge multi-word entities into single tokens (e.g., Class A instead of ClassA)""",
        )

        # Load chromadb
        self.chromadbClient = chromadb.PersistentClient(
            path=os.path.join(self.path , "chroma_db"),
            settings=Settings(anonymized_telemetry=False),
        )

        times = []

        #print(f"\n\n-----User message-----\n{message}")

        # If the question is not well-formed
        if len(message) < 3:
            return wrong_answer_prompt(self.language)

        if self.language != "English":
            message = self.llm_handler.generate_response(translate_chunk(), f"{message}", False)

        #print(f"\n\n-----English user message-----\n{message}")

        # Compute nodes and relationships for the user message.
        times.append({'start': time.time()})
        graph_question = self.llm_graph_transformer.convert_to_graph_documents(
            [Document(page_content=message)])
        times[-1]["end"] = time.time()

        #print(f"\n\n-----GraphDocument from the question-----\n{"\n".join([f"{rel.source.id} ({rel.source.type}), {rel.type}, {rel.target.id} ({rel.target.type})" for rel in graph_question[0].relationships])}")

        # Embed the question and its ndoes/relationships
        times.append({'start': time.time()})
        graph_question[0].source.metadata["embedding"] = self.embeddings.embed_query(message)
        for i, relationship in enumerate(graph_question[0].relationships):
            # Syntactic deambiguation
            relationship.source.id = process_name_or_relationship(relationship.source.id)
            relationship.source.type = process_name_or_relationship(relationship.source.type)
            relationship.type = process_name_or_relationship(relationship.type)
            relationship.target.id = process_name_or_relationship(relationship.target.id)
            relationship.target.type = process_name_or_relationship(relationship.target.type)
            # Embedding
            relationship.source.properties["embedding"] = self.embeddings.embed_query(relationship.source.id)
            relationship.source.properties["type_embedding"] = self.embeddings.embed_query(relationship.source.type)
            relationship.properties["embedding"] = self.embeddings.embed_query(relationship.type)
            relationship.target.properties["embedding"] = self.embeddings.embed_query(relationship.target.id)
            relationship.target.properties["type_embedding"] = self.embeddings.embed_query(relationship.target.type)
            relationship.properties["triple_embedding"] = self.embeddings.embed_query(
                f"{relationship.source.id} {relationship.type} {relationship.target.id}")
        times[-1]["end"] = time.time()

        #####
        # Search similar entities
        #####

        times.append({'start': time.time()})

        # Load chromadb
        collection_entities = self.chromadbClient.get_or_create_collection(name="graph_entities", metadata={"hnsw:space":"cosine", "distance_function": "cosine"})
        collection_types = self.chromadbClient.get_or_create_collection(name="graph_types", metadata={"hnsw:space":"cosine", "distance_function": "cosine"})
        collection_chunks = self.chromadbClient.get_or_create_collection(name="graph_chunks", metadata={"hnsw:space":"cosine", "distance_function": "cosine"})
        collection_relationships = self.chromadbClient.get_or_create_collection(name="graph_relationships", metadata={"hnsw:space":"cosine", "distance_function": "cosine"})
        collection_triples = self.chromadbClient.get_or_create_collection(name="graph_triples", metadata={"hnsw:space":"cosine", "distance_function": "cosine"})
        
        answer = ""
        
        context_data = ""

        while graph_question[0].relationships:

            all_entities = []
            for rel in graph_question[0].relationships:

                # --- Source ---

                # Source types
                source_types = collection_types.query(
                    query_embeddings=rel.source.properties["type_embedding"],
                    n_results=30,
                )
                source_types = [{"id": id, "distance": distance} for (id, distance) in zip(source_types["ids"][0], source_types["distances"][0]) if distance < 0.7] if source_types else []
                #print(f"\n\n-----Types for {rel.source.type}-----\n{"\n".join([f"{type}" for type in source_types])}")
                
                # Source entities
                allowed_source_entities = self.graph.get_entities(types=[id["id"] for id in source_types])["entity"].to_list() if source_types else []
                source_entities = collection_entities.query(
                    query_embeddings=rel.source.properties["embedding"],
                    ids=allowed_source_entities,
                    n_results=5,
                ) if allowed_source_entities else []
                source_entities = [{"id": id, "distance": distance} for (id, distance) in zip(source_entities["ids"][0], source_entities["distances"][0]) if distance < 0.5] if source_entities else []
                #print(f"\n\n-----Entities for {rel.source.id}-----\n{"\n".join([f"{entity}" for entity in source_entities])}")

                # Store in "all entities"
                all_entities = self._update_entries(all_entities, source_entities)

                # Get outgoing relationships (compared with question relationship and target)
                allowed_source_entities_outgoing_relationships = self.graph.get_outgoing_relationships(entities=source_entities)["relationship"].to_list() if source_entities else []
                source_entities_outgoing_relationships = []
                for emb in [rel.properties["embedding"], rel.target.properties["embedding"]]:
                    res = collection_relationships.query(
                        query_embeddings=emb,
                        ids=allowed_source_entities_outgoing_relationships,
                        n_results=5,
                    ) if allowed_source_entities_outgoing_relationships else []
                    source_entities_outgoing_relationships.extend([{"id": id, "distance": distance} for (id, distance) in zip(res["ids"][0], res["distances"][0]) if distance < 0.75] if res else [])
                #print(f"\n\n-----Outgoing relationships for {rel.type} and {rel.target.id} for {rel.source.id} similar entities-----\n{"\n".join([f"{relationship}" for relationship in source_entities_outgoing_relationships])}")
                
                # 1-Hop target entities
                allowed_one_hop_target_entities = self.graph.get_triples(rel=[id["id"] for id in source_entities_outgoing_relationships], source=[id["id"] for id in source_entities])["target"].to_list() if source_entities_outgoing_relationships and source_entities else []
                one_hop_target_entities = []
                for emb in [rel.source.properties["embedding"], rel.properties["embedding"], rel.target.properties["embedding"]]:
                    res = collection_entities.query(
                        query_embeddings=rel.target.properties["embedding"],
                        ids=allowed_one_hop_target_entities,
                        n_results=5,
                    ) if allowed_one_hop_target_entities else []
                    one_hop_target_entities.extend([{"id": id, "distance": distance} for (id, distance) in zip(res["ids"][0], res["distances"][0]) if distance < 0.75] if res else [])
                #print(f"\n\n-----Entities 1-hop from {rel.source.id}-----\n{"\n".join([f"{entity}" for entity in one_hop_target_entities])}")

                # Store in "all entities"
                all_entities = self._update_entries(all_entities, one_hop_target_entities)

                # --- Target ---

                # Target types
                target_types = collection_types.query(
                    query_embeddings=rel.target.properties["type_embedding"],
                    n_results=20,
                )
                target_types = [{"id": id, "distance": distance} for (id, distance) in zip(target_types["ids"][0], target_types["distances"][0]) if distance < 0.7] if target_types else []
                #print(f"\n\n-----Types for {rel.target.type}-----\n{"\n".join([f"{type}" for type in target_types])}")
                
                # Target entities
                allowed_target_entities = self.graph.get_entities(types=[id["id"] for id in target_types])["entity"].to_list() if target_types else []
                target_entities = collection_entities.query(
                    query_embeddings=rel.target.properties["embedding"],
                    ids=allowed_target_entities,
                    n_results=5,
                ) if allowed_target_entities else []
                target_entities = [{"id": id, "distance": distance} for (id, distance) in zip(target_entities["ids"][0], target_entities["distances"][0]) if distance < 0.5] if target_entities else []
                #print(f"\n\n-----Entities for {rel.target.id}-----\n{"\n".join([f"{entity}" for entity in target_entities])}")

                # Store in "all entities"
                all_entities = self._update_entries(all_entities, target_entities)

                # Get incoming relationships
                allowed_target_entities_incoming_relationships = self.graph.get_incoming_relationships(entities=target_entities)["relationship"].to_list() if target_entities else []
                target_entities_incoming_relationships = []
                for emb in [rel.properties["embedding"], rel.source.properties["embedding"]]:
                    res = collection_relationships.query(
                        query_embeddings=rel.properties["embedding"],
                        ids=allowed_target_entities_incoming_relationships,
                        n_results=5,
                    ) if allowed_target_entities_incoming_relationships else []
                    target_entities_incoming_relationships.extend([{"id": id, "distance": distance} for (id, distance) in zip(res["ids"][0], res["distances"][0]) if distance < 0.75] if res else [])
                #print(f"\n\n-----Incoming relationships for {rel.type} for {rel.target.id} similar entities-----\n{"\n".join([f"{relationship}" for relationship in target_entities_incoming_relationships])}")
                
                # 1-Hop source entities
                allowed_one_hop_source_entities = self.graph.get_triples(rel=[id["id"] for id in target_entities_incoming_relationships], target=[id["id"] for id in target_entities])["source"].to_list() if target_entities_incoming_relationships and target_entities else []
                one_hop_source_entities = []
                for emb in [rel.source.properties["embedding"], rel.properties["embedding"], rel.target.properties["embedding"]]:
                    res = collection_entities.query(
                        query_embeddings=rel.source.properties["embedding"],
                        ids=allowed_one_hop_source_entities,
                        n_results=5,
                    ) if allowed_one_hop_source_entities else []
                    one_hop_source_entities.extend([{"id": id, "distance": distance} for (id, distance) in zip(res["ids"][0], res["distances"][0]) if distance < 0.75] if res else [])
                #print(f"\n\n-----Entities 1-hop to {rel.source.id}-----\n{"\n".join([f"{entity}" for entity in one_hop_source_entities])}")

                # Store in "all entities"
                all_entities = self._update_entries(all_entities, one_hop_source_entities)
                all_entities = all_entities[:35]

            
            # Get "all entities"
            if all_entities:

                all_entity_descriptions = self.graph.get_entity_descriptions(entities=[id["id"] for id in all_entities], distances=[id["distance"] for id in all_entities])

                context_data += "\n" + dataframe_to_text(all_entity_descriptions, context_name='')
            
            # Search for the most similar triples based on the question
            question_triples = collection_triples.query(
                query_embeddings=graph_question[0].source.metadata["embedding"],
                n_results=5,
            )
            question_triples = [{"id": id, "distance": distance} for (id, distance) in zip(question_triples["ids"][0], question_triples["distances"][0]) if distance < 0.5] if question_triples else []
            #print(f"\n\n-----Triples for {message}-----\n{"\n".join([f"{triple}" for triple in question_triples])}")
        
            # No triple found: return wrong answer prompt
            if question_triples:
                # Get entities & references associated to the retrieved triples
                entities_from_question_triples = self.graph.get_entites_from_triples(triples=[id["id"] for id in question_triples], distances=[id["distance"] for id in question_triples])

                # No entities found: return wrong answer prompt
                if entities_from_question_triples.empty:
                    
                    context_data += "\n" + dataframe_to_text(entities_from_question_triples, context_name='')

            times[-1]["end"] = time.time()

            prompt = graph_prompt(self.language, user_type, house_type, region, context_data)
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            answer = prompt
            break
                
                
            """ # Triples containing source entities
            if not source_entities:
                triples_having_source_entities = []
            else:
                triples_having_source_entities = collection_triple_embeddings.query(
                    query_embeddings=rel.properties["triple_embeddings"],
                    where={
                        "$or": [
                                {"source": {"$in": list(map(lambda x: x[0], source_entities))}},
                                {"target": {"$in": list(map(lambda x: x[0], source_entities))}}
                            ]
                    },
                    n_results=max(1, GRAPH_PARAMETER["TOP_TRIPLES"] // len(graph_question[0].relationships) // 2),
                )
                #print(f"\n\n-----Triples for {rel.source.id + " " + rel.type + " " + rel.target.id} having {list(map(lambda x: x[0], source_entities))} (threshold: {GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]})-----\n{triples_having_source_entities}")
                triples_having_source_entities = [(id, dist, metadata) for id, dist, metadata in zip(triples_having_source_entities['ids'][0], triples_having_source_entities['distances'][0], triples_having_source_entities['metadatas'][0]) if dist <= GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]]
            
            # Target entities
            target_entities = collection_entity_embeddings.query(
                query_embeddings=rel.target.properties["embeddings"],
                n_results=GRAPH_PARAMETER["TOP_ENTITIES"],
            )
            #print(f"\n\n-----Entities for {rel.target.id} (threshold: {GRAPH_PARAMETER["ENTITIES_DISTANCE_THRESHOLD"]})-----\n{target_entities}")
            target_entities = [(id, dist) for id, dist in zip(target_entities['ids'][0], target_entities['distances'][0]) if dist <= GRAPH_PARAMETER["ENTITIES_DISTANCE_THRESHOLD"]]

            # Triples containing target entities
            if not target_entities or properties_mode:
                triples_having_target_entities = []
            else:
                triples_having_target_entities = collection_triple_embeddings.query(
                    query_embeddings=rel.properties["triple_embeddings"],
                    where={
                        "$or": [
                                {"source": {"$in": list(map(lambda x: x[0], target_entities))}},
                                {"target": {"$in": list(map(lambda x: x[0], target_entities))}}
                            ]
                    },
                    n_results=max(1, GRAPH_PARAMETER["TOP_TRIPLES"] // len(graph_question[0].relationships) // 2),
                )
                #print(f"\n\n-----Triples for {rel.source.id + " " + rel.type + " " + rel.target.id} having {list(map(lambda x: x[0], target_entities))} (threshold: {GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]})-----\n{triples_having_target_entities}")
                triples_having_target_entities = [(id, dist, metadata) for id, dist, metadata in zip(triples_having_target_entities['ids'][0], triples_having_target_entities['distances'][0], triples_having_target_entities['metadatas'][0]) if dist <= GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]]
            
            # Merge triples
            triples_having_source_target_entities = triples_having_source_entities + triples_having_target_entities
            triples_having_source_target_entities = sorted(triples_having_source_target_entities, key=lambda x: x[1])
            triples_having_source_target_entities = [tup for i, tup in enumerate(triples_having_source_target_entities) if tup[0] not in {x[0] for x in triples_having_source_target_entities[:i]}]
            triples_having_source_target_entities = triples_having_source_target_entities[:GRAPH_PARAMETER["TOP_MERGED_TRIPLES"] // len(graph_question[0].relationships)]
            #print(f"\n\n-----All triples for {rel.source.id} and {rel.target.id}-----\n{triples_having_source_target_entities}")

            all_triples.extend(triples_having_source_target_entities)
            #all_triples = sorted(all_triples, key=lambda x: x[1])
            all_triples = [tup for i, tup in enumerate(all_triples) if tup[0] not in {x[0] for x in all_triples[:i]}]
            #all_triples = all_triples[:GRAPH_PARAMETER["TOP_MERGED_TRIPLES"]]
        
        # Add triples wothout costraints
        if GRAPH_PARAMETER["TOP_TRIPLES"] < GRAPH_PARAMETER["TOP_MERGED_TRIPLES"]:
            triples_without_costraints = collection_triple_embeddings.query(
                query_embeddings=rel.properties["triple_embeddings"],
                n_results=GRAPH_PARAMETER["TOP_MERGED_TRIPLES"] - GRAPH_PARAMETER["TOP_TRIPLES"],
            )
            #print(f"\n\n-----Triples for {rel.source.id + " " + rel.type + " " + rel.target.id} (threshold: {GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]})-----\n{triples_without_costraints}")
            triples_without_costraints = [(id, dist, metadata) for id, dist, metadata in zip(triples_without_costraints['ids'][0], triples_without_costraints['distances'][0], triples_without_costraints['metadatas'][0]) if dist <= GRAPH_PARAMETER["TRIPLES_DISTANCE_THRESHOLD"]]

            all_triples.extend(triples_without_costraints)
            all_triples = sorted(all_triples, key=lambda x: x[1])
            all_triples = [tup for i, tup in enumerate(all_triples) if tup[0] not in {x[0] for x in all_triples[:i]}]
            all_triples = all_triples[:GRAPH_PARAMETER["TOP_MERGED_TRIPLES"]]
            
        # Merged triples
        #print(f"\n\n-----All triples-----\n{all_triples}")
        
        # No triple found: return wrong answer prompt
        if not all_triples:
            break
    
        # Get entity and triple names
        values_clause = " ".join(f"(<{metadata["source"]}> <{metadata["rel"]}> <{metadata["target"]}> {dist})" for id, dist, metadata in all_triples)
        query = f""
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
        ""
        #triples = sparql_query(query, self.rdf_graph)

        # No triple found: return wrong answer prompt
        if triples.empty:
            break
        
        triples["prev_chunk_content"] = triples["prev_chunk_content"].str.replace("\n", " ", regex=False)
        triples["chunk_content"] = triples["chunk_content"].str.replace("\n", " ", regex=False)
        triples["next_chunk_content"] = triples["next_chunk_content"].str.replace("\n", " ", regex=False)
        #triples["content"] = triples.apply(lambda row: self.__merge_three_overlapping_strings(row["prev_chunk_content"], row["chunk_content"], row["next_chunk_content"]), axis=1)
        triples["content"] = triples["chunk_content"]
        #triples.loc[triples["content"].duplicated(), "content"] = " "

        triples["source_entity"] = triples["source_entity_name"]
        triples["target_entity"] = triples["target_entity_name"]
        triples["relationship"] = triples["relationship_name"]

        context_data = dataframe_to_text(triples[["source_entity", "relationship", "target_entity", "description", "document_url", "content"]], context_name='')
        #context_data = dataframe_to_text(triples[["source_entity_name", "relationship_name", "target_entity_name", "description", "document_url"]], context_name='')

        end = time.time()
        #print(f"\n\n'Searching in the db' execution time: {end - start:.6f} seconds")
        
        prompt = graph_prompt(self.language, user_type, house_type, region, context_data)
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        answer = prompt
        break """
                
        # No entities extracted from the question: fallback to only the question embedding
        while answer == "":
            
            # Search for the most similar triples based on the question

            triples = collection_triples.query(
                query_embeddings=graph_question[0].source.metadata["embedding"],
                n_results=30,
            )
            triples = [{"id": id, "distance": distance} for (id, distance) in zip(triples["ids"][0], triples["distances"][0]) if distance < 0.5] if triples else []
            print(f"\n\n-----Triples for {message}-----\n{"\n".join([f"{triple}" for triple in triples])}")
        
            # No triple found: return wrong answer prompt
            if not triples:
                break

            # Get entities & references associated to the retrieved triples
            all_entities = self.graph.get_entites_from_triples(triples=[id["id"] for id in triples], distances=[id["distance"] for id in triples])

            # No entities found: return wrong answer prompt
            if all_entities.empty:
                break
            
            context_data = dataframe_to_text(all_entities, context_name='')

            prompt = graph_prompt(self.language, user_type, house_type, region, context_data)
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            end = time.time()

            answer = prompt
            break
        
        if answer == "":
            answer = wrong_answer_prompt(self.language)
            
        return answer


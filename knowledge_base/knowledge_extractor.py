"""Module to create a knowledge base from a text files."""

import hashlib
import os
import re

import joblib
import numpy as np
from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rdflib import FOAF, OWL, RDF, RDFS, XSD, BNode, Graph, Literal, Namespace
from tqdm import tqdm

from llm import LLMHandler


class KnowledgeExtractor:
    """_Class to create a knowledge base from a text files._"""

    def __init__(self, provider: str, model: str, embedding: str):
        """Initialize the KnowledgeExtractor.

        Args:
            provider (str): _Description of the model provider._
            model (str): _Description of the model name._
            embedding (str): _Description of the embedding model name._
        """
        # Initialize the LLMHandler and embedding model.
        llm_handler = LLMHandler(
            provider=provider, model=model, temperature=0.0, language=None
        )

        self.embeddings = init_embeddings(model=embedding, provider=provider)

        self.llm_graph_transformer = LLMGraphTransformer(
            llm=llm_handler.get_model(),
            # node_properties=True,
            # relationship_properties=True,
            ignore_tool_usage=True,
        )

    def __process_name_or_relationship(self, s: str) -> str:
        """_Process the name or relationship to make it more readable._
        Args:
            str (str): _Description of the string to process._
        Returns:
            str: _Description of the processed string._
        """
        s = s.replace("_", " ")
        s = s.title()
        return s

    def __normalize_l2(self, x: np.ndarray) -> list:
        """_Normalize the input array to unit length using L2 normalization._
        Args:
            x (np.ndarray): _Description of the input array._
        Returns:
            list: _Description of the normalized array._
        """
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm).tolist()

    def __remove_non_alphanumerical(self, s: str) -> str:
        strin = re.sub("[^A-Za-z0-9]", "", s)
        h = hashlib.md5(s.encode()).hexdigest()
        return strin + "_" + h

    def run(
        self,
        file_name: str,
        load_existing: bool,
        html_links: list[str] = None,
        pdf_links: list[str] = None,
        use_embedding: bool = True,
    ) -> None:
        """_Main function to create the knowledge base._
        Args:
            load_existing (bool): _Description of the load_existing parameter._
            html_links (list[str], optional): _Description of the html_links parameter._ Defaults to None.
            pdf_links (list[str], optional): _Description of the pdf_links parameter._ Defaults to None.
            use_embedding (bool, optional): _Description of the use_embedding parameter._ Defaults to True.
        """

        # Initialize the variables
        dir_path = os.path.dirname(os.path.realpath(__file__))
        html_docs = None
        pdf_docs = []

        # Checking if files folder is present
        if not os.path.exists(os.path.join(dir_path, "files")):
            os.makedirs(os.path.join(dir_path, "files"))

        if load_existing:
            try:
                print("Trying to load existing graph_documents.joblib")
                graph_documents = joblib.load(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "files",
                        "graph_documents.joblib",
                    )
                )
            except FileNotFoundError:
                print(
                    "No existing knowledge base found. Please create a new one or set load_existing to False."
                )
                return
        else:
            # Check if the user wants to load an existing knowledge base
            if load_existing:
                try:
                    print("Trying to load existing html_docs.joblib")
                    html_docs = joblib.load(
                        os.path.join(dir_path, "files", "html_docs.joblib")
                    )
                except FileNotFoundError:
                    print(
                        "No existing HTML documents found. Please create a new one or set load_existing to False."
                    )
                    return
            else:
                if html_links is not None:
                    html_loader = AsyncHtmlLoader(html_links)
                    html_docs = html_loader.load()
                    # Save the html_docs to a file for later use
                    joblib.dump(
                        html_docs, os.path.join(dir_path, "files", "html_docs.joblib")
                    )

            html2text = Html2TextTransformer()
            html_docs_transformed = html2text.transform_documents(html_docs)

            # Check if the user wants to load an existing pdf knowledge base
            if load_existing:
                try:
                    print("Trying to load existing pdf_docs.joblib")
                    pdf_docs = joblib.load(
                        os.path.join(dir_path, "files", "pdf_docs.joblib")
                    )
                except FileNotFoundError:
                    print(
                        "No existing PDF documents found. Please create a new one or set load_existing to False."
                    )
                    return
            else:
                if pdf_links is not None and len(pdf_links) > 0:
                    for p in tqdm(pdf_links, desc="Fetching PDFs: "):
                        pdf_docs.append(PyPDFLoader(p).load())

                for i, pdf_doc_pages in enumerate(
                    tqdm(pdf_docs, desc="Processing PDFs: ")
                ):
                    source = ""
                    pages = ""
                    for j, page in enumerate(pdf_doc_pages):
                        source = page.metadata["source"]
                        pages += page.page_content
                    pdf_docs[i] = Document(
                        page_content=pages, metadata={"source": source}
                    )

                # Save the pdf_docs to a file for later use
                joblib.dump(
                    pdf_docs, os.path.join(dir_path, "files", "pdf_docs.joblib")
                )

            if pdf_docs is None and len(pdf_docs) != 0:
                all_docs = pdf_docs + html_docs_transformed
            else:
                all_docs = html_docs_transformed

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=[";", ".", ","],
            )

            split_documents = text_splitter.split_documents(all_docs)

            print(f"> Number of HTML documents: {len(html_docs)}")
            print(f"> Number of PDF documents: {len(pdf_docs)}")
            print(f"> Number of documents: {len(split_documents)}")
            print(f"> Number of chunks: {len(all_docs)}")
            print("Starting to convert to graph documents...")

            graph_documents = self.llm_graph_transformer.convert_to_graph_documents(
                split_documents
            )

            print("Ending to convert to graph documents...")

            # Create the knowledge base
            joblib.dump(
                graph_documents,
                os.path.join(dir_path, "files", "graph_documents.joblib"),
            )

        if use_embedding:
            if load_existing:
                try:
                    print(
                        "Trying to load existing graph_documents_and_embeddings.joblib"
                    )
                    graph_documents = joblib.load(
                        os.path.join(
                            dir_path,
                            "files",
                            "graph_documents_and_embeddings.joblib",
                        )
                    )
                except FileNotFoundError:
                    print(
                        "No existing knowledge base found. Please create a new one or set load_existing to False."
                    )
                    return
            else:
                # Syntactic disambiguation
                for i, doc in enumerate(
                    tqdm(all_docs, desc="Syntactic disambiguation of documents: ")
                ):
                    # For each chunk in this "full document" -> filter chunks from "graph document"
                    for j, chunk in enumerate(
                        [
                            doc_chunk
                            for doc_chunk in graph_documents
                            if doc_chunk.source.metadata["source"]
                            == doc.metadata["source"]
                        ]
                    ):

                        # Entities (nodes)
                        for _, node in enumerate(chunk.nodes):

                            node.id = self.__process_name_or_relationship(node.id)
                            node.type = self.__process_name_or_relationship(node.type)

                        # Relationships
                        for k, rel in enumerate(chunk.relationships):

                            rel.source.id = self.__process_name_or_relationship(
                                rel.source.id
                            )
                            rel.type = self.__process_name_or_relationship(rel.type)
                            rel.target.id = self.__process_name_or_relationship(
                                rel.target.id
                            )

                for i, doc in enumerate(tqdm(all_docs, desc="Embedding documents: ")):
                    # doc.page_content = {'text': doc.page_content, 'embedding': embeddings.embed_query(doc.page_content)}

                    # For each chunk in this "full document" -> filter chunks from "graph document"
                    for j, chunk in enumerate(
                        tqdm(
                            [
                                doc_chunk
                                for doc_chunk in graph_documents
                                if doc_chunk.source.metadata["source"]
                                == doc.metadata["source"]
                            ]
                        )
                    ):

                        chunk.source.page_content = {
                            "text": chunk.source.page_content,
                            "embedding": self.embeddings.embed_query(
                                chunk.source.page_content
                            ),
                        }

                        # Entities (nodes)
                        for k, node in enumerate(chunk.nodes):
                            node.id = {
                                "text": node.id,
                                "embedding": self.embeddings.embed_query(node.id),
                            }
                            node.type = {
                                "text": node.type,
                                "embedding": self.embeddings.embed_query(node.type),
                            }

                        # Relationships
                        for k, rel in enumerate(chunk.relationships):
                            rel.type = {
                                "text": rel.type,
                                "embedding": self.embeddings.embed_query(rel.type),
                            }

                # Export the knowledge base
                joblib.dump(
                    graph_documents,
                    os.path.join(
                        dir_path, "files", "graph_documents_and_embeddings.joblib"
                    ),
                )

                # Reducing embedding dimensions
                for doc in tqdm(
                    graph_documents, desc="Reducing embedding dimensions: "
                ):
                    for node in doc.nodes:
                        node.id["embedding"] = self.__normalize_l2(
                            node.id["embedding"][:512]
                        )  # node.id["embedding"]
                        node.type["embedding"] = self.__normalize_l2(
                            node.type["embedding"][:512]
                        )  # node.id["embedding"]
                        # print(node.id["text"], len(node.id["embedding"]), node.type["text"], len(node.type["embedding"]))
                    for rel in doc.relationships:
                        rel.type["embedding"] = self.__normalize_l2(
                            rel.type["embedding"][:512]
                        )  # rel.type["embedding"]
                        # print(rel.source.id, rel.type["text"], len(rel.type["embedding"]), rel.target.id)
                    doc.source.page_content["embedding"] = self.__normalize_l2(
                        doc.source.page_content["embedding"][:512]
                    )  # doc.source.page_content["embedding"]

        # Initialize RDF Graph
        rdf_graph = Graph()

        # Define Namespaces
        EX = Namespace("http://example.org/data#")
        ONTO = Namespace("http://example.org/ontology#")

        # Bind namespaces to prefixes for readability
        rdf_graph.bind("ex", EX)
        rdf_graph.bind("onto", ONTO)
        rdf_graph.bind("rdf", RDF)
        rdf_graph.bind("rdfs", RDFS)
        rdf_graph.bind("owl", OWL)
        rdf_graph.bind("foaf", FOAF)

        # --- Define Classes ---
        classes = [
            "Document",
            "Entity",
            "Relationship",
            "RelTriple",
            "Chunk",
            "Property",
            # "CommunityReport",
            # "Community",
            # "Finding"
        ]

        for cls in tqdm(classes, desc="Defining classes: "):
            rdf_graph.add((ONTO[cls], RDF.type, OWL.Class))

        # --- Define Properties ---

        # Object Properties
        object_properties = {
            # Chunks & documents
            "hasChunk": {
                "domain": "Document",
                "range": "Chunk",
                "type": OWL.ObjectProperty,
            },
            "belongsToDocument": {
                "domain": "Chunk",
                "range": "Document",
                "type": OWL.ObjectProperty,
            },
            "hasNext": {
                "domain": "Chunk",
                "range": "Chunk",
                "type": OWL.ObjectProperty,
            },
            "hasPrevious": {
                "domain": "Chunk",
                "range": "Chunk",
                "type": OWL.ObjectProperty,
            },
            # Entity
            "hasType": {
                "domain": "Entity",
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "hasProperty": {
                "domain": "Entity",
                "range": "Property",
                "type": OWL.ObjectProperty,
            },
            # Relationship
            "relatesTarget": {
                "domain": "Entity",
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "relatesSource": {
                "domain": "Entity",
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "hasSource": {
                "domain": ["Relationship", "RelTriple"],
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "isSourceOf": {
                "domain": "Entity",
                "range": "Relationship",
                "type": OWL.ObjectProperty,
            },
            "hasTarget": {
                "domain": ["Relationship", "RelTriple"],
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "isTargetOf": {
                "domain": "Entity",
                "range": "Relationship",
                "type": OWL.ObjectProperty,
            },
            # References
            "hasEntity": {
                "domain": ["Document", "Chunk"],
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "hasRelationship": {
                "domain": ["Document", "Chunk", "RelTriple"],
                "range": "Relationship",
                "type": OWL.ObjectProperty,
            },
            "belongsToChunk": {
                "domain": ["Entity", "Relationship"],
                "range": "Chunk",
                "type": OWL.ObjectProperty,
            },
            "belongsToDocumentRef": {
                "domain": ["Entity", "Relationship"],
                "range": "Document",
                "type": OWL.ObjectProperty,
            },
        }

        # Datatype Properties
        datatype_properties = {
            # Document
            # "hasTitle": {
            #    "domain": ["Document"],#, "Community", "CommunityReport"],
            #    "range": XSD.string,
            #    "type": OWL.DatatypeProperty
            # },
            # "hasDescription": {
            #    "domain": ["Document"],
            #    "range": XSD.string,
            #    "type": OWL.DatatypeProperty
            # },
            "hasUri": {
                "domain": ["Document", "Chunk"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            "hasContent": {
                "domain": ["Document", "Chunk"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            "hasContentEmbedding": {
                "domain": ["Document", "Chunk"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            # Entities
            "hasName": {
                "domain": ["Entity", "Relationship", "Property"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            "hasNameEmbedding": {
                "domain": ["Entity", "Relationship", "Property"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            "hasValue": {
                "domain": ["Property"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            "hasValueEmbedding": {
                "domain": ["Property"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            # "hasHumanReadableId": {
            #     "domain": ["Entity", "Relationship"],
            #     "range": XSD.integer,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasWeight": {
            #     "domain": "Relationship",
            #     "range": XSD.integer,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasSourceName": {
            #     "domain": "Relationship",
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasTargetName": {
            #     "domain": "Relationship",
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasText": {
            #     "domain": "TextUnit",
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasNToken": {
            #     "domain": ["TextUnit"],
            #     "range": XSD.integer,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasLevel": {
            #     "domain": ["Community", "CommunityReport"],
            #     "range": XSD.integer,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasRank": {
            #     "domain": ["Community", "CommunityReport", "Relationship"],
            #     "range": XSD.integer,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasSummary": {
            #     "domain": ["CommunityReport", "Finding"],
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasExplanation": {
            #     "domain": "Finding",
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasFullContent": {
            #     "domain": "CommunityReport",
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
            # "hasFullContentJSON": {
            #     "domain": "CommunityReport",
            #     "range": XSD.string,
            #     "type": OWL.DatatypeProperty
            # },
        }

        # Add Object Properties to the Graph
        for prop, details in tqdm(
            object_properties.items(), desc="Defining properties: "
        ):
            rdf_graph.add((ONTO[prop], RDF.type, details["type"]))
            rdf_graph.add((ONTO[prop], RDFS.domain, ONTO[details["domain"]]))
            rdf_graph.add((ONTO[prop], RDFS.range, ONTO[details["range"]]))

        # Add Datatype Properties to the Graph
        for prop, details in tqdm(
            datatype_properties.items(), desc="Defining properties: "
        ):
            rdf_graph.add((ONTO[prop], RDF.type, details["type"]))
            # Handle multiple domains
            domains = (
                details["domain"]
                if isinstance(details["domain"], list)
                else [details["domain"]]
            )
            for domain in domains:
                rdf_graph.add((ONTO[prop], RDFS.domain, ONTO[domain]))
            rdf_graph.add((ONTO[prop], RDFS.range, details["range"]))

        # For each "full document"
        for i, doc in enumerate(
            tqdm(all_docs, desc="Processing documents into the graph: ")
        ):

            # Document
            doc_id = doc.metadata["source"]
            doc_uri = EX[f"Document_{doc_id}"]
            rdf_graph.add((doc_uri, RDF.type, ONTO.Document))
            rdf_graph.add((doc_uri, ONTO.hasUri, Literal(doc_id, datatype=XSD.string)))
            # rdf_graph.add((doc_uri, ONTO.hasContent, Literal(doc.page_content["text"], datatype=XSD.string)))
            # rdf_graph.add((doc_uri, ONTO.hasContentEmbedding, Literal(doc.page_content["embedding"], datatype=XSD.string)))

            # if hasattr(doc.metadata, 'title'):
            #    rdf_graph.add((doc_uri, ONTO.hasTitle, Literal(doc.metadata.title, datatype=XSD.string)))
            # if hasattr(doc.metadata, 'description'):
            #    rdf_graph.add((doc_uri, ONTO.hasDescription, Literal(doc.metadata.description, datatype=XSD.string)))

            # For each chunk in this "full document" -> filter chunks from "graph document"
            for j, chunk in enumerate(
                [
                    doc_chunk
                    for doc_chunk in graph_documents
                    if doc_chunk.source.metadata["source"] == doc_id
                ]
            ):

                # Chunk
                chunk_uri = EX[f"Chunk_{doc_id}_{j}"]
                rdf_graph.add((chunk_uri, RDF.type, ONTO.Chunk))
                rdf_graph.add(
                    (chunk_uri, ONTO.hasUri, Literal(chunk_uri, datatype=XSD.string))
                )
                rdf_graph.add(
                    (
                        chunk_uri,
                        ONTO.hasContent,
                        Literal(chunk.source.page_content["text"], datatype=XSD.string),
                    )
                )
                rdf_graph.add(
                    (
                        chunk_uri,
                        ONTO.hasContentEmbedding,
                        Literal(
                            chunk.source.page_content["embedding"], datatype=XSD.string
                        ),
                    )
                )
                rdf_graph.add((doc_uri, ONTO.hasChunk, chunk_uri))
                rdf_graph.add((chunk_uri, ONTO.belongsToDocument, doc_uri))

                if j > 0:
                    previous_chunk_uri = EX[f"Chunk_{doc_id}_{j-1}"]
                    rdf_graph.add((previous_chunk_uri, ONTO.hasNext, chunk_uri))
                    rdf_graph.add((chunk_uri, ONTO.hasPrevious, previous_chunk_uri))

                # Entities (nodes)
                for k, node in enumerate(chunk.nodes):

                    node_id = node.id["text"]
                    node_id_embedding = node.id["embedding"]

                    entity_id = self.__remove_non_alphanumerical(node_id)
                    entity_uri = EX[f"Entity_{entity_id}"]
                    rdf_graph.add((entity_uri, RDF.type, ONTO.Entity))
                    rdf_graph.add(
                        (
                            entity_uri,
                            ONTO.hasName,
                            Literal(node_id, datatype=XSD.string),
                        )
                    )
                    rdf_graph.add(
                        (
                            entity_uri,
                            ONTO.hasNameEmbedding,
                            Literal(node_id_embedding, datatype=XSD.string),
                        )
                    )

                    node_type = node.type["text"]
                    node_type_embedding = node.type["embedding"]

                    entity_type_id = self.__remove_non_alphanumerical(node_type)
                    entity_type_uri = EX[f"Entity_{entity_type_id}"]
                    rdf_graph.add((entity_type_uri, RDF.type, ONTO.Entity))
                    rdf_graph.add(
                        (
                            entity_type_uri,
                            ONTO.hasName,
                            Literal(node_type, datatype=XSD.string),
                        )
                    )
                    rdf_graph.add(
                        (
                            entity_type_uri,
                            ONTO.hasNameEmbedding,
                            Literal(node_type_embedding, datatype=XSD.string),
                        )
                    )

                    rdf_graph.add((entity_uri, ONTO.hasType, entity_type_uri))

                    for key, value in node.properties.items():
                        property_id = self.__remove_non_alphanumerical(key)
                        property_uri = EX[f"Property_{property_id}_{entity_id}"]
                        rdf_graph.add((property_uri, RDF.type, ONTO.Property))
                        rdf_graph.add((entity_uri, ONTO.hasProperty, property_uri))
                        rdf_graph.add(
                            (
                                property_uri,
                                ONTO.hasName,
                                Literal(key, datatype=XSD.string),
                            )
                        )
                        rdf_graph.add(
                            (
                                property_uri,
                                ONTO.hasValue,
                                Literal(value, datatype=XSD.string),
                            )
                        )

                    rdf_graph.add((chunk_uri, ONTO.hasEntity, entity_uri))
                    rdf_graph.add((chunk_uri, ONTO.hasEntity, entity_type_uri))
                    rdf_graph.add((doc_uri, ONTO.hasEntity, entity_uri))
                    rdf_graph.add((doc_uri, ONTO.hasEntity, entity_type_uri))

                    rdf_graph.add((entity_uri, ONTO.belongsToChunk, chunk_uri))
                    rdf_graph.add((entity_uri, ONTO.belongsToDocument, doc_uri))
                    rdf_graph.add((entity_type_uri, ONTO.belongsToChunk, chunk_uri))
                    rdf_graph.add((entity_type_uri, ONTO.belongsToDocument, doc_uri))

                # Relationships
                for k, rel in enumerate(chunk.relationships):

                    rel_type = rel.type["text"]
                    rel_type_embedding = rel.type["embedding"]

                    rel_id = self.__remove_non_alphanumerical(rel_type)
                    rel_uri = EX[f"Relationship_{rel_id}"]
                    rdf_graph.add((rel_uri, RDF.type, ONTO.Relationship))
                    rdf_graph.add(
                        (rel_uri, ONTO.hasName, Literal(rel_type, datatype=XSD.string))
                    )
                    rdf_graph.add(
                        (
                            rel_uri,
                            ONTO.hasNameEmbedding,
                            Literal(rel_type_embedding, datatype=XSD.string),
                        )
                    )

                    source_id = self.__remove_non_alphanumerical(rel.source.id)
                    source_uri = EX[f"Entity_{source_id}"]
                    rdf_graph.add((rel_uri, ONTO.hasSource, source_uri))
                    rdf_graph.add((source_uri, ONTO.isSourceOf, rel_uri))

                    target_id = self.__remove_non_alphanumerical(rel.target.id)
                    target_uri = EX[f"Entity_{target_id}"]
                    rdf_graph.add((rel_uri, ONTO.hasTarget, target_uri))
                    rdf_graph.add((target_uri, ONTO.isTargetOf, rel_uri))

                    rdf_graph.add((source_uri, ONTO.relatesTarget, target_uri))
                    rdf_graph.add((target_uri, ONTO.relatesSource, source_uri))

                    bnode = BNode()
                    rdf_graph.add((bnode, RDF.type, ONTO.RelTriple))
                    rdf_graph.add((bnode, ONTO.hasSource, source_uri))
                    rdf_graph.add((bnode, ONTO.hasRelationship, rel_uri))
                    rdf_graph.add((bnode, ONTO.hasTarget, target_uri))

                    rdf_graph.add((chunk_uri, ONTO.hasRelationship, rel_uri))
                    rdf_graph.add((doc_uri, ONTO.hasRelationship, rel_uri))

                    rdf_graph.add((rel_uri, ONTO.belongsToChunk, chunk_uri))
                    rdf_graph.add((rel_uri, ONTO.belongsToDocument, doc_uri))

        # Serialize as Turtle
        turtle_data = rdf_graph.serialize(format="turtle")

        # Save to a file
        export_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "files", f"{file_name}.ttl"
        )
        with open(export_file, "w", encoding="utf-8") as f:
            f.write(turtle_data)

        print(f"RDF graph has been serialized to '{str(export_file)}'")

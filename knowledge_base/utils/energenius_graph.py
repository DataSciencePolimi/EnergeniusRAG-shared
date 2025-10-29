from collections import defaultdict
from rdflib import FOAF, OWL, RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef
from rdflib.plugins.sparql import prepareQuery
import pandas as pd

class EnergeniusGraph:
        
    rdf_graph = None
    
    # Define Namespaces
    EX = Namespace("http://example.org/data#")
    ONTO = Namespace("http://example.org/ontology#")


    def __init__(self, ):
        
        # Initialize RDF Graph
        self.rdf_graph = Graph()
        

    def load_ontology(self, ):

        # Bind namespaces to prefixes for readability
        self.rdf_graph.bind("ex", self.EX)
        self.rdf_graph.bind("onto", self.ONTO)
        self.rdf_graph.bind("rdf", RDF)
        self.rdf_graph.bind("rdfs", RDFS)
        self.rdf_graph.bind("owl", OWL)
        self.rdf_graph.bind("foaf", FOAF)

        # Define Classes
        classes = [
            "Document",
            "Chunk",
            "Entity",
            "EntityType",
            "Relationship",
            "Triple",
            # "Community",
        ]

        for cls in classes:
            self.rdf_graph.add((self.ONTO[cls], RDF.type, OWL.Class))

        # --- Define Properties ---

        # Object Properties
        object_properties = {
            # Documents
            "hasChunk": {
                "domain": "Document",
                "range": "Chunk",
                "type": OWL.ObjectProperty,
            },
            "belongsToDocument": {
                "domain": ["Chunk", "Entity", "EntityType", "Relationship", "Triple"],
                "range": "Document",
                "type": OWL.ObjectProperty,
            },
            # Chunks
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
            "belongsToChunk": {
                "domain": ["Entity", "EntityType", "Relationship", "Triple"],
                "range": "Chunk",
                "type": OWL.ObjectProperty,
            },
            # Entity
            "hasType": {
                "domain": "Entity",
                "range": "EntityType",
                "type": OWL.ObjectProperty,
            },
            "isTypeOf": {
                "domain": "EntityType",
                "range": "Entity",
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
                "domain": ["Relationship", "Triple"],
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "isSourceOf": {
                "domain": "Entity",
                "range": "Relationship",
                "type": OWL.ObjectProperty,
            },
            "hasTarget": {
                "domain": ["Relationship", "Triple"],
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "isTargetOf": {
                "domain": "Entity",
                "range": "Relationship",
                "type": OWL.ObjectProperty,
            },
            "composes": {
                "domain":  ["Relationship", "Entity"],
                "range": "Triple",
                "type": OWL.ObjectProperty,
            },
            # References
            "hasEntity": {
                "domain": ["Document", "Chunk", "EntityType"],
                "range": "Entity",
                "type": OWL.ObjectProperty,
            },
            "hasRelationship": {
                "domain": ["Document", "Chunk", "Triple"],
                "range": "Relationship",
                "type": OWL.ObjectProperty,
            },
        }

        # Datatype Properties
        datatype_properties = {
            # Documents & Chunks
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
            # Entities, Relationships & Properties
            "hasName": {
                "domain": ["Entity", "EntityType", "Relationship"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
            "hasDescription": {
                "domain": ["Entity", "Relationship", "Triple"],
                "range": XSD.string,
                "type": OWL.DatatypeProperty,
            },
        }

        # Add Object Properties to the Graph
        for prop, details in object_properties.items():
            self.rdf_graph.add((self.ONTO[prop], RDF.type, details["type"]))
            # Handle multiple domains
            domains = (
                details["domain"]
                if isinstance(details["domain"], list)
                else [details["domain"]]
            )
            for domain in domains:
                self.rdf_graph.add((self.ONTO[prop], RDFS.domain, self.ONTO[domain]))
            self.rdf_graph.add((self.ONTO[prop], RDFS.range, self.ONTO[details["range"]]))

        # Add Datatype Properties to the Graph
        for prop, details in datatype_properties.items():
            self.rdf_graph.add((self.ONTO[prop], RDF.type, details["type"]))
            # Handle multiple domains
            domains = (
                details["domain"]
                if isinstance(details["domain"], list)
                else [details["domain"]]
            )
            for domain in domains:
                self.rdf_graph.add((self.ONTO[prop], RDFS.domain, self.ONTO[domain]))
            self.rdf_graph.add((self.ONTO[prop], RDFS.range, details["range"]))


    def add(self, *args, **kwargs):

        self.rdf_graph.add(*args, **kwargs)


    def load_from_file(self, filename):
        self.rdf_graph.parse(filename, format="turtle")


    def save_to_file(self, filename):
    
        turtle_data = self.rdf_graph.serialize(format="turtle")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(turtle_data)
        print(f"RDF graph has been serialized to '{str(filename)}'")


    def get_entities(self, types=None, entities=None):
        type_filter = ""
        if types:
            type_filter = f"""
                ?entity onto:hasType ?type .
                VALUES ?type {{ {" ".join(f"<{type}>" for type in types)} }} .
            """
        entity_filter = ""
        if entities:
            entity_filter = f"""
                VALUES ?entity {{ {" ".join(f"<{entity}>" for entity in entities)} }} .
            """
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?entity ?name ?description
            WHERE {{
                ?entity a onto:Entity .
                {type_filter}
                {entity_filter}
                OPTIONAL {{ ?entity onto:hasName ?name . }}
                OPTIONAL {{ ?entity onto:hasDescription ?description . }}
            }}
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_entity_descriptions(self, entities, distances):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?name ?description (SAMPLE(?reference) AS ?reference)
            WHERE {{
                ?entity a onto:Entity ;
                    onto:hasName ?name ;
                    onto:hasDescription ?description ;
                    onto:belongsToDocument ?document .
                ?document onto:hasUri ?reference .
                VALUES (?entity ?distance) {{ {" ".join(f"(<{entity}> {distance})" for (entity, distance) in zip(entities, distances))} }} .
            }}
            GROUP BY ?entity ?name ?description
            ORDER BY ASC(?distance)
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_entity_references(self, entities, distances):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?reference
            WHERE {{
                ?entity a onto:Entity ;
                    onto:belongsToDocument ?document .
                ?document onto:hasUri ?reference .
                VALUES (?entity ?distance) {{ {" ".join(f"(<{entity}> {distance})" for (entity, distance) in zip(entities, distances))} }} .
            }}
            ORDER BY ASC(?distance)
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_entites_from_triples(self, triples, distances):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?name ?description (SAMPLE(?reference) AS ?reference)
            WHERE {{
                ?triple a onto:Triple .
                ?entity a onto:Entity ;
                    onto:hasName ?name ;
                    onto:hasDescription ?description ;
                    onto:belongsToDocument ?document ;
                    onto:composes ?triple.
                ?document onto:hasUri ?reference .
                VALUES (?triple ?distance) {{ {" ".join(f"(<{triple}> {distance})" for (triple, distance) in zip(triples, distances))} }} .
            }}
            GROUP BY ?entity ?name ?description
            ORDER BY ASC(?distance)
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_types(self, entity=None):
        query_entity = f"?type onto:isTypeOf <{entity}> ." if entity else ""
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?type ?name
            WHERE {{
                ?type a onto:EntityType ;
                    onto:hasName ?name .
                {query_entity}
            }}
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_relationships(self):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?relationship ?name
            WHERE {{
                ?relationship a onto:Relationship ;
                    onto:hasName ?name .
            }}
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_incoming_relationships(self, entities):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?source ?relationship ?target
            WHERE {{
                ?relationship a onto:Relationship ;
                            onto:hasSource ?source ;
                            onto:hasTarget ?target .
                ?source onto:relatesTarget ?target .
                ?target onto:relatesSource ?source .
                ?source onto:isSourceOf ?relationship .
                ?target onto:isTargetOf ?relationship .
                VALUES (?target ?distance) {{
                    {" ".join(f"(<{entity["id"]}> {entity["distance"]})" for entity in entities)}
                }}
            }}
            ORDER BY ASC(?distance)
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_outgoing_relationships(self, entities):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?source ?relationship ?target
            WHERE {{
                ?relationship a onto:Relationship ;
                            onto:hasSource ?source ;
                            onto:hasTarget ?target .
                ?source onto:relatesTarget ?target .
                ?target onto:relatesSource ?source .
                ?source onto:isSourceOf ?relationship .
                ?target onto:isTargetOf ?relationship .
                VALUES (?source ?distance) {{
                    {" ".join(f"(<{entity["id"]}> {entity["distance"]})" for entity in entities)}
                }}
            }}
            ORDER BY ASC(?distance)
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_triples(self, source=None, rel=None, target=None):
        query_source = f"VALUES ?source {{ {" ".join(f"<{s}>" for s in source)} }} ." if source else ""
        query_rel = f"VALUES ?relationship {{ {" ".join(f"<{r}>" for r in rel)} }} ." if rel else ""
        query_target = f"VALUES ?target {{ {" ".join(f"<{t}>" for t in target)} }} ." if target else ""
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?triple ?description ?source ?relationship ?target
            WHERE {{
                ?triple a onto:Triple ;
                    onto:hasDescription ?description ;
                    onto:hasSource ?source ;
                    onto:hasRelationship ?relationship ;
                    onto:hasTarget ?target .
                {query_source}
                {query_rel}
                {query_target}
            }}
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_chunks(self):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?chunk ?content
            WHERE {{
                ?chunk a onto:Chunk ;
                    onto:hasContent ?content .
            }}
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_triples_and_chunks(self):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?triple ?prev_chunk_content ?chunk_content ?next_chunk_content ?source_entity ?source_entity_name ?relationship ?relationship_name ?target_entity ?target_entity_name
            WHERE {{
                ?triple onto:hasRelationship ?relationship ; rdf:type onto:Triple ; onto:hasSource ?source_entity ; onto:hasTarget ?target_entity ; onto:belongsToChunk ?chunk .
                ?source_entity onto:hasName ?source_entity_name .
                ?relationship onto:hasName ?relationship_name .
                ?target_entity onto:hasName ?target_entity_name .
                ?chunk onto:hasContent ?chunk_content .

                OPTIONAL {{ ?chunk onto:hasNext ?next_chunk . }}
                OPTIONAL {{ ?next_chunk onto:hasContent ?next_chunk_content . }}
                OPTIONAL {{ ?chunk onto:hasPrevious ?prev_chunk . }}
                OPTIONAL {{ ?prev_chunk onto:hasContent ?prev_chunk_content . }}
            }}
            GROUP BY ?triple
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def get_entity_triples(self, entity):
        query = f"""
            PREFIX onto: <{self.ONTO}>
            PREFIX ex: <{self.EX}>
            SELECT ?description
            WHERE {{
                <{entity}> onto:composes ?triple .
                ?triple a onto:Triple ;
                    onto:hasDescription ?description .
            }}
        """
        return self.sparql_to_dataframe(self.rdf_graph.query(prepareQuery(query)))


    def sparql_to_dataframe(self, results) -> pd.DataFrame:

        try:
            # Extract variable (column) names from the query result
            columns = results.vars  # Get the variable names from the query results

            # Process the results and convert them into a list of dictionaries
            data = []
            for row in results:
                # Dynamically build a row dict
                row_data = {str(var): row[var] for var in columns}
                data.append(row_data)

            # Convert the data into a DataFrame
            df = pd.DataFrame(data, columns=[str(var) for var in columns])
            return df
        
        except Exception:
            return pd.DataFrame()


    def __str__(self):
        return '\n'.join(f"{node}: {neighbors}" for node, neighbors in self.graph.items())

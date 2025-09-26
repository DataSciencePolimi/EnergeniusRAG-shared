"""Helpers for graph operations and SPARQL queries."""
import pandas as pd
import numpy as np

from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery


def process_name_or_relationship(text: str) -> str:
    """Process a string to make it more readable.
    Args:
        text (str): The string to process.
    """
    text = text.replace("_", " ")
    text = text.title()
    return text


def normalize_l2(x):
    """Normalize a vector or a matrix to L2 norm.

    Args:
        x (np.ndarray): The input vector or matrix to normalize.

    Returns:
        np.ndarray: The normalized vector or matrix.
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


def sparql_query(query: str, rdf_graph: Graph) -> pd.DataFrame:
    """Execute a SPARQL query on the RDF graph and return the results as a DataFrame.
    Args:
        query (str): The SPARQL query to execute.
        rdf_graph (Graph): The RDF graph to query.
    Returns:
        pd.DataFrame: The results of the query as a DataFrame.
    """
    try:
        # Prepare and execute the query
        query = prepareQuery(query)
        results = rdf_graph.query(query)

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


def dataframe_to_text(df: pd.DataFrame, column_delimiter: str = "|", context_name: str = "") -> str:
    """Convert a DataFrame to a text representation.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        column_delimiter (str, optional): The delimiter to use between columns. Defaults to "|".
        context_name (str, optional): The context name to include in the output. Defaults to "".

    Returns:
        str: The text representation of the DataFrame.
    """
    if df.empty:
        return ""

    header_text = f"-----{context_name}-----\n" if context_name else ""
    header_text += column_delimiter.join(df.columns) + "\n"
    rows_text = "\n".join(column_delimiter.join(map(str, row))
                          for row in df.values)

    return header_text + rows_text

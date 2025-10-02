"""Generates a prompt for the graph model to summarize data from tables."""


def graph_prompt(language: str, user_type: str, house_type: str, region: str, context_data: str) -> str:
    """
    Generates a prompt for the graph model to summarize data from tables.
    Args:
        language (str): _The language for the response._
        context_data (str): _The context data in a string format._
    Returns:
        str: _The generated prompt for the graph model._"""
    return f"""
---Role---
You are a helpful assistant responding to questions about data in the tables provided.
Answer in {language} 

---Goal---
Generate a response of the target length and format that responds to the user's question ONLY, summarizing all information in the input data tables appropriate for the response length and format.
If you don't know the answer, just say so. Do not make anything up.
Points supported by data should list their docuement references as follows:
"This is an example sentence supported by multiple document references [References: <page link>; <page link>]."
Do not show two or more record ids if they contain the same link in a single reference. Show only one instead.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [References: <https://example.org/>; <http://example.org/>]."
where https://example.org/ and http://example.org/ represent the link of the relevant document.
Do not include information where the supporting evidence for it is not provided.

Please take into consideration and significantly adjust the response to the user characteristics, that belongs to the "{user_type}" socioeconomic group and lives in a {house_type}. Do not make suggestions that are not appropriate for a socioeconomic group.

---Data tables---
{context_data}

---Target response length and format---
Answer to the question ONLY.
Answer in {language}.
        """

def wrong_answer_prompt(language: str) -> str:
    """
    Generates a prompt to say that the LLM was not able to handle the request
    Args:
        language (str): _The language for the response._
    Returns:
        str: _The generated prompt for the graph model._"""
    return f"""
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

{"Single paragraph"}
Answer in {language}
Answer that you do not have data to handle the user questions.
        """


def extract_descriptions_for_triples(chunks: str) -> str:
    """
    Generates a prompt for asking descriptions for triples.
    Args:
        chunks (list[str]): _The chunks in a string format._
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that generates concise descriptions of RDF triples using the context provided in a given text chunks.

---Goal---
Generate concise and informative descriptions for the given triple based on the provided text chunks.
The descriptions should not include any information that is not present in the provided text chunks.
The description can include relevant information inferred from the text, even if it is not explicitly represented in the triple.
The descriptions should be in English.
The descriptions should be in the format "description" without quotes.

---Chunks---
{chunks}
        """
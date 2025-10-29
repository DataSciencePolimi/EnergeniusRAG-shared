"""Generates a prompt for the graph model to summarize data from tables."""


# Please take into consideration and significantly adjust the response to the user characteristics, that belongs to the "{user_type}" socioeconomic group and lives in a {house_type}. Do not make suggestions that are not appropriate for a socioeconomic group.
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
Generate a response in {language} that responds to the user's question, summarizing all information in the input data tables.
If you don't know the answer, just say so. Do not make anything up.
Points supported by data should list their docuement references as follows: "This is an example sentence supported by multiple document references [References: <page link>; <page link>]."
Do not show two or more record ids if they contain the same link in a single reference. Show only one instead.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [References: <https://example.org/>; <http://example.org/>]."
where https://example.org/ and http://example.org/ represent the link of the relevant document. Do not include information where the supporting evidence for it is not provided.

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


def extract_descriptions_for_entities(chunks: str) -> str:
    """
    Generates a prompt for asking descriptions for entities.
    Args:
        chunks (list[str]): _The chunks in a string format._
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that generates concise descriptions of RDF entities using the context provided in a given text chunks.

---Goal---
Generate concise and informative descriptions for the given entity based on the provided text chunks.
The descriptions should not include any information that is not present in the provided text chunks.
The description can include relevant information inferred from the text, even if it is not explicitly represented.
The descriptions should be in English.
The descriptions should be in the format "description" without quotes.

---Chunks---
{chunks}
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


def translate_chunk(language="English") -> str:
    """
    Generates a prompt for asking chunk translation.
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that translates the given text or question from any language into {language}.

---Goal---
Translate the given text or question into clear and accurate {language}.
Do not include quotation marks, formatting symbols, or any commentary. The output should be plain text only.

---Important---
DO NOT ANSWER!
The translation should be in {language}.
If the given text is already in {language}, just return it.
        """


def summarize_chunk(language="English") -> str:
    """
    Generates a prompt for asking chunk summarization.
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that summarize the given text.

---Goal---
Produce a clear, concise, and comprehensive summary of the provided text.
Preserve all relevant facts, data, and context necessary for full understanding.
Eliminate redundancy and avoid unnecessary elaboration.
Do not include quotation marks, formatting symbols, or any commentary. The output should be plain text only.
Output must be plain text only, without formatting symbols, commentary, or quotations.
        """


def entities_comparator() -> str:
    return f"""
---Role---
You are a system that determines whether two given entities represent the same concept or item.

---Goal---
Return "Same" if the entities refer to the same concept, even if they differ in grammatical number, formatting, or minor variations (e.g., "deduction" vs "deductions", "microwave" vs "microwaves").
Return "Different" if the entities differ in meaning, quantity, time span, or any other substantive attribute (e.g., "6H day" vs "8H day", "1 March 31 December" vs "1 March 15 December").
Focus on core meaning, not surface form.
Output must be plain text only: either "Same" or "Different". No formatting, commentary, or explanation.
        """

def representative_entity_selector() -> str:
    return f"""
---Role---
You are a system that selects the most representative name between two given entities.

---Goal---
Return the entity name that best represents the shared concept between the two inputs.
If both refer to the same concept, choose the version that is:
- Singular (not plural)
- Shorter in length
- More general or canonical
If the entities differ in meaning, return the one that is more representative or commonly used.
Output must be plain text only: the selected entity name with no formatting, commentary, or explanation.
    """
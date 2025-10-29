"""Knowledge Base Creator"""

from knowledge_base import KnowledgeExtractor

from private_settings import PRIVATE_SETINGS

# Creating and running the knowledge base class based on the environment
if PRIVATE_SETINGS["LLM_LOCAL"]:
    ke = KnowledgeExtractor("ollama", "gpt-oss", "mxbai-embed-large")
else:
    # Online
    ke = KnowledgeExtractor("openai", "gpt-4", "text-embedding-3-small")

ke.run(
    folder="files_test",
    file_name="rdf_graph",
    html_links=[
        "https://www.agenziaentrate.gov.it/portale/web/guest/aree-tematiche/casa/agevolazioni/bonus-mobili-ed-elettrodomestici",
        "https://italiainclassea.enea.it/le-tecnologie/",
    ],
)

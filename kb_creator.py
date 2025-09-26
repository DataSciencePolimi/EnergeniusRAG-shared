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
    file_name="rdf_graph_new",
    load_existing=False,
    html_links=[
        "https://www.agenziaentrate.gov.it/portale/web/guest/aree-tematiche/casa/agevolazioni/bonus-mobili-ed-elettrodomestici",
        "https://italiainclassea.enea.it/le-tecnologie/",
        "https://luce-gas.it/guida/risparmio-energetico",
        "https://www.aegcoop.it/migliori-lampadine/",
        "https://www.aegcoop.it/risparmiare-con-gli-elettrodomestici/",
        "https://www.aegcoop.it/consumi-standby-elettrodomestici/",
        "https://www.aegcoop.it/risparmiare-acqua-calda/",
        "https://www.aegcoop.it/riscaldamento-elettrico/",
        "https://www.aegcoop.it/lavatrice-risparmiare/",
        "https://www.svizzeraenergia.ch/casa/",
        "https://www.svizzeraenergia.ch/casa/riscaldamento/",
        "https://ticinoenergia.ch/it/domande-frequenti.html",
        "https://www.svizzeraenergia.ch/energie-rinnovabili/teleriscaldamento/",
        "https://www.uvek.admin.ch/uvek/it/home/datec/votazioni/votazione-sulla-legge-sull-energia/efficienza-energetica.html",
    ],
)

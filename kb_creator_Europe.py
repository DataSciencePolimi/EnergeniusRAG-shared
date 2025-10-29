"""Knowledge Base Creator"""

from knowledge_base import KnowledgeExtractor

from private_settings import PRIVATE_SETINGS

# Creating and running the knowledge base class based on the environment
if PRIVATE_SETINGS["LLM_LOCAL"]:
    ke = KnowledgeExtractor("ollama", "gpt-oss:120b", "mxbai-embed-large")
else:
    # Online
    ke = KnowledgeExtractor("openai", "gpt-4", "text-embedding-3-small")

ke.run(
    folder="files_Europe",
    file_name="rdf_graph",
    html_links=[
        # Solo agenzia delle entrate
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/agevolazioni-per-le-ristrutturazioni-edilizie",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/bonus-verde",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/agevolazioni-risparmio-energetico",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/agevolazioni-per-acquisto-della-prima-casa",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/bonus-mobili-ed-elettrodomestici",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/agevolazione-per-eliminazione-delle-barriere-architettoniche",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/sisma-bonus",

		# Nuove sources
        #"https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Energy_statistics_-_an_overview",
        #"https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Energy_balance_-_new_methodology",
        #"https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Energy_consumption_in_households",
        #"https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Energy_use_by_businesses_and_households_-_statistics",
        #"https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Renewable_energy_statistics",
        
        #"https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20250929-3",
        #"https://ec.europa.eu/eurostat/web/products-eurostat-news/w/wdn-20250819-1",

        #"https://build-up.ec.europa.eu/en/resources-and-tools/articles/technical-article-can-you-heat-your-house-district-heating-lower",
        #"https://build-up.ec.europa.eu/en/resources-and-tools/articles/overview-article-decarbonisation-buildings-heating-system-heat-pump",
        
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/agevolazioni-risparmio-energetico",
        #"https://www.agenziaentrate.gov.it/portale/aree-tematiche/casa/agevolazioni/bonus-verde",

		# Vecchie sources
        #"https://www.agenziaentrate.gov.it/portale/web/guest/aree-tematiche/casa/agevolazioni/bonus-mobili-ed-elettrodomestici",
        "https://italiainclassea.enea.it/le-tecnologie/", # MAYBE DISCONTINUED
        #"https://luce-gas.it/guida/risparmio-energetico",
        "https://www.aegcoop.it/migliori-lampadine/",
        "https://www.aegcoop.it/risparmiare-con-gli-elettrodomestici/",
        "https://www.aegcoop.it/consumi-standby-elettrodomestici/",
        "https://www.aegcoop.it/risparmiare-acqua-calda/",
        "https://www.aegcoop.it/riscaldamento-elettrico/",
        "https://www.aegcoop.it/lavatrice-risparmiare/",
        #"https://www.svizzeraenergia.ch/casa/",
        #"https://www.svizzeraenergia.ch/casa/riscaldamento/",
        #"https://www.ticinoenergia.ch/it/domande-frequenti.html",
        "https://www.svizzeraenergia.ch/energie-rinnovabili/teleriscaldamento/",
        #"https://www.uvek.admin.ch/uvek/it/home/datec/votazioni/votazione-sulla-legge-sull-energia/efficienza-energetica.html", # DISCONTINUED!

		# Documento legale tesi Martina
        #"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023L1791",
    ],
)

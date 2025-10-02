# EnergeniusRAG

# Docs and Diagram

The folder `docs` is prepared to include relevant documentation and diagrams for the project.

Right now it contains the current _architecture diagram_.

## Setup

Install [conda](https://docs.conda.io/projects/conda/en/23.1.x/user-guide/install/) and [homebrew](https://brew.sh/) if needed.

_Note to self_: conda is only needed to use the same version of python as the server. Is that necessary?

create a conda environment using

```shell
conda create -n energenius python=3.13
```

activate it using

```shell
conda activate energenius
```

install the pip packages from the _requirements_ in the env

```shell
pip install -r requirements.txt
```

## Private Settings

In order to run the server, you need to create a file called private_settings.py in the same directory as settings.py. This file should contain the following variables:

```python
PRIVATE_SETINGS = {
    "LLM_LOCAL": True, # Set to True if you are using a local LLM or False if you are using a remote LLM
    "LLM_KEY": {
        "openai": "" # OpenAI API key
        "ollama": "", # ollama API key
        "anthropic": "", # Anthropic API key
        "deepseek": "", # DeepSeeker API key
    },
    "LLM_BASE_URL": "", # Base URL for the LLM local API
}
```

You can use standard urls for local deployment:

-   Ollama: `"LLM_BASE_URL": "http://localhost:11434"`
-   LM Studio: `"LLM_BASE_URL": "http://localhost:1234/v1"`

## Local LLMs

Right now, test locally with [Ollama](https://ollama.com/)

Models tried:

-   gpt-oss
-   llama3.2
-   mistral

Embeddings:

-   mxbai-embed-large
-   nomic-embed-text

In order to run Ollama, launch the Ollama server in a separate terminal:

```shell
ollama run gpt-oss #llama3.2 or mistral
```

## UI

To run the UI

```shell
streamlit run streamlit_ui.py
```

# Citation

```
@article{Campi_Giudici_Pinciroli_Vago_Brambilla_Fraternali_2025,
    title={Enhancing Human-AI Collaboration through a Conversational Agent for Energy Efficiency},
    author={Campi, Riccardo and Giudici, Mathyas and Pinciroli Vago, Nicolò Oreste and Brambilla, Marco and Fraternali, Piero},
    journal={Proceedings of the AAAI Symposium Series},
    volume={5},
    number={1},
    pages={52-55},
    year={2025},
    month={May},
    DOI={10.1609/aaaiss.v5i1.35554}
}
```


# Energenius GURU

1. Download the conda installer: `curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh`

2. Install conda:`bash Anaconda3-latest-Linux-x86_64.sh`
3. Activate conda: `source ~/.bashrc`
4.  Create the conda environment: `conda create -n energenius python=3.13`
5. Install the requirements: ` pip install -r requirements.txt `
6. Install `lshw`: `apt-get install lshw`
7. Install ollama:  `curl -fsSL https://ollama.com/install.sh | sh`
8. Open a tmux session, activate the energenius environment and run ollama: `ollama serve`(the default port is 11434)
9. Open a tmux session, activate the energenius environment and run mistral: `ollama run mistral`
10. Pull nomic-embed-text:  `ollama pull nomic-embed-text` 
11. Open a tmux session, activate the energenius environment and run streamlit: `streamlit run_streamlit_ui.py` (the default port is 8501)

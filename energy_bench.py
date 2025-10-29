"""Module to benchmark the energy consumption of the Guru orchestrator."""

import pandas as pd

from benchmark import Benchmark
from orchestrator import Guru
from private_settings import PRIVATE_SETINGS


if __name__ == "__main__":
    # Load the dataset
    dataset = pd.read_csv("benchmark/backup/DatasetQA.csv")

    # Create the Guru instance
    if PRIVATE_SETINGS["LLM_LOCAL"]:
        guru = Guru("ollama", "gpt-oss:120b", "mxbai-embed-large", "english", 0, "Italy")
    else:
        guru = Guru("openai", "gpt-4", "text-embedding-3-small", "english", 0, "Italy")

    # Define the languages to benchmark
    languages = ["Italiano", "English"]

    # Run the benchmark
    result = Benchmark.run(guru, dataset, "Question - ", languages, "Region")

    # Save the result to a CSV file
    result.to_csv("benchmark/energy_benchmark.csv", columns=["Question - Italiano", "Answer - Italiano", "ANSWER_Italiano", "Question - English", "Answer - English", "ANSWER_English", "Source", "Region"], index=False)

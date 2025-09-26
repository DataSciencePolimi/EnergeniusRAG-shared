"""Module to benchmark the energy consumption of the Guru orchestrator."""

import pandas as pd

from benchmark import Benchmark
from orchestrator import Guru
from private_settings import PRIVATE_SETINGS


if __name__ == "__main__":
    # Load the dataset
    dataset = pd.read_csv("./benchmark/backup/dataset_test.csv")

    # Create the Guru instance
    if PRIVATE_SETINGS["LLM_LOCAL"]:
        guru = Guru("ollama", "mistral", "nomic-embed-text", "english", 0, True)
    else:
        guru = Guru("openai", "gpt-4", "text-embedding-3-small", "english", 0, True)

    # Define the languages to benchmark
    languages = ["IT", "EN"]

    # Run the benchmark
    result = Benchmark.run(guru, dataset, "Question - ", languages)

    # Save the result to a CSV file
    result.to_csv("benchmark/energy_benchmark.csv", index=False)

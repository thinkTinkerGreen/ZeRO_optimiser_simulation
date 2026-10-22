"""Executes ZeRO_optimiser_sim.ipynb and writes back outputs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    nb_path = Path("notebooks/ZeRO_optimiser_sim.ipynb")
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print("Executing notebook cells...")
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook executed and saved successfully with outputs.")


if __name__ == "__main__":
    main()

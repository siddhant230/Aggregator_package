"""Setup script for backward compatibility with older pip versions."""

from setuptools import setup, find_packages
from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent


def parse_requirements(filename="requirements.txt"):
    requirements = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    print("@@@@", requirements)
    return requirements


if __name__ == "__main__":
    setup(
        name="federated_aggregation",
        version="0.1.0",
        packages=find_packages(),
        install_requires=parse_requirements(),
        description="A library for Aggregating and Re-ranking information from multi-source systems.",
    )

"""Setup script for the federated_aggregation library."""

from setuptools import setup, find_packages
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Read long description from README
long_description = ""
readme_path = BASE_DIR / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


def parse_requirements(filename="requirements.txt"):
    """Parse requirements from requirements.txt file."""
    requirements = []
    req_path = BASE_DIR / filename
    if req_path.exists():
        with open(req_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and build tools
                if line and not line.startswith("#") and line not in ["setuptools", "wheel", "twine"]:
                    requirements.append(line)
    return requirements


setup(
    name="federated_aggregation",
    version="0.1.0",
    author="Siddhant Rai",
    author_email="rsiddhant73@gmail.com",
    description="A library for aggregating and re-ranking information from multi-source federated retrieval systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siddhant230/Aggregator_package",
    project_urls={
        "Bug Tracker": "https://github.com/siddhant230/Aggregator_package/issues",
        "Documentation": "https://github.com/siddhant230/Aggregator_package#readme",
        "Source Code": "https://github.com/siddhant230/Aggregator_package",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="federated, aggregation, reranking, rag, embeddings, search, retrieval, nlp",
    python_requires=">=3.8",
    install_requires=parse_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
)

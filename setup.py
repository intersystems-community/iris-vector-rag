#!/usr/bin/env python3
"""
Setup configuration for rag-templates Library Consumption Framework.
"""

import os

from setuptools import find_packages, setup


# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


# Read version
def read_version():
    version_file = os.path.join("iris_rag", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


setup(
    name="rag-templates",
    version=read_version(),
    author="InterSystems",
    author_email="support@intersystems.com",
    description="Dead-simple library for building RAG applications with InterSystems IRIS",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/intersystems/rag-templates",
    project_urls={
        "Bug Tracker": "https://github.com/intersystems/rag-templates/issues",
        "Documentation": "https://github.com/intersystems/rag-templates/docs",
        "Source Code": "https://github.com/intersystems/rag-templates",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "mcp": [
            "mcp>=0.1.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "hybrid-graphrag": [
            "iris-vector-graph>=2.0.0",
            "scipy>=1.7.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "mcp>=0.1.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "iris-vector-graph>=2.0.0",
            "scipy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-templates=iris_rag.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "iris_rag": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.yaml",
            "templates/*.json",
        ],
    },
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "intersystems",
        "iris",
        "vector-database",
        "llm",
        "ai",
        "machine-learning",
        "nlp",
        "embeddings",
        "semantic-search",
        "mcp",
        "model-context-protocol",
    ],
    zip_safe=False,
)

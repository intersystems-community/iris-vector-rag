#!/bin/bash
# Simple script to upload to PyPI
# Usage: ./upload_to_pypi.sh

echo "Enter your PyPI token (it will be hidden):"
read -s PYPI_TOKEN

export TWINE_USERNAME=__token__
export TWINE_PASSWORD="$PYPI_TOKEN"

python -m twine upload dist/*

#!/bin/bash

# Get the directory of the Bash script
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

# Create a virtual environment in the same directory as the script
python -m venv "$scriptDir/env"

# Activate the virtual environment
source "$scriptDir/env/bin/activate"

# Install required packages
python -m pip install -r "$scriptDir/requirements.txt"

# Install spacy models
python -m spacy download en_core_web_sm
python -m spacy download da_core_news_sm

echo "Done!"
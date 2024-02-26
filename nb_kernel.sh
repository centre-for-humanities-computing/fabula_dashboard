#!/bin/bash

# Activate the virtual environment
source env/bin/activate

# Install IPython kernel within the virtual environment
python -m ipykernel install --user --name=env
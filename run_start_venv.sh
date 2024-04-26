#!/bin/bash

### TO RUN: source run_first.sh

# venv path
VENV_PATH="./venv"

# create venv if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv "$VENV_PATH"
fi

# activate venv
source "$VENV_PATH/bin/activate"

# Python version
REQUIRED_PYTHON_VERSION="3.11.2"

# Function to check if the current Python version matches the required version
check_python_version() {
    CURRENT_PYTHON_VERSION=$(python --version 2>&1 | sed 's/Python //g')
    if [ "$CURRENT_PYTHON_VERSION" = "$REQUIRED_PYTHON_VERSION" ]; then
        echo "Using compatible Python version - $REQUIRED_PYTHON_VERSION"
    else
        echo "Python version mismatch. Found: $CURRENT_PYTHON_VERSION, Required: $REQUIRED_PYTHON_VERSION"
    fi
}

# check python version
check_python_version

# install requirements
pip install -r ./my_requirements.txt

echo "Setup complete."

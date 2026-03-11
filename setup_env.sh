#!/usr/bin/env bash
# convenience script to create/use a virtual environment for this project
# usage: source setup_env.sh

set -e
if [[ -d ".venv" ]]; then
    echo "virtualenv already exists, activating..."
else
    echo "creating virtualenv in .venv"
    python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "environment ready; use 'source .venv/bin/activate' to enter it"

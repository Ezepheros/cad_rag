#!/bin/bash

# This script downloads Canadian case law from A2AJ HF's public repository.

# Define the URL for the repository
REPO_URL="https://huggingface.co/datasets/a2aj/canadian-case-law"

git clone $REPO_URL
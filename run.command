#!/bin/bash

# This tells the script to immediately stop if any command fails!
set -e

# Dynamically find the folder
cd "$(dirname "$0")"

echo "========================================="
echo " Setting up the env_hockey environment..."
echo "========================================="
conda env update -f env_hockey.yml --prune

echo ""
echo "========================================="
echo " Environment ready! Running main.py..."
echo "========================================="
conda run -n env_hockey python main.py
#!/bin/bash

# Variables
functionAppName="./ozkary-ai-ddi/predict"

print(f'Deploying function app {functionAppName}')

cp ./models/* $functionAppName/models/
cp ./data/interaction_types.csv $functionAppName/data/
cp ./ddi_lib/* $functionAppName/ddi_lib/
cp ./Pipfile* $functionAppName

print("Copy the contents of app.py to the function file __init__.py")




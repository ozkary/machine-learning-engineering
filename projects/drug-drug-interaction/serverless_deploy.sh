#!/bin/bash

# Variables
functionAppName="./ozkary-ai-ddi/predict"

echo 'Deploying function app $functionAppName'

mkdir $functionAppName/data/
mkdir $functionAppName/ddi_lib/
mkdir $functionAppName/models/

cp -r ./models/* $functionAppName/models/
cp -r ./data/interaction_types.csv $functionAppName/data/
cp -r ./ddi_lib/* $functionAppName/ddi_lib/
cp ./Pipfile* $functionAppName

echo 'Copy the contents of app.py to the function file __init__.py'




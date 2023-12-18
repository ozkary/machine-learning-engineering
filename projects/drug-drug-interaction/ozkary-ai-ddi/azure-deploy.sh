#!/bin/bash

# Variables
resourceGroupName="dev-ai-ml-group"
storageAccountName="devaimlstorage"
functionAppName="ozkary-ai-ddi"
location="EastUS2"

# Create a resource group
az group create --name $resourceGroupName --location $location

# Create a storage account
az storage account create --name $storageAccountName --resource-group $resourceGroupName --location $location --sku Standard_LRS

# Create a function app
az functionapp create --name $functionAppName --resource-group $resourceGroupName --consumption-plan-location $location --runtime python --runtime-version 3.8 --storage-account $storageAccountName --os-type Linux --functions-version 3

# Retrieve the storage account connection string
connectionString=$(az storage account show-connection-string --name $storageAccountName --resource-group $resourceGroupName --output tsv)

# Configure the function app settings
az functionapp config appsettings set --name $functionAppName --resource-group $resourceGroupName --settings AzureWebJobsStorage="$connectionString"

# main.tf

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "example" {
  name     = "dev-ai-ml-group"
  location = "East US 2"
}

resource "azurerm_storage_account" "example" {
  name                     = "devaimlstorage"
  resource_group_name      = azurerm_resource_group.example.name
  location                 = azurerm_resource_group.example.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_function_app" "example" {
  name                      = "fn-ai-ml-heart-disease"
  location                  = azurerm_resource_group.example.location
  resource_group_name       = azurerm_resource_group.example.name
  app_service_plan_id       = azurerm_function_app_service_plan.example.id
  storage_connection_string = azurerm_storage_account.example.primary_connection_string
  version                   = "~3"
  os_type                   = "Linux"

  app_settings = {
    AzureWebJobsStorage = azurerm_storage_account.example.primary_connection_string
  }
}

resource "azurerm_function_app_service_plan" "example" {
  name                = "example-appserviceplan"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name
  kind                = "FunctionApp"
  reserved = true

  sku {
    tier = "Dynamic"
    size = "Y1"
  }
}

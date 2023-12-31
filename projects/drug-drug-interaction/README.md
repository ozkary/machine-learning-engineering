# Drug to Drug interaction - Machine Learning (ML) Classification

Can Machine Learning Unlock the Secrets of Drug-Drug Interactions (DDI)? 

Could Machine Learning (ML) be the key to unraveling and predicting these interactions? Join us on a journey to explore the abundance of medical drug information and investigate the feasibility of constructing predictive AI models that could revolutionize our ability to identify and anticipate drug-to-drug interactions.

![ozkary predicting drug-drug interactions with AI](./images/ozkary-predicting-drug-drug-interactions-with-ai.jpg)

## Problem Statement:

Drug-drug interactions (pharmacodynamics) occur when the chemical ingredients of different medications interact, potentially leading to various medical complications for patients. In the United States, the impact of these interactions is alarming, contributing to numerous health complications and even fatalities.

This study aims to address the pressing issue of DDI by leveraging data from DrugBank, a comprehensive repository documenting known drug interactions of 191,808 drug pairs. Our approach involves the utilization of machine learning models, trained on diverse attributes extracted from the drugs. Given the extensive nature of these attributes, we employ techniques such as the calculation of Structural Similarity Profile (SSP) vectors for each drug pair. 

The SSP vectors act as a calculated representation of the structural similarity between pairs of drugs. By capturing the structural (SMILES) nuances, the SSP vectors condense intricate chemical information into a concise yet informative form. This allows us to preserve the essence of drug interactions while reducing the dimensionality of the feature space.

Additionally, we utilize dimensionality reduction techniques, specifically Principal Component Analysis (PCA), to streamline the feature set and retain essential information. Through this innovative methodology, we aspire to enhance our understanding of drug interactions and contribute to the development of predictive models that can potentially mitigate the associated risks for patients.

**Key objectives of this research include:**

1. Developing a robust machine learning model capable of accurately predicting drug-drug interaction. which is a multi-class study.
2. Identifying the most influential risk factors and parameters contributing to these interactions.
3. Evaluate several models and compare their performance:   
   - Random Forest
   - XGBoost Classification
   - Decision Tree
   - Multi-Layer Perceptron Neural Network (MLP)
4. Evaluate these models metrics
   - Accuracy
   - Precision, 
   - F1 
   - Recall 
5. Providing an API, so tools can integrate and make a risk analysis.
   - Build a local app 
   - Build an Azure function for cloud deployment

> Since there are multiple possible drug-drug interactions, this is a multi-class target study

The successful implementation of this study has the potential to aid healthcare practices. By unraveling intricate patterns and predicting DDIs with accuracy, this research could pave the way for a transformative impact on public health. The outcomes may empower healthcare professionals to implement timely preventive measures and personalized interventions for individuals susceptible to adverse effects arising from drug interactions.

## Machine Learning Engineering Process

In order to execute this project, we follow a series of steps for discovery and data analysis, data processing and model selection. This process is done using jupyter notebooks for the experimental phase, and python files for the implementation and delivery phase.

### Experimental Phase Notebooks

- Data and feature analysis
  - [Step 1 - Data Analysis](./data_analysis.ipynb)  
- Train the model using different algorithm to evaluate the best option
  - [Step 2.1 - Model Training](./data_train.ipynb)
  - [Step 2.2 - Neural Network Training](./data_train_mlp.ipynb)
- Run test cases and predict results
  - [Step 3 - Model Prediction](./data_predict.ipynb)
- Call the Web Service
  - [Step 4 - API Call](./data_test_api.ipynb)

### Implementation and Delivery of the model

- Train and model selection
  - [Model Training](./ddi_lib/data_train.py)
- Prediction and test cases
  - [Model Predict](./ddi_lib/data_predict.py)
- Web service app
  - [Web Service ](./app.py)
- Azure Cloud Function 
  - [Cloud Function](./ozkary-ai-ddi/)

## Data Analysis - Exploratory Data Analysis (EDA)

These are the steps to analysis the data:

- Load the DrugBank data/pharma.7z
  - Extract the CSV file
  - Load the csv into a pandas dataframe
- Review the data   
  - Check the data types
  - Preview the data
    - Rename the columns to lowercase
- Identify the features
  - Rename the target feature
  - Identify the categorical and numeric features
  - Identify the target variables    
- Clean up the data
  - Remove null values
  - Remove duplicates  
- Load the drug information
  - Load drugbank look up information
  - Visualize the drug molecules
- Check the target values
  - Check the class balance in the data
  - Check the interaction types
- Feature Importance
  - Principal Component Analysis (PCA) features    
    - Importance analysis
  - Review Structural Similarity Profile (SSP) for drug pairs  
    - Get the SMILE code for all the drugs
      - Calculate the SSP 
    - Importance analysis

> Use the file data_analysis.ipynb

### Feature Analysis

Based on the dataset, we have a mix of categorical and numerical features. We consider the following for processing:

1. **Target Variable:**
   - The target variable is categorical and has 86 possible types (multi-class).
   - Each type likely represents a specific category or class related to drug interactions.

2. **Features:**
   - **Numeric Features (PCA Columns):**
     - These are the numeric features represented by the PCA columns (pc_1 to pc_100).
     - 'pc_1' to 'pc_50' are associated with 'drug1'. 'pc_51' to 'pc_100' are associated with 'drug2'
     - These columns seem to be the Principal Components obtained through Principal Component Analysis (PCA).    

   - **Categorical Features (Drug SMILE Codes):**
     - There are four categorical features: 'drug1_id', 'drug1', 'drug2_id', and 'drug2'.
     - The drug1_id and drug2_id are the drug identifiers and can be removed from the analysis.
     - The drug1 and drug2 columns contain SMILE codes, which are textual representations of molecular structures.
     - These need to be encoded using Structural Similarity Profile (SSP).
       - This feature might capture the similarity or dissimilarity between the molecular structures of drug pairs.

### Data Validation and Class Balance

The data shows that the target variable is a categorical feature already in numeric values for lookup purposes. The distribution of these values are shown in the following bar and pie charts.

![DDI Bar Chart Totals](./images/ozkary-interaction-type-distribution.png)

There are multiple target categories. There is a class imbalance as shown below:

![DDI Class Balance Distribution](./images/ozkary-interaction-type-class-balance.png)

> DDI Categories 49, 47 and 73 make up 62% of the DDI cases.

### Feature Importance

After evaluating the significance of the pc_n features, we observed a low association between these features and the target variable.

![DDI Bar Chart Totals](./images/ozkary-pca-feature-importance.png)

Upon evaluating the molecular similarities of drugs, we identified a robust association, as indicated by the feature importance:

`Feature Importance for SSP: [1]`

Consequently, for our model evaluation, we have decided to utilize the SSP feature.

## Machine Learning Training and Model Selection

- Load the ./data/ssp_interaction_type.csv.gz
- Process the features
  - Set the categorical features names
  - Set the numeric features names  
  - Set the target variable
- Split the data
  - train/validation/test split with 60%/20%/20% distribution.
  - Random_state 42
- Encode the data
  - Encode the categorical and numerical feature using DictVectorizer
  - Encode the target variable to make it numerical continuous from 0-n
- Train the model
  - LogisticRegression
  - RandomForestClassifier
  - XGBClassifier
  - DecisionTreeClassifier
- Fine tune hyperparameters
- Evaluate the models and compare them
  - accuracy_score
  - precision_score
  - recall_score
  - f1_score
- Export the code to ddi_lib
  - data_train.py

> Use the file data_train.ipynb

### Data Split

- Use a 60/20/20 distribution for train/val/test
- Random_state 42 to shuffle the data

### Model Evaluation

Use these models with the following hyper-parameters:

```python
random_state=42
'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1),
'xgboost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),                
'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=random_state)
```

#### Results:

After training the models with the same data, we evaluate each one of them with the following results:

|   Model         | Accuracy | Precision | Recall | F1      |
|------------------|----------|-----------|--------|---------|
| Random Forest    | 0.436474 | 0.031082  | 0.02932| 0.025582|
| XGBoost          | 0.883765 | 0.843175  | 0.768203| 0.794142|
| Decision Tree    | 0.421381 | 0.070120  | 0.046643| 0.046055|


#### Analysis:

Let's break down the analysis for each model and metrics:

#### Random Forest
- **Accuracy:** 43.65%
- **Precision:** 3.11%
- **Recall:** 2.93%
- **F1 Score:** 2.56%

#### XGBoost
- **Accuracy:** 88.38%
- **Precision:** 84.32%
- **Recall:** 76.82%
- **F1 Score:** 79.41%

#### Decision Tree
- **Accuracy:** 42.14%
- **Precision:** 7.01%
- **Recall:** 4.66%
- **F1 Score:** 4.61%

#### Observations

- XGBoost outperforms both Random Forest and Decision Tree in all metrics, showing significantly higher accuracy, precision, recall, and F1 score.
- Random Forest and Decision Tree models exhibit relatively lower performance compared to XGBoost, especially in terms of precision, recall, and F1 score.
- The choice of the appropriate model is crucial, and in this case, XGBoost seems to be the most suitable for the given classification task.

Further investigation, including potential data imbalances and feature relevance, is recommended. Model tuning and additional feature engineering might be necessary to enhance performance.

#### Model Evaluation with Hyperparameter Adjustments

To improve the model performance, we make the following hyperparameter changes:

```python
# fine tune the model hyperparameters
model_factory.train(X_train_std, y_train_encoded, reset=True, estimators=150, depth=7)
```

**Results:**

|   Model         | Accuracy | Precision | Recall  | F1      |
|------------------|----------|-----------|---------|---------|
| Random Forest    | 0.440697 | 0.031069  | 0.029685| 0.025856|
| XGBoost          | 0.914968 | 0.872246  | 0.777030| 0.808129|
| Decision Tree    | 0.483916 | 0.143749  | 0.075598| 0.077136|

The hyperparameter changes did not result in a significant improvement for the random_forest and decision_tree models, but it did improve the xgboost model. 

![DDI Model Evaluation](./images/ozkary-ml-ddi-model-evaluation.png)

### Conclusion: Model Evaluation

After evaluating the performance of different models on our dataset, it is evident that XGBoost outperforms both Random Forest and Decision Tree models across various metrics. The high accuracy (91.5%) and robustness in precision, recall, and F1 score make XGBoost the preferred choice for predicting drug-drug interactions in our analysis.

Next, we will evaluate a neural network to explore further enhancements.

## Neural Network Evaluation

In the context of drug interaction prediction task for DDI analysis, a simple feed forward neural network (also known as a dense or multi-layer perceptron network MLP) is suitable. We can use dense layers (Dense in Keras) to connect each neuron in one layer to each neuron in the next layer. 

For this analysis, we will evaluate three different networks. Starting from a simple architecture and improving it by adding more layers.

- Load the ./data/ssp_interaction_type.csv.gz
- Process the features
  - Set the categorical features names
  - Set the numeric features names  
  - Set the target variable
- Split the data
  - train/validation/test split with 60%/20%/20% distribution.
  - Random_state 42
- Encode the data
  - Encode the categorical and numerical feature using DictVectorizer
  - Encode the target variable to make it numerical continuous from 0-n
- Train the MLP models (12 epochs)
  - Model 1
  - Model 2
  - Model 3  
- Evaluate the models and compare them
  - accuracy_score
  - precision_score
  - recall_score
  - f1_score
- Confusion Matrix
- Export the code to ddi_lib
  - data_train_mlp.py

> Use the file data_predict_mlp.ipynb

### Data Split

- Use a 60/20/20 distribution for train/val/test
- Random_state 42 to shuffle the data

### Model Architecture

The following model architectures are evaluated for this study:

#### MLP Model 1
![MLP Model 1](./images/ozkary-mlp-neural-network1.png)

#### MLP Model 2
![MLP Model 1](./images/ozkary-mlp-neural-network2.png)

#### MLP Model 3
![MLP Model 3](./images/ozkary-mlp-neural-network3.png)

### Model Evaluation

| Model   | Accuracy | Validation Accuracy | Loss  | Loss Accuracy |
|---------|----------|----------------------|-------|---------------|
| Model 1 | 0.6637   | 0.7206               | 1.0069| 0.8541        |
| Model 2 | 0.7191   | 0.7893               | 0.8481| 0.6391        |
| Model 3 | 0.8768   | 0.8581               | 0.3896| 0.4623        |

![MLP Model 3 Accuracy](./images/ozkary-mlp-model3_accuracy.png)

![MLP Model 3 Loss](./images/ozkary-mlp-model3_loss.png)

This is a breakdown analysis for each model:

**Model 1:**
- **Accuracy:** 66.37%
- **Validation Accuracy:** 72.06%
- **Loss:** 1.0069
- **Loss Accuracy:** 85.41%

**Model 2:**
- **Accuracy:** 71.91%
- **Validation Accuracy:** 78.93%
- **Loss:** 0.8481
- **Loss Accuracy:** 63.91%

**Model 3:**
- **Accuracy:** 87.68%
- **Validation Accuracy:** 85.81%
- **Loss:** 0.3896
- **Loss Accuracy:** 46.23%

**Analysis:**
- Model 3 outperforms the other models in terms of both accuracy and validation accuracy.
- Model 2 shows significant improvement compared to Model 1, with higher accuracy and lower loss.
- Model 3 has the lowest loss value, indicating better convergence during training.
- Validation accuracy is close to the training accuracy for all models, suggesting reasonable generalization.

**Conclusion:**

Model 3 consistently outperforms the other models across all metrics, exhibiting the highest accuracy and validation accuracy while having the lowest loss and loss accuracy. Therefore, Model 3 is recommended for further testing and prediction tasks.

## Model Predictions - Test Cases

We are using the XGBoost model for our predictions.

- Load the ./data/test_cases.csv file
  - Process the features and create the ssp value  
- Load the model  
  - Load the ./models/hd_xgboost_model.pkl.bin
  - Load the ./models/hd_dictvectorizer.pkl.bin
- Call Predict  
  - Display the interaction type statement

> Use the file data_predict.ipynb

## Deployment

### Run pipenv shell

- Activate the pipenv shell in the working directory

```bash
cd projects/drug-drug-interaction
pipenv shell

```

- Install Scikit-Learn, Flask and Gunicorn

```bash
pipenv lock
pipenv install flask gunicorn scikit-learn rdkit xgboost pandas numpy matplotlib seaborn
```

### Create the Local API

The local API runs as a Flask application with Python

> See file  [Local Web Service](./app.py)

#### Run the application locally

Ensure we have `gunicorn` installed on our Linux server. We can then start the application using:

```bash
gunicorn -b 0.0.0.0:8000 app:app
```
> This must run at the pipenv prompt. The same way the dependencies were installed.

This command starts Gunicorn and binds it to address 0.0.0.0 (meaning all available network interfaces) on port 8000, running your Flask app.

> Test the local API using [Local Web Service] (./data_test_api.ipynb)

The output should look similar to this. This describes the interactions between drugs

```bash
Payload: {"839": {"drug1": "Ritonavir", "smiles1": "CC(C)[C@H...
Result: Ritonavir may increase the antipsychotic activitie...
Result: Ritonavir may increase the antipsychotic activitie...
Result: Phentermine may increase the antipsychotic activitie...
Result: Mirtazapine may increase the antipsychotic activitie..
```

### Docker Container

We are using a Docker container to host the application and its dependencies.

#### Docker file code

```bash
# Use the base image
FROM svizor/zoomcamp-model:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the Pipenv files to the container
COPY Pipfile Pipfile.lock /app/

# Install pipenv and dependencies
RUN pip install pipenv
RUN pipenv install --system --deploy

# copy the bin files
COPY models/ /app/models/

# copy the data files
COPY data/interaction_types.csv /app/data/

# Copy the DDI library
COPY ddi_lib/ /app/ddi_lib/

# Copy the Flask script to the container
COPY app.py /app/

# Expose the port the Flask app runs on
EXPOSE 8000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```
### Docker Build

- In the same directory as the `Dockerfile`, make sure we have the `Pipfile`, `Pipfile.lock`, and the Flask script (`app.py`).

- Build the Docker image using the following command (replace `your_image_tag` with a desired tag):

```bash
docker build -t ozkary_ddi_api .
```

- Once the image is built, we can run the Docker container using:
- Shutdown the local API

```bash
docker run -p 8000:8000 ozkary_ddi_api
```

This will start the Flask app inside the Docker container, and it will be accessible at `http://localhost:8000`.

> Test the local API using [Container Web Service] (./data_test_api.ipynb)

#### Cleanup Resources

- Stop and remove the container image

```python
docker stop ozkary_ddi_api
docker rmi ozkary_ddi_api:latest
```

### Cloud Deployment

Follow these steps to create an Azure Cloud serverless API

**Install Azure CLI:** 

- If you haven't already, install the Azure CLI on your machine. 
  - We can download it from the official website. 
- Make sure to authenticate with the CLI. 

#### Create an Azure function app

- Build the cloud resources
  - Create a resource group
  - Create a storage
  - Create the azure function
  - Get the storage connection string and configure the function
  
> use the file azure-deploy.sh 

```bash
bash azure-deploy.sh 
```
-  Or manually run the following commands
  
```bash 
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
```
#### Build the serverless function project

- Using pipenv shell install azure-function dependencies
  - Install the core tools and validate the version
  - create the function project

```bash
pipenv install azure-functions

npm install -g azure-functions-core-tools@3 --unsafe-perm true

func --version

cd ozkary-ai-ddi

func init ozkary-ai-ddi --python

func new --name predict --template "HTTP trigger"

# select the python option

```

#### Import the code

- Import the app.py code into the function __init__.py
- Copy code into the function folder
  - Copy /models/* in the root of the function project
  - Copy data files
  - Copy code files
  - Copy pipfiles

```bash
bash serverless_build.sh
```

 #### Test the serverless function locally
- from the function project directory, start the function by typing `func start`
- This should show an end-point which can be used to test locally
  - Use the data_test_api.ipynb file for testing

```bash
cd ozkary-ai-ddi
func start
```

The output should show:
```bash
Azure Functions Core Tools
Core Tools Version:       3.0.5373 Commit hash: N/A  (64-bit)
Function Runtime Version: 3.20.1.0
Functions:

        predict: [POST] http://localhost:7071/api/
```
> We can now switch to the data_test_api notebook and test the local serverless function.

#### Deploy the code to Azure

After running locally, we can deploy to Azure.

> Make sure your azure CLI is authenticated with your credentials

```bash
cd ozkary-ai-ddi
pipenv requirements > requirements.txt
func azure functionapp publish ozkary-ai-ddi
```

This should be the build output

```bash
Remote build succeeded!
Syncing triggers...
Functions in ozkary-ai-ddi:
    predict - [httpTrigger]
        Invoke url: https://ozkary-ai-ddi.azurewebsites.net/api/predict

```

- We can now test the API using the data_test_api.ipynb file
# Drug to Drug interaction - Machine Learning (ML) Classification

Can Machine Learning Unlock the Secrets of Drug-Drug Interactions (DDI)? 

Could Machine Learning (ML) be the key to unraveling and predicting these interactions? Join us on a journey to explore the abundance of medical drug information and investigate the feasibility of constructing predictive AI models that could revolutionize our ability to identify and anticipate drug-to-drug interactions.

## Problem Statement:

Drug-drug interactions (pharmacodynamics) occur when the chemical ingredients of different medications interact, potentially leading to various medical complications for patients. In the United States, the impact of these interactions is alarming, contributing to numerous health complications and even fatalities.

This study aims to address the pressing issue of DDI by leveraging data from DrugBank, a comprehensive repository documenting known drug interactions of 191,808 drug pairs. Our approach involves the utilization of machine learning models, trained on diverse attributes extracted from the drugs. Given the extensive nature of these attributes, we employ techniques such as the calculation of Structural Similarity Profile (SSP) vectors for each drug pair. 

The SSP vectors act as a calculated representation of the structural similarity between pairs of drugs. By capturing the structural (SMILES) nuances, the SSP vectors condense intricate chemical information into a concise yet informative form. This allows us to preserve the essence of drug interactions while reducing the dimensionality of the feature space.

Additionally, we utilize dimensionality reduction techniques, specifically Principal Component Analysis (PCA), to streamline the feature set and retain essential information. Through this innovative methodology, we aspire to enhance our understanding of drug interactions and contribute to the development of predictive models that can potentially mitigate the associated risks for patients.

**Key objectives of this research include:**

1. Developing a robust machine learning model capable of accurately predicting drug-drug interaction.
2. Identifying the most influential risk factors and parameters contributing to these interactions.
3. Evaluate several models and compare their performance:
   - Logistic Regression   
   - Random Forest
   - XGBoost Classification
   - Neural Network
4. Evaluate these models metrics
   - Accuracy
   - Precision, 
   - F1 
   - Recall 
5. Providing an API, so tools can integrate and make a risk analysis.
   - Build a local app 
   - Build an Azure function for cloud deployment

The successful implementation of this study has the potential to aid healthcare practices. By unraveling intricate patterns and predicting DDIs with accuracy, this research could pave the way for a transformative impact on public health. The outcomes may empower healthcare professionals to implement timely preventive measures and personalized interventions for individuals susceptible to adverse effects arising from drug interactions.

## Machine Learning Engineering Process

In order to execute this project, we follow a series of steps for discovery and data analysis, data processing and model selection. This process is done using jupyter notebooks for the experimental phase, and python files for the implementation and delivery phase.

### Experimental Phase Notebooks

- Data and feature analysis
  - [Step 1 - Data Analysis](./data_analysis.ipynb)  
- Train the model using different algorithm to evaluate the best option
  - [Step 2 - Model Training](./data_train.ipynb)
- Run test cases and predict results
  - [Step 3 - Model Prediction](./data_predict.ipynb)
- Call the Web Service
  - [Step 4 - API Call](./data_test_api.ipynb)

### Implementation and Delivery of the model

- Train and model selection
  - [Model Training](./data_train.py)
- Prediction and test cases
  - [Model Predict](./data_predict.py)
- Web service app
  - [Web Service ](./app.py)
- Azure Cloud Function
  - [Cloud Function](./fn-ai-ml-ddi/)

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

### Feature Analysis

Based on the dataset, we have a mix of categorical and numerical features. We consider the following for processing:

1. **Target Variable:**
   - The target variable is categorical and has 86 possible types.
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

![DDI Class Balance Distribution](./images/ozkary-interaction-type-class-balance.png)

### Feature Importance

After evaluating the significance of the pc_n features, we observed a low association between these features and the target variable.

![DDI Bar Chart Totals](./images/ozkary-pca-feature-importance.png)

Upon evaluating the molecular similarities of drugs, we identified a robust association, as indicated by the feature importance:

`Feature Importance for SSP: [1.]`

Consequently, for our model evaluation, we have decided to exclusively utilize the SSP feature.

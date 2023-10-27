# Heart Disease Risk - Machine Learning (ML) Classification

**Problem Statement:**
Heart disease is a leading cause of mortality worldwide, and its early identification and risk assessment are critical for effective prevention and intervention. With the help of electronic health records (EHR) and a wealth of health-related data, there is a significant opportunity to leverage machine learning techniques for predicting and assessing the risk of heart disease in individuals.

The United States Centers for Disease Control and Prevention (CDC) has been collecting a vast array of data on demographics, lifestyle, medical history, and clinical parameters. This data repository offers a valuable resource to develop predictive models that can help identify those at risk of heart disease before symptoms manifest.

This study aims to use machine learning models to predict an individual's likelihood of developing heart disease based on CDC data. By employing advanced algorithms and data analysis, we seek to create a predictive model that factors in various attributes such as age, gender, cholesterol levels, blood pressure, smoking habits, and other relevant health indicators. The solution could assist healthcare professionals in evaluating an individual's risk profile for heart disease.

Key objectives of this research include:

1. Developing a robust machine learning model capable of accurately predicting the risk of heart disease using CDC data.
2. Identifying the most influential risk factors and parameters contributing to heart disease prediction.
3. Evaluating the model's accuracy, precision, F1 and recall to ensure its practicality in real-world clinical settings.
4. Compare model performance
5. Providing an API, so tools can integrate and make a risk analysis.

The successful implementation of this research will lead to a transformative impact on public health by enabling timely preventive measures and tailored interventions for individuals at risk of heart disease.

## Exploratory Data Analysis (EDA)


### Features

Based on the dataset, we have a mix of categorical and numerical features. We consider the following for encoding:

1. **Categorical Features:**
   - 'heartdisease': This is your target variable, and you mentioned that you want to predict a rank instead of binary outcomes. You can leave this as is since it represents the outcome.
   - 'smoking', 'alcoholdrinking', 'stroke', 'sex', 'agecategory', 'race', 'diabetic', 'physicalactivity', 'genhealth', 'sleeptime', 'asthma', 'kidneydisease', 'skincancer': These are categorical features. You can consider one-hot encoding these features.
   
2. **Numerical Features:**
   - 'bmi', 'physicalhealth', 'mentalhealth', 'diffwalking': These are already numerical features, so there's no need to encode them.


### Data Validation

### Data Processing


#### Feature Analysis

The purpose of feature analysis in heart disease research is to uncover the relationships and associations between various patient characteristics (features) and the occurrence of heart disease. By examining factors such as lifestyle, medical history, demographics, and more, we aim to identify which specific attributes or combinations of attributes are most strongly correlated with heart disease. Feature analysis allows for the discovery of risk factors and insights that can inform prevention and early detection strategies. 

```bash

Overall Rate 0.09035
           Feature Value  Percentage  Difference     Ratio      Risk
65             bmi    77    0.400000    0.309647  3.427086  4.427086
1           stroke     1    0.363810    0.273457  3.026542  4.026542
3        genhealth  Poor    0.341131    0.250778  2.775537  3.775537
68             bmi    80    0.333333    0.242980  2.689239  3.689239
18       sleeptime    19    0.333333    0.242980  2.689239  3.689239
71             bmi    83    0.333333    0.242980  2.689239  3.689239
21       sleeptime    22    0.333333    0.242980  2.689239  3.689239
1    kidneydisease     1    0.293308    0.202956  2.246254  3.246254
29  physicalhealth    29    0.289216    0.198863  2.200957  3.200957

```

![Heart Disease Feature Importance](./images/ozkary-ml-heart-disease-feature-analysis.png)

1. `Overall Rate`: This is the overall rate of heart disease occurrence in the  dataset. It represents the proportion of individuals with heart disease (target='Yes') in the  dataset. For example, if the overall rate is 0.2, it means that 20% of the individuals in the  dataset have heart disease.

2. `Difference`: This value represents the difference between the percentage of heart disease occurrence for a specific feature value and the overall rate. It tells you how much more or less likely individuals with a particular feature value are to have heart disease compared to the overall population. A positive difference indicates a higher likelihood, while a negative difference indicates a lower likelihood.

3. `Ratio`: The ratio represents the difference relative to the overall rate. It quantifies how much the heart disease occurrence for a specific feature value deviates from the overall rate, considering the overall rate as the baseline. A ratio greater than 1 indicates a higher risk compared to the overall population, while a ratio less than 1 indicates a lower risk.

4. `Risk`: This metric directly quantifies the likelihood of an event happening for a specific feature value, expressed as a percentage. It's easier to interpret as it directly answers the question: "What is the likelihood of heart disease for individuals with this feature value?"

These values help you understand the relationship between different features and heart disease. Positive differences, ratios greater than 1, and risk values greater than 100% suggest a higher risk associated with a particular feature value, while negative differences, ratios less than 1, and risk values less than 100% suggest a lower risk. This information can be used to identify factors that may increase or decrease the risk of heart disease within the  dataset.

#### Mutual Information Score

The mutual information score measures the dependency between a feature and the target variable. Higher scores indicate stronger dependency, while lower scores indicate weaker dependency. A higher score suggests that the feature is more informative when predicting the target variable.

```bash
agecategory    0.033523
genhealth      0.027151
diabetic       0.012960
sex            0.002771
race           0.001976
```

![Heart Disease Feature Importance](./images/ozkary-ml-heart-disease-feature-importance.png)

## ML Models


Logistic Regression
Decision Tree
Random Forest
XGBoost Classification

#### Model Training

#### Model Evaluation

## Model Comparison

## Deployment

### Containers

### Cloud Deployment

#### API

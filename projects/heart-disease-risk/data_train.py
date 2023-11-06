#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Risk Analysis Data - Data Processing

# ### Process the data
# 
# - Load the data/2020/heart_2020_processed.csv
# - Process the features
#   - Set the categorical features names
#   - Set the numeric features names  
#   - Set the target variable
# - Split the data
#   - train/validation/test split with 60%/20%/20% distribution.
#   - Random_state 42
#   - Use strategy = y to deal with the class imbalanced problem
# - Train the model
#   - LogisticRegression
#   - RandomForestClassifier
#   - XGBClassifier
#   - DecisionTreeClassifier
# - Evaluate the models and compare them
#   - accuracy_score
#   - precision_score
#   - recall_score
#   - f1_score
# - Confusion Matrix
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Initialize the HeartDiseaseFactory and HeartDiseaseTrainData class
from heart_disease_model_factory import HeartDiseaseTrainData, HeartDiseaseModelFactory


# open the csv file and read it into a pandas dataframe to understand the data
df_source = pd.read_csv('./data/2020/heart_2020_processed.csv', sep=',', quotechar='"')

# save the original set of data
df = df_source.copy()

df.head()


# Process the features

# set the target feature
target = 'heartdisease'

train_data = HeartDiseaseTrainData(df, target)
cat_features, num_features = train_data.process_features()

# split the data in train/val/test sets
# use 60%/20%/20% distribution with seed 1
# use stratified sampling to ensure the distribution of the target feature is the same in all sets
X_train, X_val, y_train, y_val, X_test, y_test = train_data.split_data(test_size=0.2, random_state=42)

print(X_val.head())


# hot encode the categorical features for the train data
model_factory = HeartDiseaseModelFactory(cat_features, num_features)
X_train_std = model_factory.preprocess_data(X_train[cat_features + num_features], True)

# hot encode the categorical features for the validation data
X_val_std = model_factory.preprocess_data(X_val[cat_features + num_features], False)


# Train the model
model_factory.train(X_train_std, y_train)


# Evaluate the model
df_metrics = model_factory.evaluate(X_val_std, y_val)
df_metrics.head()


df_metrics[['model','accuracy', 'precision', 'recall', 'f1']].head()


# plot df_metrics with the model name on the y-axis and metrics on the x-axis for all models and all metrics
# Sort the DataFrame by a metric (e.g., accuracy) to display the best-performing models at the top
df_metrics.sort_values(by='accuracy', ascending=False, inplace=True)
# Define the models, metrics, and corresponding scores
models = df_metrics['model']
metrics =['accuracy', 'precision', 'recall', 'f1']
scores = df_metrics[['accuracy', 'precision', 'recall', 'f1']]

# Set the positions for the models
model_positions = np.arange(len(models))

# Define the width of each bar group
bar_width = 0.15

# Create a grouped bar chart
plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    plt.barh(model_positions + i * bar_width, scores[metric.lower()], bar_width, label=metric)

      # Add score labels over the bars
    for index, row in df_metrics.iterrows():
        score = row[metric.lower()]
        plt.text(score, index, f'{score:.3f}', va='top', ha='center', fontsize=9)

# Customize the chart
plt.yticks(model_positions, models)
plt.xlabel('Score')
plt.ylabel('ML Models')
plt.title('Model Comparison for Heart Disease Prediction')
plt.legend(loc='upper right')

plt.savefig('./images/ozkary-ml-heart-disease-model-evaluation.png')
# Display the chart
# plt.show()



# ## Confusion Matrix Analysis


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cms = []
model_names = []
total_samples = []

for model_name in df_metrics['model']:
    model_y_pred = df_metrics[df_metrics['model'] == model_name]['y_pred'].iloc[0]

    # Compute the confusion matrix
    cm = confusion_matrix(y_val, model_y_pred)    
    cms.append(cm)
    model_names.append(model_name)
    total_samples.append(np.sum(cm))    

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Loop through the subplots and plot the confusion matrices
for i, ax in enumerate(axes.flat):
    cm = cms[i]    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.6)
    
    # Set labels, title, and value in the center of the heatmap
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), 
           xticklabels=["No Heart Disease", "Heart Disease"], yticklabels=["No Heart Disease", "Heart Disease"],
           title=f'{model_names[i]} (n={total_samples[i]})\n')

    # Loop to annotate each quadrant with its count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="gray")
            
    ax.title.set_fontsize(12)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.xaxis.set_label_position('top')

# Adjust the layout
plt.tight_layout()

plt.savefig('./images/ozkary-ml-heart-disease-model-confusion-matrix.png')
# plt.show()


# get the metrics grid with total samples for confusion matrix analysis
scores = df_metrics[['model','accuracy', 'precision', 'recall', 'f1']] 
scores['total'] = total_samples

scores.head()

print(cms)


# ## Save the model
# 
# - Save the best performing model
# - Save the encoder

# get the model and the dictionary vectorizer
model = model_factory.models[model_name]
encoder = model_factory.encoder

# Save the XGBoost model to a file
xgb_model_filename = './bin/hd_xgboost_model.pkl.bin'
with open(xgb_model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the DictVectorizer to a file
dv_filename = './bin/hd_dictvectorizer.pkl.bin'
with open(dv_filename, 'wb') as dv_file:
    pickle.dump(encoder, dv_file)



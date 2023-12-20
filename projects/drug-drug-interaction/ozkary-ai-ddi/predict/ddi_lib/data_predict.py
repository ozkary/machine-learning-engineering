#!/usr/bin/env python
# coding: utf-8

# Drug to Drug Interaction - Predicting the interaction 

import os
import numpy as np
import pandas as pd
import sklearn
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem


class DDIModelLoader():
    """ 
    Class to load a model from a pickle file and make predictions
    """
    def __init__(self, model_path, encoder_path=None):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoder = None
        self.load_model()

    def load_model(self):
        """ 
        Load the model from the pickle file
        Load encoder if use_encoder is set
        """

        with open(self.model_path, 'rb') as model:
            self.model = pickle.load(model)

        if self.encoder_path is not None:
            with open(self.encoder_path, 'rb') as encoder:
                self.encoder = pickle.load(encoder)        

    def predict(self, X):

        # Transform the data
        # X_encoded = self.encoder.transform(X)
        
        # Predict the results
        y_pred = self.model.predict(X)
        return y_pred


class DDIPredictor:
    """
    Maps the predictions to the original labels and meaning
    """
    def __init__(self, model_path, encoder_path, data_path):
        self.model = DDIModelLoader(model_path, encoder_path)
        self.interactions = None
        self.data_path = data_path
        self.drug_pca_lookup = None
        self.load_drug_pca_lookup()

    def load_drug_pca_lookup(self):
        """ 
        Load the drug pca lookup from pickle file
        """
        if self.drug_pca_lookup is None:
            pca_drugs = pd.read_csv(f'{self.data_path}drugbank_pca50.csv.gz',index_col=0)
            pca_drugs['name'] = pca_drugs['name'].str.lower()
            # convert drugs to a dictionary using name in lowercase as the key
            self.drug_pca_lookup = pca_drugs.set_index('name').T.to_dict('list')
            
    def predict(self, X):
        """ 
        Predict the results and map them to the original labels
        """       
        return self.model.predict(X)    
    
    def feature_names(self):
        """ 
        Return the feature names
        """
        # if the property feature name exists, return it
        if hasattr(self.model.model, 'feature_names'):
            return self.model.feature_names
        
        return None
    
    def build_model_input(self, rxs, features=101):
        """ 
        Build the message to be returned using a pca lookup
        """        
        # select the mean of the pc_ columns from the dict_drugs
        mean_pca = np.mean(list(self.drug_pca_lookup.values()), axis=0).tolist()

        #for each unique drug name in rxs do a lookup to get the pca values
        inputs = []
        for rx in rxs:
            # from the dict get all the keys with the drug names
            pca_drug = []            
            for label in ['drug1','drug2']:    
                name = rx[label].lower()                                
                pca = self.drug_pca_lookup[name] if name in self.drug_pca_lookup else mean_pca       
                pca_drug = pca_drug + pca
            pca_drug = pca_drug + [rx['ssp']]            
            inputs = inputs + [pca_drug]
      
        # convert the list to a numpy array to see the shape
        X = np.array(inputs)
        print(X.shape)
        
        return X
    
    def build_model_message(self, ssp_values, features=101):
        """ 
        Build the message to be returned
        """

        X = np.zeros((len(ssp_values), features))
        for index, ssp in enumerate(ssp_values):            
            X[index,-1] = ssp

        print(X.shape)        
        return X

    def get_ddi_description(self, results):
        # check if the list is empty

        if len(results) == 0:
            return None
        
        if self.interactions is None:
            self.load_interactions()

        notes = []
        for result in results:
            # select the row with ddi_type = result
            ddi_type = int(result['result']) + 1
            ddi_row = self.interactions.loc[self.interactions['ddi_type'] == ddi_type]
            
            # add one to the result to match the encoding during training            
            note = f'No interaction found for {result["drug1"]} and {result["drug2"]}'
            if ddi_row is not None and len(ddi_row) > 0:
                note = ddi_row['description'].to_string(index=False).replace('#Drug1', result['drug1']).replace('#Drug2', result['drug2'])
            notes.append(note)
            
        return notes
        
    def load_interactions(self):
        """ 
        Load the interactions from csv        
        """
        df = pd.read_csv(f'{self.data_path}interaction_types.csv')

        #rename the columns to lowercase and replace spaces with underscore
        df.columns = map(str.lower, df.columns)
        df.columns = df.columns.str.replace(' ', '_')

        # remove the DDI type text from the ddi_type column
        df['ddi_type'] = df['ddi_type'].str.replace('DDI type ', '')

        # cast the ddi_type column to integer
        df['ddi_type'] = df['ddi_type'].astype(int)
        self.interactions = df

    def calculate_ssp(self, smiles_drug1, smiles_drug2):

        """ 
        Structural Similarity Profile (SSP) for drug pairs  
        """

        # check if the SMILE code is valid
        if smiles_drug1 is None or smiles_drug2 is None:
            return 0
        
        try:
            mol_drug1 = Chem.MolFromSmiles(smiles_drug1)
            mol_drug2 = Chem.MolFromSmiles(smiles_drug2)

            fp_drug1 = AllChem.GetMorganFingerprintAsBitVect(mol_drug1, 2, nBits=1024)
            fp_drug2 = AllChem.GetMorganFingerprintAsBitVect(mol_drug2, 2, nBits=1024)

            array_fp_drug1 = np.array(list(fp_drug1.ToBitString())).astype(int)
            array_fp_drug2 = np.array(list(fp_drug2.ToBitString())).astype(int)

            tanimoto_similarity = np.sum(np.logical_and(array_fp_drug1, array_fp_drug2)) / np.sum(np.logical_or(array_fp_drug1, array_fp_drug2))

            return tanimoto_similarity
        except:
            return 0
        

def load_test_cases():
    """ 
    Load the test cases from the csv file
    """
    df_test_cases = pd.read_csv('./data/test_cases.csv')
    # make all columns lowercase and replace spaces with underscores
    df_test_cases.columns = [col.lower().replace(' ', '_') for col in df_test_cases.columns]

    # convert the data into a drug pair using the prescription column
    prescriptions = {}
    for index, row in df_test_cases.iterrows():
        rx = row['prescription']
        if rx not in prescriptions:        
            prescriptions[rx] = {}
        
        # get the key count to start bulding the properties
        key_count = 1 if len(prescriptions[rx]) == 0 else 2
        drug = f'drug{key_count}'
        smile = f'smiles{key_count}'        
        prescriptions[rx][drug] = row['drug_name'] 
        prescriptions[rx][smile] = row['smiles'] 
   
    return prescriptions    

def predict(data, path = './'):
    """ 
    Predict the DDI for the given data
    """
    
    # load the model
    data_path = os.path.join(path,'data/')
    model_path = os.path.join(path,'models')
    print('resources',model_path, data_path)

    model_file = f'{model_path}/ozkary_ddi_xgboost.pkl.bin'
    encoder_file = F'{model_path}/ozkary_ddi_encoder.pkl.bin'

    predictor = DDIPredictor(model_file, encoder_file, data_path)

    prescriptions = data

    if not isinstance(prescriptions, dict):
        raise Exception(f'Invalid data type. Expected a dictionary {data}')

    # for each drug pair calculate the structural similarity profile
    for rx in prescriptions:        
        drug1 = prescriptions[rx]['smiles1']
        drug2 = prescriptions[rx]['smiles2']
        prescriptions[rx]['ssp'] = predictor.calculate_ssp(drug1, drug2)
        
    print(prescriptions.values())

    # select all the ssp values from the list
    ssp_values = [item['ssp'] for item in prescriptions.values()]    
    # X = predictor.build_model_message(ssp_values)
    X = predictor.build_model_input(prescriptions.values())
    
    # run a prediction
    y_pred = predictor.predict(X)
    print('Predictions ', y_pred)
    # for each y_pred value add it to the dictionary    
    for index, item in enumerate(prescriptions.values()):
        item['result'] = y_pred[index]  

    # print the results
    print(prescriptions.values())

    # load the interactions types file
    result = predictor.get_ddi_description(prescriptions.values())
    
    return result


# add a main function for the entry point to the program
if __name__ == '__main__':
    os.system('clear')
    print('Running DDI main function')

    prescriptions = load_test_cases()
    results = predict(prescriptions)
    for result in results:
        print(result)

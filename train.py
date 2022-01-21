import pandas as pd
import numpy as np
import random
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

import pickle


def get_embeddings(df, model):
    ''' get list of Sentences from df; return df containing list of embeddings according to model '''
    
    sentences = df['Sentence'].tolist()
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
    for i in range(len(embeddings)):
        embeddings[i] = list(embeddings[i])

    df['Embeddings'] = embeddings.tolist()
    
    return df


def do_PLS(df, m):
    ''' Given a dataframe with high-d embeddings from model m, add a column with PLS-reduced embeddings '''
    
    PLS_labels = {'warm': [0, 1], 'comp': [1, 0], 'cold': [0, -1], 'incomp': [-1, 0],
                  'warm_comp': [1, 1], 'warm_incomp': [-1, 1], 'cold_comp': [1, -1], 'cold_incomp': [-1, -1]}
                  
    embeddings = np.array(df['Embeddings'].tolist())
    labels = [PLS_labels[row['Target']] for i, row in df.iterrows()]

    # fit PLS model
    pls = PLSRegression(n_components=10, scale = True)
    pls.fit(embeddings, labels)
    
    # transform training data 
    PLS_embeddings = pls.transform(embeddings)
    df['PLS_embeddings'] = PLS_embeddings.tolist()
    
    # save PLS model to transform test data later 
    filename = 'pls_model_' + m + '.sav'
    pickle.dump(pls, open(filename, 'wb'))
    
    return df
    

def do_PCA(df, m):
    ''' Given a dataframe with high-d embeddings from model m, add a column with PCA-reduced embeddings '''
             
    embeddings = np.array(df['Embeddings'].tolist())

    # fit PCA model
    pca = PCA(n_components=10)
    pca.fit(embeddings)
    
    # transform training data 
    pca_embeddings = pca.transform(embeddings)
    df['PCA_embeddings'] = pca_embeddings.tolist()
    
    # save PLS model to transform test data later 
    filename = 'pca_model_' + m + '.sav'
    pickle.dump(pca, open(filename, 'wb'))
    
    return df

    
def get_centroid(df, label, use_PLS=True, use_PCA=False):
    ''' Given df and direction label (warm, cold, etc.), return centroid of vectors for that direction '''
    
    temp_df = df[df['Target'] == label]
    
    if temp_df.shape[0] == 0:
        print('No data for label ', label)
        exit()
        
    if use_PLS and 'PLS_embeddings' in temp_df:  
        print('Using PLS embeddings ... ')   
        vecs = temp_df['PLS_embeddings'].tolist()
    elif use_PCA and 'PCA_embeddings' in temp_df:  
        print('Using PCA embeddings ... ')   
        vecs = temp_df['PCA_embeddings'].tolist()
    else:
        print('Using original sentence embeddings ...')
        vecs = temp_df['Embeddings'].tolist()
        
    vecs = np.array(vecs)
    
    return np.mean(vecs, axis=0)   
    

def normalize(v):
    ''' Return normalized vector v  '''
    
    return v / np.linalg.norm(v)
    

def compute_rotation_matrix(df, model_name, polar_model='original', PLS=False, PCA=False):
    ''' Compute the rotation matrix.
     
        polar_model: 'original' or 'axis_rotated'
        PLS: True or False
        PCA: True or False 
    '''
    
    #NB: consider moving this to train
    
    
    if polar_model == 'original':
     
        # compute centroids for each direction
        competence_center = normalize(get_centroid(df, 'comp', use_PLS=PLS, use_PCA=PCA))
        incompetence_center = normalize(get_centroid(df, 'incomp', use_PLS=PLS, use_PCA=PCA))
        warm_center = normalize(get_centroid(df, 'warm', use_PLS=PLS, use_PCA=PCA))
        cold_center = normalize(get_centroid(df, 'cold', use_PLS=PLS, use_PCA=PCA))

        # generate rotation matrix 
        dir_comp = normalize(np.array([competence_center - incompetence_center]))
        dir_warm = normalize(np.array([warm_center - cold_center]))
        dir = np.concatenate((dir_comp, dir_warm), axis=0)
        dir_T_inv = np.linalg.pinv(dir)    
        
    elif polar_model == 'axis_rotated':

        # compute centroids for each direction 
        warm_comp_center = normalize(get_centroid(df, 'warm_comp', use_PLS=PLS, use_PCA=PCA))
        cold_comp_center = normalize(get_centroid(df, 'cold_comp', use_PLS=PLS, use_PCA=PCA))
        warm_incomp_center = normalize(get_centroid(df, 'warm_incomp', use_PLS=PLS, use_PCA=PCA))
        cold_incomp_center = normalize(get_centroid(df, 'cold_incomp', use_PLS=PLS, use_PCA=PCA))

        # generate rotation matrix 
        dir_comp_warm = normalize(np.array([warm_comp_center - cold_incomp_center]))
        dir_incomp_warm = normalize(np.array([warm_incomp_center - cold_comp_center]))
        dir = np.concatenate((dir_comp_warm, dir_incomp_warm), axis=0)
        dir_T_inv = np.linalg.pinv(dir)      
    
    else:
        print("Argument polar_model must be one of 'original' or 'axis_rotated'")
        exit()
        
    # save rotation matrix 
    if PLS: 
        np.save('rotation_' + polar_model + '_PLS_' + model_name + '.npy', dir_T_inv)  
    elif PCA:       
        np.save('rotation_' + polar_model + '_PCA_' + model_name + '.npy', dir_T_inv)  
    else:
        np.save('rotation_' + polar_model + '_none_' + model_name + '.npy', dir_T_inv)  
        
    return


if __name__ == "__main__":

    # for reproducibility
    rseed = 1
    np.random.seed(seed=rseed)
    random.seed(rseed) 
    
    
    # sentence embedding model
    model_name = 'roberta-large-nli-mean-tokens' # can be any model here: https://www.sbert.net/docs/pretrained_models.html
    model = SentenceTransformer(model_name)

    
    # get sentence embeddings for single-adjective training data - can skip if embeddings already exist
    if not os.path.isfile('embeddings/training_all_one_adjective_' + model_name + '.csv'):
        train_df = pd.read_csv('data/training_all_one_adjective.csv')
        train_df = get_embeddings(train_df, model)
        train_df.to_csv('embeddings/training_all_one_adjective_' + model_name + '.csv')
    else:
        print('Embeddings appear to exist, moving on ...')

    # get sentence embeddings for double-adjective training data - can skip if embeddings already exist
    if not os.path.isfile('embeddings/training_all_two_adjectives_' + model_name + '.csv'):
        train_df = pd.read_csv('data/training_all_two_adjectives.csv')
        train_df = get_embeddings(train_df, model)
        train_df.to_csv('embeddings/training_all_two_adjectives_' + model_name + '.csv')
    else:
        print('Embeddings appear to exist, moving on ...')

    # read in embeddings and combine in tot_df
    df_single_adj = pd.read_csv('embeddings/training_all_one_adjective_' + model_name + '.csv')
    df_double_adj = pd.read_csv('embeddings/training_all_two_adjectives_' + model_name + '.csv')

    tot_df = pd.concat([df_single_adj, df_double_adj], ignore_index=True)
    for i, row in tot_df.iterrows():
        tot_df.at[i, 'Embeddings'] = json.loads(row['Embeddings'])


    # do PLS dimensionality reduction (optional)
    tot_df = do_PLS(tot_df, model_name)
    
    # do PCA dimensionality reduction (optional)
    #tot_df = do_PLS(tot_df, model_name)

    # compute and save the rotation matrix
    compute_rotation_matrix(tot_df, model_name, polar_model='axis_rotated', PLS=True, PCA=False)
    
    

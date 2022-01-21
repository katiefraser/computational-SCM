import pandas as pd
import numpy as np
import random
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import pickle
from scipy.spatial.transform import Rotation as R

from train import get_embeddings, do_PLS, do_PCA, get_centroid, normalize, compute_rotation_matrix
from test import rotate_data, compute_warmth_competence, compute_1d_accuracy, compute_2d_accuracy, compute_accuracy
    
        
if __name__ == "__main__":

    #optimal sentence embedding model
    model_name = 'roberta-large-nli-mean-tokens' # can be any model here: https://www.sbert.net/docs/pretrained_models.html
    
    #comparison models
    #model_name = 'stsb-roberta-large'
    #model_name = 'paraphrase-distilroberta-base-v2'
    #model_name = 'average_word_embeddings_glove.840B.300d'
    #model_name = 'paraphrase-mpnet-base-v2'

    model = SentenceTransformer(model_name)

    # these files contain the training sentences for all folds
    train_df_one_adj = pd.read_csv('data/cross_validation/training_one_adjective.csv')
    train_df_two_adj = pd.read_csv('data/cross_validation/training_two_adjectives.csv')
    
    test_results = {}
    
    # four functionalities:
    functional_testing = ['basic_functionality', 'negation', 'semantic_composition', 'syntax']
    
    # six model configurations:
    for test_case in functional_testing:
        test_results[test_case] = {'original': {'none': [], 'PCA': [], 'PLS': []}, 'axis_rotated': {'none': [], 'PCA': [], 'PLS': []}}


    for fold in range(5):
    
        print('=== FOLD ' + str(fold) + ' ===')

        # use different random seed in each fold 
        rseed = fold

        np.random.seed(seed=rseed)
        random.seed(rseed)
        
        ############
        # TRAINING #
        ############
        

        # get sentence embeddings for single-adjective training data - can skip if embeddings already exist
        if not os.path.isfile('embeddings/cross_validation/training_one_adjective_' + model_name + '_fold' + str(fold) + '.csv'):
            train_df = train_df_one_adj[train_df_one_adj['Fold'] == fold]
            train_df = get_embeddings(train_df, model)
            train_df.to_csv('embeddings/cross_validation/training_one_adjective_' + model_name + '_fold' + str(fold) + '.csv')
        else:
            print('Embeddings appear to exist, moving on ...')   

        # get sentence embeddings for double-adjective training data - can skip if embeddings already exist
        if not os.path.isfile('embeddings/cross_validation/training_two_adjectives_' + model_name + '_fold' + str(fold) + '.csv'):
            train_df = train_df_two_adj[train_df_two_adj['Fold'] == fold]
            train_df = get_embeddings(train_df, model)
            train_df.to_csv('embeddings/cross_validation/training_two_adjectives_' + model_name + '_fold' + str(fold) + '.csv')
        else:
            print('Embeddings appear to exist, moving on ...')    

        # read in embeddings for this fold
        df_single_adj = pd.read_csv('embeddings/cross_validation/training_one_adjective_' + model_name + '_fold' + str(fold) + '.csv')
        df_double_adj = pd.read_csv('embeddings/cross_validation/training_two_adjectives_' + model_name + '_fold' + str(fold) + '.csv')

        tot_df = pd.concat([df_single_adj, df_double_adj], ignore_index=True)
        for i, row in tot_df.iterrows():
            tot_df.at[i, 'Embeddings'] = json.loads(row['Embeddings'])

        # do PLS dimensionality reduction 
        tot_df = do_PLS(tot_df, model_name)

        # do PCA dimensionality reduction 
        tot_df = do_PCA(tot_df, model_name)
        
        # compute rotation matrices for each column in Table 2
        compute_rotation_matrix(tot_df, model_name, polar_model='original', PLS=False, PCA=False)
        compute_rotation_matrix(tot_df, model_name, polar_model='original', PLS=False, PCA=True)
        compute_rotation_matrix(tot_df, model_name, polar_model='original', PLS=True, PCA=False)

        compute_rotation_matrix(tot_df, model_name, polar_model='axis_rotated', PLS=False, PCA=False)
        compute_rotation_matrix(tot_df, model_name, polar_model='axis_rotated', PLS=False, PCA=True)
        compute_rotation_matrix(tot_df, model_name, polar_model='axis_rotated', PLS=True, PCA=False)
        

        #############
        #  TESTING  #
        #############
                
        for test_case in functional_testing: 
    
        
            # check if the embeddings already exist; if not, generate them
            if not os.path.isfile('embeddings/cross_validation/testing_' + test_case + '_' + model_name + '_fold' + str(fold) + '.csv'):
                test_df = pd.read_csv('data/cross_validation/testing_' + test_case + '.csv')
                test_df = test_df[test_df['Fold'] == fold]
                test_df = get_embeddings(test_df, model)
                test_df.to_csv('embeddings/cross_validation/testing_' + test_case + '_' + model_name + '_fold' + str(fold) + '.csv')
            else:
                print('Embeddings appear to exist, moving on ...')

            # load embeddings for test data
            test_df = pd.read_csv('embeddings/cross_validation/testing_' + test_case + '_' + model_name + '_fold' + str(fold) + '.csv')
            for i, row in test_df.iterrows():
                test_df.at[i, 'Embeddings'] = json.loads(row['Embeddings'])
                
            ## Compute accuracy for each column in Table 2 ##

            test_df = compute_warmth_competence(test_df, model_name, polar_model='original', PLS=False, PCA=False)
            test_results[test_case]['original']['none'].append(compute_accuracy(test_df))
            
            test_df = compute_warmth_competence(test_df, model_name, polar_model='original', PLS=False, PCA=True)
            test_results[test_case]['original']['PCA'].append(compute_accuracy(test_df))
            
            test_df = compute_warmth_competence(test_df, model_name, polar_model='original', PLS=True, PCA=False)
            test_results[test_case]['original']['PLS'].append(compute_accuracy(test_df))
            
            test_df = compute_warmth_competence(test_df, model_name, polar_model='axis_rotated', PLS=False, PCA=False)
            test_results[test_case]['axis_rotated']['none'].append(compute_accuracy(test_df))
            
            test_df = compute_warmth_competence(test_df, model_name, polar_model='axis_rotated', PLS=False, PCA=True)
            test_results[test_case]['axis_rotated']['PCA'].append(compute_accuracy(test_df))
                        
            test_df = compute_warmth_competence(test_df, model_name, polar_model='axis_rotated', PLS=True, PCA=False)
            test_results[test_case]['axis_rotated']['PLS'].append(compute_accuracy(test_df))
            

            
    ### RESULTS SUMMARY ###

    print()
    print('== Cross-validation for embedding model:', model_name, '==')
    print()

    for test_case in functional_testing:
        print('--', test_case, '--')
    
        for polar_model in ['original', 'axis_rotated']:
            for dim_reduction in ['none', 'PCA', 'PLS']:
                print('Model:', polar_model, '+', dim_reduction, ':', round(100*np.mean(test_results[test_case][polar_model][dim_reduction]), 1), '(' + str(round(100*np.std(test_results[test_case][polar_model][dim_reduction]), 1)) +')')        
            
            




import pandas as pd
import numpy as np
import random
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
import pickle
from scipy.spatial.transform import Rotation as R

from train import get_embeddings, normalize

    
def rotate_data(arr):

    ''' Rotate array by 45 degrees  '''

    r = R.from_euler('z', 45, degrees=True)
    arr = list(arr)

    for i in range(len(arr)):
        arr[i] = list(arr[i])
        arr[i].append(0) # add z-coordinate
        
    arr = np.array(arr)
    arr = r.apply(arr)
    
    arr = np.delete(arr, 2, 1) #remove z-coordinate

    return arr
    

def compute_warmth_competence(df, model_name, polar_model='original', PLS=False, PCA=False):
    ''' Given df and arguments, compute the warmth and competence values '''

    if PLS:

        # get saved PLS model
        print('Loading PLS model ...')
        pls = pickle.load(open('pls_model_' + model_name + '.sav', 'rb'))

        # do PLS dimensionality reduction 
        print('Doing PLS dimensionality reduction ...')
        PLS_embeddings = pls.transform(np.array(df['Embeddings'].tolist())) 
        embeddings = [normalize(s) for s in PLS_embeddings]
        
        dir_T_inv = np.load('rotation_' + polar_model + '_PLS_' + model_name + '.npy')

        
    elif PCA:
    
        # get saved PCA model
        print('Loading PCA model ...')
        pca = pickle.load(open('pca_model_' + model_name + '.sav', 'rb'))

        # do PCA dimensionality reduction 
        print('Doing PCA dimensionality reduction ...')
        PCA_embeddings = pca.transform(np.array(df['Embeddings'].tolist())) 
        embeddings = [normalize(s) for s in PCA_embeddings]
        
        dir_T_inv = np.load('rotation_' + polar_model + '_PCA_' + model_name + '.npy')
        
    else:
    
        embeddings = df['Embeddings'].tolist()
        dir_T_inv = np.load('rotation_' + polar_model + '_none_' + model_name + '.npy')



    # project to 2D warmth-competence plane (with rotation for axis-rotated POLAR)
    print('Computing warmth and competence ...')
    if polar_model == 'original':
        SCM_embeddings = np.array(np.matmul(embeddings, dir_T_inv))
    else:
        SCM_embeddings = rotate_data(np.array(np.matmul(embeddings, dir_T_inv)))

    # make warmth and competence columns 
    df['Competence'] = SCM_embeddings[:,0].tolist()
    df['Warmth'] = SCM_embeddings[:,1].tolist()
    
    return df


def compute_1d_accuracy(label, c, w):
    ''' Given gold label, competence value c, and warmth value w, return 1 if hypothesis matches gold and 0 otherwise. '''

    result = 0

    if label == 'warm' and w >= 0:
        result = 1
    elif label == 'cold' and w < 0:
        result = 1
    elif label ==  'comp' and c >= 0:
        result = 1
    elif label == 'incomp' and c < 0:
        result = 1
        
    return result


def compute_2d_accuracy(label, c, w):
    ''' Given gold label, competence value c, and warmth value w, return 1 if hypothesis matches gold and 0 otherwise. '''

    result = 0
    
    if label == 'warm_comp' and w >= 0 and c >= 0:
        result = 1
    elif label == 'cold_comp' and w < 0 and c >= 0:
        result = 1
    elif label ==  'warm_incomp' and w >= 0 and c < 0:
        result = 1
    elif label == 'cold_incomp' and w < 0 and c < 0:
        result = 1        
    
    return result       


def compute_accuracy(df):
    ''' Given df containing columns 'Target' (gold labels), 'Warmth', and 'Competence', return accuracy. '''

    if 'Target' in df:
        acc = []
    
        # heuristic: if target labels contain an underscore, use 2-D accuracy 
        # (i.e. projected point should lie in correct quadrant)
        # otherwise use 1D accuracy (correct along the relevant axis only)
    
        labels = list(set(df['Target'].tolist()))  
        use_1d_accuracy = True
    
        for label in labels:
            if '_' in label:
                 use_1d_accuracy = False

        # compute correctness for each point 
        for index, row in df.iterrows():
            if use_1d_accuracy:
                acc.append(compute_1d_accuracy(row['Target'], row['Competence'], row['Warmth']))
            else:
                acc.append(compute_2d_accuracy(row['Target'], row['Competence'], row['Warmth']))
                
        return np.mean(acc)
    
    else:
        print('No target label information available')
        return 0
        


if __name__ == "__main__":

    # sentence embedding model
    model_name = 'roberta-large-nli-mean-tokens' # can be any model here: https://www.sbert.net/docs/pretrained_models.html
    model = SentenceTransformer(model_name)

    # test file should contain sentences and, optionally, labels
    test_dir = 'data'
    test_filename = 'testing_all_basic_functionality' #assume CSV file

    # check if the embeddings already exist; if not, generate them
    if not os.path.isfile('embeddings/' + test_filename + '_' + model_name + '.csv'):
        train_df = pd.read_csv(test_dir + '/' + test_filename + '.csv')
        train_df = get_embeddings(train_df, model)
        train_df.to_csv('embeddings/' + test_filename + '_' + model_name + '.csv')
    else:
        print('Embeddings appear to exist, moving on ...')

    # load embeddings for test data
    test_df = pd.read_csv('embeddings/' + test_filename + '_' + model_name + '.csv')
    for i, row in test_df.iterrows():
        test_df.at[i, 'Embeddings'] = json.loads(row['Embeddings'])
        
        
    # load the saved dimensionality reduction model and rotation matrix; compute warmth and competence
    test_df = compute_warmth_competence(test_df, model_name, polar_model='axis_rotated', PLS=True, PCA=False)

    # save to file 
    print('Outputting file ... ')
    test_df.to_csv('output/' + test_filename + '_' + model_name + '.csv', index=False)
    
    # optionally -- compute accuracy with respect to gold labels 
    accuracy = compute_accuracy(test_df)
    print('ACCURACY: ', accuracy)
        


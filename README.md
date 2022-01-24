# computational-SCM
A computational approach to the Stereotype Content Model, as described in our paper Computational Modelling of Stereotype Content in Text.
 
The [Stereotype Content Model (SCM)](https://www.sciencedirect.com/science/article/pii/S1364661306003299) provides a theoretical basis for understanding human social cognition. It proposes that the two primary, universal dimensions underlying human stereotypes are *warmth* (friendliness, trustworthiness) and *competence* (ability, intelligence, skill). As a step towards automatically identifying and understanding stereotypes in text, we provide a computational model to estimate warmth and competence from free-text sentences.

The model makes use of [sentence-transformers](https://www.sbert.net/) and is based on the POLAR framework proposed by [Mathew et al.](https://arxiv.org/pdf/2001.09876.pdf). We have only tested the code using python 3.9.5 and on a CPU. 


# Training
To train your SCM model, run train.py

```
python3 train.py
```

By default, this uses the optimal model from the paper (i.e., with PLS dimensionality reduction, axis-rotation, and using the 'roberta-large-nli-mean-tokens' model). All of these options can be modified within the program.

The output of running the program will be a saved a model, e.g. pls_model_roberta-large-nli-mean-tokens.sav.

# Testing

Once you have a saved model, you can apply it to test data using test.py to compute warmth and competence values for unseen data.

```
python3 test.py
```

By default, this will process the file data/testing_all_basic_functionality.csv. This will create an output file: output/testing_all_basic_functionality_<embedding-model>.csv, which contains the test sentences as well as the associated sentence embeddings and the estimated warmth and competence values. Because the test file also contains Target labels, the program computes an accuracy with respect to the gold labels. On this test case, the result should look something like this:
  
```
Batches: 100%|████████████████████████████████████| 8/8 [00:15<00:00,  1.91s/it]
Loading PLS model ...
Doing PLS dimensionality reduction ...
Computing warmth and competence ...
Outputting file ... 
ACCURACY:  0.9914893617021276
```
                                                                                                                                          
# Reproducibility

 To reproduce the cross-validation results from the paper, run:
                                                                      
```
python3 reproduce.py
```

Again, by default this will generate results for the 'roberta-large-nli-mean-tokens' sentence embedding model, although this can be modified in the code to be any of the other models considered in the paper, or indeed any sentence embedding model of your choosing. The code runs 5-fold cross validation for the six model configurations (original POLAR and axis-rotated POLAR, each with PCA, PLS, or no dimensionality reduction). It uses the data files found in data/cross_validation. If you run this program, you should reproduce the results from Table 2 in the paper. 
                                                                      
```
== Cross-validation for embedding model: roberta-large-nli-mean-tokens ==

-- basic_functionality --
Model: original + none : 95.0 (3.8)
Model: original + PCA : 95.4 (3.3)
Model: original + PLS : 96.2 (2.8)
Model: axis_rotated + none : 97.9 (2.3)
Model: axis_rotated + PCA : 97.9 (2.3)
Model: axis_rotated + PLS : 97.9 (2.3)
-- negation --
Model: original + none : 95.3 (1.6)
Model: original + PCA : 95.3 (2.8)
Model: original + PLS : 95.3 (1.6)
Model: axis_rotated + none : 95.8 (3.1)
Model: axis_rotated + PCA : 96.2 (2.4)
Model: axis_rotated + PLS : 95.8 (2.6)
-- semantic_composition --
Model: original + none : 73.9 (9.8)
Model: original + PCA : 77.9 (8.2)
Model: original + PLS : 77.7 (10.3)
Model: axis_rotated + none : 81.6 (8.2)
Model: axis_rotated + PCA : 78.8 (7.9)
Model: axis_rotated + PLS : 84.4 (7.7)
-- syntax --
Model: original + none : 70.2 (16.7)
Model: original + PCA : 70.1 (14.0)
Model: original + PLS : 71.0 (12.3)
Model: axis_rotated + none : 72.1 (9.3)
Model: axis_rotated + PCA : 72.0 (11.4)
Model: axis_rotated + PLS : 78.7 (11.2)
```
                                                                      
# Data

Included in the repository are:

- training sentences containing one and two adjectives (generated from the Seed Lexicon provided by [Nicolas et al., 2021](https://onlinelibrary.wiley.com/doi/full/10.1002/ejsp.2724)). 
- test sentences with labels. Note that these test sentences overlap with the training sentences and are only intended to be used to confirm that the software is working properly.
- the cross-validation data, as discussed in Section 3.1.3 of the paper. In this case, there is no overlap between train and test data in each fold. There are four test files, intended to test basic functionality, negation, semantic composition, and syntactic variability.
- the manual annotations for warmth and competence for each word in the Seed Lexicon, averaged over three annotators. 

In the paper, we also use StereoSet as a validation set; that data is available here: https://stereoset.mit.edu/

Finally, we also describe a Twitter data collection effort related to women and older adults. That data cannot be publicly re-distributed under the Twitter Terms of Service, but interested researchers may contact us for more information. 
                                                                  

# Citation

If you use this code, please cite our paper:

```
@journal{}
```

# Contact

If you have any questions or comments, please contact Katie at kathleen.fraser@nrc-cnrc.gc.ca                                                                      

                                                                      

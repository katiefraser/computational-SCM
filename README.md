# computational-SCM
A computational approach to the Stereotype Content Model


# Training
To train your SCM model, run train.py

```
python3 train.py
```

By default, this uses the optimal model from the paper (i.e., with PLS dimensionality reduction, axis-rotation, and using the roberta-large-NLI model). All of these options can be modified within the program.

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
                                                                      
                                                                      
                                                                      
                                                                   

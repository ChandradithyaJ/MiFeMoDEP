# MiFeMoDEP

## Introduction

We propose MiFeMoDEP, a hybrid Mixed Feature Model for Defect Prediction. Building upon the ideas of the related works, we propose a file-level classification model, which is passed to LIME to rank the lines. Our model deviates from the others with a mixed feature pathway, where the semantic information and flow information are extracted independently and mixed before passing it to a classifier.

We also built a dataset for line-level testing from the Defectors Dataset. Each file or commit is given in the Defectors Dataset, and a list of defective lines is the target. This is in contrast to the regularly used dataset, where lines are individual rows.

First, the neural network that encodes the PDG is trained with another simple neural network acting as a classifier. Transfer Learning is performed where the weights of the PDG Encoder are transferred to MiFeMoDEP, and a Random Forest Classifier is trained using the concatenated CodeBERT and PDG embeddings.

*Note: To download [`Defectors dataset`](https://zenodo.org/records/7570822)*

## Test your data on MiFeMoDEP as follows:
* Download Repository from GitHub
* Get the file path of the file you want to check<br>
* Follow the steps below for testing a Source Code or Git Diff file:
  * For checking a Source Code file, run one of the following command:(add the filepath to the variable `path_to_file` in the `MiFeMoDEP_SourceCode_TestSingleSourceCode.py` file)
    *   `python ./MiFeMoDEP/MiFeMoDEP/SourceCode/MiFeMoDEP_SourceCode_TestSingleSourceCode.py` or 
    *  `python3 ./MiFeMoDEP/MiFeMoDEP/SourceCode/MiFeMoDEP_SourceCode_TestSingleSourceCode.py` (i.e. run the file `MiFeMoDEP_SourceCode_TestSingleSourceCode.py`)
  * For checking a Git Diff file, run one of the following command:(add the filepath to the variable `path_to_file` in the `MiFeMoDEP_SourceCode_TestSingleCommit.py` file)
    *   `python ./MiFeMoDEP/MiFeMoDEP/JIT/MiFeMoDEP_JIT_TestSingleCommit.py` or 
    *   `python3 ./MiFeMoDEP/MiFeMoDEP/JIT/MiFeMoDEP_JIT_TestSingleCommit.py` (i.e. run the file `MiFeMoDEP_JIT_TestSingleCommit.py`)
* Output will be displayed on terminal

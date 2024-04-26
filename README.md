# MiFeMoDEP

## Introduction

We propose MiFeMoDEP, a Hybrid Mixed Feature Model for Defect Prediction. Building upon the ideas of the related works, we propose a file-level classification model, which is passed to LIME to rank the lines. Our model deviates from the others with a mixed feature pathway, where the semantic information and flow information are extracted independently and then mixed before passing it to a classifier.

We also replicated two other Models namely **[DeepLineDP](https://ieeexplore.ieee.org/document/9689967)**(for source code) and **[JITLine](https://arxiv.org/abs/2103.07068)**(for JIT changes).

We also built datasets for line-level testing from the Defectors Dataset. In the Defectors Dataset, each file or commit is given, and a list of defective lines is the target. This is in contrast to the regularly used dataset where lines are individual rows in the dataset.

### Proess for **Source Code**
First, the source code is given to CodeBERT which extracts semantic information from the code(in batch size of 64, which is the limit of CodeBERT) in the form of embeddings. These embedddings are passed to a per-trained PCA model to focus on more important fatures of  code, rather than all the features. Also, source code is given to [JOERN](https://joern.io/)(a graph processing tool) which gives us a graph(with the information of AST, PDG, and CFG) with 20 different edges, now this graph is given to a RGCN(relational graph convulutional netwrork), which can extract information of nodes from it's neighbours and aggrgate the details at each node.

Now, output of both of them are concatanated and sent to a random forest classifier which predicts whether the file is defective or not. If the file is defective, then A LimeTabularExplainer is created with the training features, and each file in the test dataset, along with the trained Random Forest Classifer are passed to it to extract an explanation. This explanation has scores for each line, which are then used to rank the lines based on their defectiveness.

### Process for **JIT Code Changes**
The only difference for JIT defect prediction is that the input JIT cdoe changes to the model is preprocessed such that only the source code and the plus/minus characters that represent the addition or removal of a line are present.

*Note: To download [`Defectors dataset`](https://zenodo.org/records/7570822)*

## Test your data on MiFeMoDEP as follows:
* Download Release-2 from GitHub Releases
* Store the file you want to check in MiFeMoDEP(which is main folder)<br>
* *Note: You should have downloaded **JOERN** from their official website, so as to run the below files.*
* Follow the steps below for testing a Source Code or JIT code changes file:
  * For checking a Source Code file, run one of the following command:(add the filepath to the variable `filepath` in the `complete_pipeline_MiFeMoDEP_for_single_input.py` file)
    *   `python ./MiFeMoDEP/MiFeMoDEP/SourceCode/complete_pipeline_MiFeMoDEP_for_single_input.py` or 
    *  `python3 ./MiFeMoDEP/MiFeMoDEP/SourceCode/complete_pipeline_MiFeMoDEP_for_single_input.py` (i.e. run the file `complete_pipeline_MiFeMoDEP_for_single_input.py`)
  * For checking a JIT cdoe changes, run one of the following command:(add the filepath to the variable `path_to_file` in the `MiFeMoDEP_SourceCode_TestSingleCommit.py` file)
    *   `python ./MiFeMoDEP/MiFeMoDEP/JIT/MiFeMoDEP_JIT_TestSingleCommit.py` or 
    *   `python3 ./MiFeMoDEP/MiFeMoDEP/JIT/MiFeMoDEP_JIT_TestSingleCommit.py` (i.e. run the file `MiFeMoDEP_JIT_TestSingleCommit.py`)
* Output will be displayed on terminal

# python-nlp-dependency-parser
NLP Dependency Parsing using Perceptron and Chu-Liu-Edmonds. The task is building an unlabeled parse tree for a POS-tagged sentence.

## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`numpy`|  `1.14.5`|
|`matplotlib`|  `3.0.0`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`DepOptimizer.py`| Helper class for the Chu-Liu-Edmonds algorithmm|
|`DependencyParser.py`| Parser class, including Structured Perceptron|
|`ProgressBar.py`| Progress Bar class|
|`generate_competition_files.py`| exmaple for generating trees for unlabled files|
|`chu_liu.py`| Chu-Liu-Edmonds algorithmm|
|`features.py`| old features functions, more modular|
|`features_v2.py`| modified features, including McDonald features, less modular|
|`dependency_parser.py`| example for training a model|
|`calc_accuracy_main.py`| example for evaluating a model|
|`utils.py`| other utility functions|
|`FeaturesAnalysis.ipynb`|features analysis|
|`ModelsComparison.ipynb`| comparison of various models|
|`DataAnalysis.ipynb`| first analysis of the problem|
|`*.wtag` | labeled samples| 
|`*.unlabeld` | samples without labels| 
|`*.weights`| Checkpoint files for the models' weights (inferring/continual learning)|

## Generating Labeled Files Example
1. Prepare 2 pretrained models from `/pretrained`
2. Evaluate them on the test file
3. Generate labels for the unlabeled file
4. Validate the generated file (accuracy should be 100%).
 
`python generate_competition_files.py`

## Training and Testing Example
1. change models parameters (features, feature threshold, number of training iterations) in `dependency_parser.py`
2. run `python dependency_parser.py`
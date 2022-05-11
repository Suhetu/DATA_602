# DATA_602_Project

This repository currently contains a Jupyter Notebook of Exploratory Data Analysis performed on my data selection of 'NASA Asteroid Data-Potentially Hazardous Asteroids' as well as a notebook containing classification performed on the dataset to identify Potentially Hazardous Asteroids, or PHAs.

This repository also contains pre-trained classifier models as pickle objects exported from my notebook in case anyone wants to try them out :)


The notebook is best viewed using nbviewer, https://nbviewer.org/ . Kindly use the nbviewer as the pandas profiling report is visible on it, else not.



## Data:
The data source is: https://ssd.jpl.nasa.gov/tools/sbdb_query.html#!#results 
The data which I have queried has dimensions of 612,011 rows and 74 columns.
Glossary for data features present in the dataset can be found here: https://cneos.jpl.nasa.gov/glossary/PHA.html 


## Objective:
This project aims to identify observed asteroids if they are potentially hazardous or not. This goal has been accomplished using certain machine learning classification algorithms present in the scikit-learn library.


## Overview of ML content present in the Jupyter notebook:
•	The notebook starts by loading the data and the immediate major task performed is feature engineering. Pandas profiling was used to identify important features and to avoid problematic features which were cardinal or were collinear or were all unique.

•	Once the important features were explained and a few new features were derived from the original features, developing machine learning models started.

•	I would like to bring to your notice that the classification problem at hand is highly imbalanced, so the number of true observations are very less (about 1%) but are very important to detect too (having high risk). Hence, I used recall as the significant scoring metric and all my results/comments/model observations would be based on this.

•	The first classification model I used was Logistic Regression since the model is considered a really good baseline model in terms of performance and time. Using this model, I achieved a recall score of 87% for the minority class and 93% macro average recall.

•	The next classification algorithm I used was Logistic regression with L2 Regularization. This model gave a significant improvement with a recall score of 95% for the minority class and 97% macro average recall.

•	Then Decision Trees classification algorithm was implemented, and this model surprised me the most. This model gave perfect (100%)  precision, recall and accuracy. There were zero false positive and false negatives, and every test observation was predicted perfectly.

•	Random Forest classification was then implemented which gave really similar results to the Logistic Regression with L2 Regularization model. I got 95% minority class recall score and 97% macro average recall score.

•	Then a Support Vector Classifier was implemented. Support Vector Classifier also performed really well, getting a recall score of 99% for the minority class. This model also gave very few false positives and false negatives.

•	A KNN classifier was implemented after the SVC and this model surprised me the most. This model gave horrible results with recall score of 28% for the minority class. This model’s results were then ignored.

•	The last classification model implemented was an ensemble. This ensemble consisted of the following models:
    Logistic Regression with Regularization
    Decision Trees
    Random Forest
    Support Vector Classifier
    
All the hyperparameters were tuned using GridsearchCV and the best estimator models of each respective model stated above were used as ‘soft’ voters in the voting classifier algorithm.


## Key Takeaways:
1. Our data was highly imbalanced with about 99% data belonging to class 0 and less than 1% data belonging to class 1. That's why we focused on the recall score to measure our model performance.
2. Models which performed really well were Logistic Regression with regularization, Decision Tree, Random Forest, Support Vector Classifier and the ensemble built consisting of above models.
3. Decision Trees model strangely gave 100% recall score, to which I'm more scared of rather happy. I'm glad the model was capable of predicting the asteroid class perfectly as the classification problem is a high risk one.
4. My first choice of model for this particular classification task at hand would be the Decision Tree but a really close and almost tie would be the ensemble model because both these models give really good recall scores for class 1.
5. Machine Learning is a very powerful tool.


## Final Model:
The best model according to the performance measured in terms of recall, was the Decision Tree classifier. This model gave amazing results while at the same time not consuming much time compared to SVC and KNN.



## References:
https://scikit-learn.org/stable/modules/preprocessing.html
https://medium.com/@ritesh.110587/correlation-between-categorical-variables-63f6bd9bf2f7
https://datascience.stackexchange.com/questions/65839/macro-average-and-weighted-average-meaning-in-classification-report
https://elitedatascience.com/imbalanced-classes
https://pandas-profiling.ydata.ai/docs/master/index.html
https://datascience.stackexchange.com/questions/36862/macro-or-micro-average-for-imbalanced-class-problems
https://inside.getyourguide.com/blog/2020/9/30/what-makes-a-good-f1-score
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

## Special References:
https://github.com/appliedecon/data602-lectures/blob/main/logistic-regression/logistic-regression.ipynb
https://github.com/appliedecon/data602-lectures/blob/main/trees/trees.ipynb
https://github.com/appliedecon/data602-lectures/blob/main/supervised-algorithms/knn-nb-svm.ipynb
https://github.com/appliedecon/data602-lectures/blob/main/ensembles/ensembles.ipynb

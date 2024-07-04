## Project Premise 

I've been contracted as a data scientist to look at recent data from a regional airline designing flight deals for young professionals flying their airline. They'd like a model that accurately predicts which customers are more likely to purchase travel insurance since previous research has shown those customers are most likely to buy flight deals of this type, so they'd like to use one of our models for subsequent customer targeting. 

## Project Approach

This project uses the [Travel Insurance dataset](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data) from Kaggle to train and test both single-classifier and ensemble supervised learning models. 

For the single classifiers model selection proceeds as follows:
1. Split data
2. Hyperparametric tuning
3. Conduct Kfolds cross-validation on training set
5. Repartition data
6. Calculate training set performance
7. Calculate testing set performance
8. Assess 2-3 models side-by-side

Ensemble models model selection varies, but is built using the tuned models from the single-classifier models that came before. 

## Project Findings
After conducting exploratory data analysis on the 1987 customer dataset, this project uses logistic regression, K-nearest neighbors, a decision tree, 
and categorical Naive Bayes classifiers to set a predictive baseline. We then proceed to implement random forest, stacked generalization, and voting classifier models to explore how much improvement in predictive performance can be generated using these ensemble methods. 
Once the models have been compared side by side and evaludated across multiple metrics, we select the random classifier model specification that yielded the best performance as the final product to the client. 

## Relevant Files
Please check out the travel_insurance_functions.py file to see the helper functions used, and requirements.txt for module/package information. 

# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

This dataset contains data about bank marketing campaigns based on phone calls to potential clients. The campaing has as target to
convince the potential clients to make a term deposit with the bank. We seek to predict whether or not the potential client would 
accept to make a term deposit with the bank.


**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The first, Scikit-learn pipeline, that uses logistic regression with hyperdrive to find the best hyperparameters scored 0.91685% accuracy.
The AutoML pipeline scored 0.91591% accuracy and the algorithm used for this is VotingEnsemble.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The data used for predictions are from : https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv .
First of all the data are cleaned with the ```clean_data``` in ```train.py```. The rows with missing values are dropped and categorical fields are converted to numerical fields. The data are splitted to train and test set with ratio 0.85:0.15. 

Tha algorithm used for the training is Logistic Regression. The two hyperparameters of the Logistic Regression are tuned with the hyperdrive to find the model with the best accuracy on the test set.  The two hyperparameters are the following:

```C```: The inverse of the reqularization strength. The smaller the number the stronger the regularization.
```max_iter```: Maximum number of iterations to converge. 


**Benefits of the parameter sampler**

I chose the ```RandomParameterSampling```, the hyperparameters are randomly selected from the search space. The search space for the two hyperaparameters is the following:

```
   '--C' : choice(0.1,1,10,100,500),
   '--max_iter': choice(50,100,300)
```
where the choice define discrete space over the values. The benefits of the ```RandomParameterSampling```, is that it is more fast than for example the ```GridParameterSampling``` where all the possible values from the search space are used, and it supports early termination of low-performance runs.

**Benefits of the early stopping policy**

I chose the ```BanditPolicy``` which is an "aggressive" early stopping policy with the meaning that cuts more runs than a conservative one like the ```MedianStoppingPolicy```, so it saves computational time. There are three configuration parameters ```slack_factor, evaluation_interval(optional), delay_evaluation(optional)```. 
* ```slack_factor/slack_amount``` : (factor)The slack allowed with respect to the best performing training run.(amount) specifies the allowable slack as an absolute amount, instead of a ratio.

* ```evaluation_interval``` : (optional) The frequency for applying the policy.

* ```delay_evaluation``` : (optional) Delays the first policy evaluation for a specified number of intervals.

I set ```evaluation_interval=2, slack_factor=0.1```.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

In AutoML many pipelines produced simultaneously that run different algorithms and parameters in an automated way. 
In order to setup an AutoML train you have to config some parameters : 

* ```experiment_timeout_minutes=30``` : It has been set that way and we can't change it for the purpose of this assignment.

* ```task='classification'``` : We have a classification task to do, we seek to predict whether or not the potential client would accept to make a term deposit with the bank.

* ```compute_target ``` : The compute target with specific ```vm_size``` and ```max_nodes```.

* ```training_data``` : The data on which the algorithm will be trained.

* ```label_column_name``` : The name of the column that contains the labels of the train data.

* ```n_cross_validations=3 ``` : It is how many cross validations to perform when user validation data is not specified. 

* ```primary_metric = 'accuracy'``` : The metric that Automated Machine Learning will optimize for model selection. We have set the 'accuracy'.

* ```enable_early_stopping = True``` : Whether to enable early termination if the score is not improving in the short term. 


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

As we can see there are no sifnificant differences in accuracy. Both models trained with the same dataset and with early stopping policy. 






## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

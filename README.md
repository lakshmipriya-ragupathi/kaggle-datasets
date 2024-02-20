# kaggle-datasets
Exploring Datasets from Kaggle

## Traffic Sign Classifier 

_Trafficsign_classifier_DeepSMOTE_YOLOv8_InceptionV3.ipynb_

A Computer Vision, Deep Learning Task.

dataset link: https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset 

notebook link: https://www.kaggle.com/code/lakshmipriya190803/classifier-deepsmote-yolov8-inceptionv3-resnet

In this notebook, I have explored traffic sign classification - a multiclass class classfication problem using German Traffic Sign Dataset. Due to heavy imbalance of data, I have implemented DeepSMOTE with the help of this paper: https://ieeexplore.ieee.org/document/9694621, and used transfer learning technique to classify the traffic signs. 

I plan on creating a web app using the trained model for traffic sign recognition in real time.

## Reddit Comment Analysis

_reddit-comments-analysis-using-nlp-and-dl.ipynb_

A Natural Language Processing, Deep Learning Task

dataset link: https://www.kaggle.com/datasets/armitaraz/chatgpt-reddit

notebook link: https://www.kaggle.com/code/lakshmipriya190803/reddit-comments-analysis-using-nlp-and-dl

As part of an Open Source Project for SWOC 2024, we are supposed to perfom Sentimental Analysis on Reddit Comments on the topic of ChatGPT from various SubReddits. The dataset is not labelled, so I have used VADER Sentiment Analyser to obtain the Labels: Positive, Neutral and Negative. Then I have used various Deep Learning Approaches starting with Simple RNN, and made my way towards BERT model.

I plan on using hugging face models in the context of unsupervised learning without using "VADERised" data

## Binary Classification

_plaayground_binary_classification-using-ml-and-dl.ipynb_

An ML Task

dataset link: https://www.kaggle.com/competitions/playground-series-s4e1

notebook link: https://www.kaggle.com/code/lakshmipriya190803/logistic-svm-nb-rf-xgb-cb-lightgbm-0-885

As part of Playgound Series, in this notebook, I learnt A LOT. First the dataset looked imblanced, so I used Oversampling(SMOTE) and Undersampling on the datasets and compared the ROC for various models on both the training samples (there was no notable difference). 

I started off with Logistic Regression and struggled my way in SVM, ended up learning about CuML library using which I put the GPUs on fire. Even then the performance was just about satifactory, so explored further with Gradient Boosting Classifiers - XGBoost gave the best performance and achieved 0.89 score in the final submission.


## Exploratory Data Analysis

_hate-speech-detection.ipynb_ : An EDA on Sentiment Dataset

_eda-on-dollar-price-dialy.ipynb_ : An EDA on Time Series Dataset

dataset link: https://www.kaggle.com/datasets/raoofiali/ohlc-dollar-daily-price

notebook link: https://www.kaggle.com/code/lakshmipriya190803/beginner-friendly-eda-and-time-series-forecasting


# 🐦 Twitter Sentiment Analysis
Logistic Regression + TF-IDF for Tweet Classification

<img width="1536" height="1024" alt="Twitter Sentiment Analysis Header Design" src="https://github.com/user-attachments/assets/b74a4ee4-2d9d-45cc-a742-57cdc9afaa93" />


"End-to-end Twitter Sentiment Analysis using Logistic Regression & TF-IDF — from raw tweets to accurate sentiment predictions."

# Introduction
This project is part of my work in Machine Learning and Natural Language Processing (NLP), showcasing my ability to apply end-to-end ML workflows to real-world problems.
By leveraging Logistic Regression for classification and TF-IDF Vectorization for feature extraction, the system analyzes tweets and determines their sentiment as Positive, Negative, or Neutral.
It demonstrates my skills in data preprocessing, feature engineering, model building, and evaluation, as well as my ability to present results in a clear and actionable format.

# Project Overview
This project focuses on analyzing tweets to classify their sentiment as Positive, Negative, or Neutral.
It applies Natural Language Processing (NLP) techniques using TF-IDF Vectorization for text feature extraction and Logistic Regression for classification.
The aim is to demonstrate how machine learning can be applied to real-world social media analytics.

# Key Features
* Cleans and preprocesses raw Twitter text by removing mentions, hashtags, URLs, and extra symbols.
* Uses TF-IDF Vectorizer to convert text into numerical features.
* Trains a Logistic Regression model for sentiment classification.
* Evaluates the model using Accuracy, Precision, Recall, and F1-score.
* Visualizes sentiment distribution and performance metrics

# Workflow
* Collect Twitter Data
* Clean and Preprocess Text
* Extract Features with TF-IDF
* Train Logistic Regression Model
* Evaluate and Visualize Results
* Predict on New Tweets

# Project Structure
* data/ – Contains the dataset.
* notebooks/ – Jupyter Notebooks for exploration and training.
* src/ – Source code for preprocessing, training, and prediction.
* README.md – Project documentation.

# Tech Stack
* Programming Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* Model: Logistic Regression
* Feature Extraction: TF-IDF Vectorizer
* Tools: colab Notebook, Git, GitHub

# How It Works
* Collect and clean Twitter data.
* Preprocess text by removing noise and normalizing words.
* Convert text into numerical vectors using TF-IDF.
* Train a Logistic Regression classifier on the processed data.
* Evaluate performance and visualize results.
* Predict sentiment for new tweets.

# Model Performance
* Accuracy: 57%
*  precision      - recall    -  f1-score     - support

 - Irrelevant     -  0.56     -  0.38       -  0.46    - 10392
 -  Negative      -  0.66     -  0.70       -  0.68    - 18033
 -   Neutral      -  0.51     -  0.57       -  0.54    - 14654
 -   Positive     -  0.61     -  0.64       -  0.63    - 16665


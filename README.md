# NLP-Project

# SMS Spam Detection using NLP



## Project Description

This repository contains a comprehensive Natural Language Processing (NLP) project focused on building a classification model to distinguish between legitimate (**Non-Spam/Ham**) and unwanted (**Spam**) SMS messages. The entire workflow, from data cleaning and feature engineering to model training and evaluation, is documented in the **Jupyter Notebook** (`SMS_NLP_Project.ipynb`).

The project uses NLP techniques like **Tokenization**, **Stemming**, **Stopword Removal**, and **TF-IDF Vectorization** to transform text data into numerical features suitable for a machine learning classifier, which appears to be a **Support Vector Classifier (SVC)** based on the notebook imports.


## Files in this Repository

| File Name | Description |
| :--- | :--- |
| **SMS\_NLP\_Project.ipynb** | The main Jupyter Notebook where all the NLP processing, model training, and performance evaluation takes place. |
| **SMS\_train (1).csv** | The dataset used for training the classification model. It contains the message bodies and their corresponding `Label` (Spam or Non-Spam). |
| **SMS\_test (2).csv** | The dataset used for independent testing and final evaluation of the trained model's performance. |
| **README.md** | This file, providing an overview and instructions for the project. |


## Technical Approach (NLP Pipeline)

The `SMS_NLP_Project.ipynb` notebook implements the following core steps:

### 1. Data Preprocessing and Cleaning
* **Loading:** Reading the separate training and testing CSV files into pandas DataFrames.
* **Text Cleaning:** Standardizing the text by removing special characters, numbers, and converting text to lowercase.
* **Tokenization:** Breaking the messages into individual words (tokens) using the `TweetTokenizer`.
* **Stopword Removal:** Eliminating common, less informative words (e.g., "the", "a", "is") using the **NLTK corpus**.
* **Stemming/Lemmatization:** Reducing words to their root form using a **Stemmer** (e.g., `SnowballStemmer`) to improve feature consolidation.

### 2. Feature Extraction
* **TF-IDF Vectorization:** Converting the cleaned text into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)**. This method weighs words based on their importance across the entire corpus.

### 3. Model Training and Evaluation
* **Model:** A **Support Vector Classifier (SVC)** is trained on the TF-IDF vectors from the training data.
* **Metrics:** The model's performance is evaluated using the test data to assess its accuracy in correctly classifying messages as Spam or Non-Spam.


## Prerequisites

To run this project, you need a Python environment with the following libraries:

* **pandas**
* **numpy**
* **nltk** (for NLP tasks)
* **scikit-learn** (`sklearn`)
    * `TfidfVectorizer`
    * `train_test_split`
    * `SVC`

### Installation

You can install the primary dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn nltk

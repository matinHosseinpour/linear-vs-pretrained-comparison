import pandas as pd
import numpy as np
import json
import re
import nltk
import time
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the preprocessing function
def preprocess_text(text):
    preprocess_text = text

    # Remove white space
    preprocess_text = " ".join(preprocess_text.split())

    # Remove Twitter handles
    preprocess_text = re.sub(r'@([A-Za-z0-9_]+)', '', preprocess_text)

    # Convert to lowercase
    preprocess_text = preprocess_text.lower()

    # Remove special characters and numbers
    preprocess_text = re.sub(r'[^a-zA-Z\s]', '', preprocess_text)

    # Tokenizing
    word_tokens = word_tokenize(preprocess_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    preprocess_text = [word for word in word_tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in preprocess_text]

    # Join tokens back into a string
    preprocessed_text = ' '.join(stems)

    return preprocessed_text

files = [
    {
        'dataset': 'sst2',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'sst2',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'movie_reviews',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'movie_reviews',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imdb',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imdb',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'yelp',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'yelp',
        'pre_process': False,
        'class_weight': False
    },
    # Imbalance - class weight
    {
        'dataset': 'imbalanced_sst2',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_sst2',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_movie_reviews',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_movie_reviews',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_imdb',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_imdb',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_yelp',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_yelp',
        'pre_process': False,
        'class_weight': False
    },
    # Imbalance + class weight
    {
        'dataset': 'imbalanced_sst2',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_sst2',
        'pre_process': False,
        'class_weight': True
    },
    {
        'dataset': 'imbalanced_movie_reviews',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_movie_reviews',
        'pre_process': False,
        'class_weight': True
    },
    {
        'dataset': 'imbalanced_imdb',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_imdb',
        'pre_process': False,
        'class_weight': True
    },
    {
        'dataset': 'imbalanced_yelp',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_yelp',
        'pre_process': False,
        'class_weight': True
    },
]

for file in files:
    print(file)
    start_time = time.time()
    # 1. Read the dataset using Pandas
    data = pd.read_csv(f'../data/' + file['dataset'] + '/' + file['dataset'] + '.csv')

    labels = data['sentiment'].tolist()
    label_mapping = {"negative": 0, "positive": 1}  # Define mapping for labels
    labels = [label_mapping[label] for label in labels]
    data["label"] = labels

    # 2. Preprocess the data (if necessary)
    data.dropna(subset=['review', 'label'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 3. Prepare texts and labels
    texts = data['review'].tolist()
    if file['pre_process']:
        texts = data['review'].apply(preprocess_text).tolist()
    labels = data['label'].tolist()

    # 4. Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # 5. Define the Multinomial Naive Bayes classifier
    mnb_classifier = MultinomialNB()
    if file['class_weight']:
        mnb_classifier = MultinomialNB(class_prior=[0.5, 0.5])

    # 6. Implement k-Fold Cross-Validation and Collect Metrics
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    overall_true_labels = []
    overall_pred_labels = []
    class_names = np.unique(y).astype(str)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"Processing Fold {fold}...")
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        mnb_classifier.fit(X_train_fold, y_train_fold)
        y_pred_fold = mnb_classifier.predict(X_test_fold)
        
        overall_true_labels.extend(y_test_fold)
        overall_pred_labels.extend(y_pred_fold)
        
        accuracy = accuracy_score(y_test_fold, y_pred_fold)
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_test_fold, y_pred_fold, labels=class_names, zero_division=0
        )
        
        fold_metric = {
            'fold': fold,
            'accuracy': accuracy,
            'values': {
                'train': {str(key): str(value) for key, value in Counter(y_train_fold).items()},
                'test': {str(key): str(value) for key, value in Counter(y_test_fold).items()}
            },
            'metrics_per_class': []
        }
        
        for idx, class_name in enumerate(class_names):
            class_metric = {
                'class': class_name,
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1_score[idx],
            }
            fold_metric['metrics_per_class'].append(class_metric)
        
        fold_metrics.append(fold_metric)

    # 7. Compute Overall Metrics
    overall_accuracy = accuracy_score(overall_true_labels, overall_pred_labels)
    overall_precision, overall_recall, overall_f1_score, overall_support = precision_recall_fscore_support(
        overall_true_labels, overall_pred_labels, labels=class_names, zero_division=0
    )

    evaluation_time = time.time() - start_time

    # Ensure overall_accuracy is a native Python float
    overall_accuracy = float(overall_accuracy)

    # Prepare overall metrics dictionary
    overall_metrics = {
        "time": evaluation_time,
        'accuracy': overall_accuracy,
        'values': {str(k): str(v) for k, v in data['sentiment'].value_counts().to_dict().items()},
        'total_data': len(data['sentiment']),
        'metrics_per_class': []
    }

    for idx, class_name in enumerate(class_names):
        class_metric = {
            'class': str(class_name),
            'precision': float(overall_precision[idx]),
            'recall': float(overall_recall[idx]),
            'f1_score': float(overall_f1_score[idx]),
        }
        overall_metrics['metrics_per_class'].append(class_metric)

    # 8. Output Metrics to a JSON File
    output_data = {
        'fold_metrics': fold_metrics,
        'overall_metrics': overall_metrics
    }

    model_name = 'mnb_tfidf.json'
    output_path = '../models_results/' + file['dataset'] + '/mnb/'
    if file['pre_process']:
        output_path += 'pre_'
    else :
        output_path += ''
        
    if file['class_weight']:
        output_path += 'cw_'
    else :
        output_path += ''
        
    output_path += model_name
        
    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print("\nMetrics have been written to " + output_path)

import pandas as pd
import numpy as np
import json
import re
import nltk
import time
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
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
    # Remove white space
    text = " ".join(text.split())

    # Remove Twitter handles
    text = re.sub(r'@([A-Za-z0-9_]+)', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenizing
    word_tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokens if word not in stop_words]

    # Do not perform stemming
    # stemmer = PorterStemmer()
    # stems = [stemmer.stem(word) for word in words]

    # Join tokens back into a string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

# Function to load GloVe embeddings
def load_glove_embeddings(glove_file_path, expected_dim):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line_number, line in enumerate(f, 1):
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == expected_dim:
                embeddings_index[word] = coefs
            else:
                print(f"Warning: Line {line_number}: Embedding for word '{word}' has length {len(coefs)} instead of {expected_dim}")
    return embeddings_index

# Function to get text embedding
def get_text_embedding(text, embeddings_index, embedding_dim):
    words = text.split()
    valid_embeddings = []
    for word in words:
        embedding = embeddings_index.get(word)
        if embedding is not None:
            if embedding.shape != (embedding_dim,):
                print(f"Warning: Embedding for word '{word}' has shape {embedding.shape}")
            valid_embeddings.append(embedding)
    if valid_embeddings:
        # Average the embeddings
        mean_embedding = np.mean(valid_embeddings, axis=0)
        if mean_embedding.shape != (embedding_dim,):
            print(f"Warning: mean_embedding has shape {mean_embedding.shape} instead of ({embedding_dim},)")
    else:
        # If no words are found in embeddings, return a zero vector
        mean_embedding = np.zeros(embedding_dim)
    return mean_embedding

files = [
    {
        'dataset': 'dbpedia_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'dbpedia_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'yahooa_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'yahooa_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'dbpedia_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'dbpedia_50000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'yahooa_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'yahooa_50000',
        'pre_process': False,
        'class_weight': False
    },
    # Imbalance - class weight
    {
        'dataset': 'imbalanced_dbpedia_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_dbpedia_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_yahooa_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_yahooa_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_dbpedia_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_dbpedia_50000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_yahooa_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_yahooa_50000',
        'pre_process': False,
        'class_weight': False
    },
    # Imbalance + class weight
    {
        'dataset': 'imbalanced_dbpedia_10000',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_dbpedia_10000',
        'pre_process': False,
        'class_weight': True
    },
    {
        'dataset': 'imbalanced_yahooa_10000',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_yahooa_10000',
        'pre_process': False,
        'class_weight': True
    },
    {
        'dataset': 'imbalanced_dbpedia_50000',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_dbpedia_50000',
        'pre_process': False,
        'class_weight': True
    },
    {
        'dataset': 'imbalanced_yahooa_50000',
        'pre_process': True,
        'class_weight': True,
    },
    {
        'dataset': 'imbalanced_yahooa_50000',
        'pre_process': False,
        'class_weight': True
    },
]

files = [
    {
        'dataset': 'dbpedia_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'dbpedia_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'yahooa_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'yahooa_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'dbpedia_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'dbpedia_50000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'yahooa_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'yahooa_50000',
        'pre_process': False,
        'class_weight': False
    },
    # Imbalance - class weight
    {
        'dataset': 'imbalanced_dbpedia_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_dbpedia_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_yahooa_10000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_yahooa_10000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_dbpedia_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_dbpedia_50000',
        'pre_process': False,
        'class_weight': False
    },
    {
        'dataset': 'imbalanced_yahooa_50000',
        'pre_process': True,
        'class_weight': False,
    },
    {
        'dataset': 'imbalanced_yahooa_50000',
        'pre_process': False,
        'class_weight': False
    },
]

files = [
    {
        "dataset": "dbpedia_10000_5",
        "pre_process": True,
        "class_weight": False,
    },
    {
        "dataset": "dbpedia_10000_5",
        "pre_process": False,
        "class_weight": False,
    },
    # Imbalance - class weight
    {
        "dataset": "imbalanced_dbpedia_10000_5",
        "pre_process": True,
        "class_weight": False,
    },
    {
        "dataset": "imbalanced_dbpedia_10000_5",
        "pre_process": False,
        "class_weight": False,
    },
    {
        "dataset": "yahooa_10000_5",
        "pre_process": True,
        "class_weight": False,
    },
    {
        "dataset": "yahooa_10000_5",
        "pre_process": False,
        "class_weight": False,
    },
    # Imbalance - class weight
    {
        "dataset": "imbalanced_yahooa_10000_5",
        "pre_process": True,
        "class_weight": False,
    },
    {
        "dataset": "imbalanced_yahooa_10000_5",
        "pre_process": False,
        "class_weight": False,
    },
]
for file in files:
    print(file)
    start_time = time.time()
    # 1. Read the dataset using Pandas
    data = pd.read_csv(f'../data/' + file['dataset'] + '/' + file['dataset'] + '.csv')

    labels = data['sentiment'].tolist()
    data["label"] = labels

    # 2. Preprocess the data (if necessary)
    data.dropna(subset=['review', 'label'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 3. Prepare texts and labels
    texts = data['review'].tolist()
    if file['pre_process']:
        texts = data['review'].apply(preprocess_text).tolist()
    labels = data['label'].tolist()

    # Load GloVe embeddings
    glove_file_path = '../glove.6B.100d.txt'  # Replace with your GloVe file path
    embedding_dim = 100  # For GloVe 100d embeddings

    print("Loading GloVe embeddings...")
    embeddings_index = load_glove_embeddings(glove_file_path, embedding_dim)
    print("Loaded GloVe embeddings.")

    # 4. Convert text data to embeddings
    print("Converting texts to embeddings...")
    embeddings_list = []
    for idx, text in enumerate(texts):
        mean_embedding = get_text_embedding(text, embeddings_index, embedding_dim)
        if mean_embedding.shape != (embedding_dim,):
            print(f"Warning: Text at index {idx} has embedding of shape {mean_embedding.shape}")
        embeddings_list.append(mean_embedding)
    X = np.array(embeddings_list)
    print("Texts converted to embeddings.")
    print(f"Shape of X: {X.shape}")

    y = np.array(labels)

    # 5. Define the Logistic Regression classifier
    lr_classifier = LogisticRegression(random_state=42)
    if file['class_weight']:
        lr_classifier = LogisticRegression(class_weight={0:10, 1:2}, random_state=42)

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
        
        lr_classifier.fit(X_train_fold, y_train_fold)
        y_pred_fold = lr_classifier.predict(X_test_fold)
        
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

    model_name = 'lr_glove.json'
    output_path = '../models_results/' + file['dataset'] + '/lr/'
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

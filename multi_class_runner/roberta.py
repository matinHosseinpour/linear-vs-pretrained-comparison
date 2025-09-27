import pandas as pd
import numpy as np
import json
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a custom Dataset class
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        text = str(self.reviews[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc='Training'):
        optimizer.zero_grad()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader):
    model.eval()
    total_loss = 0
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss / len(data_loader), pred_labels, true_labels

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

    # Drop missing values
    data.dropna(subset=['review', 'sentiment'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Ensure labels are integers starting from 0
    data["label"] = data['sentiment'].astype(int)
    if data["label"].min() != 0:
        data["label"] = data["label"] - data["label"].min()

    num_labels = data["label"].nunique()
    print(f"Number of unique labels: {num_labels}")
    print(f"Labels: {data['label'].unique()}")

    # 2. Preprocess the data
    data.dropna(subset=['review', 'label'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 3. Prepare texts and labels
    if file['pre_process']:
        data['processed_review'] = data['review'].apply(preprocess_text)
        texts = data['processed_review'].tolist()
    else:
        texts = data['review'].tolist()
    labels = data['label'].tolist()

    # 4. Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # 5. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, max_length=128)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer, max_length=128)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model with correct num_labels
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # Training for 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training
    for epoch in range(3):  # Training for 3 epochs
        print(f"Epoch {epoch + 1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train loss: {train_loss}")

    # Evaluation
    val_loss, y_pred, y_true = eval_model(model, test_loader)
    print(f"Validation loss: {val_loss}")

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_labels), zero_division=0
    )

    evaluation_time = time.time() - start_time

    # Prepare overall metrics dictionary
    overall_metrics = {
        "time": evaluation_time,
        'accuracy': accuracy,
        'values': {str(k): str(v) for k, v in data['sentiment'].value_counts().to_dict().items()},
        'total_data': len(data['sentiment']),
        'metrics_per_class': []
    }

    class_names = list(range(num_labels))
    for idx, class_name in enumerate(class_names):
        class_metric = {
            'class': str(class_name),
            'precision': precision[idx],
            'recall': recall[idx],
            'f1_score': f1_score[idx],
        }
        overall_metrics['metrics_per_class'].append(class_metric)

    # Output Metrics to a JSON File
    output_data = {
        'overall_metrics': overall_metrics
    }

    model_name = 'roberta.json'
    output_path = '../models_results/' + file['dataset'] + '/roberta/'
    if file['pre_process']:
        output_path += 'pre_'
    else :
        output_path += ''
        
    if file['class_weight']:
        output_path += 'cw_'
    else :
        output_path += ''
        
    output_path += model_name

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print("\nMetrics have been written to " + output_path)

    # Free up memory
    del model
    torch.cuda.empty_cache()

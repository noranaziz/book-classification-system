import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- STEP 1: PARTITION DATASET ----------
# The labeled training dataset is divided into 3 parts:
## train1: used to train agent1 (XGBoost)
## train2: used to train agent2 (MLP Neural Network)
## train3: labels are removed to make it an unlabeled dataset

# load the training dataset
data = pd.read_csv('train_data.csv')

# ensure relevant columns are selected (genre and words - preprocessed/cleaned summaries)
data = data[['genre', 'words']]

# split into 3 equal parts
train1, temp = train_test_split(data, test_size=2/3, random_state=42, stratify=data['genre'])
train2, train3 = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['genre'])

# remove labels from train3 to simulate unlabeled data
train3_unlabeled = train3.drop(columns='genre')



# ---------- STEP 2: INITIAL TRAINING ----------
# train each agent independently:
## agent1 (XGBoost) is trained using train1
## agent2 (MLP) is trained using train2
## both agents are evaluated against the testing dataset
## stats are outputted

# load testing data
test_data = pd.read_csv('test_data.csv')
test_labels = pd.read_csv('test_labels.csv')

# initialize vectorizer + label encoder
vectorizer = TfidfVectorizer(max_features=5000)
label_encoder = LabelEncoder()

# fit vectorizer on data
vectorizer.fit(pd.concat([train1['words'], train2['words'], test_data['words']], axis=0))

# encode labels w/ LabelEncoder
label_encoder.fit(data['genre'])

# extract features + labels for train1 and train2
X_train1 = vectorizer.transform(train1['words']).toarray()
y_train1 = label_encoder.transform(train1['genre'])
X_train2 = vectorizer.transform(train2['words']).toarray()
y_train2 = label_encoder.transform(train2['genre'])

# extract features + labels for testing dataset
X_test = vectorizer.transform(test_data['words']).toarray()
y_test = label_encoder.transform(test_labels['genre'])

# scale features
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_train2 = scaler.transform(X_train2)
X_test = scaler.transform(X_test)

# train agent 1 (XGBoost)
agent_1 = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
agent_1.fit(X_train1, y_train1)

# train agent 2 (MLP)
agent_2 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
agent_2.fit(X_train2, y_train2)

# evaluate agents on test dataset
# agent 1 (XGBoost)
y_pred_agent1 = agent_1.predict(X_test)
# agent 2 (MLP)
y_pred_agent2 = agent_2.predict(X_test)



# ---------- STEP 3: MUTUAL LEARNING ----------
# 1. prediction phase:
## both agents make predictions for train3
## confidence scores (posterior probabilities) are recorded
# 2. relabeling phase:
## each data point in train3 are relabeled based on the agent w/ the highest confidence
### if agent 1 (XGBoost) predicts w/ higher confidence, use its label
### otherwise, agent 2's (MLP) label is used
# 3. combine training data:
## a new training dataset is created for each agent by combining:
### train1 + train3
### train2 + train3
# 4. both agents are retrained w/ their respective combined datasets
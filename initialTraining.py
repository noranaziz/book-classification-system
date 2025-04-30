import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load dataset
df = pd.read_csv('cleaned_books.csv')

# Set threshold per genre
threshold = 1500

# Count genre occurrences
genre_counts = df['genre'].value_counts()
print(genre_counts)

# Keep top 7 genres
valid_genres = genre_counts[genre_counts >= threshold].index
df = df[df['genre'].isin(valid_genres)]

# Split datasets into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Obtain genre from test dataset
test_labels = test_df['genre']
test_df = test_df.drop(columns=['genre'])

print("Training dataset:")
print(train_df['genre'].unique())
print(train_df['genre'].value_counts())

# Save the new datasets
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
test_labels.to_csv('test_labels.csv', index=False)

# Define feature and target columns
X_train = train_df['words']
y_train = train_df['genre']

X_test = test_df['words']
y_test = test_labels

# Encode target column 'y' to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Convert the text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert the training and test sets to dense arrays (for compatibility with models)
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# Normalize the data
scaler = StandardScaler()
X_train_tfidf = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = scaler.transform(X_test_tfidf)

# Save the TF-IDF vectorizer and scaler
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Open a file to write the results
with open('initTrainResults.txt', 'w') as f:
    def train_and_evaluate_model(model, model_name):
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        f.write(f'{model_name} Accuracy: {accuracy * 100:.2f}%\n')
        
        # Print classification report
        class_names = label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        f.write(f"\n{model_name} Classification Report:\n{report}\n")

        # Save model
        joblib.dump(model, f'{model_name.lower().replace(" ", "_")}_model.pkl')

    try:
        # ------------------- XGBoost Model -------------------
        xgb_model = xgb.XGBClassifier()
        train_and_evaluate_model(xgb_model, 'XGBoost')
        print("xgboost trained")

        # ------------------- SVM Model -------------------
        svm_model = SVC(kernel='linear')
        train_and_evaluate_model(svm_model, 'SVM')
        print('svm trained')
        
        # ------------------- Random Forest Model -------------------
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
        train_and_evaluate_model(rf_model, 'Random Forest')
        print('random forest trained')
        
        # ------------------- Logistic Regression Model -------------------
        lr_model = LogisticRegression(max_iter=1000)
        train_and_evaluate_model(lr_model, 'Logistic Regression')
        print('logistic regression trained')

        # ------------------- Naive Bayes -------------------
        nb_model = MultinomialNB()
        nb_model.fit(vectorizer.transform(X_train), y_train)
        y_pred_nb = nb_model.predict(vectorizer.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred_nb)
        f.write(f'Naive Bayes Accuracy: {accuracy * 100:.2f}%\n')
        # Classification report
        class_names = label_encoder.classes_
        report = classification_report(y_test, y_pred_nb, target_names=class_names)
        f.write(f"\nNaive Bayes Classification Report:\n{report}\n")
        print('naive bayes trained')
        
        # ------------------- MLP Neural Network -------------------
        mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        train_and_evaluate_model(mlp_model, 'MLP Neural Net')
        print('mlp neural net trained')
        
        # ------------------- Ensemble (Voting Classifier - MLP + XGBoost + Logistic Regression) -------------------
        ensemble_model = VotingClassifier(estimators=[
            ('mlp', mlp_model),
            ('xgb', xgb_model),
            ('lr', lr_model)
        ], voting='soft')  # soft = use predicted probabilities

        train_and_evaluate_model(ensemble_model, 'Voting Ensemble (MLP + XGB + LR)')
        print('voting ensemble (MLP + XGB + LR) trained')

    except Exception as e:
        f.write(f"An error occurred: {str(e)}\n")
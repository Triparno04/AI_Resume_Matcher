import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# ✅ 1. Load your diverse dataset
df = pd.read_csv("data/combined_resume_job_dataset.csv")

# ✅ 2. Combine resume text + job description into 1 string
df['text'] = df['resume_text'].fillna('') + " " + df['job_description'].fillna('')
X = df['text']
y = df['label']

# ✅ 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 4. Convert to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ 5. Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# ✅ 6. Evaluate on test set
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# ✅ 7. Save model + vectorizer
joblib.dump(clf, "scripts/resume_classifier_model.pkl")
joblib.dump(vectorizer, "scripts/resume_vectorizer.pkl")

print("✅ Model and vectorizer saved.")

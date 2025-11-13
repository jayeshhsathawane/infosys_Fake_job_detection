# =========================================================
#   FAKE JOB DETECTION PIPELINE — COMPLETE PROJECT
# =========================================================

# ----------------------------
# Day 2 — Data Understanding
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('fake_job_postings.csv')

print("Dataset loaded successfully.")
print("\nSample Data:\n", df.head())
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values per Column:\n", df.isnull().sum())

# Target distribution
print("\nTarget (fraudulent) Distribution:\n", df['fraudulent'].value_counts())

# Basic statistics
print("\nDataset Summary:\n", df.describe(include='all'))

print("""
Insights:
1) Many fake job postings have missing company profiles or descriptions.
2) Fake jobs often lack company logos or have unrealistic salary ranges.
3) Dataset is imbalanced — most jobs are real.
""")

# ----------------------------
# Day 3 — Text Cleaning & Preprocessing
# ----------------------------
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

print("\nCleaning text data...")
df['clean_description'] = df['description'].apply(clean_text)

df['word_count_before'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))
df['word_count_after'] = df['clean_description'].apply(lambda x: len(x.split()))

print("\nAverage word count before:", df['word_count_before'].mean())
print("Average word count after:", df['word_count_after'].mean())
print("Text cleaning completed.")

# ----------------------------
# Day 3.5 — Feature Correlation & Insights
# ----------------------------
from wordcloud import WordCloud

features = ['has_company_logo', 'telecommuting', 'employment_type', 'required_experience', 'fraudulent']
subset = df[features]

print("\nFeature Distribution Grouped by Fraudulent:")
for col in ['has_company_logo', 'telecommuting', 'employment_type']:
    print(f"\nFeature: {col}")
    print(df.groupby('fraudulent')[col].value_counts(normalize=True))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.countplot(x='has_company_logo', hue='fraudulent', data=df)
plt.title('Company Logo vs Fraudulent')

plt.subplot(1, 3, 2)
sns.countplot(x='telecommuting', hue='fraudulent', data=df)
plt.title('Remote Work vs Fraudulent')

plt.subplot(1, 3, 3)
sns.countplot(x='employment_type', hue='fraudulent', data=df)
plt.title('Employment Type vs Fraudulent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# WordClouds for real vs fake jobs
real_text = " ".join(df[df['fraudulent'] == 0]['description'].dropna())
fake_text = " ".join(df[df['fraudulent'] == 1]['description'].dropna())

real_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
fake_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_text)

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.imshow(real_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Real Job Descriptions (Green)')

plt.subplot(1,2,2)
plt.imshow(fake_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Fake Job Descriptions (Red)')
plt.show()

# ----------------------------
# Day 4 — Feature Extraction
# ----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_description'].fillna(''))

print("\nTF-IDF Shape:", X_tfidf.shape)
print("Sample Features:", tfidf.get_feature_names_out()[:10])

sum_words = X_tfidf.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in tfidf.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

print("\nTop 15 Words with Highest TF-IDF Scores:")
for w, s in words_freq[:15]:
    print(f"{w}: {s:.4f}")

# ----------------------------
# Day 5 — Logistic Regression Model
# ----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("""
Interpretation:
High accuracy due to class imbalance (most jobs real).
Recall for fake jobs may be low — use class weights or SMOTE later to improve.
""")

# ----------------------------
# Day 6 — Model Saving & Predictions
# ----------------------------
import joblib

probs = lr_model.predict_proba(X_test)[:, 1]
rand_idx = np.random.choice(len(y_test), 5, replace=False)

for i in rand_idx:
    text = df.iloc[y_test.index[i]]['description'][:200]
    print(f"\nJob Description (first 200 chars):\n{text}\nPredicted Probability of Fake: {probs[i]:.3f}")

joblib.dump(lr_model, 'fake_job_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved successfully.")

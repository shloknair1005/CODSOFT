# CODSOFT
Projects made for the CODSOFT internship tasks

Task1: 🎬 Movie Genre Classification

This project classifies movie plots into genres using Support Vector Machines (SVM). The dataset is based on IMDb movie descriptions, and includes multiple genres like drama, comedy, horror, and more.

📁 Dataset

The dataset consists of:
- `train_data.txt` — contains plot, genre, and ID
- `test_data.txt` — contains only plot and ID
- `test_data_solution.txt` — contains the test set's true genres

Each plot is associated with a single genre.

🧹 Preprocessing

- Lowercased all text
- Removed special characters and extra whitespace
- Applied TF-IDF vectorization (with 1-2 grams and 5000 features)

🧠 Model

The main classifier is a **Support Vector Machine (SVM)**. Logistic Regression and Naive Bayes were also tested for comparison.

📊 Evaluation

Evaluation includes:
- Accuracy score
- Precision, recall, and F1-score
- Confusion matrix visualized as a heatmap
- Genre distribution plots

🚀 Future Improvements

- Use word embeddings (Word2Vec, GloVe)
- Try transformer models (e.g., BERT)
- Handle multi-label genres
- Address class imbalance with resampling


Created by **Shlok Nair** 


Task2: 📱 Spam SMS Detection

A simple yet effective machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using natural language processing and a Naive Bayes model.

📁 Dataset

Used the "SMS Spam Collection Dataset" (from UCI/Kaggle), containing labeled SMS messages with `spam` or `ham` tags.

🧹 Preprocessing

- Converted text to lowercase
- Removed punctuation, digits, and extra spaces
- Vectorized text using TF-IDF

🧠 Model

Used **Multinomial Naive Bayes** — fast, interpretable, and surprisingly accurate for text classification.

📊 Evaluation

- Accuracy, precision, recall, and F1-score
- Confusion matrix (visualized as a heatmap)
- Charts for class distribution and most frequent words

🚀 Future Work

- Experiment with deep learning (e.g., LSTM, Transformers)
- Add live message prediction through a web UI
- Integrate spam keyword highlighting


Created by **Shlok Nair** 


Task3: 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression on real-world transaction data.

📦 Dataset

Used the **Credit Card Transactions Fraud Detection** dataset by **Kartik Shenoy** from Kaggle. It contains anonymized details of over 500,000 transactions split into training (`fraudTrain.csv`) and testing (`fraudTest.csv`) datasets.

🔧 Preprocessing

- Dropped heavy and personally identifiable columns (`merchant`, `cc_num`, etc.)
- One-hot encoded categorical features (`category`, `gender`, `state`)
- Converted `dob` to age for consistency
- Mapped boolean-like string fields (`true`/`false`) to 1/0
- Handled missing values and ensured column consistency across train/test

🧠 Model

Used **Logistic Regression** with `class_weight='balanced'` to account for extreme class imbalance. Standardized the feature values using `StandardScaler`.

📊 Evaluation

Two modes of testing were used:
- **Train-Test Split (from `fraudTrain.csv`)**
  - Accuracy: ~89%
  - ROC AUC: ~91%

- **Separate Validation (`fraudTest.csv`)**
  - Accuracy: ~93%
  - ROC AUC: ~90%

Visualizations included:
- Class imbalance plots
- Confusion matrices
- Classification report (precision, recall, F1-score)

🚀 Future Improvements

- Experiment with ensemble models like **Random Forest** or **XGBoost**
- Apply **SMOTE** or **ADASYN** for better handling of data imbalance
- Deploy the model as a REST API for real-time fraud detection


Created by Shlok Nair



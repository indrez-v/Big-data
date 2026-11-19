# Customer Sentiment and Product Recommendation System

A Flask dashboard that analyzes customer review sentiment and recommends products using a hybrid recommender system.

## Features

- Sentiment analysis with Logistic Regression, Linear SVM, and Naive Bayes
- Model accuracy, precision, recall, F1, confusion matrix, and cross-validation reports
- Customer complaint and aspect analysis
- Product, category, and review insights
- Hybrid recommendations combining:
  - Item-based collaborative filtering
  - Category preferences
  - Review sentiment
- Interactive review prediction

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

The application creates `data/reviews.csv` from `generate_dataset.py` if the dataset is missing.

## Main Files

- `app.py` - Flask application, sentiment models, recommender, and API routes
- `generate_dataset.py` - Generates the sample review dataset
- `templates/index.html` - Dashboard interface
- `requirements.txt` - Python dependencies

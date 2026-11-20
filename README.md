# Customer Sentiment and Product Recommendation System

A Flask web application that analyzes customer reviews, identifies common complaints, and recommends products using sentiment analysis and a hybrid recommender system.

## Features

- Classifies reviews as positive, neutral, or negative using Logistic Regression, Linear SVM, and Naive Bayes.
- Compares model accuracy, precision, recall, and F1 score, and reports confusion matrices and cross-validation results.
- Summarizes frequent complaint topics and phrases.
- Displays product, category, review, and customer insights.
- Recommends products using item-based collaborative filtering, category preferences, and review sentiment.
- Predicts sentiment and aspects for text entered by a user.

## Project Structure

```text
sentiment_recsys/
├── .gitignore
├── README.md
├── app.py                 # Root entry point for starting the Flask app
├── data/
│   └── reviews.csv        # Sample customer reviews
├── models/
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   └── naive_bayes.joblib
├── notebooks/
│   └── sentiment_analysis.ipynb
├── outputs/
│   ├── evaluation_report.json
│   └── sentiment_model_comparison.csv
├── requirements.txt       # Python package dependencies
└── src/
    ├── __init__.py
    ├── app.py             # Flask routes, model training, and recommender
    ├── generate_dataset.py  # Sample review dataset generator
    └── templates/
        └── index.html     # Web dashboard
```

When the app starts, it trains and saves the sentiment classifiers under `models/`, then writes evaluation metrics to `outputs/`. The notebook provides a separate place to explore the dataset and evaluation results.

## Requirements

- Python 3
- pip

## Installation and Setup (Windows PowerShell)

1. Clone the repository and move into the project directory:

   ```powershell
   git clone https://github.com/indrez-v/Big-data.git
   cd Big-data
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use the environment's Python directly in the following commands: `.\.venv\Scripts\python.exe`.

3. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Run the Application

From the project directory, with the virtual environment activated, run:

```powershell
python app.py
```

Wait for the startup message in the terminal, then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in a browser. Stop the server with `Ctrl+C`.

The app reads reviews from `data/reviews.csv`. If the file is missing, it generates a sample dataset automatically. To regenerate the dataset manually, run:

```powershell
python -m src.generate_dataset
```

## Explore the Analysis Notebook

Install Jupyter if it is not already available:

```powershell
python -m pip install notebook
```

Start Jupyter from the project directory and open `notebooks/sentiment_analysis.ipynb`:

```powershell
python -m notebook
```

The notebook imports the project analysis, so it also trains the models and refreshes the files in `models/` and `outputs/`.

## Dependencies

Dependencies are listed in `requirements.txt`: Flask, pandas, NumPy, scikit-learn, and joblib.

The dashboard loads Chart.js from a CDN, so an internet connection is needed to display its charts.

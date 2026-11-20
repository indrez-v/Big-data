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
├── app.py                 # Flask app, model training, analysis, recommender, and API routes
├── generate_dataset.py    # Creates the sample review dataset
├── requirements.txt       # Python package dependencies
├── data/
│   └── reviews.csv        # Sample customer reviews
└── templates/
    └── index.html         # Web dashboard
```

The sentiment models are trained when the app starts and kept in memory; trained model files are not currently saved to a `models/` directory.

## Requirements

- Python 3
- pip

## Installation and Setup (Windows PowerShell)

1. Clone the repository and move into the project directory:

   ```powershell
   git clone <repository-url>
   cd sentiment_recsys
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
python generate_dataset.py
```

## Dependencies

Dependencies are listed in `requirements.txt`: Flask, pandas, NumPy, and scikit-learn.

The dashboard loads Chart.js from a CDN, so an internet connection is needed to display its charts.

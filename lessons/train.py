from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"


def build_pipeline():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    report = classification_report(y_test, model.predict(x_test))
    return model, report


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    model, report = train(frame["text"], frame["label"])
    print(report)
    return model


if __name__ == "__main__":
    main()

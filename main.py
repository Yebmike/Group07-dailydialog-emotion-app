
import os
import re
import string
import json
import argparse
import numpy as np
import pandas as pd

# Streamlit (UI)
import streamlit as st

# Embeddings
import gensim.downloader as api

# Models & metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
import joblib

# Plots
import matplotlib.pyplot as plt

# TensorFlow / Keras for ANN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers

# preparing path for github and streamlit
from pathlib import Path
# -----------------------------
# Paths & constants
# -----------------------------

DATA_DIR = Path(__file__).parent / "data"
DIALOGUES_JSON = DATA_DIR / "dialogues.json"


# Canonical emotion set (7 classes)
EMOTION_TO_ID = {
    "no emotion": 0, "no_emotion": 0, "none": 0, "neutral": 0, "noemotion": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "happiness": 4, "joy": 4,
    "sadness": 5, "sad": 5,
    "surprise": 6
}
ID_TO_EMOTION = {
    0: "no_emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}
EMOTIONS = [ID_TO_EMOTION[i] for i in range(7)]

# ---------------------------------------
# Text processing utilities & embeddings
# ---------------------------------------
def simple_tokenize(s: str):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"\b\w+\b", s)

@st.cache_resource(show_spinner=False)
def load_glove(dim: int = 100):
    """Load GloVe vectors via gensim. Choose from 50/100/200/300 dims."""
    return api.load(f"glove-wiki-gigaword-{dim}")

def sentence_to_glove(tokens, glove, dim=100):
    vecs = [glove[t] for t in tokens if t in glove]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

@st.cache_data(show_spinner=True)
def build_embeddings(df: pd.DataFrame, glove_dim=100):
    glove = load_glove(glove_dim)
    xs = []
    for txt in df["text"].tolist():
        toks = simple_tokenize(txt)
        xs.append(sentence_to_glove(toks, glove, dim=glove_dim))
    X = np.vstack(xs)
    y = df["emotion"].values
    return X, y

# -----------------------------
# Local data loader (JSON)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dailydialog_local(json_path: str = DIALOGUES_JSON) -> pd.DataFrame:
    """
    Load DailyDialog from a local dialogues.json created from the dataset.
    Expected structure (typical):
      [
        {
          "data_split": "train" | "validation" | "test",
          "turns": [
            {"utterance": "...", "emotion": "happiness" | 4 | ... },
            ...
          ]
        },
        ...
      ]

    We output a DataFrame with columns:
      text: str  (utterance)
      emotion: int in [0..6] using our canonical mapping
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Could not find {json_path}. Place dialogues.json under ./data/"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for d in data:
        turns = d.get("turns", [])
        for t in turns:
            text = t.get("utterance", "")
            emo_val = t.get("emotion", "no emotion")

            # Normalize emotion to int id
            if isinstance(emo_val, int):
                # already id-like; clamp to [0..6] just in case you are asking
                emo_id = int(emo_val)
                if emo_id not in ID_TO_EMOTION:
                    emo_id = 0
            else:
                # string label; normalize and map
                key = str(emo_val).strip().lower()
                emo_id = EMOTION_TO_ID.get(key, 0)

            if text and text.strip():
                rows.append({"text": text.strip(), "emotion": emo_id})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Loaded dialogues.json but it contained no usable rows.")
    return df.reset_index(drop=True)

# -----------------------------
# Models
# -----------------------------
def train_logreg(X_train, y_train, C=2.0, max_iter=2000, random_state=42):
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

def build_ann(input_dim: int, num_classes: int = 7, hidden=128, dropout=0.2, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(hidden // 2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_ann(X_train, y_train, X_val, y_val, epochs=12, batch_size=128, hidden=128, dropout=0.2, lr=1e-3):
    model = build_ann(X_train.shape[1], len(EMOTIONS), hidden=hidden, dropout=dropout, lr=lr)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    return model, hist.history

# -----------------------------
# Model Evaluation helpers
# -----------------------------
def evaluate_classifier(name, clf, X_test, y_test):
    y_prob = clf.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0
    )

    lb = LabelBinarizer().fit(list(range(len(EMOTIONS))))
    y_test_bin = lb.transform(y_test)
    try:
        auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
        auc_micro = roc_auc_score(y_test_bin, y_prob, average="micro", multi_class="ovr")
    except ValueError:
        auc_macro, auc_micro = np.nan, np.nan

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(EMOTIONS))))
    return {
        "name": name,
        "report": report,
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "cm": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        ylabel="True label",
        xlabel="Predicted label",
        title=title
    )
    # annotate counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_roc_curves(y_test, y_prob, title):
    lb = LabelBinarizer().fit(list(range(len(EMOTIONS))))
    y_bin = lb.transform(y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(EMOTIONS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# -----------------------------
# Inference helper
# -----------------------------
def predict_with_glove(text: str, model, glove_dim=100):
    glove = load_glove(glove_dim)
    toks = simple_tokenize(text)
    vec = sentence_to_glove(toks, glove, dim=glove_dim).reshape(1, -1)
    probs = model.predict_proba(vec)
    idx = int(np.argmax(probs))
    return probs.flatten(), EMOTIONS[idx]

# -----------------------------
# Streamlit App
# -----------------------------
def run_app():
    st.set_page_config(page_title="DailyDialog Emotion Classifier (Local JSON)", page_icon="🧠", layout="wide")
    st.title("DailyDialog Emotion Detection • GloVe + LR / ANN")
    st.caption(" A JSON data in ./data/dialogues.json, Comparing Logistic Regression and ANN.")

    with st.expander("Page Info"):
        st.markdown(
            """
**Data source:** Local `data/dialogues.json`  
**Classes:** no_emotion, anger, disgust, fear, happiness, sadness, surprise  
**Embeddings:** GloVe (glove-wiki-gigaword) averaged per sentence  
**Metrics:** Precision, Recall, F1, ROC-AUC (macro, micro)  
"""
        )

    # Path controls
    st.sidebar.header("Data path")
    data_path = st.sidebar.text_input("dialogues.json path", DIALOGUES_JSON)

    # Data
    st.header("1) Load & Explore")
    try:
        df = load_dailydialog_local(data_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.sample(min(1000, len(df)), random_state=7).reset_index(drop=True))
    with c2:
        st.metric("Total utterances", len(df))
        counts = pd.Series(df["emotion"]).map(ID_TO_EMOTION).value_counts().reindex(EMOTIONS, fill_value=0)
        st.bar_chart(counts)

    # Embeddings
    st.header("2) Build GloVe Embeddings")
    glove_dim = st.selectbox("GloVe dimension", [50, 100, 200, 300], index=1)
    with st.spinner("Building embeddings..."):
        X, y = build_embeddings(df, glove_dim=glove_dim)
    st.success(f"Embeddings ready: {X.shape[0]} samples × {X.shape[1]} features")

    # Split
    st.header("3) Train / Val / Test Split")
    test_size = st.slider("Test size", 0.1, 0.3, 0.2, 0.05)
    val_size = st.slider("Validation size (from train)", 0.05, 0.2, 0.1, 0.05)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, random_state=42, stratify=y_train_full)
    st.write(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Train models
    st.header("4) Train Models")
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Logistic Regression")
        c_val = st.number_input("C (inverse regularization)", min_value=0.01, max_value=10.0, value=2.0, step=0.25)
        if st.button("Train Logistic Regression"):
            with st.spinner("Training..."):
                logreg = train_logreg(X_train, y_train, C=c_val)
            st.session_state["logreg"] = logreg
            st.success("Logistic Regression trained.")
    with cr:
        st.subheader("Artificial Neural Network")
        epochs = st.number_input("Epochs", 5, 50, 12, 1)
        batch = st.number_input("Batch size", 32, 512, 128, 32)
        hidden = st.number_input("Hidden units", 32, 512, 128, 32)
        drop = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05)
        if st.button("Train ANN"):
            with st.spinner("Training..."):
                ann, hist = train_ann(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch, hidden=hidden, dropout=drop)
            st.session_state["ann"] = ann
            st.session_state["ann_hist"] = hist
            st.success("ANN trained.")

    # Evaluate
    st.header("5) Evaluate And Compare")
    ready = [m for m in ["logreg", "ann"] if m in st.session_state]
    if not ready:
        st.info("Train at least one model to see evaluation.")
    else:
        tabs = st.tabs([m.upper() for m in ready])
        for mname, tab in zip(ready, tabs):
            with tab:
                if mname == "logreg":
                    clf = st.session_state["logreg"]
                    metrics = evaluate_classifier("LogReg", clf, X_test, y_test)
                else:
                    class KerasWrap:
                        def __init__(self, model): self.model = model
                        def predict_proba(self, X): return self.model.predict(X, verbose=0)
                    clf = KerasWrap(st.session_state["ann"])
                    metrics = evaluate_classifier("ANN", clf, X_test, y_test)

                # Summary
                st.subheader("Summary metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Macro F1", f"{metrics['report']['macro avg']['f1-score']:.3f}")
                c2.metric("Macro Precision", f"{metrics['report']['macro avg']['precision']:.3f}")
                c3.metric("Macro Recall", f"{metrics['report']['macro avg']['recall']:.3f}")
                st.metric("ROC-AUC (macro)", f"{metrics['auc_macro']:.3f}" if not np.isnan(metrics["auc_macro"]) else "N/A")
                st.metric("ROC-AUC (micro)", f"{metrics['auc_micro']:.3f}" if not np.isnan(metrics["auc_micro"]) else "N/A")

                # Report
                rep_df = pd.DataFrame(metrics["report"]).T
                st.write("Classification report")
                st.dataframe(rep_df.round(3))

                # Confusion matrix
                st.write("Confusion matrix")
                fig_cm = plot_confusion_matrix(metrics["cm"], f"{metrics['name']} — Confusion Matrix")
                st.pyplot(fig_cm)

                # ROC curves
                st.write("ROC curves (one-vs-rest)")
                fig_roc = plot_roc_curves(y_test, metrics["y_prob"], f"{metrics['name']} — ROC by Class")
                st.pyplot(fig_roc)

    # Inference
    st.header("6) Try it: classify a headline")
    headline = st.text_input("Enter a short headline", "Nation mourns after helicopter crash claims government officials")
    model_choice = st.selectbox(
        "Model for inference",
        ["Logistic Regression"] + (["ANN"] if "ann" in st.session_state else [])
    )
    if st.button("Predict"):
        if model_choice == "Logistic Regression":
            if "logreg" not in st.session_state:
                st.warning("Train Logistic Regression first.")
            else:
                probs, top = predict_with_glove(headline, st.session_state["logreg"], glove_dim=glove_dim)
                st.write("Top emotion:", f"**{top}**")
                st.bar_chart(pd.Series(probs, index=EMOTIONS))
        else:
            if "ann" not in st.session_state:
                st.warning("Train the ANN first.")
            else:
                glove = load_glove(glove_dim)
                vec = sentence_to_glove(simple_tokenize(headline), glove, dim=glove_dim).reshape(1, -1)
                probs = st.session_state["ann"].predict(vec, verbose=0).flatten()
                top = EMOTIONS[int(np.argmax(probs))]
                st.write("Top emotion:", f"**{top}**")
                st.bar_chart(pd.Series(probs, index=EMOTIONS))

    # Final caption with icon (JPEG)
    icon_path = DATA_DIR / "2.jpg"  # Sabi
    ICON_SIZE = 96

    col_i, col_t = st.columns([3, 20])
    with col_i:
        try:
            st.image(str(icon_path), width=ICON_SIZE)  # small icon
        except Exception:
            st.write("")  # ignore if missing
    with col_t:
        st.caption("hallelujah!!! 🙌🙌🙌.")



# -----------------------------
# CLI Trainer
# -----------------------------
def run_cli():
    os.makedirs("artifacts", exist_ok=True)
    print(f"Loading local data from {DIALOGUES_JSON} ...")
    df = load_dailydialog_local(DIALOGUES_JSON)
    X, y = build_embeddings(df, glove_dim=100)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = train_logreg(Xtr, ytr, C=2.0)
    yprob_lr = lr.predict_proba(Xte)
    ypred_lr = np.argmax(yprob_lr, axis=1)
    rep_lr = classification_report(yte, ypred_lr, target_names=EMOTIONS, zero_division=0, output_dict=True)
    lb = LabelBinarizer().fit(list(range(7)))
    auc_lr = roc_auc_score(lb.transform(yte), yprob_lr, average="macro", multi_class="ovr")
    print(f"LogReg Macro F1: {rep_lr['macro avg']['f1-score']:.3f}  Macro AUC: {auc_lr:.3f}")
    joblib.dump(lr, "artifacts/logreg_glove.joblib")

    # ANN
    print("Training ANN...")
    ann, _ = train_ann(Xtr, ytr, Xte[:len(Xte)//5], yte[:len(yte)//5], epochs=12, batch_size=128)
    yprob_nn = ann.predict(Xte, verbose=0)
    ypred_nn = np.argmax(yprob_nn, axis=1)
    rep_nn = classification_report(yte, ypred_nn, target_names=EMOTIONS, zero_division=0, output_dict=True)
    auc_nn = roc_auc_score(lb.transform(yte), yprob_nn, average="macro", multi_class="ovr")
    print(f"ANN   Macro F1: {rep_nn['macro avg']['f1-score']:.3f}  Macro AUC: {auc_nn:.3f}")
    ann.save("artifacts/ann_glove.keras")

    json.dump({"logreg": rep_lr, "ann": rep_nn}, open("artifacts/metrics.json", "w"))
    print("Artifacts saved in ./artifacts")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cli", action="store_true", help="Run CLI training and save artifacts")
    args, unknown = parser.parse_known_args()

    if args.train_cli:
        run_cli()
    else:
        # Let Streamlit take over when run via `streamlit run`.
        try:
            run_app()
        except SystemExit:
            pass

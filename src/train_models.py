import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


INPUT_FILES = {
    "premier_league": "../data/processed/premier_league_features.csv",
    "bundesliga": "../data/processed/bundesliga_features.csv",
    "league_1": "../data/processed/league_1_features.csv"
}

MODEL_DIR = "../models"
PLOT_DIR = "../plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Loaded: {df.shape}")
    return df


def prepare_data(df):

    y = df["FTR"].map({
        "A": 0,
        "D": 1,
        "H": 2
    })

    drop_cols = [
        "Date", "Div", "Time",
        "HomeTeam", "AwayTeam",
        "Referee", "Season", "FTR"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    leakage_cols = [
        "Over2.5",
        "HS",
        "HST",
        "AST"
    ]

    X = X.drop(columns=leakage_cols, errors="ignore")

    return X, y


def split_data(X, y):

    split = int(len(X) * (1 - TEST_SIZE))

    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:]
    )


def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def get_models():

    return {
        "Logistic": LogisticRegression(max_iter=3000),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(n_neighbors=15),

        "SVM": SVC(probability=True)
    }


def train_models(models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):

    probs = {}
    trained_models = {}
    results = []

    print("\nTraining models...\n")

    for name, model in models.items():

        print(f"{name}")

        if name == "XGBoost":
            X_train_xgb = X_train.copy()
            X_test_xgb = X_test.copy()

            X_train_xgb.columns = [str(i) for i in range(X_train.shape[1])]
            X_test_xgb.columns = [str(i) for i in range(X_test.shape[1])]

            model.fit(X_train_xgb, y_train)
            pred = model.predict(X_test_xgb)
            prob = model.predict_proba(X_test_xgb)

        elif name in ["Logistic", "KNN", "SVM"]:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)

        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average="weighted")
        recall = recall_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")

        print(f"{name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}\n")

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        probs[name] = prob
        trained_models[name] = model

    return trained_models, probs, results


def plot_confusion_matrix(
    y_true,
    y_pred,
    league_name,
    model_name
):

    cm = confusion_matrix(y_true, y_pred)

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,

        annot=True,
        fmt="d",

        annot_kws={
            "size": 16,
            "weight": "bold"
        },

        cmap=sns.color_palette(
            [
                "#D8F3DC",
                "#95D5B2",
                "#52B788",
                "#2D6A4F",
                "#1B4332"
            ],
            as_cmap=True
        )
    )

    plt.title(
        f"{league_name} — {model_name}",
        fontsize=22,
        weight="bold"
    )

    plt.xlabel(
        "Predicted",
        fontsize=15
    )

    plt.ylabel(
        "Actual",
        fontsize=15
    )

    plt.xticks(
        [0.5, 1.5, 2.5],
        ["Away Win", "Draw", "Home Win"],
        fontsize=12
    )

    plt.yticks(
        [0.5, 1.5, 2.5],
        ["Away Win", "Draw", "Home Win"],
        fontsize=12,
        rotation=0
    )

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"{league_name}_{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")

def plot_feature_importance(model, feature_names, title, league_name, model_name):

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    idx = np.argsort(importances)[::-1][:15]

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 6))

    plt.bar(
        range(len(idx)),
        importances[idx],
        color="#40916C"
    )

    plt.xticks(
        range(len(idx)),
        [feature_names[i] for i in idx],
        rotation=35,
        ha="right",
        fontsize=11
    )

    plt.yticks(fontsize=11)

    plt.title(
        title,
        fontsize=20,
        weight="bold"
    )

    plt.ylabel(
        "Importance",
        fontsize=14
    )

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"{league_name}_{model_name}_importance.png"
    )

    plt.savefig(
        save_path,
        dpi=400
    )

    plt.close()

    print(f"Saved: {save_path}")


def evaluate_ensemble(probs, y_test, league_name):

    avg_probs = np.mean(list(probs.values()), axis=0)
    final_pred = np.argmax(avg_probs, axis=1)

    acc = accuracy_score(y_test, final_pred)

    print("ENSEMBLE:", round(acc, 4))
    print("\nREPORT:")
    print(classification_report(y_test, final_pred))

    plot_confusion_matrix(y_test, final_pred, league_name, "XGBoost")

    return acc, final_pred


def save_models(models, scaler, feature_names, league_name):

    save_path = os.path.join(MODEL_DIR, f"{league_name}_ensemble.pkl")

    bundle = {
        "models": models,
        "scaler": scaler,
        "features": feature_names
    }

    joblib.dump(bundle, save_path)

    print(f"Saved model: {save_path}")


def process_league(name, path):

    print(f"\n==============================")
    print(f"LEAGUE: {name}")
    print(f"==============================")

    df = load_data(path)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    models = get_models()

    trained_models, probs, results = train_models(
        models,
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled
    )

    import pandas as pd

    results_df = pd.DataFrame(results)
    print("\nRESULTS:")
    print(results_df)

    acc, final_pred = evaluate_ensemble(probs, y_test, name)

    feature_names = list(X.columns)

    plot_feature_importance(
        trained_models["RandomForest"],
        feature_names,
        f"{name} - RandomForest Importance",
        name,
        "rf"
    )

    plot_feature_importance(
        trained_models["XGBoost"],
        feature_names,
        f"{name} - XGBoost Importance",
        name,
        "xgb"
    )

    save_models(trained_models, scaler, feature_names, name)


def main():

    for name, path in INPUT_FILES.items():
        process_league(name, path)


if __name__ == "__main__":
    main()
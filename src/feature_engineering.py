import pandas as pd
import numpy as np
import os

from mrmr import mrmr_classif
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch


INPUT_FILES = {
    "premier_league": "../data/processed/premier_league_clean.csv",
    "bundesliga": "../data/processed/bundesliga_clean.csv",
    "league_1": "../data/processed/league_1_clean.csv"
}

OUTPUT_DIR = "../data/processed"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Season", "Date"]).reset_index(drop=True)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Over2.5"] = (df["FTHG"] + df["FTAG"] > 2).astype(int)
    return df


def create_team_features(df: pd.DataFrame) -> pd.DataFrame:

    df["HomeOver2.5Perc"] = df.groupby(
        ["Season", "HomeTeam"]
    )["Over2.5"].transform("mean") * 100

    df["AwayOver2.5Perc"] = df.groupby(
        ["Season", "AwayTeam"]
    )["Over2.5"].transform("mean") * 100

    return df


def create_last5_features(df: pd.DataFrame) -> pd.DataFrame:

    df["AvgLast5HomeGoalsScored"] = df.groupby(
        ["Season", "HomeTeam"]
    )["FTHG"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["AvgLast5HomeGoalsConceded"] = df.groupby(
        ["Season", "HomeTeam"]
    )["FTAG"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["Last5HomeOver2.5Count"] = df.groupby(
        ["Season", "HomeTeam"]
    )["Over2.5"].transform(lambda x: x.rolling(5, min_periods=1).sum())

    df["Last5HomeOver2.5Perc"] = df.groupby(
        ["Season", "HomeTeam"]
    )["Over2.5"].transform(lambda x: x.rolling(5, min_periods=1).mean() * 100)

    df["AvgLast5AwayGoalsScored"] = df.groupby(
        ["Season", "AwayTeam"]
    )["FTAG"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["AvgLast5AwayGoalsConceded"] = df.groupby(
        ["Season", "AwayTeam"]
    )["FTHG"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["Last5AwayOver2.5Count"] = df.groupby(
        ["Season", "AwayTeam"]
    )["Over2.5"].transform(lambda x: x.rolling(5, min_periods=1).sum())

    df["Last5AwayOver2.5Perc"] = df.groupby(
        ["Season", "AwayTeam"]
    )["Over2.5"].transform(lambda x: x.rolling(5, min_periods=1).mean() * 100)

    return df


def select_features(df: pd.DataFrame):

    df = df.drop(columns=["FTHG", "FTAG", "HTHG", "HTAG"], errors="ignore")

    df = df.dropna(axis=1, thresh=int(0.7 * len(df)))

    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    if "Over2.5" in num_cols:
        num_cols.remove("Over2.5")

    X = df[num_cols]
    y = df["Over2.5"]

    selected = mrmr_classif(X=X, y=y, K=20)
    print("mRMR selected:", selected)

    scaler = StandardScaler()
    df[selected] = scaler.fit_transform(df[selected])

    corr = df[selected].corr(method="spearman")
    dist = sch.distance.pdist(corr)
    linkage = sch.linkage(dist, method="average")

    cluster_ids = sch.fcluster(linkage, 0.5, criterion="distance")

    final_features = []

    for cid in pd.Series(cluster_ids).unique():
        cluster = corr.columns[pd.Series(cluster_ids) == cid]
        best = cluster[np.argmax(df[cluster].var())]
        final_features.append(best)

    print("FINAL FEATURES:", final_features)

    return df, final_features


def save_dataset(df: pd.DataFrame, features: list, output_path: str):

    cat_cols = [
        "Div", "Time",
        "HomeTeam", "AwayTeam",
        "FTR", "HTR",
        "Referee", "Season"
    ]

    available_cols = ["Date"] + cat_cols + features + ["Over2.5"]
    available_cols = [c for c in available_cols if c in df.columns]

    final_df = df[available_cols]

    final_df = final_df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
    final_df = final_df.fillna(0)

    final_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Final shape: {final_df.shape}")


def process_league(name: str, input_path: str):

    print(f"\nFeature engineering: {name}")

    df = load_data(input_path)

    df = add_target(df)
    df = create_team_features(df)
    df = create_last5_features(df)

    df, features = select_features(df)

    output_path = os.path.join(OUTPUT_DIR, f"{name}_features.csv")

    save_dataset(df, features, output_path)


def main():

    for name, path in INPUT_FILES.items():
        process_league(name, path)


if __name__ == "__main__":
    main()
import pandas as pd
import os
from typing import List


BASE_PATH = "../data/raw"

LEAGUES = {
    "Premier_League": "premier_league_clean.csv",
    "Bundesliga": "bundesliga_clean.csv",
    "League_1": "league_1_clean.csv"
}

OUTPUT_DIR = "../data/processed"


def get_csv_files(path: str) -> List[str]:
    files = []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            files.append(os.path.join(path, file))

    print(f"Found {len(files)} CSV files in {path}")
    return sorted(files)


def load_league(path: str) -> pd.DataFrame:
    files = get_csv_files(path)

    df_list = []

    for file in files:
        print(f"Reading: {file}")
        temp = pd.read_csv(file)

        if not temp.empty:
            df_list.append(temp)

    if not df_list:
        raise ValueError(f"No valid data in {path}")

    df = pd.concat(df_list, ignore_index=True).copy()

    print(f"Combined shape: {df.shape}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df["Date"].astype(str).str.strip()

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    print(f"NaT: {df['Date'].isna().sum()}")
    return df


def get_season(date):
    if pd.isna(date):
        return None

    if date.month >= 8:
        return f"{date.year}/{date.year+1}"
    return f"{date.year-1}/{date.year}"

def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df["Season"] = df["Date"].apply(get_season)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    after = len(df)
    print(f"Removed rows: {before - after}")

    return df


def save(df: pd.DataFrame, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Final shape: {df.shape}")


def process_league(league_name: str, filename: str):

    print(f"\nProcessing: {league_name}")

    path = os.path.join(BASE_PATH, league_name)

    if not os.path.exists(path):
        print(f"⚠Missing folder: {path}")
        return

    df = load_league(path)

    df = parse_dates(df)
    df = add_season(df)
    df = clean(df)

    df = df.sort_values("Date").reset_index(drop=True)

    save(df, filename)


def main():

    for league, filename in LEAGUES.items():
        process_league(league, filename)


if __name__ == "__main__":
    main()
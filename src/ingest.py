import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os


load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def load_raw_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case
    'Gr Liv Area' → 'gr_liv_area'
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )
    return df

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Order and PID — these are just row identifiers, not features."""
    return df.drop(columns=["order", "pid"], errors="ignore")

def save_to_postgres(df: pd.DataFrame, table_name: str):
    """
    Load the cleaned DataFrame into PostgreSQL.
    if_exists='replace' drops and recreates the table each run —
    good for development, you'd use 'append' in production pipelines.
    """
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Saved {len(df)} rows to table '{table_name}'")

if __name__ == "__main__":
    RAW_PATH = "data/raw/AmesHousing.csv"

    df = load_raw_data(RAW_PATH)
    df = clean_column_names(df)
    df = drop_id_columns(df)

    # Save raw (but cleaned) version to postgres
    save_to_postgres(df, "housing_raw")

    # Also save a local copy to data/processed/ for reference
    df.to_csv("data/processed/housing_cleaned.csv", index=False)
    print("Done — data is in PostgreSQL and data/processed/")
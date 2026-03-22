import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# LOAD FROM POSTGRESQL
load_dotenv()
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
df = pd.read_sql("SELECT * FROM housing_raw", engine)

# CLEAN + ENGINEER
cols_to_drop = ["pool_qc", "misc_feature", "alley", "fence", "misc_val"]
df = df.drop(columns=cols_to_drop, errors="ignore")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(0)

df["house_age"] = df["yr_sold"] - df["year_built"]
df["remod_age"] = df["yr_sold"] - df["year_remod_add"]
df["total_sf"]  = df["total_bsmt_sf"] + df["gr_liv_area"]

#  RUN MODEL TO GET PREDICTIONS 

neighborhood_labels = df["neighborhood"].copy()

le = LabelEncoder()
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

X = df_encoded.drop(columns=["saleprice"])
y = df_encoded["saleprice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# BUILD DASHBOARD CSV 

dashboard_full = pd.DataFrame({
    "neighborhood"  : neighborhood_labels,
    "saleprice"     : df["saleprice"],
    "overall_qual"  : df["overall_qual"],
    "total_sf"      : df["total_sf"],
    "year_built"    : df["year_built"],
    "house_age"     : df["house_age"],
    "gr_liv_area"   : df["gr_liv_area"],
    "garage_cars"   : df["garage_cars"],
    "yr_sold"       : df["yr_sold"],
})
dashboard_full.to_csv("data/processed/dashboard_full.csv", index=False)
print(f"Saved dashboard_full.csv — {len(dashboard_full)} rows")

# Actual vs predicted for view 3
predictions = pd.DataFrame({
    "actual"        : y_test.values,
    "predicted"     : y_pred.round(0),
    "error"         : y_test.values - y_pred.round(0),
    "abs_error"     : abs(y_test.values - y_pred.round(0)),
    "neighborhood"  : neighborhood_labels.iloc[y_test.index].values,
    "overall_qual"  : df["overall_qual"].iloc[y_test.index].values,
    "total_sf"      : df["total_sf"].iloc[y_test.index].values,
})
predictions.to_csv("data/processed/dashboard_predictions.csv", index=False)
print(f"Saved dashboard_predictions.csv — {len(predictions)} rows")

print("\nAll dashboard data exported to data/processed/")
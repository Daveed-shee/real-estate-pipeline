import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

#LOAD DATA FROM POSTGRESQL

load_dotenv()
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
df = pd.read_sql("SELECT * FROM housing_raw", engine)
print(f"Loaded {len(df)} rows from PostgreSQL")


# DROP HIGH-NULL COLUMNS


cols_to_drop = ["pool_qc", "misc_feature", "alley", "fence", "misc_val"]
df = df.drop(columns=cols_to_drop, errors="ignore")


# HANDLE REMAINING NULLS


for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(0)


# FEATURE ENGINEERING


df["house_age"]       = df["yr_sold"] - df["year_built"]      # age at time of sale
df["remod_age"]       = df["yr_sold"] - df["year_remod_add"]  # years since last remodel
df["total_sf"]        = df["total_bsmt_sf"] + df["gr_liv_area"]  # total livable sqft
df["price_per_sqft"]  = df["saleprice"] / df["total_sf"].replace(0, np.nan)  # unit price


# ENCODE CATEGORICAL COLUMNS

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))


# DEFINE FEATURES AND TARGET 

X = df.drop(columns=["saleprice", "price_per_sqft"])
y = df["saleprice"]


# TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")


# TRAIN THE MODEL 


model = GradientBoostingRegressor(
    n_estimators=300,    # number of trees
    learning_rate=0.05,  # how much each tree corrects — smaller = more careful
    max_depth=4,         # how deep each tree grows — controls overfitting
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained")


# EVALUATE

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE : ${rmse:,.0f}")
print(f"  R²   : {r2:.4f}")


# SAVE PREDICTIONS 


results = pd.DataFrame({"actual": y_test, "predicted": y_pred.round(0)})
results["error"] = results["actual"] - results["predicted"]
results.to_csv("data/processed/predictions.csv", index=False)
print("Predictions saved to data/processed/predictions.csv")


# FEATURE IMPORTANCE PLOT

feat_imp = pd.Series(model.feature_importances_, index=X.columns)
top15    = feat_imp.nlargest(15).sort_values()

plt.figure(figsize=(8, 6))
top15.plot(kind="barh", color="steelblue")
plt.title("Top 15 features by importance")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("dashboard/screenshots/feature_importance.png", dpi=150)
plt.show()
print("Feature importance plot saved")
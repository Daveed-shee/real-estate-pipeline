import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import shap

# REBUILD THE MODEL


load_dotenv()
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

df = pd.read_sql("SELECT * FROM housing_raw", engine)

cols_to_drop = ["pool_qc", "misc_feature", "alley", "fence", "misc_val"]
df = df.drop(columns=cols_to_drop, errors="ignore")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(0)

df["house_age"]  = df["yr_sold"] - df["year_built"]
df["remod_age"]  = df["yr_sold"] - df["year_remod_add"]
df["total_sf"]   = df["total_bsmt_sf"] + df["gr_liv_area"]
df["price_per_sqft"] = df["saleprice"] / df["total_sf"].replace(0, np.nan)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["saleprice", "price_per_sqft"])
y = df["saleprice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
)
model.fit(X_train, y_train)
print("Model rebuilt — generating SHAP explanations...")


# BUILD SHAP EXPLAINER 


explainer   = shap.TreeExplainer(model)
sample      = X_test.sample(200, random_state=42)
shap_values = explainer.shap_values(sample)


# SUMMARY PLOT 

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("dashboard/screenshots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP summary plot saved")


# WATERFALL PLOT FOR ONE HOUSE 


explanation = shap.Explanation(
    values        = shap_values[0],
    base_values   = float(np.array(explainer.expected_value).flatten()[0]),
    data          = sample.iloc[0],
    feature_names = X.columns.tolist()
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig("dashboard/screenshots/shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP waterfall plot saved")


# DEPENDENCE PLOT — overall_qual


plt.figure()
shap.dependence_plot("overall_qual", shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("dashboard/screenshots/shap_dependence_qual.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP dependence plot saved")

print("\nAll SHAP charts saved to dashboard/screenshots/")
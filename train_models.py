import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATIENT ADMISSION FORECASTING
# MODEL TRAINING PIPELINE
# ============================================================

print("=" * 65)
print("PATIENT ADMISSION FORECASTING")
print("MODEL TRAINING STARTED")
print("=" * 65)


# ============================================================
# 1. LOAD DATA
# ============================================================

data_path = "data/model_dataset.csv"

df = pd.read_csv(
    data_path,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

print("\nDataset loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "day",
    "month",
    "year",
    "day_of_week",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos"
]

target = "admissions"

X = df[features]
y = df[target]


# ============================================================
# 3. TIME-BASED TRAIN TEST SPLIT
# ============================================================

# We do NOT randomly shuffle the data.
# Earlier dates = training
# Later dates = testing

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTRAINING ROWS:", len(X_train))
print("TESTING ROWS :", len(X_test))

print(
    "\nTraining period:",
    df["date"].iloc[0].date(),
    "to",
    df["date"].iloc[split_index - 1].date()
)

print(
    "Testing period :",
    df["date"].iloc[split_index].date(),
    "to",
    df["date"].iloc[-1].date()
)


# ============================================================
# 4. CREATE FOUR ML MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=6,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "-" * 65)
    print("TRAINING:", name)
    print("-" * 65)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # R2
    r2 = r2_score(
        y_test,
        predictions
    )

    # Store
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Lower RMSE is better for selecting the model
results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)


print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_mae = results_df.iloc[0]["MAE"]
best_rmse = results_df.iloc[0]["RMSE"]
best_r2 = results_df.iloc[0]["R2"]


print("\n")
print("=" * 65)
print("SELECTED MODEL")
print("=" * 65)

print("Model :", best_model_name)
print("MAE   :", round(best_mae, 4))
print("RMSE  :", round(best_rmse, 4))
print("R2    :", round(best_r2, 4))


# ============================================================
# 8. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

model_package = {

    "model": best_model,

    "model_name":
        best_model_name,

    "features":
        features,

    "last_date":
        df["date"].max(),

    "last_admissions":
        float(df["admissions"].iloc[-1])
}


joblib.dump(
    model_package,
    "models/best_model.pkl"
)


# ============================================================
# 10. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ============================================================
# 11. SAVE TEST PREDICTIONS
# ============================================================

test_df = df.iloc[split_index:].copy()

test_df["predicted_admissions"] = (
    best_model.predict(X_test)
)

test_df.to_csv(
    "outputs/test_predictions.csv",
    index=False
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance":
            best_model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    print("\nFeature importance saved.")


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nCreated files:")

print("models/best_model.pkl")
print("outputs/model_comparison.csv")
print("outputs/test_predictions.csv")

if hasattr(best_model, "feature_importances_"):
    print("outputs/feature_importance.csv")

print("\nNext step:")
print("7-DAY ADMISSION FORECAST + STREAMLIT DASHBOARD")

print("=" * 65)
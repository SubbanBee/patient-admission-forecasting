import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PATIENT ADMISSION FORECASTING
# 7-DAY FORECAST
# ============================================================

print("=" * 65)
print("PATIENT ADMISSION FORECASTING")
print("7-DAY FORECAST")
print("=" * 65)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/daily_admissions.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

print("\nHistorical observations:", len(df))

print(
    "Last historical date:",
    df["date"].max().date()
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

package = joblib.load(
    "models/best_model.pkl"
)

model = package["model"]
model_name = package["model_name"]
features = package["features"]


print("Loaded model:", model_name)


# ============================================================
# 3. CREATE WORKING HISTORY
# ============================================================

history = df[
    ["date", "admissions"]
].copy()


# ============================================================
# 4. FORECAST FUNCTION
# ============================================================

def create_features(date, history):

    month = date.month
    day = date.day
    year = date.year
    day_of_week = date.dayofweek

    # Previous observed admission
    lag_1 = history["admissions"].iloc[-1]

    # Seven observations previously
    lag_7 = history["admissions"].iloc[-7]

    # Previous seven observations
    rolling_mean_7 = (
        history["admissions"]
        .iloc[-7:]
        .mean()
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    dow_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    dow_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    return pd.DataFrame([{

        "day": day,

        "month": month,

        "year": year,

        "day_of_week": day_of_week,

        "lag_1": lag_1,

        "lag_7": lag_7,

        "rolling_mean_7": rolling_mean_7,

        "month_sin": month_sin,

        "month_cos": month_cos,

        "dow_sin": dow_sin,

        "dow_cos": dow_cos

    }])[features]


# ============================================================
# 5. GENERATE 7-DAY FORECAST
# ============================================================

forecast_rows = []

last_date = history["date"].max()

for i in range(1, 8):

    future_date = (
        last_date +
        pd.Timedelta(days=i)
    )

    X_future = create_features(
        future_date,
        history
    )

    prediction = model.predict(
        X_future
    )[0]

    # Admissions cannot be negative
    prediction = max(
        0,
        prediction
    )

    prediction = round(
        prediction,
        2
    )

    forecast_rows.append({

        "date": future_date,

        "predicted_admissions":
            prediction

    })

    # Add prediction to history
    history.loc[len(history)] = [
        future_date,
        prediction
    ]


# ============================================================
# 6. CREATE FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecast_rows
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("7-DAY ADMISSION FORECAST")
print("=" * 65)

print(
    forecast_df.to_string(
        index=False
    )
)


# ============================================================
# 8. FORECAST SUMMARY
# ============================================================

total_forecast = (
    forecast_df[
        "predicted_admissions"
    ].sum()
)

average_forecast = (
    forecast_df[
        "predicted_admissions"
    ].mean()
)

peak_day = forecast_df.loc[
    forecast_df[
        "predicted_admissions"
    ].idxmax()
]


print("\n")
print("TOTAL EXPECTED ADMISSIONS:",
      round(total_forecast, 2))

print("AVERAGE DAILY ADMISSIONS:",
      round(average_forecast, 2))

print(
    "PEAK FORECAST DATE:",
    peak_day["date"].date()
)

print(
    "PEAK FORECAST:",
    round(
        peak_day[
            "predicted_admissions"
        ],
        2
    )
)


# ============================================================
# 9. SAVE FORECAST
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

forecast_df.to_csv(
    "outputs/7_day_forecast.csv",
    index=False
)


print("\n")
print("=" * 65)
print("FORECAST SAVED")
print("=" * 65)

print(
    "outputs/7_day_forecast.csv"
)
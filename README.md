# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.The dataset is first loaded and cleaned by removing unnecessary spaces, converting the time column into datetime format, sorting the data in chronological order, and filling missing values using interpolation.
2. Next, feature engineering is performed by creating time-based features such as sine and cosine of the hour, and generating lag features to include previous values of temperature, pollution, and solar radiation. Rows with missing values after this process are removed.
3. Then, the data is split into training and testing sets in an 80:20 ratio without shuffling, and a Random Forest model is trained using the training data to predict the target variables.
4. Finally, the model is evaluated using performance metrics like R² score and mean absolute error, graphs are plotted for visualization, and the trained model is used to predict future values based on the latest available data
    

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: VARSHA M
RegisterNumber:25001006 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()


df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)


cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)


df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)


processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)


features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]
# Print summary of feature engineering
print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)


split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]

models = {}
results = {}


target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:
    y_train, y_test = train[target], test[target]
    
    # Random Forest with high-depth logic for complex weather patterns
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    models[target] = model
    
    # Store metrics for interpretation
    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }


fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]
    
    axes[i, 0].plot(res['actual'][-150:], label='Actual', color='black', alpha=0.4, linewidth=2)
    axes[i, 0].plot(res['preds'][-150:], label='Predicted', color=color, linestyle='--', linewidth=2)
    axes[i, 0].set_title(f"{label}: Actual vs Predicted\n$R^2$: {res['r2']:.3f} | MAE: {res['mae']:.2f}")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    importances = pd.Series(models[target].feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', ax=axes[i, 1], color=color, alpha=0.7)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()

last_row = processed_df.iloc[-1]
latest_data = pd.DataFrame([{
    'hum': last_row['hum'], 'pressure': last_row['pressure'], 'wind_speed': last_row['wind_speed'],
    'illumination': last_row['illumination'], 'co2': last_row['co2'],
    'hour_sin': last_row['hour_sin'], 'hour_cos': last_row['hour_cos'],
    'tem_lag1': last_row['tem'], 'pm2_5_lag1': last_row['pm2_5'], 'tsr_lag1': last_row['tsr']
}])

print("\n--- NEXT STEP PREDICTIONS (Using Latest Data) ---")
for target in targets:
    pred_val = models[target].predict(latest_data)[0]
    print(f"Predicted {target_meta[target][0]}: {pred_val:.2f} {target_meta[target][1]}")

*/
```
## Output:
<img width="1233" height="465" alt="image" src="https://github.com/user-attachments/assets/f3e29d96-2629-4c48-817f-83929a868b62" />
<img width="1197" height="462" alt="image" src="https://github.com/user-attachments/assets/65b60395-8bdf-430e-a0ba-f37d17403c91" />
<img width="1310" height="595" alt="image" src="https://github.com/user-attachments/assets/96b6a954-ee3d-412b-8038-fc5a46eaee4c" />


## Result:

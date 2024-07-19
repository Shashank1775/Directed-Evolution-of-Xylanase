import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load and preprocess the data
df = pd.read_csv(r'C:\Users\shash\Downloads\ChatBoxPython\CodingProjects\XylaneFinalData\trainingdata3')
print(df.shape)
def remove_duplicate_columns(df):
    base_names = df.columns.str.replace(r'\.\d+', '', regex=True)
    seen = set()
    keep_columns = []
    for col, base in zip(df.columns, base_names):
        if base not in seen:
            keep_columns.append(col)
            seen.add(base)
    return df[keep_columns]

df_unique = remove_duplicate_columns(df)
df_unique = df_unique.loc[:, ~df_unique.columns.duplicated()]
df = df_unique.drop(['OGT_Source', 'Topt_Source', 'ogt_groups'], axis=1)
df = df.dropna()
df = df.sample(frac=1, random_state=121).reset_index(drop=True)

labels = df["Topt"]
features = df.drop(["Topt"], axis=1)
print(features.columns)
# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)
print(features.shape)
# Split the data
Topt_features_train, Topt_features_test, Topt_labels_train, Topt_labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the models
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
}

# Evaluate models
results = {}
for model_name, model in models.items():
    model.fit(Topt_features_train, Topt_labels_train)
    predictions = model.predict(Topt_features_test)
    
    mse = mean_squared_error(Topt_labels_test, predictions)
    mae = mean_absolute_error(Topt_labels_test, predictions)
    rmse = mean_squared_error(Topt_labels_test, predictions, squared=False)
    r2 = r2_score(Topt_labels_test, predictions)
    
    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R^2): {r2}")
    print("\n")
    
    features =  model.feature_importances_
    for i in features:
        print(i)

best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]
joblib_file = f"best_{best_model_name.replace(' ', '_').lower()}_model.pkl"
joblib.dump(best_model, joblib_file)
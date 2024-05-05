import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer  # Import SimpleImputer from sklearn.impute
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.metrics import r2_score

# Your code continues...


# Load data
def load_data():
    housing = pd.read_csv("Real-Estate-House-Price-Prediction\housingdata.csv")
    return housing

# Train-test split
def train_test_split_data(data, test_size=0.2):
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=42)
    return train_set, test_set

# Stratified train-test split
def stratified_train_test_split(data):
    splt = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in splt.split(data, data['CHAS']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set

# Feature scaling pipeline
def create_pipeline():
    my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scalar', StandardScaler()),
    ])
    return my_pipeline

# Model training
def train_model(data, labels):
    model = RandomForestRegressor()  # You can change the model here if needed
    model.fit(data, labels)
    return model

# Evaluate model using cross-validation
def evaluate_model(model, data, labels):
    scores = cross_val_score(model, data, labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

# Load saved model
def load_saved_model(filename):
    model = load(filename)
    return model

# Save trained model
def save_model(model, filename):
    dump(model, filename)

# Generate output function
def generate_output():
    # Load data
    housing = load_data()

    # Train-test split
    train_set, test_set = train_test_split_data(housing)
    
    # Stratified train-test split
    strat_train_set, strat_test_set = stratified_train_test_split(housing)

    # Feature scaling pipeline
    my_pipeline = create_pipeline()
    
    # Prepare data
    housing_prepared = my_pipeline.fit_transform(strat_train_set.drop("MEDV", axis=1))
    housing_labels = strat_train_set["MEDV"].copy()
    
    # Train model
    model = train_model(housing_prepared, housing_labels)
    
    # Evaluate model
    rmse_scores = evaluate_model(model, housing_prepared, housing_labels)
    mean_rmse = rmse_scores.mean()
    
    return f"Mean RMSE: {mean_rmse}"

if __name__ == "__main__":
    print(generate_output())

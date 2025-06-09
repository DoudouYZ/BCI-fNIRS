import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Preprocessing import get_epochs

epochs = get_epochs()
# Define a seed for reproducibility
seed = 42
np.random.seed(seed)
# Extract features as before (for example: averaging the signal over a time window)
tmin_window, tmax_window = 0, 10
data = epochs.copy().crop(tmin=tmin_window, tmax=tmax_window).get_data()
features = data.mean(axis=2)  # [n_epochs, n_channels]
labels = epochs.events[:, -1]  # getting labels from epoch events

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build a pipeline with StandardScaler, PCA and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Dimensionality reduction
    ('classifier', RandomForestClassifier(random_state=42))
])

# Setup a parameter grid to search over PCA components and Random Forest hyperparameters
param_grid = {
    'pca__n_components': [None, 5, 10],  # if None, all components are kept
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on the test set
test_score = grid_search.score(X_test, y_test)
print("Test set score:", test_score)


import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from Preprocessing import get_epochs
import matplotlib.pyplot as plt
import seaborn as sns

epochs = get_epochs()
# Define a seed for reproducibility
seed = 42
np.random.seed(seed)
# Extract features as before (for example: averaging the signal over a time window)
tmin_window, tmax_window = 0, 10
data = epochs.copy().crop(tmin=tmin_window, tmax=tmax_window).get_data()
features = data.mean(axis=2)  # [n_epochs, n_channels]
print(data.shape, features.shape)
quit()
# Scale the features (it's important to scale before PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
labels = epochs.events[:, -1]  # getting labels from epoch events

# Set the number of PCA components (adjust this as needed)
n_components = 5
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA components
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['Label'] = labels  # Add labels for coloring

# Use Seaborn's pairplot to create a grid of all components plotted against each other
sns.pairplot(df_pca, hue='Label', diag_kind='kde')
plt.suptitle("Pairwise PCA Components", y=1.02)
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build a pipeline with StandardScaler, PCA and RandomForestClassifier
pipeline_PCA = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Dimensionality reduction
    ('feature_selection', SelectKBest(score_func=f_classif)),  # Select top k features based on ANOVA F-values
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid_pca = {
    'feature_selection__k': [5, 10, 'all'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10]
}


# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline_PCA, param_grid_pca, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on the test set
test_score = grid_search.score(X_test, y_test)
print("Test set score:", test_score)
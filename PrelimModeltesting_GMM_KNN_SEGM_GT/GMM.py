import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

file_name = "./Heart_Disease_Classification_Dataset_Cleaned.csv"
data = pd.read_csv(file_name)


if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

X = data.drop(columns=['target'])
y = data['target']

# handle missing values by imputing with the mean of each feature
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# njromali the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

def evaluate_gmm(X, y, n_components_list, covariance_types):
    best_model = None
    best_score = float('inf')
    best_params = None
    results = []

    for n_components in n_components_list:
        for covariance_type in covariance_types:
            try:
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
                gmm.fit(X)
                
                # Predict cluster membership
                cluster_labels = gmm.predict(X)
                
                # Metrics
                aic = gmm.aic(X)
                bic = gmm.bic(X)
                sil_score = silhouette_score(X, cluster_labels) if n_components > 1 else -1
                mse = mean_squared_error(y, cluster_labels)  # Treat the cluster as a prediction

                results.append({
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'aic': aic,
                    'bic': bic,
                    'silhouette_score': sil_score,
                    'mse': mse
                })
                
                # Select the best model based on the lowest BIC
                if bic < best_score:
                    best_score = bic
                    best_model = gmm
                    best_params = (n_components, covariance_type)
            except Exception as e:
                print(f"Error with n_components={n_components}, covariance_type={covariance_type}: {e}")

    return best_model, best_params, results

n_components_list = range(1, 11)  # Try 1 to 10 components
covariance_types = ['full', 'tied', 'diag', 'spherical']  # Different covariance types

best_gmm, best_params, evaluation_results = evaluate_gmm(X_scaled, y, n_components_list, covariance_types)

print(f"Best GMM parameters: n_components={best_params[0]}, covariance_type={best_params[1]}")
print("\nEvaluation Results:")
for result in evaluation_results:
    print(result)

results_df = pd.DataFrame(evaluation_results)
results_df.to_csv('PrelimModeltesting_GMM_KNN_SEGM_GT/gmm_evaluation_results.csv', index=False)



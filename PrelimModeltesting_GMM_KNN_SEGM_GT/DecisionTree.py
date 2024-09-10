import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_name = "./Heart_Disease_Classification_Dataset_Cleaned.csv"
data = pd.read_csv(file_name)


if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

X = data.drop(columns=['target'])
y = data['target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def evaluate_decision_tree(X_train, X_test, y_train, y_test, max_depth_list, min_samples_split_list, criteria_list):
    best_model = None
    best_score = 0
    best_params = None
    results = []

    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            for criterion in criteria_list:
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    random_state=42
                )
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                results.append({
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'criterion': criterion,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                if f1 > best_score:
                    best_score = f1
                    best_model = dt
                    best_params = (max_depth, min_samples_split, criterion)

    return best_model, best_params, results

max_depth_list = range(1, 11)  # Try tree depths from 1 to 10
min_samples_split_list = [2, 5, 10]  # Try different minimum samples for a split
criteria_list = ['gini', 'entropy']  # Different split quality criteria

# eveaulate the models
best_dt, best_params, evaluation_results = evaluate_decision_tree(X_train, X_test, y_train, y_test, max_depth_list, min_samples_split_list, criteria_list)

print(f"Best Decision Tree parameters: max_depth={best_params[0]}, min_samples_split={best_params[1]}, criterion={best_params[2]}")
print("\nEvaluation Results:")
for result in evaluation_results:
    print(result)

results_df = pd.DataFrame(evaluation_results)
results_df.to_csv('PrelimModeltesting_GMM_KNN_SEGM_GT/decision_tree_evaluation_results.csv', index=False)


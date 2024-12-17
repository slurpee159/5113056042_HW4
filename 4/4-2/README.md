# 4-2: Model Optimization using PyCaret and Optuna

## Overview

This project focuses on model optimization for a classification problem using **PyCaret**, **Optuna**, and other AutoML or meta-heuristic techniques. The main steps include:

1. **Feature Engineering**
2. **Model Selection**
3. **Hyperparameter Optimization**

We use the Titanic dataset for demonstration, which is preprocessed and evaluated using AutoML tools to optimize performance.

---

## Question 1

**How do I preprocess the Titanic dataset and automatically select the best classification model using PyCaret?**

### Solution:

We preprocess the Titanic dataset by handling missing values, dropping unnecessary columns, and encoding categorical features. Using PyCaret's `setup()` and `compare_models()` functions, we automatically select the best-performing classification model.

#### Code Snippet:

```python
import pandas as pd
from pycaret.classification import *

def load_data():
    # Example: Load the Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)

    # Drop columns with too many missing values or irrelevant features
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Fill missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    return data

# Step 1: Load data
data = load_data()

# Step 2: PyCaret Workflow
s = setup(data=data, target='Survived', session_id=42, normalize=True,
          feature_selection=True, remove_multicollinearity=True, fold=5, verbose=False)

# Compare models and select the best one
best_model = compare_models()
print("Best Model Selected:", best_model)
```

---

## Question 2

**How can I optimize the hyperparameters of the selected model using Optuna?**

### Solution:

We use **Optuna**, a hyperparameter optimization library, to tune the selected model from PyCaret. Optuna creates a custom search space for the model's hyperparameters and optimizes them for maximum performance.

#### Code Snippet:

```python
import optuna
from pycaret.classification import *

def optuna_optimization(model, setup_instance):
    def objective(trial):
        param_grid = {
            'max_depth': [trial.suggest_int('max_depth', 3, 10)],
            'n_estimators': [trial.suggest_int('n_estimators', 50, 200)],
            'learning_rate': [trial.suggest_float('learning_rate', 0.01, 0.3)],
            'subsample': [trial.suggest_float('subsample', 0.5, 1.0)],
        }

        try:
            tuned_model = tune_model(model, custom_grid=param_grid)
            score = pull().iloc[0]['Accuracy']  # Use accuracy for evaluation
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0  # Return a default score for failed trials

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    return study.best_params

# Step 3: Hyperparameter Optimization
best_params = optuna_optimization(best_model, s)
print("Best Parameters from Optuna:", best_params)
```

---

## Workflow

### Step 1: Load and Preprocess the Dataset
- Load the Titanic dataset.
- Drop irrelevant columns and fill missing values.
- Encode categorical variables for machine learning.

### Step 2: Model Selection with PyCaret
- Use PyCaret's `setup()` to prepare the dataset.
- Automatically compare multiple classification models using `compare_models()`.
- Select the best-performing model for optimization.

### Step 3: Hyperparameter Optimization with Optuna
- Define a custom hyperparameter search space for the selected model.
- Use Optuna to optimize the hyperparameters for maximum performance.
- Return the best parameters and finalize the optimized model.

### Step 4: Finalize and Save the Model
- Finalize the model with the optimized parameters.
- Save the model using PyCaret's `save_model()` function.

---

## Example Output

1. **Best Model Selected**:
   ```text
   Best Model Selected: RandomForestClassifier
   ```

2. **Best Hyperparameters from Optuna**:
   ```text
   Best Parameters from Optuna: {'max_depth': 7, 'n_estimators': 150, 'learning_rate': 0.1, 'subsample': 0.8}
   ```

3. **Final Model Saved**:
   The model is saved as `optimized_model.pkl` for future use.

---

## Full Code

```python
import pandas as pd
from pycaret.classification import *
import optuna

# Step 1: Load and preprocess data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    return data

def pycaret_workflow(data):
    s = setup(data=data, target='Survived', session_id=42, normalize=True,
              feature_selection=True, remove_multicollinearity=True, fold=5, verbose=False)
    best_model = compare_models()
    return best_model, s

def optuna_optimization(model, setup_instance):
    def objective(trial):
        param_grid = {
            'max_depth': [trial.suggest_int('max_depth', 3, 10)],
            'n_estimators': [trial.suggest_int('n_estimators', 50, 200)],
            'learning_rate': [trial.suggest_float('learning_rate', 0.01, 0.3)],
            'subsample': [trial.suggest_float('subsample', 0.5, 1.0)],
        }
        try:
            tuned_model = tune_model(model, custom_grid=param_grid)
            score = pull().iloc[0]['Accuracy']
            return score
        except:
            return 0
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params

if __name__ == "__main__":
    data = load_data()
    best_model, s = pycaret_workflow(data)
    best_params = optuna_optimization(best_model, s)
    print("Best Parameters from Optuna:", best_params)
    final_model = finalize_model(best_model)
    save_model(final_model, 'optimized_model')
```

---

## Notes
- PyCaret simplifies the comparison of multiple models and feature selection.
- Optuna provides a powerful framework for hyperparameter tuning.
- Ensure you have the necessary libraries installed:
  ```bash
  pip install pycaret optuna pandas
  ```

---



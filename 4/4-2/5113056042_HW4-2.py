import pandas as pd
from pycaret.classification import *
import optuna

# Step 1: Load and preprocess data
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

# Step 2: Feature Engineering and Model Selection using PyCaret
def pycaret_workflow(data):
    s = setup(
        data=data,
        target='Survived',
        session_id=42,
        normalize=True,
        feature_selection=True,
        remove_multicollinearity=True,
        fold=5,
        verbose=False
    )

    # Compare models and select the best one
    best_model = compare_models()
    return best_model, s

# Step 3: Hyperparameter Optimization with Optuna
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

if __name__ == "__main__":
    # Step 1: Load data
    data = load_data()

    # Step 2: Feature Engineering and Model Selection
    best_model, setup_instance = pycaret_workflow(data)

    # Step 3: Hyperparameter Optimization
    best_params = optuna_optimization(best_model, setup_instance)

    print("Best Parameters from Optuna:", best_params)

    # Finalize model with optimized parameters
    final_model = finalize_model(best_model)
    
    # Save the model
    save_model(final_model, 'optimized_model')

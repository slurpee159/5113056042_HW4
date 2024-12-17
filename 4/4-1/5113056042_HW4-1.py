import pandas as pd
from pycaret.classification import *

# Load the Titanic dataset (using seaborn's built-in Titanic dataset for simplicity)
def load_titanic_data():
    import seaborn as sns
    titanic = sns.load_dataset('titanic')

    # Preprocess the data
    titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1)

    # Encode categorical variables
    titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
    titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Handle missing values
    titanic['age'].fillna(titanic['age'].median(), inplace=True)
    titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

    # Drop rows with any remaining missing values
    titanic.dropna(inplace=True)

    return titanic

# Load and preprocess the Titanic data
data = load_titanic_data()

# Set up PyCaret classification experiment
target = 'survived'
exp = setup(data=data, target=target, session_id=123, use_gpu=False)

# Compare models (PyCaret will automatically try multiple classification models)
results = compare_models(n_select=16)

# Print the top models
print("Top 16 Models:")
print(results)

# Optionally save the results
save_experiment('titanic_classification_experiment')

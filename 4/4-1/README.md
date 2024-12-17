# 4-1 Comparing 16 Machine Learning Models using PyCaret on Titanic Dataset

## Overview

This project uses PyCaret to compare 16 machine learning classification models on a multi-feature Titanic dataset. PyCaret automates the end-to-end machine learning workflow, allowing efficient model comparison and selection.

---

## Question 1

**How can I efficiently compare 16 machine learning models for a classification problem using PyCaret?**

### Solution:

We use PyCaret's `compare_models()` function, which automates the evaluation of multiple machine learning algorithms for a classification task. The Titanic dataset, which includes multiple features, is preprocessed and used as input.

---

## Question 2

**How do I preprocess the Titanic dataset for use in machine learning models?**

### Solution:

1. Drop unnecessary columns such as `deck`, `embark_town`, and others.
2. Encode categorical features like `sex` and `embarked`.
3. Handle missing values by filling them with the median or mode.
4. Drop rows with any remaining missing values.

The preprocessing ensures the dataset is clean and suitable for PyCaret's automated machine learning pipeline.

---

## Workflow

### Step 1: Load and Preprocess the Titanic Dataset

We load the Titanic dataset using Seaborn and preprocess it:

1. Drop unnecessary columns such as `deck`, `embark_town`, and others.
2. Encode categorical features like `sex` and `embarked`.
3. Handle missing values by filling them with the median or mode.
4. Drop rows with remaining missing values.

#### Code Snippet:

```python
import pandas as pd
from pycaret.classification import *

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

# Load data
data = load_titanic_data()
```

---

### Step 2: Set Up the PyCaret Classification Experiment

We configure the PyCaret classification setup to automatically prepare the data and enable comparison of models.

#### Code Snippet:

```python
# Set up PyCaret classification experiment
target = 'survived'
exp = setup(data=data, target=target, session_id=123, use_gpu=False)
```

---

### Step 3: Compare Models

We use PyCaret's `compare_models()` function to evaluate and rank 16 machine learning models. PyCaret automatically trains and evaluates models based on metrics like accuracy, F1-score, and AUC.

#### Code Snippet:

```python
# Compare models and select the top 16
results = compare_models(n_select=16)

# Print the top models
print("Top 16 Models:")
print(results)
```

---

### Step 4: Save the Experiment

We save the PyCaret experiment for future reference or reloading.

#### Code Snippet:

```python
# Save the experiment
save_experiment('titanic_classification_experiment')
```

---

## Example Output

1. **Top 16 Models**: The output displays the ranked list of 16 machine learning models evaluated by PyCaret.

   ```text
   Top 16 Models:
   [Logistic Regression, Random Forest, SVM, ...]
   ```

2. **Model Comparison Table**: PyCaret generates a table of metrics such as Accuracy, AUC, Recall, Precision, and F1-score for each model.

3. **Experiment Saved**: The experiment is saved as `titanic_classification_experiment` for future reuse.

---

## Notes

- PyCaret simplifies the comparison of multiple models, saving time on manual implementation.
- The Titanic dataset is multi-feature and ideal for binary classification tasks.
- Ensure PyCaret and Seaborn are installed before running the script:
  ```bash
  pip install pycaret seaborn pandas
  ```

---

## Full Code

```python
import pandas as pd
from pycaret.classification import *

def load_titanic_data():
    import seaborn as sns
    titanic = sns.load_dataset('titanic')

    # Preprocess the data
    titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1)

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
    # Encode categorical variables
    titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
    titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Handle missing values
    titanic['age'].fillna(titanic['age'].median(), inplace=True)
    titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

    # Drop rows with any remaining missing values
    titanic.dropna(inplace=True)

```

---

## Results

![image](https://github.com/user-attachments/assets/cadffded-f4a1-4014-b2cc-1ba65216be8f)




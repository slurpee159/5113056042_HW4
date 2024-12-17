
# 4-1 Pycarat to compare ML agorithms on classification problem 16 Model 
Titanic Dataset Classification using PyCaret

This project demonstrates how to use PyCaret for machine learning model comparison on the Titanic dataset. 
The primary goal is to evaluate and compare the performance of 16 classification models based on various metrics.

## Features

- Automated machine learning model comparison using PyCaret.
- Includes preprocessing steps for the Titanic dataset (feature selection, encoding, and missing value handling).
- Outputs the top-performing models ranked by Accuracy, AUC, Recall, Precision, F1 Score, and other metrics.

## Installation

To run this program, follow these steps:

1. Clone the repository.
    ```bash
    git clone https://github.com/your-repository/titanic-pycaret-classification.git
    cd titanic-pycaret-classification
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Python script to preprocess the Titanic dataset and compare the performance of 16 classification models:
```bash
python 5113056042_HW4.py
```

## Results

The program compares 16 machine learning models and outputs their performance metrics. Below is a preview of the results:

![image](https://github.com/user-attachments/assets/cadffded-f4a1-4014-b2cc-1ba65216be8f)


## License

This project is licensed under the MIT License. Feel free to use and modify the code.

## Contributing

If you would like to contribute, feel free to fork the repository and submit a pull request.


# 4-2 HW4-2 對 model optimization using pycarat or optuna or other AutoML,meta-heuristic 1. Feature engineering, 2. model selection , 3 . training 超參數優化

# Project Overview
This project demonstrates the use of PyCaret and Optuna to optimize machine learning models. The workflow includes feature engineering, model selection, and hyperparameter optimization.

## Key Steps
1. **Feature Engineering**: Preprocess the Titanic dataset, including handling missing values, removing irrelevant columns, and encoding categorical variables.
2. **Model Selection**: Use PyCaret to evaluate and select the best model from various classifiers.
3. **Hyperparameter Optimization**: Use Optuna to fine-tune the selected model's parameters.

## Results
### Model Comparison
Based on the PyCaret workflow, the top-performing models were:
- **Extra Trees Classifier**
  - Accuracy: 0.7031
  - AUC: 0.7202
  - F1 Score: 0.5389
- **Light Gradient Boosting Machine**
  - Accuracy: 0.7016
  - AUC: 0.7316
  - F1 Score: 0.5697
- **Random Forest Classifier**
  - Accuracy: 0.7015
  - AUC: 0.7421
  - F1 Score: 0.5662

### Best Parameters from Optuna
The optimized hyperparameters for the selected model were:
- `max_depth`: 10
- `n_estimators`: 60
- `learning_rate`: 0.227
- `subsample`: 0.829

## Prerequisites
### Required Python Libraries
To run the code, ensure the following libraries are installed:

```bash
pip install pandas pycaret optuna scikit-learn
```

## Usage
1. Clone the repository or download the script.
2. Ensure the required libraries are installed.
3. Run the script:

```bash
python script_name.py
```

4. The optimized model will be saved as `optimized_model.pkl`.

## Notes
- The `setup()` function in PyCaret is used for preprocessing and initializing the modeling environment.
- Optuna is integrated with PyCaret's `tune_model` to perform efficient hyperparameter optimization.
- The dataset used is the Titanic dataset, loaded directly from an online source.

## Troubleshooting
- Ensure all required libraries are installed.
- If PyCaret or Optuna raises parameter-related errors, verify the parameter grid aligns with the selected model's supported parameters.


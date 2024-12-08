
# Titanic Dataset Classification using PyCaret

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

| Model                         | Accuracy | AUC   | Recall | Precision | F1 Score | Kappa  | MCC    | TT (Sec) |
|-------------------------------|----------|-------|--------|-----------|----------|--------|--------|----------|
| Light Gradient Boosting Machine | 0.8185   | 0.8444| 0.7236 | 0.7934    | 0.7531   | 0.6103 | 0.6153 | 0.043    |
| Gradient Boosting Classifier    | 0.8106   | 0.8430| 0.6819 | 0.7958    | 0.7316   | 0.5874 | 0.5936 | 0.057    |
| Random Forest Classifier        | 0.8009   | 0.8376| 0.6724 | 0.7355    | 0.7288   | 0.5724 | 0.5777 | 0.019    |
| Ada Boost Classifier            | 0.7945   | 0.8394| 0.6829 | 0.7335    | 0.7212   | 0.5607 | 0.5655 | 0.017    |
| Logistic Regression             | 0.7988   | 0.8454| 0.6946 | 0.7418    | 0.7321   | 0.5687 | 0.5695 | 0.273    |
| ...                             | ...      | ...   | ...    | ...       | ...      | ...    | ...    | ...      |

## License

This project is licensed under the MIT License. Feel free to use and modify the code.

## Contributing

If you would like to contribute, feel free to fork the repository and submit a pull request.

---

**Contact**  
Your Name  
Your Email  
Your GitHub Profile: [GitHub](https://github.com/your-profile)

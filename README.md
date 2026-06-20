# Diabetes Prediction using Machine Learning

A machine learning project that predicts whether a patient has diabetes based on diagnostic health measurements. The project compares multiple classification algorithms — **SVM (with different kernels)** and **K-Nearest Neighbors** — to find the best-performing model on the well-known **PIMA Indians Diabetes Dataset**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

## Overview

Diabetes is one of the most common chronic diseases worldwide, and early prediction can help with timely medical intervention. This project builds and evaluates several supervised machine learning classifiers to predict diabetes diagnosis (`Outcome`: 0 = No Diabetes, 1 = Diabetes) using patient health metrics such as glucose level, BMI, blood pressure, and age.

## Dataset

The project uses the **PIMA Indians Diabetes Dataset**, which contains 768 records of female patients with 8 medical predictor variables and 1 target variable.

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | Target variable (0 = Non-diabetic, 1 = Diabetic) |

**Shape:** 768 rows × 9 columns

> Note: The dataset file (`diabetes.csv`) is not included in this repository. You can download it from [Kaggle's PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the project directory.

## Project Workflow

1. **Data Loading & Exploration** — Loaded the dataset and examined its shape, structure, and summary statistics.
2. **Data Cleaning** — Identified biologically invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` (zero is not a valid medical value for these features) and treated them as missing data.

   | Column | Zero values found |
   |---|---|
   | Glucose | 5 |
   | BloodPressure | 35 |
   | SkinThickness | 227 |
   | Insulin | 374 |
   | BMI | 11 |

3. **Missing Value Imputation** — Replaced the invalid zeros with `NaN` and imputed them using the **median** of each respective column.
4. **Feature/Target Split** — Separated the dataset into features (`X`) and target (`y = Outcome`).
5. **Train-Test Split** — Split data into 80% training and 20% testing sets (`random_state=42`, stratified by outcome).
6. **Feature Scaling** — Applied `StandardScaler` to normalize features (important for distance-based models like SVM and KNN).
7. **Model Training & Evaluation** — Trained multiple classifiers and evaluated them using accuracy, confusion matrix, and classification report.
8. **Model Comparison** — Compared all models on test accuracy and visualized the results with a bar chart.

## Models Used

| Model | Configuration |
|---|---|
| SVM (Linear Kernel) | `kernel='linear'` |
| SVM (RBF Kernel) | `kernel='rbf'` |
| SVM (Sigmoid Kernel) | `kernel='sigmoid'` |
| SVM (Polynomial Kernel) | `kernel='poly'`, `degree=3` |
| K-Nearest Neighbors | `n_neighbors=5` (default) |

## Results

Model performance on the test set (154 samples):

| Model | Accuracy |
|---|---|
| **KNN (k=5)** | **75.32%** |
| SVM (RBF) | 74.03% |
| SVM (Polynomial) | 71.43% |
| SVM (Linear) | 70.13% |
| SVM (Sigmoid) | 70.13% |

**Best performing model:** K-Nearest Neighbors (k=5), with an accuracy of **75.32%** and the best balance of precision/recall across both classes.

A bar chart comparing all model accuracies is generated at the end of the notebook for quick visual reference.

> Detailed confusion matrices and classification reports (precision, recall, F1-score) for each model are available in the notebook output.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Download the dataset (`diabetes.csv`) and place it in the project directory (update the file path in the notebook if needed — it currently points to `/content/diabetes.csv`, set up for Google Colab).

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook Diabetes_Prediction.ipynb
   ```
2. Run all cells sequentially to:
   - Load and clean the data
   - Train all five models
   - View accuracy scores, confusion matrices, and classification reports
   - Compare models via the accuracy bar chart

## Tech Stack

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — data visualization
  - `scikit-learn` — model building and evaluation (`SVC`, `KNeighborsClassifier`, `StandardScaler`, `train_test_split`)

## Future Improvements

- Perform hyperparameter tuning (e.g., `GridSearchCV`) for SVM and KNN to improve accuracy.
- Try additional models such as Logistic Regression, Random Forest, or XGBoost.
- Use cross-validation instead of a single train-test split for more robust evaluation.
- Address class imbalance (roughly 65% non-diabetic vs. 35% diabetic) using techniques like SMOTE.
- Add feature importance/explainability analysis (e.g., SHAP values).
- Deploy the best model as a simple web app (e.g., using Flask or Streamlit) for interactive predictions.

## License

This project is open source and available under the [MIT License](LICENSE).

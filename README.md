# Brain Stroke Prediction and Data Visualization

This repository contains the final year project focusing on predicting brain stroke risk using machine learning techniques and providing an interactive dashboard for data visualization and risk assessment.

## Project Description

The project addresses the significant public health concern of brain strokes by developing a tool that leverages machine learning to predict stroke risk based on health and lifestyle factors. It includes a comprehensive data visualization dashboard to explore stroke patterns and an interactive interface for users to assess their individual risk. The project utilized the 2021 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

## Features

* **Machine Learning Model:** Application and evaluation of various classification algorithms (Decision Tree, Logistic Regression, Random Forest, K-Nearest Neighbors) for stroke risk prediction.
* **Data Preprocessing:** Techniques including handling missing values, attribute generalization, and stratified sampling to prepare the BRFSS data.
* **Feature Selection:** Identification of key stroke risk factors using statistical methods (Chi-Square Test of Independence and Cramer's V).
* **Interactive Dashboard:** A user-friendly interface built with Tkinter/CustomTkinter for:
    * Visualizing data preprocessing steps and results.
    * Displaying model evaluation metrics and comparisons.
    * Providing an interactive tool for individual stroke risk prediction.
* **Model Evaluation:** Comprehensive assessment using metrics like Accuracy, Precision Macro, Recall Macro, F1-Score Macro, and ROC_AUC.

## Methodology

The project followed an Agile methodology, involving iterative steps of data understanding, preprocessing, feature selection, model training, evaluation, and dashboard development.

* **Data Source:** Utilized the 2021 Behavioral Risk Factor Surveillance System (BRFSS) dataset.
* **Requirement Analysis & Feature Selection:** Identified potential risk factors based on literature review and validated using Chi-Square Test of Independence and Cramer's V. Key selected features include Heart Disease, Age Group, High Blood Pressure, Diabetes, and Income Group.
* **Data Preprocessing:** Cleaned data, handled missing values (imputation with mode), performed attribute generalization, and applied stratified sampling to manage data imbalance and size.
* **Model Training & Evaluation:** Trained and tested multiple classification models. Addressed class imbalance in training data using SMOTE. Evaluated models using k-fold cross-validation and various performance metrics.
* **Dashboard Development:** Created a multi-page dashboard using Tkinter/CustomTkinter to visualize results and provide the prediction interface.
* **Model Testing:** Validated the selected model's performance using a subset of the BRFSS 2019 dataset.

## Technology Stack

* **Programming Language:** Python
* **Libraries:**
    * `pandas`, `numpy`: Data manipulation and analysis
    * `scipy.stats`: Statistical tests (Chi-Square, Cramer's V)
    * `sklearn` (Scikit-learn): Machine learning models and evaluation
    * `imbalanced-learn`: Handling imbalanced datasets (SMOTE)
    * `matplotlib`, `seaborn`: Data visualization
    * `tkinter`, `CustomTkinter`: Desktop application GUI
* **Tools:** Kaggle Notebooks, PyCharm Community

## Results

Among the evaluated models, Logistic Regression demonstrated the most balanced performance for stroke prediction on the test set, achieving:

* **Accuracy:** 0.71
* **Precision Macro:** 0.54
* **Recall Macro:** 0.72
* **F1-Score Macro:** 0.50
* **ROC_AUC:** 0.79

The test case using BRFSS 2019 data showed reasonable adaptability, particularly in recall, though with a slight decrease in overall accuracy compared to the original test set.

## Installation and Usage

(This section assumes the project is a Python application with a Tkinter GUI. You may need to adjust based on your actual project structure and dependencies.)

The project code is available on the master branch.

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    ```

    ```bash
    cd <repository_folder>
    ```

2.  Install the required Python libraries:

    ```bash
    pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn CustomTkinter
    ```

3.  Run the main application script:

    ```bash
    python main_script_name.py # Replace main_script_name.py with your actual script name
    ```

The interactive dashboard GUI should open, allowing you to explore data visualizations and use the stroke prediction function.

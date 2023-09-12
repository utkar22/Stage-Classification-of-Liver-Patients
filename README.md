# Stage-Classification-of-Liver-Patients
Our objective in this mini-project is to explore various machine learning models and techniques to achieve optimal results in terms of Receiver Operating Characteristic (ROC) score, in regards to classifying the stage of liver patients.

## Objective
Our goal is to implement advanced machine learning techniques to maximize the ROC score to analyse the stage of liver patients.

## Approach
We employed the following techniques to optimize our predictive models:

1. **Recursive Feature Elimination (RFE):**
   Utilized Logistic Regression with RFE to iteratively improve the subset of features used for prediction.

2. **Random Forest Classifier:**
   Implemented the Random Forest Classifier, an ensemble model that combines multiple decision trees for robust predictions.

3. **AdaBoost (Adaptive Boosting):**
   Used AdaBoost to create a strong classifier by combining multiple weak classifiers, focusing on difficult cases.

By applying these techniques, we aimed to optimize our predictive models for the highest ROC score on the given dataset. The resulting predictions were saved in an output CSV file for further analysis.

## Results
The achieved ROC score for our optimized predictive models was **0.82258**.


## Code
The programming for this assignment was done in Python. You can find the code in the `src/code.py` file.

## License
This project is licensed under the [MIT License](LICENSE).

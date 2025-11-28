Dataset source (tabular, CSV):
  - UCI ML Repository “Estimation of Obesity Levels Based on Eating Habits and Physical Condition”.

Basics installations required
  - pip install pandas numpy scikit-learn matplotlib seaborn streamlit


| Model              | Accuracy | F1-score |
| ------------------ | -------- | -------- |
| LogisticRegression | 0.9905   | 0.9897   |
| KNN                | 0.9504   | 0.9455   |
| SVM_linear         | 0.9882   | 0.9872   |
| SVM_rbf            | 0.9929   | 0.9923   |
| DecisionTree       | 0.9858   | 0.9846   |
| RandomForest       | 0.9669   | 0.9635   |
| GradientBoosting   | 0.9905   | 0.9897   |


SVM with RBF kernel is clearly best by both accuracy and F1-score, so that is the model you should deploy in Streamlit

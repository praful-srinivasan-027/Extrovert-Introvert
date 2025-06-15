# Extrovert vs Introvert Personality Classification

This project predicts whether a person is an **Introvert** or an **Extrovert** based on behavioral features using machine learning. It is implemented in Python using scikit-learn and demonstrates data preprocessing, pipeline construction, model evaluation, and basic fairness investigation.

---

## Dataset

The dataset contains the following features:

- Time_spent_Alone (numeric)  
- Social_event_attendance (numeric)  
- Going_outside (numeric)  
- Friends_circle_size (numeric)  
- Post_frequency (numeric)  
- Stage_fear (categorical: Yes/No)  
- Drained_after_socializing (categorical: Yes/No)  

**Target variable:** Personality  
- Values: Introvert or Extrovert

---

## Objective

To build a classification model that can accurately predict a person's personality type while also investigating whether certain features (like Post_frequency) disproportionately affect predictions.

---

## Project Highlights

- Constructed a full preprocessing pipeline using ColumnTransformer
- Performed imputation, scaling, and encoding
- Trained a RandomForestClassifier with class_weight='balanced'
- Evaluated using cross-validation and separate test set
- Investigated model bias caused by over-reliance on a single feature
- Built an alternate model excluding Post_frequency for fairer predictions

---

## Preprocessing

- Numerical features: Imputed using median, scaled using StandardScaler
- Categorical features: Imputed using most frequent value, encoded with OrdinalEncoder

The numerical and categorical pipelines were combined into a single ColumnTransformer.

---

## Model Evaluation

- Classifier: RandomForestClassifier with max_depth=5, max_features='sqrt'
- Cross-validation (10-fold) used to measure generalization
- Accuracy and classification reports printed for both training and test sets
- Final model accuracy ~91–92%

---

## Bias Handling

A strong correlation between Post_frequency = 0 and Introvert led to biased predictions.

**Mitigation steps:**
- Trained an alternate model excluding Post_frequency
- Compared performance of both models
- Found improved balance in predictions after removal

---

## Usage

You can call the final prediction function like this:

```python
predict_personality_new(
    Time_spent_Alone=4,
    Social_event_attendance=5,
    Going_outside=3,
    Friends_circle_size=8,
    Stage_fear="No",
    Drained_after_socializing="Yes",
    model=forest_clas_new
)

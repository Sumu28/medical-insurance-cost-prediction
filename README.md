# Predicting Medical Insurance Costs

**What actually drives a person's yearly medical bill, and how well can machine learning predict it?**

Insurers need to estimate how much each customer is likely to cost. This project takes a large insurance dataset of 100,000 people, cleans it up, and tests a range of models on two questions: *how much will someone's medical costs be?* (regression) and *which cost band, or which cost group, do they fall into?* (classification).

It is the project behind the paper "A Comparative Study of Regression Models for Medical Insurance Cost Prediction in Sustainable Healthcare Systems", presented at the CCCD 2026 International Conference in Bengaluru. 
(https://drive.google.com/file/d/1vVJKPizwWRKtu2fBugKVfj3BEyG6mHYY/view?usp=sharing)
> Group project. Sumukha Sagar, Kavya Kumar, Laura Pérez, Ethan Bochereau.

---

## The data

A public [Kaggle dataset](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction) of **100,000 people and 54 columns**: demographics, income, lifestyle, health measures such as BMI, blood pressure and cholesterol, chronic conditions, hospital visits, claims history, and the insurance plan. The target is `annual_medical_cost`. *(Check the dataset's licence and whether it is synthetic, and say so here. The CSV is large, so link to it rather than uploading it.)*

## What I did

1. **Looked at the data:** shape, types, missing values and duplicates (there were none).
2. **Handled missing values:** only `alcohol_freq` had gaps, about 30% of rows, so I dropped that column instead of losing 30,000 rows.
3. **Removed outliers:** values more than 4 standard deviations from the mean in 12 columns (income, BMI, blood pressure, cholesterol, HbA1c, costs and claims). This removed about 5,000 rows and left **94,995**.
4. **Explored the data** with charts of demographics and the main health and cost variables.
5. **Chose features** for the regression by correlation with cost, leaving out the two premium columns because they give the answer away.
6. **Compared feature choices:** the top-10 correlated features against 10 random ones and 10 I picked by hand.
7. **Turned cost into categories:** four equal bands (Very Cheap, Cheap, Moderate, Expensive) for classification, plus a simpler "is this person in the top 25% most expensive?" question.
8. **Trained and compared** linear regression, random forest, decision trees, logistic regression, Naive Bayes and gradient boosting, checked with 5-fold cross-validation, ROC curves and confusion matrices, and did a grid search for some of them.

---

## Results

### Predicting the cost (regression)

| Model and features | MAE | RMSE | R² | 5-fold CV R² |
|---|---|---|---|---|
| Linear regression, top-10 correlated | 1,031 | 1,536 | 0.524 | 0.532 |
| Linear regression, 10 hand-picked | 1,245 | 1,782 | 0.360 | 0.368 |
| Linear regression, 10 random | 1,314 | 1,859 | 0.304 | 0.311 |
| **Random Forest (300 trees), top-10 correlated** | **917** | **1,472** | **0.563** | **0.573** |

Choosing features by correlation clearly beat both random and hand-picked ones, and the Random Forest was a step up from linear regression.

### Predicting the cost band from demographics (4 classes)

Using only sex, region, urban/rural, education, marital and employment status, smoking, age, BMI and number of chronic conditions:

| Model | Accuracy | 5-fold CV macro F1 |
|---|---|---|
| Logistic regression | 0.35 | 0.318 |
| Naive Bayes | 0.34 | 0.294 |
| Decision tree | 0.28 | 0.284 |

With four equal bands, random guessing gets 25%. So demographics and lifestyle alone barely help. After tuning, the best models reached about 35% accuracy.

### Spotting the top 25% most expensive people

Gradient boosting on the top-10 numeric features reached **87% accuracy, macro F1 0.81** (5-fold CV 0.81). For the expensive group it found 59% of them (recall) and was right 86% of the time when it flagged someone (precision).

### What it tells us

Who a person is (age, sex, region, lifestyle) says little about their cost. What they have already used (claims, hospital days, chronic conditions, risk score) says much more.

---

## How to read these numbers

Two things matter here, and I would rather you hear them from me than find them yourself.

**1. The strongest features describe costs that have already happened.** The correlation-based top 10 include `total_claims_paid`, `claims_count` and `avg_claim_amount`. Claims are closely tied to medical cost, so the models are partly predicting cost from its own consequences. That is fine for understanding what drives cost, but it would not work for pricing a new customer who has no claims history.

**2. Later results in the notebook are inflated by a leak.** In the "high cost" step, the notebook adds a `high_cost` column that is directly calculated from the target. Further down, the features are chosen again from every numeric column, which now includes `high_cost`. The regression table in that last part therefore shows much higher scores (R² around 0.74 to 0.80), and I have not used them here. The results above are from before that column was added. *(To confirm, print `selected_features` after that second selection and check whether `high_cost` is in it. Then fix it by excluding `high_cost` and `cost_category` from the feature search, and rerun.)*

## Other limitations

- **Removing outliers removes the expensive people.** The 4-standard-deviation rule also dropped about 950 rows with extreme costs, which are exactly the cases an insurer cares about most.
- **The written validation description does not match the code.** The notebook text says an 80/20 stratified split, 5-fold CV on the training set, and tuning for random forest and gradient boosting. The code uses a mix of 70/30 and 80/20 splits, runs the CV on the whole cleaned dataset, and tunes the linear, decision tree and classification models. This README describes what the code does.
- **Some printed labels are wrong.** The CV output for gradient boosting, decision tree and Naive Bayes says "Random Forest CV R2", but those numbers are macro F1 scores for different models.
- **Features were chosen by correlation on the full dataset** before the train/test split.

## What I would do next

- Fix the `high_cost` leak and rerun the tuned comparison
- Predict cost without claims features, to see what can be known at sign-up
- Keep high-cost cases instead of removing them as outliers
- Try gradient boosting and tuned random forests for regression
- Explain the predictions with SHAP

---

## Tech stack

Python · Pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Google Colab

## Run it yourself

1. Download `medical_insurance.csv` from the Kaggle link above.
2. Open the notebook in Colab or Jupyter and put the CSV where the notebook expects it (the first cell mounts Google Drive).
3. Install the libraries if needed: `pip install pandas numpy scikit-learn scipy matplotlib seaborn`.
4. Run the cells in order.

## Authors

Sumukha Sagar, Kavya Kumar, Laura Pérez, Ethan Bochereau
School of Computing, Dublin City University

## License

MIT

# Module 2 — Analytics Pipeline

## What I did in this module

For this module, I worked with the Titanic dataset to perform data cleaning, exploratory data analysis, visualization, and predictive modeling.

I divided the work into two Python files so that the EDA and modeling steps are easier to follow.

First, `01_eda.py` loads the Titanic dataset using `sns.load_dataset('titanic')`. I save the original dataset as `titanic.csv`, check its structure and missing values, clean the data, perform the required analysis, and save the cleaned dataset as `cleaned_titanic.csv`.

After that, `02_modeling.py` reads the cleaned dataset and continues with the classification models, imbalance handling, Random Forest tuning, and the regression task.

## Run order

I run the files in this order:

```bash
python analytics/01_eda.py
```

This performs the initial data loading, profiling, cleaning, EDA, and saves the cleaned dataset.

Then:

```bash
python analytics/02_modeling.py
```

This reads `cleaned_titanic.csv` and continues with the modeling and evaluation part.

## Dataset profile

The original Titanic dataset contains **891 rows and 15 columns**.

I first checked the missing values in the dataset and applied the missing-value rules given in the project.

| Column        | Missing count | Missing % | What I did                                                     |
| ------------- | ------------: | --------: | -------------------------------------------------------------- |
| `age`         |           177 |    19.87% | Filled missing values using the median, which was 28.00        |
| `deck`        |           688 |    77.22% | Dropped the column because the missing percentage was too high |
| `embark_town` |             2 |     0.22% | Removed the rows containing missing values                     |
| `embarked`    |             2 |     0.22% | Removed the rows containing missing values                     |

The `age` column has 19.87% missing values, so I used the median value of 28.00 to fill them.

The `deck` column has 77.22% missing values. Because such a large portion of the column is missing, I dropped it instead of trying to fill most of the values.

For `embark_town` and `embarked`, the missing percentage is below 5%, so I removed the affected rows.

These decisions were made using the missing-value threshold specified in the project.

## Univariate analysis

I checked the distributions of `age` and `fare` using histograms and box plots.

The IQR method gave the following outlier counts:

* Age: **65 outliers**
* Fare: **114 outliers**

For `fare`, the calculated values were:

* Mean = **32.097**
* Median = **14.454**
* Mode = **8.050**

Since the mean is greater than the median and the median is greater than the mode, the fare distribution is **right-skewed**.

I also saved the age and fare plots inside the `plots` folder.

## Survival analysis

I then looked at survival rates using different passenger groups.

### Survival by sex

| Sex    | Survival rate |
| ------ | ------------: |
| Female |        0.7404 |
| Male   |        0.1889 |

The survival rates are quite different between the two groups. I calculated the rates by dividing the number of survivors by the total number of passengers in each group.

### Survival by passenger class

| Passenger class | Survival rate |
| --------------- | ------------: |
| 1               |        0.6262 |
| 2               |        0.4728 |
| 3               |        0.2424 |

The survival rate changes across the three passenger classes. The values show an association between passenger class and survival in this dataset.

### Survival by sex and passenger class

| Group             | Survival rate |
| ----------------- | ------------: |
| Female, 1st class |        0.9674 |
| Female, 2nd class |        0.9211 |
| Female, 3rd class |        0.5000 |
| Male, 1st class   |        0.3689 |
| Male, 2nd class   |        0.1574 |
| Male, 3rd class   |        0.1354 |

Looking at both variables together gives more detail than looking at sex or passenger class separately. The survival rate changes between the groups depending on both the passenger's sex and class.

## Correlation analysis

For the correlation matrix, I used the six columns specified in the project:

`survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`.

I did not include `adult_male` and `alone` because they are derived boolean columns rather than separate measured features.

The two strongest correlations based on absolute correlation value were:

* `pclass` and `fare` = **-0.5482**
* `sibsp` and `parch` = **0.4145**

The sign tells the direction of the relationship, while the absolute value was used to identify the strongest correlations.

I also created a heatmap to make the correlations easier to see.

## Charts and data story

I created several charts to understand the survival patterns in the dataset.

### 1. Survival rate by sex

This chart compares the survival rates between male and female passengers. It gives a quick visual representation of the difference that was already seen in the survival-rate calculation.

### 2. Survival rate by passenger class

This chart shows how the survival rate changes across first, second, and third class. The differences are easier to see visually than from the table alone.

### 3. Survival by sex and class

This grouped bar chart combines sex and passenger class. It allows me to compare passengers within the same class as well as compare the different groups.

### 4. Fare distribution by survival

This box plot compares the fare values of passengers who survived with those who did not. It helps show how fare is distributed across the two survival groups.

### 5. Age and fare by survival

This scatter plot uses age and fare together and separates the observations based on survival. There is some overlap between the groups, but it also helps show the relationship between age, fare, and the passenger data.

## Standardization check

As an additional EDA check, I standardized `age` and `fare` using the z-score approach.

This step was only used for the EDA check. I did not use these standardized columns directly in the modeling stage because the modeling pipeline performs its own preprocessing.

After standardization, the values were approximately:

| Column   | Mean after standardization | Standard deviation |
| -------- | -------------------------: | -----------------: |
| `age_z`  |                2.71749e-16 |                  1 |
| `fare_z` |                1.39871e-16 |                  1 |

The means are effectively zero and the standard deviations are 1, which confirms that the standardization worked as expected.

# Predictive Modeling

## Train/test split

After cleaning, the dataset contains **889 rows**.

I divided the data into training and testing sets using an 80/20 stratified split with `random_state=42`.

I used stratification because the target variable is not equally distributed between the survived and not-survived classes. Stratification keeps a similar class proportion in both the training and testing datasets.

## Preprocessing

For the modeling part, I used a `Pipeline` together with `ColumnTransformer`.

For the numeric features, I used median imputation followed by standardization.

For the categorical features `sex` and `embarked`, I used most-frequent imputation and one-hot encoding.

The preprocessing is fitted only on the training data and then applied to the test data. This avoids using information from the test set while training the model.

## Class balance

The cleaned target data had the following distribution:

| Survived | Count | Percentage |
| -------: | ----: | ---------: |
|        0 |   439 |     61.74% |
|        1 |   272 |     38.26% |

The two classes are not perfectly balanced, so I used the stratified split to keep their proportions similar in both datasets.

## Classification models

I trained three classification models using the same train/test split:

* Logistic Regression
* Decision Tree
* Random Forest

The results from the test set were:

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |    0.809 |    0.7833 | 0.6912 | 0.7344 |  0.861 |
| Decision Tree       |    0.764 |    0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest       |   0.8034 |    0.7619 | 0.7059 | 0.7328 | 0.8237 |

I also generated confusion matrices and ROC curves for all three models.

For the Decision Tree, I created a tree visualization using `plot_tree` so that the learned splits could be viewed.

## Imbalance handling

I compared three approaches for handling the class imbalance:

| Strategy                  | Precision | Recall |     F1 |
| ------------------------- | --------: | -----: | -----: |
| Baseline                  |    0.7619 | 0.7059 | 0.7328 |
| `class_weight='balanced'` |    0.7536 | 0.7647 | 0.7591 |
| SMOTE                     |    0.7612 | 0.7500 | 0.7556 |

In my run, the `class_weight='balanced'` approach gave the highest F1 value of **0.7591** among the three approaches.

For SMOTE, I applied the oversampling only to the training data so that the test data was not used when generating the synthetic samples.

## Random Forest tuning

I used `GridSearchCV` to tune the Random Forest parameters.

The best parameters from the search were:

```text
{'classifier__max_depth': 5,
 'classifier__max_features': 'sqrt',
 'classifier__n_estimators': 200}
```

The best cross-validation F1 score was **0.7408**.

After refitting the best Random Forest configuration with `oob_score=True`, the OOB score was **0.8214**.

The grid search was performed on the complete pipeline, so the preprocessing steps were also handled correctly during cross-validation.

## Regression task

Along with the classification models, I also performed the required regression task to predict `fare`.

I used multivariate linear regression and obtained:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 19.7531 |
| RMSE        | 41.2701 |
| R²          |  0.3471 |
| Adjusted R² |  0.3081 |

I also created a residual plot to check the behavior of the prediction errors.

The residual check used in the script indicates evidence of heteroscedasticity. I considered both the residual plot and the numerical check rather than relying only on the visual plot.

## Model comparison

The classification and regression metrics are kept as separate groups because they measure different types of model performance.

Classification uses metrics such as accuracy, precision, recall, F1, and AUC, while regression uses MAE, RMSE, R², and Adjusted R².

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |     MAE |    RMSE |     R² | Adjusted R² |
| ------------------- | -------: | --------: | -----: | -----: | -----: | ------: | ------: | -----: | ----------: |
| Logistic Regression |    0.809 |    0.7833 | 0.6912 | 0.7344 |  0.861 |       — |       — |      — |           — |
| Decision Tree       |    0.764 |    0.7600 | 0.5588 | 0.6441 | 0.8374 |       — |       — |      — |           — |
| Random Forest       |   0.8034 |    0.7619 | 0.7059 | 0.7328 | 0.8237 |       — |       — |      — |           — |
| Linear Regression   |        — |         — |      — |      — |      — | 19.7531 | 41.2701 | 0.3471 |      0.3081 |

## Final model

Based on the test-set results from this run, I selected **Logistic Regression** as the classifier to save for deployment.

Its results were:

* Accuracy: **0.809**
* Precision: **0.7833**
* Recall: **0.6912**
* F1: **0.7344**
* AUC: **0.861**

I based this selection on the observed test-set metrics rather than only looking at the training results.

The saved model is not just the classifier by itself. I saved the complete preprocessing pipeline together with the classifier so that raw input data can be passed through the same preprocessing steps during prediction.

## Saved model

The final pipeline is saved as:

```text
artifacts/best_classifier_pipeline.joblib
```

The file contains the preprocessing steps and the classifier together.

I also tested loading the saved pipeline again using `joblib.load()` and used it to make a prediction on a held-out raw row.

The reload check produced:

```text
Prediction: 0
```

This confirms that the saved pipeline can be loaded again and used for prediction without manually preprocessing the input first.

The generated analysis files and plots are stored in the analytics outputs, plots, and artifacts directories for later review.
Execution order: run 01_eda.py first, then run 02_modeling.py.
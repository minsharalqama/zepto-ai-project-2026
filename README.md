# Zepto Data & AI Platform

## About the Project

This is my capstone project for building a small Data and AI platform based on a Zepto-style use case.

The project has three main parts:

* Data Pipeline
* Data Analytics and Machine Learning
* AI Support Assistant

I kept all three parts in the same GitHub repository.

The main idea was to collect and process data, analyse it using Python and machine learning, and then build a simple support assistant that can answer questions using policy documents.

## Project Structure

```text
zepto_ai_capstone/

├── data_pipeline/
│   ├── run_pipeline.py
│   ├── requirements.txt
│   ├── README.md
│   └── data/

├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── requirements.txt
│   ├── README.md
│   ├── titanic.csv
│   ├── cleaned_titanic.csv
│   ├── outputs/
│   ├── plots/
│   └── artifacts/

├── support_assistant/
│   ├── docs/
│   ├── ingest.py
│   ├── rag_engine.py
│   ├── main.py
│   ├── run_examples.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md

├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Setup

I used Python for the project.

First, I created a virtual environment in the project folder:

```powershell
python -m venv .venv
```

Then I activated it:

```powershell
.\.venv\Scripts\Activate.ps1
```

I checked the Python version using:

```powershell
python --version
```

After that, I updated pip:

```powershell
python -m pip install --upgrade pip
```

Finally, I installed the project dependencies:

```powershell
pip install -r requirements.txt
```

Each module also has its own `requirements.txt` file.

## 2. Data Pipeline

The first part of the project is the data pipeline.

For this part, I collected book information from `books.toscrape.com` using Python. I used `requests` and `BeautifulSoup` for the scraping.

I can run the pipeline using:

```powershell
python data_pipeline/run_pipeline.py
```

The script first finds the book categories and then collects the books from the category pages. For each book, it also opens the book's detail page to get the availability information.

After collecting the data, I cleaned and prepared it before storing it in the database.

The main flow I followed was:

```text
Find categories
      ↓
Collect books
      ↓
Open book detail pages
      ↓
Create raw dataset
      ↓
Clean the data
      ↓
Convert price to INR
      ↓
Store data in SQLite
      ↓
Run SQL queries
      ↓
Compare SQL and pandas JOIN results
```

For the price conversion, I used the fixed rate:

```text
1 GBP = 105.50 INR
```

I created two tables in the SQLite database:

```text
categories
    |
    └── books
```

The `books` table has a foreign key connected to the `categories` table.

The pipeline creates the following files:

```text
data_pipeline/scraped_books_raw.csv
data_pipeline/scraped_books_clean.csv
data_pipeline/books.db
data_pipeline/queries_output.md
```

Some of the cleaned fields are:

```text
price_gbp
rating
in_stock
price_inr
```

I also checked the data using SQL and pandas. In particular, I compared the SQL `JOIN` result with the `pandas.merge()` result to make sure both gave the same output.

## 3. Analytics

The second part of the project is the analytics and machine learning module.

For this part, I used the Titanic dataset.

I first ran the EDA script:

```powershell
python analytics/01_eda.py
```

This script loads the dataset, checks the data, handles the missing values and creates the required analysis outputs.

During the EDA, I checked things such as:

* Missing values
* Data types
* Age distribution
* Fare distribution
* Box plots
* Outliers using IQR
* Fare mean, median and mode
* Fare skewness
* Survival rate by gender
* Survival rate by passenger class
* Survival rate using gender and class
* Correlation
* Multivariate charts
* Standardization

The main output files are stored in:

```text
analytics/outputs/
analytics/plots/
```

The cleaned Titanic dataset is saved as:

```text
analytics/cleaned_titanic.csv
```

The main hand-off dataset is:

```text
analytics/titanic.csv
```

## 4. Machine Learning

After completing the EDA, I ran the modeling script:

```powershell
python analytics/02_modeling.py
```

For classification, the target variable is:

```text
survived
```

I used a stratified train/test split so that the classes were represented properly in both the training and test data.

I kept the preprocessing inside scikit-learn pipelines. This included missing-value handling, encoding and scaling.

For classification, I used three models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

I compared the models using:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

I also generated confusion matrices and ROC curves.

## 5. Class Imbalance

The Titanic target has an imbalance between the two classes, so I also tested different approaches for handling it.

The three approaches I used were:

```text
Baseline

Class Weight = Balanced

SMOTE
```

I applied SMOTE only to the training data.

The comparison results are saved in:

```text
analytics/outputs/imbalance_comparison.csv
```

## 6. Random Forest Tuning

I also tuned the Random Forest model using `GridSearchCV`.

The search results are saved in:

```text
analytics/outputs/rf_grid_search_results.csv
```

I also checked the Random Forest OOB score as part of the evaluation.

## 7. Fare Regression

Apart from the classification task, I also performed a regression task to predict fare.

For this, I used `LinearRegression` with multiple features.

I evaluated the regression model using:

```text
MAE
RMSE
R²
Adjusted R²
```

I also created a residual plot to check the model errors and look for possible heteroscedasticity.

The residual plot is saved at:

```text
analytics/plots/regression_residuals.png
```

## 8. Model Saving

After comparing the classification models, I saved the selected classifier pipeline using Joblib.

The saved model is:

```text
analytics/artifacts/best_classifier_pipeline.joblib
```

I saved the complete pipeline instead of saving only the classifier, so the preprocessing steps are also included.

I then loaded the saved pipeline again and tested it with raw input data to make sure that the saved model was working correctly.

## 9. Support Assistant

The third part of the project is the AI support assistant.

For this part, I used eight policy documents stored inside:

```text
support_assistant/docs/
```

I converted the documents into embeddings and stored them in ChromaDB.

To create the local vector database, I ran:

```powershell
python support_assistant/ingest.py
```

I used the `all-MiniLM-L6-v2` model for creating the embeddings.

The ChromaDB collection is called:

```text
zepto_policies
```

## 10. How the Support Assistant Works

The basic flow I followed is:

```text
User Question
      ↓
Check the type of question
      ↓
Policy question?
    /       \
  Yes        No
   ↓          ↓
Search      Direct
documents   answer
   ↓
Get top 3
results
   ↓
Generate answer
```

For a policy-related question, the assistant searches the local policy documents and uses the most relevant results to prepare the answer.

For a general question, it follows the direct-answer path.

I used three LangGraph nodes for this flow along with a conditional edge.

## 11. Mock Mode

I added a mock mode so that the support assistant can be tested without using an external LLM API.

For testing, I normally use:

```powershell
$env:MOCK_LLM="1"
```

In mock mode, the application does not make an external LLM API call.

For example, one policy question I tested was:

```text
What delivery fee applies to orders below INR 149?
```

I also tested a general question:

```text
What is the capital of France?
```

Both types of questions are included in the example script.

I can run the examples using:

```powershell
python support_assistant/run_examples.py
```

## 12. FastAPI

I exposed the support assistant through a FastAPI API.

First, I went to the support assistant folder:

```powershell
cd support_assistant
```

Then I started the server:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 7860
```

The main API endpoint is:

```text
POST /ask
```

I tested it using a request like:

```powershell
Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:7860/ask `
    -ContentType 'application/json' `
    -Body '{"query":"What delivery fee applies to orders below INR 149?"}'
```

The response is validated using Pydantic.

The response contains the answer, sources and confidence information.

## 13. Prompt and Validation

The support assistant uses a structured prompt.

The prompt contains:

```text
Role
Context
Task
Format
Length
Constraints
Few-shot example
```

I also added a negative constraint so that the assistant does not add unsupported information or make up policy details.

Pydantic is used to check whether the generated response follows the expected structure.

When the real LLM mode is used, the application can retry the request if the response does not match the expected format.

## 14. Docker

I also prepared the support assistant to run using Docker.

From the `support_assistant` folder, I built the image using:

```powershell
docker build -t zepto-support-assistant .
```

Then I ran it using:

```powershell
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

The FastAPI application is then available on port `7860`.

## 15. Optional Real LLM

The project also contains an optional real LLM mode.

It can be configured using environment variables:

```text
MOCK_LLM=0

GROQ_API_KEY=<your-api-key>

GROQ_MODEL=<model-name>
```

I kept the API key out of the source code and Git repository.

This part is optional because the main project can be demonstrated using mock mode.

## 16. Git Workflow

I used a feature branch while developing the project.

First, I initialized the repository on `main`:

```powershell
git init
git branch -M main
git add .
git commit -m "chore: initialize capstone repository"
```

Then I created the feature branch:

```powershell
git checkout -b feature/capstone-platform
```

I committed the three modules separately.

For the data pipeline:

```powershell
git add data_pipeline
git commit -m "feat: add data engineering pipeline"
```

For analytics:

```powershell
git add analytics
git commit -m "feat: add analytics and modeling pipeline"
```

For the support assistant:

```powershell
git add support_assistant
git commit -m "feat: add support assistant and FastAPI service"
```

After completing the work, I merged the feature branch into `main`:

```powershell
git checkout main
git merge --no-ff feature/capstone-platform -m "merge: capstone platform feature branch"
```

I can check the Git history using:

```powershell
git log --graph --oneline --decorate --all
```

## 17. Final Result

After completing the three modules, the project contains:

```text
Data Collection & ETL
        ↓
Data Analysis & Machine Learning
        ↓
AI Support Assistant
```

Through this project, I worked with:

* Python
* Pandas
* BeautifulSoup
* Requests
* SQLite
* SQL
* Scikit-learn
* SMOTE
* GridSearchCV
* Joblib
* Sentence Transformers
* ChromaDB
* LangGraph
* Pydantic
* FastAPI
* Docker
* Git

I kept the complete project in one GitHub repository with the source code, datasets, outputs, model artifact, documentation and configuration files.
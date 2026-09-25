# Module 3 — Support Assistant

## Graded baseline

For this module, I kept the LLM part deterministic and offline so that the project can be tested without needing an LLM API key. I used `MOCK_LLM` for this. The code treats anything other than exactly `0` as mock mode, so mock mode is used by default.

For embeddings, I used `sentence-transformers/all-MiniLM-L6-v2`. The embeddings are created locally and stored in ChromaDB. The model may need to download the first time it is used, but this does not require an LLM API key or account.

## Architecture

The basic flow I followed is:

```text
8 policy .txt files
        |
        v
ingest.py: one-document chunks
        |
        v
SentenceTransformer(all-MiniLM-L6-v2)
        |
        v
ChromaDB collection: zepto_policies (cosine)
        |
        v
FastAPI /ask -> LangGraph StateGraph
        |
        v
classify_intent
      |       |
   policy   general
      |       |
      v       v
retrieve_and_answer   direct_answer
      |
      +--> embed query -> Chroma top-3
      |
      +--> MOCK_LLM=1: deterministic canned answer
      |
      +--> MOCK_LLM=0: structured prompt -> Groq -> Pydantic validation/retries

All final routes return AnswerResponse(answer, sources, confidence)
```

I used LangGraph `StateGraph` to pass the shared state between the different steps. The `classify_intent` node checks whether the question is policy-related or general. Based on that, the flow goes either to `retrieve_and_answer` or `direct_answer`.

The implementation contains the three required nodes:

* `classify_intent`
* `retrieve_and_answer`
* `direct_answer`

## Pipeline stages

### Ingestion

I created `ingest.py` to read the 8 policy files from `docs/doc_01.txt` to `docs/doc_08.txt`. For this project, I treated each document as one chunk and used IDs from `doc_01` to `doc_08`.

### Embedding

I used:

```python
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

to create the embeddings. The vectors are normalized before being stored in ChromaDB.

The ChromaDB collection used for this project is:

```text
zepto_policies
```

and cosine distance is used for the retrieval.

### Retrieval

In `RAGEngine.retrieve()`, I convert the user's question into an embedding and search the `zepto_policies` collection.

I used the top 3 matching chunks from ChromaDB for the retrieved context.

### Generation

The `retrieve_and_answer` node is used to create the final answer for policy questions.

When mock mode is enabled, the application returns a deterministic answer using the retrieved context. This keeps the graded version predictable and does not require an external LLM.

When `MOCK_LLM=0`, the application uses the `PROMPT_TEMPLATE` and sends the request to Groq. The response is then checked using the `AnswerResponse` Pydantic model. If the response does not match the expected structure, the request can be retried up to two more times.

The `direct_answer` node follows a similar approach for general questions.

The query embedding and ChromaDB retrieval are used in both modes. Only the LLM-related classification and answer-generation parts change when `MOCK_LLM` is changed.

## Structured prompt template

For the optional real-LLM mode, I used a fixed prompt structure. It contains:

* **ROLE** — Zepto policy support assistant.
* **CONTEXT** — retrieved policy information.
* **TASK** — answer the question using the given context.
* **FORMAT** — return the required JSON fields.
* **LENGTH** — keep the answer concise.
* **Negative constraint** — do not add information that is not present in the context or make up policy details.
* **Few-shot example** — includes a delivery-fee question and an example of the expected JSON response.

The complete prompt is stored as `PROMPT_TEMPLATE` inside `rag_engine.py`.

## Install and build the index

I used the following steps from the repository root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r support_assistant/requirements.txt

python support_assistant/ingest.py
```

I ran `ingest.py` before starting FastAPI because the application expects the `zepto_policies` ChromaDB collection to already exist.

## Run FastAPI locally

For my normal testing, I used mock mode. Since mock mode is the default, `MOCK_LLM` can be left unset. It can also be explicitly set to `1`.

From the `support_assistant` directory, I used:

```bash
cd support_assistant

uvicorn main:app --reload --host 127.0.0.1 --port 7860
```

I then tested the API from another terminal using a policy question and a general question.

```bash
curl -X POST "http://127.0.0.1:7860/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What delivery fee applies to orders below INR 149?\"}"

curl -X POST "http://127.0.0.1:7860/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the capital of France?\"}"
```

For PowerShell, I used:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/ask -ContentType 'application/json' -Body '{"query":"What delivery fee applies to orders below INR 149?"}'

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/ask -ContentType 'application/json' -Body '{"query":"What is the capital of France?"}'
```

I also created `run_examples.py`, so both example requests can be tested directly with:

```bash
python support_assistant/run_examples.py
```

This creates:

```text
support_assistant/outputs/example_calls.md
```

The file contains the raw JSON results from the example calls.

## Docker

I also tested the application using Docker.

From the `support_assistant` directory, I used:

```bash
docker build -t zepto-support-assistant .

docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

After the container starts, the same `/ask` endpoint can be used for testing:

```text
http://127.0.0.1:7860/ask
```

The Dockerfile starts Uvicorn on port `7860`.

## Optional real LLM extension

The project also has an optional real-LLM mode. To use it, I can set:

```text
MOCK_LLM=0

GROQ_API_KEY=<your secret>

GROQ_MODEL=<currently available model name>
```

The API key should only be stored as an environment variable. I should not add it to the Python files, README, Dockerfile, or Git history.

For the main graded demonstration, I used mock mode, so the real LLM setup is not required.

## Example call transcripts

After building the ChromaDB index, I ran:

```bash
python support_assistant/run_examples.py
```

The output is saved in:

```text
support_assistant/outputs/example_calls.md
```

I used the examples to test both required cases:

1. A policy-related question.
2. An unrelated/general question.

Both examples were run using the default mock mode.

## Validated Local Run Results

### Data Pipeline

I tested the data pipeline using `books.toscrape.com`. It produced 69 book records from 3 categories:

* Travel
* Mystery
* Historical Fiction

I used the fixed conversion rate:

```text
1 GBP = 105.50 INR
```

for the `price_inr` field.

The SQLite database was created successfully. I also compared the SQL `JOIN` result with the `pandas.merge` result, and the outputs matched.

### Analytics Pipeline

The Titanic dataset had 891 rows and 15 columns before cleaning. After applying the missing-value rules, the cleaned dataset contained 889 rows.

The modeling pipeline also completed successfully with a stratified split of 711 training rows and 178 test rows.

I trained and evaluated the three classification models and also completed the imbalance comparison, Random Forest GridSearchCV, OOB evaluation, and the fare regression task.

The complete classifier pipeline was saved as:

```text
analytics/artifacts/best_classifier_pipeline.joblib
```

I loaded the saved pipeline again and tested it successfully.

### Support Assistant

For the support assistant, I processed all 8 policy documents using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and stored the embeddings in the `zepto_policies` ChromaDB collection.

I tested the LangGraph flow in mock mode with both a policy question and a general question.

The FastAPI `/ask` and `/health` endpoints returned HTTP 200 during my local testing.

I also built and ran the Docker image successfully with mock mode enabled.
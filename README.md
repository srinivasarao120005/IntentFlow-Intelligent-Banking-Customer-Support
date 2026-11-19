# IntentFlow — Explainable Banking Support Ticket Routing

**Role skills:** NLP preprocessing, TF-IDF, multiclass classification, model selection, macro-F1, probability-based abstention, human-in-the-loop decision support and ML tests.

## Business challenge
Automatically route common customer queries to one of 77 banking-support intent categories, while escalating lower-confidence cases for human review.

## Source data
[PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), CC BY 4.0. Public benchmark contains 10,003 official train and 3,080 official test examples across **77 intents**. Preserve the supplied test split to compare honestly with prior work.

## Differentiating features
- Audits empty rows, normalizes whitespace and drops duplicate training questions; **removes conflicting-label duplicate text** from training.
- Stratifies 80/20 within the official training partition for model selection; **the official test data is never used for tuning**.
- Compares word TF-IDF versus combined word+character TF-IDF, each with logistic regression.
- Chooses by validation **macro-F1** and evaluates official test accuracy, macro-F1 and top-3 accuracy.
- Chooses an abstention threshold using *validation* data at a requested 90% selective-accuracy target, and reports whether that target was actually reached.
- Reports test auto-routing coverage, accuracy on auto-routed tickets, human review rate, per-intent scores and error examples.

## Run
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.train --train data/raw/train.csv --test data/raw/test.csv --out reports/real_experiment
python -m pytest -q
```

## Predict with the trained model
```bash
python -m src.predict --model reports/real_experiment/model.joblib --text 'I lost my card and need a replacement'
```

The prediction includes a human-review/auto-route decision, confidence, and top three intents.

## Validation status
Automated tests pass and a **templated synthetic 8-intent** smoke dataset has completed the training/plotting pipeline. Its perfect scores are an artifact of simple generated examples; they are **NOT BANKING77 benchmark scores**. Actual 77-class benchmark evaluation remains pending official dataset retrieval.

## Suggested next experiment
Compare sentence embeddings and small transformers with the word+character baseline; audit errors from underrepresented intents, paraphrases, out-of-distribution requests and adversarial prompts. Avoid suggesting the model can provide financial advice.

### Example chart from the **synthetic software smoke test** (not a real dataset result)

![Synthetic smoke-test visualization](reports/synthetic_smoke/figures/selective_routing.png)

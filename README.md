<div align="center">

# Advanced Text-to-SQL Query System

**Ask questions about World Bank data in plain English and get a SQL-backed, summarised answer.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

</div>

## Overview

A natural-language interface to a World Bank indicators database. A question such as *"Are citizens in developed countries more aware of their rights under financial consumer protection laws than those in developing nations?"* goes through four stages:

1. **Few-shot retrieval**: the 5 most similar example question/SQL pairs are picked with OpenAI embeddings and a FAISS index.
2. **Indicator search**: a TF-IDF search over the `Indicator` table (name, description, source, topic) finds the indicators that best match each sentence of the question, and their ids are appended to the prompt.
3. **SQL agent**: a LangChain SQL agent (OpenAI function calling, `gpt-4o` by default) inspects the schema, writes read-only SQL and runs it.
4. **Summary**: a second, cheaper model (`gpt-4o-mini`) turns the raw rows into a short answer, with averages instead of long tables.

## How it works

```
question
   |
   +--> FewShotSelector   (OpenAI embeddings + FAISS, top 5 examples)
   +--> IndicatorSearch   (TF-IDF + cosine similarity, top N indicator ids)
   |
   v
augmented prompt --> SQL agent (LangChain, GPT-4o) --> SQLite --> rows
                                                                    |
                                       summary (GPT-4o-mini) <------+
```

Guardrails live in `prompts.py`: the agent is told never to run DML (`INSERT`, `UPDATE`, `DELETE`, `DROP`), to skip NULL and duplicate rows, to add `LIMIT 150` when asked for "all results", and to answer "I don't know" for questions unrelated to the database. Token usage for each query is printed through LangChain's OpenAI callback.

## Tech stack

Python, LangChain, OpenAI (GPT-4o, GPT-4o-mini, embeddings), FAISS, SQLAlchemy and SQLite, scikit-learn (TF-IDF), NLTK (sentence splitting), pandas.

## Project structure

```
.
├── main.py                # pipeline: examples -> indicators -> agent -> summary
├── agent.py               # WorldBankAgent: SQL agent, token tracking, summary call
├── few_shot_selector.py   # semantic example selection (FAISS)
├── indicator_search.py    # TF-IDF indicator search
├── prompts.py             # agent and summary prompts
├── config.py              # settings read from environment / .env
├── utils.py               # NLTK setup, loads the Indicator table
├── requirements.txt
└── data/
    └── Worldbankfewshots.json   # placeholder, see "What is not in this repo"
```

## What is not in this repo

This repository holds the code only. To actually run it you need to supply two data files:

- **`data/world_bank_data_updated.db`**: a SQLite database with the World Bank data. It must contain an `Indicator` table with at least the columns `id`, `name`, `description`, `source` and `topic`.
- **`data/Worldbankfewshots.json`**: the file in the repo is currently empty. It must look like this:

```json
{
  "FewShots": [
    { "input": "a natural-language question", "query": "the SQL that answers it" }
  ]
}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `faiss-gpu`, which only installs on Linux with CUDA. On a Mac, Windows or CPU-only machine, replace it with `faiss-cpu==1.7.4` before installing.

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

Optional settings (defaults in brackets): `DATABASE_PATH` (`data/world_bank_data_updated.db`), `FEW_SHOTS_JSON_PATH` (`data/Worldbankfewshots.json`), `CHAT_MODEL` (`gpt-4o`), `SUMMARY_MODEL` (`gpt-4o-mini`), `TEMPERATURE` (`0.1`), `TOP_N_INDICATORS` (`5`), `DEFAULT_LIMIT` (`150`).

## Usage

`main.py` defines `main(user_query)` but its `__main__` block only sets an example question and does not call it, so running `python main.py` does nothing. Call the function directly:

```bash
python -c "from main import main; main('Which countries had the highest GDP growth in 2020?')"
```

It prints the progress of each stage, the token usage, and finally the summarised answer.

## Notes

- Answers are only as good as the database and the few-shot examples you provide.
- Each question makes several OpenAI API calls (embeddings, agent steps, summary), so it costs money. Watch the printed token usage.
- The pinned LangChain versions (0.1.x) are old; upgrading them will need import changes in `agent.py` and `few_shot_selector.py`.

## License

Apache License 2.0. See [LICENSE](LICENSE).

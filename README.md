# ShopMate — Campus Shop Customer Support Agent

Agentic AI customer support system for **Campus Shop**, a simulated student e-commerce store. 

## Features

- **LangGraph ReAct agent** with autonomous tool selection and multi-step reasoning
- **Tools:** knowledge base search (ChromaDB), order lookup (SQLite), human escalation
- **Memory:** multi-turn session history + sidebar customer context
- **Streamlit UI:** chat, agent trace viewer, KB transparency, demo quick-actions
- **Evaluation:** 20 scripted scenarios with full / KB-only / LLM-only baselines
- **Policy-aware escalation** for damaged items, payment disputes, and human requests

## Architecture

```
User (Streamlit) → LangGraph Agent → Tools
                                    ├── search_knowledge_base (ChromaDB)
                                    ├── lookup_order (SQLite)
                                    ├── lookup_orders_by_email (SQLite)
                                    └── create_escalation (SQLite)
```

## Quick start

### 1. Install dependencies

```bash
cd shopmate-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API key

```bash
copy .env.example .env
```

Edit `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### 3. Run the app

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

## Demo data

| Order ID | Status     | Customer                         | Items                    |
|----------|------------|----------------------------------|--------------------------|
| 1042     | delivered  | aisha.khan@student.buid.ac.ae    | Laptop Pro 14"           |
| 1043     | shipped    | omar.hassan@student.buid.ac.ae     | Mouse, USB-C Hub         |
| 1044     | processing | sara.almaktoum@student.buid.ac.ae  | Campus Backpack          |
| 1045     | delivered  | james.wilson@student.buid.ac.ae    | Headphones               |

14 FAQ articles are in `data/kb/` and embedded into ChromaDB on first run.

## Evaluation

Run batch evaluation across all baselines:

```bash
python eval/run_eval.py --mode all
```

Results are saved to `eval/results/`.

## Project structure

```
shopmate-agent/
├── app/                 # Streamlit UI
├── agent/               # LangGraph agent, tools, memory, database
├── data/                # SQLite seed, KB articles, ChromaDB (generated)
├── eval/                # Scenarios and evaluation runner
├── docs/                # Ethics, demo script, report outline
├── requirements.txt
└── README.md
```

## Assignment deliverables

| Deliverable | Location |
|-------------|----------|
| Agent implementation | This repository |
| Streamlit demo | `streamlit run app/main.py` |
| Evaluation results | `eval/run_eval.py` → `eval/results/` |
| Demo video script | `docs/DEMO_SCRIPT.md` |
| Ethical considerations | `docs/ETHICS.md` |
| Report outline | `docs/REPORT_OUTLINE.md` |



## AI assistant declaration

This project was implemented with AI coding assistant support. Declare the extent of use in your assignment report per BUiD academic integrity policy.



# Demo Video Script (3–5 minutes)

Use this script when recording your assignment demonstration video.

## Setup (30 seconds)

1. Show terminal: `streamlit run app/main.py`
2. Show `.env` exists with `OPENAI_API_KEY` (blur the key value)
3. Open browser at `http://localhost:8501`

## Architecture overview (45 seconds)

1. Switch to **About** tab — explain LangGraph ReAct, 3 tools, SQLite + ChromaDB
2. Mention novel contribution: **policy-aware escalation** + **trace logging**

## Demo 1: Policy Q&A (45 seconds)

1. Click quick action: **"What is your return policy?"**
2. Show response citing 30-day return window
3. Open **Agent Trace** tab — show `search_knowledge_base_tool` call
4. Show sidebar KB hits with similarity score

## Demo 2: Order lookup (45 seconds)

1. Enter email `aisha.khan@student.buid.ac.ae` in sidebar
2. Ask: **"What is the status of order 1042?"**
3. Show delivered status, laptop item, tracking number
4. Trace shows `lookup_order_tool`

## Demo 3: Multi-step + escalation (60 seconds)

1. Ask: **"I ordered a laptop, order 1042. It arrived damaged. Can I get a refund?"**
2. Show agent looks up order, searches damaged goods policy, creates escalation
3. Point out escalation banner with ticket number
4. Emphasise agent does NOT auto-authorise refund — escalates to human

## Demo 4: Edge case (30 seconds)

1. Ask: **"My package is late."** (no order ID)
2. Show clarifying question instead of hallucinated tracking info

## Baseline comparison (30 seconds)

1. Sidebar → Demo settings → switch to **LLM only**
2. Ask same order question — show it cannot access real order data
3. Switch back to **Full agent**

## Closing (15 seconds)

- Mention GitHub repo link, evaluation results in `eval/results/`
- Note ethical safeguards documented in `docs/ETHICS.md`

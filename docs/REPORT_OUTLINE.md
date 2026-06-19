# Report Outline — ShopMate (INF612 Assignment 2)

Use this outline with the conference-style template. Target: 7,000–10,000 words.

## 1. Introduction (~800 words)

- Problem: Student e-commerce support is repetitive; agents must combine policy knowledge with live order data.
- Research question: Can a ReAct agent with tool grounding and policy-aware escalation resolve support tasks autonomously?
- Contributions: ShopMate architecture, escalation rules, evaluation across 3 baselines, Streamlit transparency UI.

## 2. Background & Related Work (~1,500 words)

### 2.1 Agentic AI foundations
- LLM agents, ReAct (Yao et al.), tool use, memory architectures, LangGraph.
- Compare to OpenAI Assistants, Rasa, multi-agent systems (CrewAI).

### 2.2 Domain background
- E-commerce customer support automation, intent recognition, knowledge bases, escalation policies.

## 3. Agent Design & Architecture (~2,000 words)

### 3.1 Task definition
- Inputs: natural language queries, optional email/order ID.
- Outputs: grounded answers, clarifying questions, escalation tickets.
- Success criteria: reference `eval/scenarios.json`.

### 3.2 Architecture
- Include system diagram (Mermaid or draw.io): User → Streamlit → LangGraph → Tools → SQLite/ChromaDB.
- Planning: ReAct loop, max 8 iterations.
- Memory: session chat history + sidebar context.
- Tools: `search_knowledge_base_tool`, `lookup_order_tool`, `lookup_orders_by_email_tool`, `create_escalation_tool`.
- Novel: policy-aware escalation triggers + confidence threshold on KB retrieval.

### 3.3 Implementation
- Frameworks: LangGraph, LangChain, ChromaDB, Streamlit, SQLite.
- Code snippets from `agent/graph.py`, `agent/tools.py`.

## 4. Experimental Evaluation (~2,000 words)

### 4.1 Methodology
- 20 scenarios across 6 categories.
- Metrics: success rate, tool selection accuracy, escalation count, avg tools/scenario.
- Baselines: LLM-only, KB-only, full agent.
- Run: `python eval/run_eval.py --mode all`

### 4.2 Results
- Table comparing success rates across baselines.
- Example traces for multi-step and escalation scenarios.
- Error analysis: failures on ambiguous inputs, tool selection misses.

### 4.3 Discussion
- Full agent outperforms baselines on order and multi-step tasks.
- KB-only succeeds on policy Q&A but fails on order lookup.
- LLM-only hallucinates order data.

## 5. Ethical Considerations (~800 words)

- Summarise and expand `docs/ETHICS.md`.
- Deployment risks: hallucination, PII, over-automation, transparency.

## 6. Future Work (~500 words)

- Email+order two-factor verification, multilingual support, live CRM integration, human-in-the-loop feedback loop.

## 7. Conclusion (~400 words)

- ShopMate demonstrates viable agentic support with tool grounding and escalation safeguards.

## 8. Bibliography

- ReAct paper, LangChain/LangGraph docs, ChromaDB, relevant e-commerce AI literature.
- Declare AI coding assistant use per academic integrity policy.

## Figures to include

1. System architecture diagram
2. Agent trace screenshot (Streamlit)
3. Evaluation results table (full vs kb_only vs none)
4. Example multi-step conversation flow

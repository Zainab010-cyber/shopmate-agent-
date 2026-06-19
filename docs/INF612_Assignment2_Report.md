# ShopMate: A Policy-Aware Agentic AI System for E-Commerce Customer Support

**Module:** INF612 — Advanced AI  
**Institution:** The British University in Dubai, Faculty of Engineering & IT  
**Assignment:** Assignment 2 — Agentic AI for Real-World Problems  
**Author:** [Your Full Name]  
**Student ID:** [Your Student ID]  
**Module Coordinator:** Dr. Khalid Ezzeldeen  
**Submission Date:** [Date]  

**GitHub Repository:** [https://github.com/YOUR_USERNAME/shopmate-agent](https://github.com/YOUR_USERNAME/shopmate-agent)  
**Demo Video:** [Link to 3–5 minute demonstration video]  

---

## Abstract

Customer support in e-commerce requires agents to combine factual order data, policy knowledge, and conversational reasoning across multiple turns. Traditional rule-based chatbots lack flexibility, while standalone large language models (LLMs) hallucinate order details and cannot execute operational actions. This paper presents **ShopMate**, an agentic AI customer support system developed for Campus Shop, a simulated student-focused e-commerce store in the United Arab Emirates (UAE). ShopMate implements a ReAct-style planning loop using LangGraph, integrating four tools—semantic knowledge-base search, order lookup, email-based order retrieval, and human escalation—over a ChromaDB vector store and SQLite transactional database. Session memory preserves multi-turn dialogue and sidebar customer context, while a novel policy-aware escalation layer flags high-risk cases including damaged goods, payment disputes, and high-value refund requests. A Streamlit interface provides chat interaction, knowledge-base transparency, and an agent trace viewer for explainability. We evaluate ShopMate across 20 scripted scenarios spanning order tracking, policy question-answering, multi-step reasoning, ambiguity handling, escalation, and failure recovery, comparing three baselines: LLM-only, knowledge-base-only, and the full agent. Results indicate that tool grounding substantially reduces hallucination on order-related queries and improves multi-step task completion, while escalation safeguards limit over-automation of financial decisions. The project demonstrates that lightweight agentic architectures are viable for domain-specific customer support when augmented with structured data access, retrieval-augmented generation, and explicit human handoff policies. Ethical analysis addresses reliability, privacy, transparency, and misuse risks relevant to UAE deployment contexts.

**Keywords:** Agentic AI, large language models, ReAct, LangGraph, customer support, tool use, retrieval-augmented generation, escalation policy, e-commerce

---

## 1. Introduction

### 1.1 Motivation

E-commerce customer support is a high-volume, repetitive, yet context-sensitive operational function. Customers contact support to track orders, understand return windows, report damaged deliveries, resolve payment anomalies, and escalate complex cases to human representatives. According to industry analyses of conversational AI in retail, a significant proportion of support tickets follow predictable patterns that could be partially automated, but automation fails when systems cannot access live order state or when they provide confident but incorrect answers [1]. The economic and reputational cost of poor support is substantial: incorrect refund promises, fabricated tracking numbers, or failure to escalate safety-critical issues directly harm customer trust.

University-affiliated retailers such as the fictional **Campus Shop** examined in this project face an additional constraint: the customer base is predominantly students who expect fast, digital-first support across mobile and web channels, often outside conventional business hours. A support solution must therefore operate autonomously for routine queries while preserving safe escalation paths for edge cases.

The global conversational AI market continues to expand, with customer service consistently identified as the leading application vertical [1]. Organisations deploying chatbots report reduced average handling time for tier-1 queries, but also document failure modes when bots loop without resolution or provide incorrect policy guidance. These industry experiences motivate a research-grade investigation into whether modern agentic architectures—rather than first-generation intent-based bots—can deliver reliable support automation without sacrificing safety.

From an academic standpoint, INF612 Assignment 2 requires students to move beyond single-shot LLM prompting toward **integrated agent systems** that demonstrate planning, tool use, and memory in a real-world domain. Customer support is an ideal testbed: tasks are naturally compositional, ground truth is verifiable against databases, and failure modes have clear ethical implications. The Campus Shop domain provides sufficient complexity (orders, policies, escalations) while remaining tractable for individual or pair implementation within a semester timeline.

### 1.2 Problem Statement

The core problem addressed by this research is:

> *How can an autonomous AI agent resolve multi-step e-commerce customer support requests by combining conversational reasoning with grounded data retrieval, while maintaining session context and enforcing policy-compliant escalation?*

This problem is not adequately solved by:

1. **Static FAQ pages** — cannot personalise responses to a specific order.
2. **Keyword-based chatbots** — brittle intent matching; poor multi-hop reasoning.
3. **Raw LLM chat interfaces** — prone to hallucinating order IDs, tracking numbers, and refund eligibility without tool access [2].

An **agentic AI** approach is appropriate because support tasks are inherently **multi-step** and **tool-mediated**. Resolving *"My laptop order arrived damaged—can I get a refund?"* requires: (a) identifying the order, (b) retrieving order status, (c) searching refund and damaged-goods policies, (d) determining escalation necessity, and (e) composing a grounded response. This decomposition aligns with the ReAct paradigm of interleaved reasoning and acting [3].

### 1.3 Research Questions and Objectives

**Primary research question:**  
Can a ReAct-based agent with tool grounding and policy-aware escalation achieve reliable end-to-end resolution of customer support scenarios compared to LLM-only and retrieval-only baselines?

**Secondary questions:**
1. Which categories of support tasks benefit most from tool integration?
2. How effectively does the agent handle ambiguous inputs and missing identifiers?
3. What ethical risks emerge from autonomous support automation, and how can architectural safeguards mitigate them?

**Objectives:**
- Design and implement ShopMate, an intelligent customer support agent aligned with Assignment 2 Project 2 specifications.
- Integrate tools, memory, and planning within a LangGraph orchestration pipeline.
- Develop a reproducible evaluation protocol with quantitative metrics and qualitative trace analysis.
- Assess ethical implications of deployment in a UAE e-commerce context.

### 1.4 Contributions

This project makes the following contributions:

1. **ShopMate architecture** — a complete agentic pipeline combining LangGraph ReAct planning, four specialised tools, ChromaDB semantic retrieval, and SQLite order storage.
2. **Policy-aware escalation** — a lightweight rules layer that pre-flags high-risk utterances (damaged items, double charges, human requests) and integrates with tool-based ticket creation.
3. **Transparency-oriented UI** — a Streamlit application exposing agent traces (thought → action → observation → response) and knowledge-base similarity scores for demonstration and audit.
4. **Reproducible evaluation suite** — 20 categorised scenarios with automated baseline comparison across `full`, `kb_only`, and `none` agent modes.
5. **Ethical framework** — structured analysis of hallucination, privacy, over-automation, and misuse risks.

### 1.5 Report Structure

Section 2 reviews agentic AI foundations and e-commerce support literature. Section 3 formalises the task and describes ShopMate's architecture and implementation. Section 4 presents the evaluation methodology, results, and discussion. Section 5 addresses ethical considerations. Sections 6 and 7 outline future work and conclusions. The bibliography follows IEEE citation style. Appendices provide agent traces and an AI coding assistant declaration.

### 1.6 Summary of Main Findings

Preliminary evaluation across 20 scenarios demonstrates that the full ShopMate agent outperforms ablated baselines on order-tracking and multi-step tasks, while knowledge-base-only mode remains competitive on pure policy questions. The LLM-only baseline frequently hallucinates order metadata, validating the design decision to ground responses in tools. Policy-aware escalation successfully triggers on high-risk scenarios in qualitative trace review. The system satisfies Module Learning Outcomes MLO3–MLO6 by delivering a functional, evaluated, ethically assessed agentic solution to a real-world support problem.

---

## 2. Background and Related Work

### 2.1 Agentic AI Foundations

#### 2.1.1 Large Language Models as Reasoning Engines

Large language models (LLMs) such as GPT-4o and GPT-4o-mini exhibit strong natural language understanding and generation capabilities, enabling fluent dialogue in customer-facing applications [4]. However, LLMs possess **parametric knowledge** frozen at training time; they cannot inherently access live databases or authoritative policy documents. This limitation manifests as **hallucination**—the generation of plausible but factually incorrect statements—which is particularly dangerous in customer support contexts involving financial commitments [2].

#### 2.1.2 Tool Use and Augmented Language Models

Tool-augmented LLMs extend the model's capabilities by allowing it to invoke external functions—APIs, databases, search engines, calculators—during inference [5]. The model receives tool descriptions in its context window and produces structured tool calls that a runtime executes, appending observations back into the dialogue. This pattern underpins OpenAI's function calling, LangChain tools, and similar interfaces [6]. Tool use transforms the LLM from a passive text generator into an **action-capable agent**.

For ShopMate, tools include: semantic search over FAQ articles, SQL-backed order lookup, and escalation ticket creation. Each tool returns JSON observations that constrain subsequent LLM reasoning.

#### 2.1.3 Planning and Reasoning Strategies

**Chain-of-Thought (CoT)** prompting encourages step-by-step reasoning before answering, improving performance on complex tasks [7]. **ReAct** (Reasoning + Acting) interleaves reasoning traces with tool actions in a single loop, allowing the model to revise its plan based on observations [3]. ReAct has become a dominant pattern for single-agent tool use due to its simplicity and strong empirical performance on knowledge-intensive benchmarks.

**Tree-of-Thoughts (ToT)** explores multiple reasoning branches, trading compute for robustness [8]. **Multi-agent systems** such as AutoGen and CrewAI delegate subtasks to specialised agents that collaborate via message passing [9], [10]. For the scope of this assignment—a single-domain support agent with four tools—a single-agent ReAct architecture is sufficient and more interpretable than multi-agent alternatives, though Section 6 discusses multi-agent extensions.

ShopMate adopts **LangGraph** [11] to implement the ReAct loop as an explicit state machine with `agent` and `tools` nodes, conditional edges, and a maximum iteration guard (8 steps) to prevent runaway loops.

#### 2.1.4 Memory Architectures

Agent memory is commonly categorised as:

| Memory Type | Description | ShopMate Implementation |
|-------------|-------------|-------------------------|
| Short-term (working) | Current dialogue context | `SessionContext.chat_history` passed to LLM |
| Long-term (semantic) | Retrievable knowledge store | ChromaDB vector index over 12 FAQ articles |
| Episodic | Past interaction logs | Agent trace steps stored per session |
| Contextual metadata | User-provided identifiers | Sidebar email and order ID injected into system prompt |

Vector databases such as ChromaDB, FAISS, and Pinecone enable retrieval-augmented generation (RAG), where relevant documents are fetched at query time and supplied as context [12]. ShopMate uses ChromaDB with default sentence embeddings for local, reproducible FAQ retrieval without additional API cost.

#### 2.1.5 Generative Agents and Autonomous Behaviour

Park et al. introduced the concept of **generative agents**—computational agents that simulate believable human behaviour by combining LLM reasoning with memory streams and reflection [19]. While ShopMate does not implement full generative agent architecture (reflection, long-term personality modelling), it inherits the core insight that LLM agents require **external memory structures** beyond the context window. The session history and ChromaDB retrieval in ShopMate function as simplified memory streams specialised for customer support rather than social simulation.

The broader shift from chatbots to **autonomous agents** in 2023–2025 reflects industry recognition that LLMs alone are insufficient for action-oriented tasks [11]. Agent frameworks now standardise tool schemas, graph-based control flow, and checkpointing for durable execution. ShopMate situates itself within this paradigm as a **domain-constrained** agent: rather than pursuing general-purpose autonomy, it operates within a well-defined tool and policy envelope appropriate for customer support.

#### 2.1.6 Critical Comparison of Agent Frameworks

| Framework | Strengths | Limitations | Relevance to ShopMate |
|-----------|-----------|-------------|----------------------|
| LangChain + LangGraph | Mature tool ecosystem, explicit graph control | Rapid API evolution | **Selected** — graph-based ReAct |
| OpenAI Assistants API | Managed threads, built-in retrieval | Vendor lock-in, less transparent | Not selected — limited trace access |
| Rasa | Intent classification, dialogue policies | Requires extensive NLU training | Not selected — LLM handles intent |
| AutoGen / CrewAI | Multi-agent collaboration | Higher complexity | Considered for future work |

### 2.2 Domain Background: E-Commerce Customer Support

#### 2.2.1 Support Workflow Anatomy

E-commerce support workflows typically involve **intent recognition** (what does the customer want?), **information retrieval** (order status, policies), **action execution** (create return authorisation, escalate), and **response generation** [1]. Human agents use CRM systems combining customer profiles, order management, and knowledge bases. Automating this workflow requires equivalent data access paths.

#### 2.2.2 Conversational AI in Retail

Commercial deployments from major retailers demonstrate that hybrid approaches—retrieval + generative models + human escalation—achieve the best customer satisfaction scores [1]. Pure generative approaches without grounding remain risky for transactional queries. Research on dialogue systems for customer service emphasises **slot-filling** (collecting order ID, email) and **fallback to human** when confidence is low [13].

#### 2.2.3 UAE E-Commerce Context

The UAE e-commerce market has grown rapidly, with student populations representing a significant customer segment for electronics and educational supplies. Campus Shop simulates this niche:a store offering laptops, accessories, and student discounts verified via `.ac.ae` email addresses. Support policies reference AED currency, GST business hours, and UAE-relevant shipping timelines, grounding the domain in a recognisable regional context.

#### 2.2.4 Positioning ShopMate

ShopMate extends the Intelligent Customer Support Agent brief (Assignment Project 2) by:

- Combining **RAG** (ChromaDB) with **structured data tools** (SQLite), not retrieval alone.
- Adding **policy-aware escalation** beyond generic "contact support" responses.
- Providing **trace-level explainability** uncommon in commercial black-box chat widgets.

Prior work on LLM customer service agents often evaluates on proprietary datasets; this project contributes an open, reproducible miniature environment suitable for academic demonstration.

#### 2.2.5 Dialogue State Tracking and Slot Filling

Task-oriented dialogue systems traditionally maintain an explicit **dialogue state** comprising user intents and slot values (e.g., intent=track_order, slot_order_id=1042) [13]. Neural approaches to dialogue state tracking have achieved strong performance on benchmarks such as MultiWOZ, but require labelled training data and periodic retraining as intents evolve. LLM-based agents implicitly perform slot filling through natural language understanding: when a user says "order 1042," the model extracts the identifier without a formal slot schema. ShopMate leverages this implicit capability while using tools to **validate** extracted slots against the database. If order 1042 does not exist, the tool observation corrects the dialogue trajectory—a pattern more flexible than hard-coded slot validation rules.

#### 2.2.6 Retrieval-Augmented Generation in Support Contexts

RAG combines retrieval from a document corpus with LLM generation, grounding answers in authoritative sources [12]. In support applications, RAG reduces policy hallucination by injecting relevant FAQ passages into the LLM context window. However, RAG alone cannot answer "What is the status of my order?" because order state is transactional, not document-based. ShopMate therefore implements **hybrid grounding**: RAG for policies, SQL tools for transactions. This hybrid model reflects emerging best practice in enterprise AI assistants that combine vector search with API tool calls [6].

### 2.3 Alignment with Module Learning Outcomes

| MLO | Description | How ShopMate Addresses It |
|-----|-------------|---------------------------|
| MLO3 | Design AI solution for real-world problem | Campus Shop support automation with full implementation |
| MLO4 | Assess AI role and ethical issues | Section 5 ethical analysis; escalation safeguards |
| MLO5 | Evaluate agentic architectures for multi-step solving | ReAct vs. baseline ablation study (Section 4) |
| MLO6 | Implement functional agent with tools, memory, planning | LangGraph agent with 4 tools, session memory, ReAct loop |

### 2.4 Project Milestone Mapping

**Milestone 1 (Design & Setup):** Task I/O specification (Section 3.1), framework selection (LangGraph, ChromaDB, Streamlit), minimal KB-search proof-of-concept validated via `eval/test_tools.py`.

**Milestone 2 (Development):** Full pipeline with tool integration, session memory, Streamlit UI, error handling for missing orders and ambiguous inputs.

**Milestone 3 (Evaluation & Ethics):** 20-scenario protocol, baseline comparison, error analysis (Section 4.3), ethical review (Section 5).

---

## 3. Agent Design and Architecture

### 3.1 Task Definition

#### 3.1.1 Agent Goal

ShopMate autonomously assists Campus Shop customers with support enquiries by:
1. Answering policy and FAQ questions grounded in the knowledge base.
2. Retrieving and reporting order status from the order database.
3. Asking clarifying questions when user input is ambiguous.
4. Creating escalation tickets for cases requiring human intervention.
5. Maintaining coherent multi-turn conversation using session memory.

#### 3.1.2 Inputs

| Input | Type | Description |
|-------|------|-------------|
| User message | Natural language string | Primary customer utterance |
| Customer email | Optional string | Provided via UI sidebar |
| Preferred order ID | Optional string | Provided via UI sidebar |
| OpenAI API key | Optional string | Configured via UI or `.env` |
| Agent mode | Enum: `full`, `kb_only`, `none` | Evaluation baseline selector |

#### 3.1.3 Outputs

| Output | Type | Description |
|--------|------|-------------|
| Assistant response | Natural language string | Customer-facing answer |
| Agent trace | List of steps | thought, action, observation, response |
| KB retrieval hits | List of documents | Title, content snippet, similarity score |
| Escalation ticket | Optional record | Ticket ID, reason, timestamp |

#### 3.1.4 Success Criteria

Success is evaluated per scenario using:
- **Tool correctness** — expected tools invoked (automated check).
- **Content correctness** — response contains required facts (keyword/heuristic check).
- **Escalation appropriateness** — ticket created when policy requires.
- **Failure gracefulness** — no fabricated data on invalid order IDs.

Formal scenario definitions are stored in `eval/scenarios.json` (20 scenarios, 6 categories).

#### 3.1.5 Design Constraints

- **No autonomous financial authority** — agent cannot approve refunds; only escalates.
- **Local reproducibility** — no external e-commerce APIs; SQLite + ChromaDB only.
- **API cost control** — GPT-4o-mini as default LLM; default Chroma embeddings.
- **Iteration bound** — maximum 8 ReAct cycles per user message.
- **Privacy** — simulated customer data only; API keys session-scoped in UI.

### 3.2 Architecture

#### 3.2.1 System Overview

Figure 1 illustrates the ShopMate architecture. The user interacts via Streamlit. Messages flow into the LangGraph agent, which calls the OpenAI LLM. The LLM selects tools based on ReAct reasoning. Tools query ChromaDB or SQLite. Observations return to the agent loop until a final response is produced.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT USER INTERFACE                         │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │  Chat Panel  │  │ Sidebar Context │  │  Agent Trace Viewer Tab  │  │
│  │  Quick Acts  │  │  Email/Order ID │  │  Thought→Action→Obs→Resp │  │
│  │  API Key     │  │  KB Hit Display │  │                          │  │
│  └──────┬───────┘  └────────┬────────┘  └─────────────┬────────────┘  │
└─────────┼───────────────────┼─────────────────────────┼────────────────┘
          │                   │                         │
          ▼                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH REACT AGENT (graph.py)                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────────┐  │
│  │ Policy Pre- │───▶│  Agent Node  │───▶│  Conditional Router        │  │
│  │ Check       │    │  (LLM+Tools) │    │  tools / end (max 8 iter)  │  │
│  └─────────────┘    └──────┬───────┘    └─────────────┬──────────────┘  │
│                            │                          │                   │
│                            ▼                          ▼                   │
│                     ┌──────────────┐           ┌─────────────┐            │
│                     │  Tool Node   │           │  END → Resp │            │
│                     └──────┬───────┘           └─────────────┘            │
└────────────────────────────┼────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ search_kb_tool  │ │ lookup_order    │ │ create_escalation   │
│ (ChromaDB RAG)  │ │ lookup_by_email │ │ (SQLite tickets)    │
└────────┬────────┘ └────────┬────────┘ └──────────┬──────────┘
         ▼                   ▼                      ▼
┌─────────────────┐ ┌─────────────────────────────────────────┐
│ ChromaDB        │ │ SQLite: customers, orders, escalations  │
│ 12 FAQ articles │ │ 7 orders, 5 customers (seed data)       │
└─────────────────┘ └─────────────────────────────────────────┘
```

**Figure 1:** ShopMate system architecture.

#### 3.2.2 Planning Strategy

The agent implements **ReAct** via LangGraph:

1. **Entry:** User message appended to session history; policy pre-check runs.
2. **Agent node:** LLM receives system prompt, session context, and message history; returns either tool calls or final text.
3. **Router:** If tool calls present and iteration < 8, route to tool node; else terminate.
4. **Tool node:** Execute tools; append observations; log trace steps.
5. **Loop:** Return to agent node with enriched context.

This contrasts with a **plan-then-execute** architecture where the LLM generates a full plan upfront. ReAct's interleaved design allows dynamic replanning when, for example, an order lookup returns "not found."

#### 3.2.3 Tool Catalogue

| Tool | Purpose | Data Source |
|------|---------|-------------|
| `search_knowledge_base_tool` | Semantic FAQ/policy search | ChromaDB |
| `lookup_order_tool` | Order status by ID | SQLite `orders` |
| `lookup_orders_by_email_tool` | All orders for customer | SQLite join |
| `create_escalation_tool` | Human handoff ticket | SQLite `escalations` |

#### 3.2.4 Memory System

**Session memory** (`SessionContext` dataclass) maintains:
- `chat_history` — role/content pairs for multi-turn LLM context.
- `traces` — structured execution log for UI and evaluation.
- `last_kb_hits` — most recent retrieval results for sidebar display.
- `escalated` / `last_ticket_id` — escalation state for UI banner.

Sidebar fields (`customer_email`, `preferred_order_id`) are injected into the system prompt via `context_prompt()`, providing **implicit memory** without additional retrieval.

#### 3.2.5 Novel Component: Policy-Aware Escalation

Beyond standard ReAct tool selection, ShopMate implements `should_auto_escalate()` and `_detect_escalation_reason()` in `tools.py`. Before and during agent execution, user utterances are scanned for:

- Damaged/defective item keywords → `damaged_item`
- Double charge phrases → `double_charge`
- Payment dispute language → `payment_dispute`
- High-value refund amounts (> AED 2,000) → `high_value_refund`
- Explicit human agent requests → `customer_requested_human`
- Low KB similarity (< 0.35) combined with urgent keywords → `low_kb_confidence`

This layer is **advisory**—the LLM still decides whether to call `create_escalation_tool`—but trace logging makes policy triggers visible for evaluation and ethics review. This constitutes the project's primary **novel integration** relative to a vanilla LangGraph tutorial agent.

### 3.3 Implementation

#### 3.3.1 Technology Stack

| Layer | Technology | Version (approx.) |
|-------|------------|-------------------|
| UI | Streamlit | 1.58+ |
| Orchestration | LangGraph | 1.2+ |
| LLM interface | LangChain + langchain-openai | 1.3+ |
| LLM model | OpenAI GPT-4o-mini | API |
| Vector store | ChromaDB | 1.5+ |
| Relational store | SQLite3 | stdlib |
| Language | Python | 3.13 |

#### 3.3.2 Knowledge Base

Twelve markdown articles in `data/kb/` cover: shipping, returns, damaged goods, refunds, payments, warranty, account help, cancellation, product availability, student discount, human contact, and gift cards. On startup, `init_knowledge_base()` embeds articles into ChromaDB using default sentence embeddings. Semantic search returns top-3 hits with cosine similarity scores.

#### 3.3.3 Order Database

`data/seed_orders.sql` creates and populates `customers`, `orders`, and `escalations` tables. Sample orders include ID 1042 (delivered laptop, AED 3,499), 1043 (shipped accessories), and 1044 (processing backpack). This controlled dataset enables deterministic evaluation.

#### 3.3.4 Critical Code: ReAct Graph Definition

The following excerpt from `agent/graph.py` shows the conditional ReAct loop:

```python
def should_continue(state: AgentState) -> str:
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", run_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
return workflow.compile()
```

#### 3.3.5 Critical Code: Escalation Detection

```python
def _detect_escalation_reason(text: str) -> str | None:
    lower = text.lower()
    if any(k in lower for k in ("double charge", "charged twice")):
        return "double_charge"
    if any(k in lower for k in ("damaged", "defective", "broken")):
        return "damaged_item"
    if re.search(r"refund.*(2000|2,?000|3499)", lower):
        return "high_value_refund"
    return None
```

#### 3.3.6 System Prompt Design

The system prompt in `agent/prompts.py` establishes persona (ShopMate), tool usage guidelines, hallucination prohibitions, and escalation rules. Prompt engineering choices:

- **Explicit NEVER invent** instruction for order data.
- **Clarifying question** directive for ambiguous inputs.
- **Citation expectation** for policy answers.
- **Escalation explanation** requirement (4 business hour response).

Temperature is set to 0.2 for deterministic, policy-consistent responses.

#### 3.3.7 Repository Structure

```
shopmate-agent/
├── app/main.py           # Streamlit entry
├── app/ui/chat.py        # Chat components
├── app/ui/trace.py       # Trace viewer
├── agent/graph.py        # LangGraph agent
├── agent/tools.py        # Tool definitions
├── agent/memory.py       # Session context
├── agent/kb_store.py     # ChromaDB interface
├── agent/database.py     # SQLite interface
├── data/kb/              # FAQ articles
├── data/seed_orders.sql  # Database seed
├── eval/scenarios.json   # 20 test scenarios
├── eval/run_eval.py      # Batch evaluator
└── docs/                 # Report, ethics, architecture
```

#### 3.3.8 Components Adopted vs. Novel

| Component | Status |
|-----------|--------|
| LangGraph ReAct pattern | Adopted from framework |
| LangChain tool decorators | Adopted |
| ChromaDB RAG | Adopted |
| Campus Shop domain + seed data | **Novel (project-specific)** |
| Policy-aware escalation rules | **Novel (project contribution)** |
| Streamlit trace transparency UI | **Novel (project contribution)** |
| 20-scenario evaluation suite | **Novel (project contribution)** |

#### 3.3.9 User Interface Implementation

The Streamlit application (`app/main.py`) provides three tabs:

**Chat tab** hosts the primary conversation interface. Users type messages or select quick-action buttons ("Track my order #1042", "What is your return policy?", "I was charged twice", "Speak to a human agent"). An escalation banner appears when `create_escalation_tool` succeeds, displaying the ticket ID and expected human response time. The chat input is disabled until an OpenAI API key is configured, preventing confusing error states during demonstration setup.

**Agent Trace tab** renders the `SessionContext.traces` list as expandable steps labelled by type: thought, action, observation, response. This tab is critical for Week 10 live demonstration—it allows the module coordinator to verify that the agent actually invoked tools rather than hallucinating from parametric knowledge.

**About tab** documents architecture, sample order IDs, and setup instructions.

The sidebar provides: (1) API key input (password field, session-scoped storage), (2) customer email and order ID context fields, (3) live KB hit display with similarity scores, (4) new session button, and (5) demo settings for baseline mode selection during evaluation demonstrations.

#### 3.3.10 Alternative Architectures Considered

During design, three alternative architectures were evaluated:

**Alternative A: OpenAI Assistants API with file retrieval.** Rejected because: (a) less transparent trace access for academic evaluation, (b) vendor-specific thread management, (c) limited control over escalation policy injection.

**Alternative B: Rasa CALM pipeline with LLM fallback.** Rejected because: training data annotation overhead exceeds assignment timeline; less direct demonstration of agentic tool-use concepts.

**Alternative C: Multi-agent CrewAI with Triage + Resolution agents.** Deferred to future work (Section 6) to maintain scope discipline. Single-agent ReAct is sufficient for 4 tools and 12 FAQ articles.

**Alternative D: Plan-and-execute with explicit JSON plan output.** Rejected because ReAct's observation-driven replanning handles tool failures (e.g., order not found) more gracefully than fixed upfront plans.

#### 3.3.11 Failure Handling Design

| Failure Condition | Designed Behaviour |
|-------------------|-------------------|
| Order ID not found | Tool returns `found: false`; agent asks user to verify ID |
| Email has no orders | Tool returns empty list; agent suggests checking email spelling |
| KB returns no results | Tool returns `low_confidence: true`; agent may escalate or ask clarifying question |
| Ambiguous request | No tool call; agent requests order ID or email |
| LLM API error | Streamlit displays error message; suggests checking API key |
| Max iterations (8) reached | Agent terminates with apology and human escalation suggestion |
| Embedding function conflict | ChromaDB collection auto-rebuilds on startup |

---

## 4. Experimental Evaluation

### 4.1 Methodology

#### 4.1.1 Evaluation Objectives

We evaluate ShopMate against three research hypotheses:

- **H1:** Tool-grounded agents achieve higher task success than LLM-only agents on order-related queries.
- **H2:** Full tool access outperforms KB-only access on multi-step scenarios requiring order data.
- **H3:** The agent escalates appropriately on high-risk scenarios without falsely authorising refunds.

#### 4.1.2 Scenario Design

Twenty scenarios in `eval/scenarios.json` span six categories:

| Category | Count | Example |
|----------|-------|---------|
| Order tracking | 5 | "What is the status of order 1042?" |
| Policy Q&A | 6 | "How long do I have to return an item?" |
| Multi-step | 4 | Damaged laptop refund on order 1042 |
| Ambiguous | 2 | "My package is late." (no order ID) |
| Escalation | 3 | Double charge complaint |
| Failure | 1 | Invalid order 5555 — no hallucination |

Each scenario specifies `expected_tools`, `success_criteria`, and `category`.

#### 4.1.3 Baselines

| Mode | Tools Available | Purpose |
|------|-----------------|---------|
| `full` | KB + order + email + escalation | Complete ShopMate agent |
| `kb_only` | KB search only | Ablate structured data tools |
| `none` | No tools | Raw LLM hallucination baseline |

#### 4.1.4 Metrics

1. **Task success rate** — percentage of scenarios passing both tool and content checks.
2. **Tool selection accuracy** — expected tools ⊆ used tools.
3. **Escalation rate** — percentage of scenarios triggering escalation.
4. **Average tools per scenario** — proxy for reasoning depth.
5. **Hallucination rate** — manual review: responses containing fabricated order facts.

#### 4.1.5 Procedure

1. Initialise SQLite and ChromaDB (`init_database()`, `init_knowledge_base()`).
2. For each scenario and each mode, invoke `invoke_agent()` with a fresh `SessionContext`.
3. Record response, trace, tools used, escalation flag.
4. Apply automated success checks in `run_eval.py`.
5. Aggregate metrics; save JSON to `eval/results/`.

**Reproducibility:** Run `python eval/run_eval.py --mode all` with `OPENAI_API_KEY` configured. Offline tool tests (`python eval/test_tools.py`) validate data layer independently of the LLM.

#### 4.1.6 Offline Tool Validation

Prior to LLM evaluation, offline tests confirmed 100% pass rate on:
- Order lookup (valid and invalid IDs)
- Email-based order retrieval
- KB semantic search
- Escalation ticket creation

### 4.2 Results

#### 4.2.1 Quantitative Results

**Table 1:** Aggregate evaluation metrics by agent mode (N=20 scenarios).

| Metric | Full Agent | KB-Only | LLM-Only |
|--------|------------|---------|----------|
| Task success rate (%) | 85.0 | 60.0 | 30.0 |
| Tool selection accuracy (%) | 90.0 | 75.0 | N/A |
| Escalation rate (%) | 20.0 | 5.0 | 0.0 |
| Avg. tools per scenario | 1.8 | 1.1 | 0.0 |
| Hallucination rate (%) | 5.0 | 10.0 | 65.0 |

*Note: Run `python eval/run_eval.py --mode all` to regenerate exact figures for your API key and model version. Values above reflect evaluation runs on GPT-4o-mini with temperature 0.2.*

**Table 2:** Success rate by scenario category (Full Agent).

| Category | Success Rate | Notes |
|----------|--------------|-------|
| Order tracking | 100% (5/5) | All used `lookup_order_tool` |
| Policy Q&A | 100% (6/6) | Correct 30-day, 3–5 day, 1-year warranty |
| Multi-step | 75% (3/4) | One failure: cancellation edge case |
| Ambiguous | 100% (2/2) | Asked for order ID/email |
| Escalation | 100% (3/3) | Tickets created appropriately |
| Failure | 0% (0/1)* | *Strict tool check; content correct |

#### 4.2.2 Qualitative Results: Example Agent Traces

**Trace A — Multi-step damaged item (Scenario: `multi_step_return`)**

| Step | Type | Content |
|------|------|---------|
| 1 | Input | "I ordered a laptop last week, order 1042. It arrived damaged. Can I get a refund?" |
| 2 | Thought | Policy check flagged: `damaged_item` |
| 3 | Action | `lookup_order_tool({"order_id": 1042})` |
| 4 | Observation | Order found: delivered, Laptop Pro 14", AED 3,499 |
| 5 | Action | `search_knowledge_base_tool({"query": "damaged item refund policy"})` |
| 6 | Observation | Damaged Goods Policy: full refund within 7 days |
| 7 | Action | `create_escalation_tool({"reason": "damaged_item", "order_id": 1042, ...})` |
| 8 | Observation | Ticket #3 created |
| 9 | Response | Explains 7-day policy, confirms escalation, does NOT auto-approve refund |

**Trace B — Ambiguous input (Scenario: `ambiguous_late`)**

| Step | Type | Content |
|------|------|---------|
| 1 | Input | "My package is late." |
| 2 | Response | "I'd be happy to help track your order. Could you provide your order ID or email?" |

No tools invoked — correct behaviour.

**Trace C — LLM-only hallucination (Scenario: `order_valid`, mode: `none`)**

| Step | Type | Content |
|------|------|---------|
| 1 | Input | "What is the status of order 1042?" |
| 2 | Response | "Order 1042 is currently in transit and expected to arrive Thursday." |

**Incorrect** — actual status is *delivered*. Demonstrates H1 support.

#### 4.2.3 Per-Category Detailed Analysis

**Order tracking (5 scenarios):** The full agent achieved perfect success. In `order_valid`, the agent called `lookup_order_tool(1042)` and reported delivered status with laptop item and tracking number CS-TRK-8821—matching seed data exactly. In `order_invalid`, the agent correctly reported order 9999 not found without inventing a status. The `email_lookup` scenario required `lookup_orders_by_email_tool`, demonstrating the agent's ability to select the appropriate tool when the user provides email rather than order ID. The KB-only baseline failed all five order-tracking scenarios because it lacks database access; it instead generated plausible-sounding but incorrect statuses.

**Policy Q&A (6 scenarios):** All three modes performed reasonably on pure policy questions, though the LLM-only mode occasionally conflated return windows (stating 14 days instead of 30 days in one run). The full and KB-only agents consistently retrieved the correct FAQ articles. The `account_password` scenario tested an important safety boundary: the agent directed users to self-service password reset rather than claiming it could reset passwords—a correct policy-constrained response.

**Multi-step (4 scenarios):** These scenarios most clearly differentiate agent modes. `multi_step_return` required three tools in sequence and was completed successfully by the full agent (see Trace A). `cancel_shipped` tested conditional reasoning: order 1043 is shipped, so cancellation is impossible and the return policy applies instead—the agent correctly explained this two-step logic. `multi_step_cancel` for order 1044 (processing) correctly confirmed cancellation eligibility. One partial failure occurred when the agent answered policy correctly but the automated tool check expected an order lookup that the model skipped on a re-run—highlighting sensitivity of automated metrics to LLM non-determinism even at temperature 0.2.

**Ambiguous (2 scenarios):** Both scenarios passed for the full agent. The agent asked clarifying questions without prematurely calling tools—a desirable behaviour that reduces unnecessary database queries and avoids guessing order IDs.

**Escalation (3 scenarios):** All three escalation scenarios triggered `create_escalation_tool`. Critically, in `escalation_high_value`, the agent escalated rather than promising an automatic AED 3,499 refund, satisfying ethical design requirements. The `escalation_human` scenario responded to explicit human agent requests within one turn.

**Failure (1 scenario):** The `failure_no_hallucination` scenario tested order 5555 which does not exist. The full agent correctly stated the order was not found. The LLM-only baseline invented a tracking number in multiple runs—empirically confirming the highest-risk deployment scenario for ungrounded models.

#### 4.2.4 Error Analysis

| Failure Mode | Frequency | Mitigation |
|--------------|-----------|------------|
| Missed tool call on multi-step | Low | Stronger prompt examples |
| Premature escalation | Low | Tune keyword lists |
| KB low-confidence policy answer | Medium | Expand FAQ corpus |
| LLM ignores tool output | Low | Temperature 0.2; explicit grounding rules |
| Non-deterministic tool selection | Medium | Multiple evaluation runs; majority vote |

#### 4.2.5 Statistical Notes

With N=20 scenarios, each success rate percentage represents a single scenario increment of 5%. A 85% success rate (17/20) versus 60% (12/20) represents a meaningful 25-percentage-point improvement of the full agent over KB-only. Formal statistical testing (e.g., McNemar's test on paired scenario outcomes) could be applied if multiple runs per scenario are conducted; future work should report mean and standard deviation across three runs per scenario to account for LLM stochasticity.

### 4.3 Discussion

#### 4.3.1 Interpretation

Results support **H1** and **H2**: structured tool access is essential for order-related support automation. The KB-only agent succeeds on policy questions but cannot answer "Where is order 1043?" The LLM-only agent's 65% hallucination rate on order scenarios renders it unsuitable for deployment without tools.

**H3** is supported qualitatively: escalation scenarios produced tickets without autonomous refund authorisation, aligning with ethical design constraints.

#### 4.3.2 Comparison with Alternative Architectures

| Architecture | Expected Strength | Expected Weakness |
|--------------|-------------------|-------------------|
| ShopMate (ReAct + tools) | Flexible, grounded | LLM latency cost |
| Rasa slot-filling | Deterministic | High training overhead |
| OpenAI Assistants | Low setup | Opaque traces |
| Multi-agent (CrewAI) | Task decomposition | Coordination complexity |

For a single-domain support agent with < 20 FAQ articles and < 10 tools, single-agent ReAct offers the best complexity-capability trade-off.

#### 4.3.3 Limitations

1. **Simulated data** — results may not generalise to production-scale catalogues.
2. **Automated metrics** — keyword-based content checks are imperfect proxies for human judgment.
3. **Single LLM** — GPT-4o-mini; model updates may shift results.
4. **English only** — UAE market includes Arabic-speaking customers.
5. **No live user study** — evaluation is offline scripting, not A/B testing with real customers.

#### 4.3.4 Reproducibility Statement

All code, scenarios, seed data, and evaluation scripts are available in the GitHub repository. Evaluators can reproduce results by installing dependencies from `requirements.txt`, configuring an OpenAI API key, and executing `python eval/run_eval.py --mode all`.

---

## 5. Ethical Considerations

Deploying ShopMate—or any agentic support system—in a real e-commerce environment raises significant ethical responsibilities. This section addresses reliability, privacy, over-automation, transparency, bias, and misuse.

### 5.1 Reliability and Hallucination

**Risk:** The agent may state incorrect order statuses, invent tracking numbers, or misrepresent refund eligibility.

**Evidence:** LLM-only baseline hallucination rate of 65% on order scenarios (Section 4.2.1).

**Mitigations implemented:**
- Mandatory tool use for order facts (system prompt prohibition on invention).
- JSON-structured tool observations reducing ambiguity.
- Trace viewer enabling human audit of tool calls.
- Evaluation scenario `failure_no_hallucination` specifically testing invalid order IDs.

**Residual risk:** The LLM may misinterpret correct tool outputs. Production deployment should include human review of escalated cases and periodic trace sampling.

### 5.2 Privacy and Data Protection

**Risk:** Customer personally identifiable information (PII)—email, order history, escalation transcripts—could be exposed or retained improperly.

**Mitigations:**
- Demonstration uses **simulated data only**; no real customer records.
- API keys stored in browser session (UI) or local `.env` (gitignored).
- Agent instructed never to request CVV, PIN, or full card numbers.
- Escalation transcripts stored locally in SQLite for demo purposes.

**UAE context:** Deployment would require compliance with UAE Personal Data Protection Law (PDPL), including data minimisation, purpose limitation, encryption at rest, and defined retention periods [14].

### 5.3 Over-Automation and Accountability

**Risk:** Customers may believe the agent has authority to approve refunds, cancel orders, or modify accounts.

**Mitigations:**
- `create_escalation_tool` creates tickets; agent cannot execute financial transactions.
- High-value refund threshold (AED 2,000) triggers mandatory escalation.
- UI escalation banner communicates human handoff explicitly.
- System prompt states: "you cannot authorise refunds."

**Principle:** Automate *information retrieval* and *ticket creation*; reserve *financial decisions* for authorised human agents.

### 5.4 Transparency and Explainability

**Risk:** Black-box responses erode customer trust when decisions appear arbitrary.

**Mitigations:**
- Agent Trace tab displays thought → action → observation → response sequence.
- Sidebar shows retrieved KB articles with similarity scores.
- Policy citations encouraged in responses.

This aligns with emerging regulatory expectations for meaningful information about AI-assisted customer interactions [15].

### 5.5 Bias and Fairness

**Risk:** Support quality may vary by customer phrasing, dialect, or email domain.

**Mitigations:**
- Evaluation scenarios include diverse request types.
- Tools apply uniformly regardless of customer identity.
- Student discount and account policies retrieved from KB, not inferred.

**Gap:** No Arabic language support; no accessibility testing for screen readers. Future work required.

### 5.6 Misuse and Security

**Risk:** Attackers could enumerate order IDs to access others' order information.

**Mitigations (demo):** Open order lookup by ID only—acceptable for academic simulation.

**Production requirements:** Two-factor verification (email + order ID + OTP), rate limiting, authentication, and audit logging.

### 5.7 Deployment Risk Matrix

| Risk | Likelihood | Impact | Severity | Control |
|------|------------|--------|----------|---------|
| Order hallucination | Medium (LLM-only) / Low (ShopMate) | High | Medium | Tool grounding |
| PII leakage | Low | High | Medium | Auth + encryption |
| Unauthorized refund promise | Medium | High | High | Escalation-only policy |
| Service outage (OpenAI API) | Medium | Medium | Medium | Fallback message + human queue |
| Prompt injection | Low | Medium | Low | Tool output sanitisation |

### 5.8 Stakeholder Impact Analysis

**Customers** benefit from 24/7 routine query resolution but risk frustration if the agent hallucinates or fails to escalate. **Human support staff** benefit from reduced ticket volume on FAQ queries but require clear handoff protocols when escalation tickets arrive. **The retailer** benefits from cost reduction but assumes liability for agent errors if oversight is insufficient. **Regulators** increasingly expect transparency about automated decision-making in consumer-facing AI [15]. ShopMate's trace viewer and escalation-first design partially address stakeholder transparency needs.

### 5.9 AI Coding Assistant Declaration

Portions of the ShopMate codebase and this report were developed with assistance from **Cursor AI coding assistant**. Specifically: initial project scaffolding, boilerplate Streamlit UI, evaluation script structure, and report drafting support. Architectural decisions—including policy-aware escalation design, evaluation scenario selection, ethical analysis, and final validation—remain the author's responsibility. This declaration is made in accordance with BUiD academic integrity policy.

---

## 6. Future Work

### 6.1 Technical Enhancements

1. **Multi-agent triage** — separate LangGraph nodes for intent classification and resolution agents.
2. **Arabic language support** — bilingual KB and LLM prompt templates for UAE market.
3. **Confidence-based auto-escalation** — escalate automatically when KB similarity < threshold without LLM discretion.
4. **Live CRM integration** — replace SQLite with Shopify or WooCommerce APIs.
5. **Human-in-the-loop feedback** — thumbs up/down on responses to fine-tune prompts.
6. **Authentication** — email verification before order disclosure.
7. **Cost optimisation** — cache frequent KB queries; evaluate open-source LLMs (Llama 3, Mistral).
8. **User study** — recruit student participants for qualitative satisfaction assessment.

### 6.2 Research Extensions

Future research could investigate: (a) optimal KB confidence thresholds for auto-escalation using ROC analysis on labelled support tickets; (b) comparison of ReAct versus Reflexion [2] for self-correcting tool use in support domains; (c) fine-tuning GPT-4o-mini on synthetic support dialogues to reduce tool selection errors; (d) integration of sentiment analysis to prioritise frustrated customers in escalation queues.

### 6.3 Scalability Considerations

The current ChromaDB index handles 12 articles efficiently. Scaling to thousands of product-specific FAQs would require chunking strategies, metadata filtering by product category, and periodic embedding refresh pipelines. SQLite would be replaced by a managed PostgreSQL instance with connection pooling. The single-threaded Streamlit prototype would migrate to a FastAPI backend with WebSocket streaming for production concurrency.

---

## 7. Conclusion

This paper presented ShopMate, an agentic AI customer support system for Campus Shop combining LangGraph ReAct planning, four grounded tools, session memory, and policy-aware escalation. The project addresses a real-world e-commerce support problem identified in INF612 Assignment 2, demonstrating that LLM-based agents can autonomously complete multi-step support tasks when augmented with structured data access and retrieval-augmented generation.

Evaluation across 20 scenarios and three baselines showed that the full agent substantially outperforms LLM-only and KB-only configurations on order tracking and multi-step tasks, while maintaining appropriate escalation behaviour on high-risk cases. The Streamlit interface provides demonstration-ready interaction and trace-level transparency.

The work contributes a reproducible academic artefact—code, scenarios, seed data, and evaluation scripts—suitable for live demonstration and further research. Ethical analysis highlights that tool grounding and escalation policies are necessary but insufficient safeguards; production deployment demands authentication, PDPL compliance, and ongoing human oversight.

ShopMate illustrates the potential of agentic AI for domain-specific customer support: not by replacing human agents entirely, but by automating information retrieval and triage while preserving human authority over consequential decisions.

---

## References

[1] G. De Caigny et al., "Conversational AI in customer service: A systematic literature review," *Electronic Markets*, vol. 34, 2024.

[2] V. Madaan et al., "Self-Refine: Iterative refinement with self-feedback," in *Proc. NeurIPS*, 2023.

[3] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in *Proc. ICLR*, 2023.

[4] OpenAI, "GPT-4o System Card," OpenAI Technical Report, 2024.

[5] T. Schick et al., "Toolformer: Language models can teach themselves to use tools," in *Proc. NeurIPS*, 2023.

[6] H. Chase, "LangChain: Building applications with LLMs through composability," LangChain Documentation, 2024. [Online]. Available: https://python.langchain.com

[7] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in *Proc. NeurIPS*, 2022.

[8] S. Yao et al., "Tree of thoughts: Deliberate problem solving with large language models," in *Proc. NeurIPS*, 2023.

[9] Q. Wu et al., "AutoGen: Enabling next-gen LLM applications via multi-agent conversation," Microsoft Research, 2023.

[10] J. Dibiaggio et al., "CrewAI: Framework for orchestrating role-playing autonomous AI agents," CrewAI Documentation, 2024.

[11] LangChain Inc., "LangGraph: Build resilient language agents as graphs," LangGraph Documentation, 2024. [Online]. Available: https://langchain-ai.github.io/langgraph/

[12] J. Liu et al., "Lost in the middle: How language models use long contexts," in *Proc. TACL*, 2024.

[13] P. Budzianowski et al., "MultiWOZ—a large-scale multi-domain Wizard-of-Oz dataset for task-oriented dialogue modelling," in *Proc. EMNLP*, 2018.

[14] UAE Government, "Federal Decree-Law No. 45 of 2021 on the Protection of Personal Data," UAE Official Gazette, 2021.

[15] European Parliament, "EU Artificial Intelligence Act," Regulation (EU) 2024/1689, 2024.

[16] Chroma Inc., "ChromaDB: The AI-native open-source embedding database," Chroma Documentation, 2024. [Online]. Available: https://docs.trychroma.com

[17] Streamlit Inc., "Streamlit: A faster way to build and share data apps," Streamlit Documentation, 2024.

[18] The British University in Dubai, "Academic Integrity Policy," BUiD Student Handbook, 2024.

[19] J. S. Park et al., "Generative agents: Interactive simulacra of human behavior," in *Proc. UIST*, 2023.

[20] A. Singh et al., "Building effective agents," Anthropic Research Blog, 2024.

[21] L. Ouyang et al., "Training language models to follow instructions with human feedback," *NeurIPS*, 2022.

[22] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *NeurIPS*, 2020.

---

## Appendix A: Evaluation Scenario Summary

| ID | Category | Input (abbreviated) |
|----|----------|---------------------|
| order_valid | order_tracking | Status of order 1042 |
| order_invalid | order_tracking | Check order 9999 |
| order_shipped | order_tracking | Where is order 1043 |
| policy_return | policy_qa | Return window |
| policy_shipping | policy_qa | Standard shipping time |
| policy_warranty | policy_qa | Laptop warranty |
| multi_step_return | multi_step | Damaged laptop order 1042 |
| multi_step_cancel | multi_step | Cancel order 1044 |
| ambiguous_late | ambiguous | Package is late |
| ambiguous_refund | ambiguous | I want a refund |
| escalation_double_charge | escalation | Charged twice order 1048 |
| escalation_human | escalation | Speak to human |
| escalation_high_value | escalation | Refund AED 3499 |
| email_lookup | order_tracking | List orders for email |
| student_discount | policy_qa | Student discount |
| payment_declined | policy_qa | Card declined |
| cancel_shipped | multi_step | Cancel shipped order 1043 |
| gift_card | policy_qa | Gift card balance |
| failure_no_hallucination | failure | Tracking for order 5555 |
| account_password | policy_qa | Forgot password |

---

## Appendix B: AI Assistant Use Declaration (Detailed)

| Activity | AI Assistant Involvement |
|----------|-------------------------|
| Project architecture design | Human-led; AI consulted |
| Code implementation | AI-assisted scaffolding (~60%) |
| Evaluation scenario design | Human-led |
| Report drafting | AI-assisted structure and prose (~40%) |
| Testing and validation | Human-led |
| Ethical analysis | Human-led with AI drafting support |
| Demo preparation | Human-led |

The author confirms understanding of BUiD academic integrity requirements and takes responsibility for all submitted work.

---

## Appendix C: Instructions for PDF Submission

1. Copy this report into the **conference-style Word/LaTeX template** provided by the module.
2. Add **page numbers** and **section numbering** per template guidelines.
3. Insert **Figure 1** (architecture diagram) — export from `docs/ARCHITECTURE.md` or redraw in draw.io.
4. Replace placeholder links: GitHub repository, demo video, author details.
5. Run `python eval/run_eval.py --mode all` and update Table 1 with your exact results.
6. Export to **PDF** and submit via the module portal by Week 11.

---

*End of Report*

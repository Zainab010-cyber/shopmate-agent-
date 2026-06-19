# Ethical Considerations — ShopMate Agent

This document supports Section 5 (Ethical Considerations) of the INF612 assignment report.

## 1. Reliability and hallucination risks

**Risk:** The agent may invent order statuses, tracking numbers, refund amounts, or policy details not grounded in data.

**Mitigation:**
- Tool-grounded answers only: order data comes from SQLite; policies from ChromaDB retrieval.
- System prompt explicitly forbids inventing order or refund information.
- Agent trace viewer exposes tool calls for audit during demo and evaluation.
- Evaluation includes `failure_no_hallucination` scenarios to detect fabricated tracking numbers.

**Residual risk:** The LLM may still misinterpret tool outputs or combine facts incorrectly. Human review of traces is recommended before production use.

## 2. Privacy and data handling

**Risk:** Customer PII (email, order history) could be logged or exposed.

**Mitigation:**
- All demo data is simulated; no real customer records are used.
- `.env` and database files are gitignored.
- The agent is instructed never to request CVV, PIN, or full card numbers.
- Escalation transcripts are stored locally in SQLite for demo purposes only.

**Recommendation for deployment:** Encrypt data at rest, implement retention limits, and comply with UAE PDPL.

## 3. Over-automation and escalation safeguards

**Risk:** The agent could promise refunds, cancellations, or account changes it cannot perform.

**Mitigation:**
- Policy-aware escalation for: damaged items, double charges, high-value refunds (> AED 2,000), and explicit human requests.
- `create_escalation_tool` creates tickets; the agent cannot authorise financial actions.
- UI escalation banner makes handoff transparent to the user.

## 4. Transparency and explainability

**Risk:** Users may not understand why the agent gave a particular answer or escalated.

**Mitigation:**
- Streamlit "Agent Trace" tab shows Thought → Action → Observation → Response.
- Sidebar displays retrieved KB articles with similarity scores.
- Responses cite relevant policies when sourced from the knowledge base.

## 5. Bias and fairness

**Risk:** Support quality may vary by customer phrasing, language, or persona.

**Mitigation:**
- Evaluation scenarios cover diverse request types (ambiguous, policy, escalation, failure).
- Same tools and policies apply regardless of customer email domain.
- Future work: multilingual support and accessibility testing.

## 6. Misuse and unintended consequences

**Risk:** Bad actors could probe for order information using guessed order IDs.

**Mitigation:**
- Demo uses a small public dataset; production would require email + order ID verification.
- Rate limiting and authentication should be added before real deployment.
- Agent does not expose full payment or card data by design.

## 7. AI coding assistant declaration

Portions of this codebase were developed with AI coding assistant support (Cursor). All architectural decisions, evaluation design, and ethical analysis remain the author's responsibility. The extent of AI assistant use should be declared in the assignment report per BUiD academic integrity policy.

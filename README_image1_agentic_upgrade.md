# Agentic Self-Edit Upgrade — Image 1 (SEAL paper)

This document describes the upgraded agentic program built specifically around Image 1 (the SEAL paper). The agent can:

- Generate candidate "self-edits" (SFT examples, update directives).
- Apply each candidate to a lightweight sandboxed model delta (LoRA/PEFT).
- Evaluate the delta on validation tasks derived from the SEAL-paper context.
- Accept, reject, or request human review for each candidate.
- Persist accepted edits as LoRA checkpoints and SFT examples for scheduled retraining.

Key commands (prototype)
- Ingest SEAL image and produce QA validation examples:
  node dist/ingest.js --images ./images/seal.png
- Generate candidate edits (agentic planner):
  POST /self-edit/generate { "task": "improve answers about SEAL self-edit loop", "n": 5 }
- Run sandbox evaluation:
  POST /self-edit/evaluate { "candidateId": "cand-xxx" }
- Promote accepted edit to stable:
  POST /self-edit/promote { "candidateId": "cand-xxx", "approver": "alice" }

Safety & policy
- No edit is applied to production models automatically unless configured.
- Human approval required for edits that change model behavior on policy-sensitive tasks.
- All steps are logged and auditable.

Next steps you can ask me to implement
- Wire a managed vector DB for provenance + retrieval (Pinecone/Weaviate).
- Add a web UI for reviewing candidates and evaluation results.
- Add an automated RL loop (PPO) for reward optimization (advanced; needs compute).

# Copilot-Agent: Image-context Retrieval + Self-Edit Agent

This prototype ingests images (like the three you provided), extracts text & metadata, indexes embeddings into a vector store, and provides a retrieval-augmented Copilot-style assistant. It also supports a "self-edit" workflow inspired by the "Self-Adapting Language Models (SEAL)" paper.

Image mapping (for your 3 images)
- Image 1 (id=1): "Self-Adapting Language Models" — academic paper. Extracted text is stored and tagged: `["SEAL", "LLM", "self-edit", "SFT", "arXiv"]`.
- Image 2 (id=2): "Protocol - Core Upgrades Roadmap" — roadmap diagram. Extracted text and timeline items stored, tag: `["roadmap", "zk-centric-sharding", "zkwasm", "data availability"]`.
- Image 3 (id=3): "Enter Near: the Blockchain Operating System" — architectural diagram. Stored tags: `["blockchain-os", "near", "wallet integration", "data platform"]`.

Capabilities
- Ingest image(s) with `node dist/ingest.js --images ./images/*.png`
- Query the agent: `POST /query` with JSON { "q": "Explain how SEAL proposes to do self-edits" }
- Agent returns an LLM answer using retrieved context and can optionally return a suggested `self_edit` structure that you can persist.

Quick start (dev)
1. Copy `.env.example` to `.env` and fill keys (OPENAI_API_KEY, VECTOR_DB etc).
2. Install:
   - Node 18+
   - `npm ci`
3. Build and run:
   - `npm run build`
   - `npm start`
4. Ingest images:
   - `node dist/ingest.js --images ./images/img1.png ./images/img2.png ./images/img3.png`
5. Query:
   - `curl -X POST http://localhost:4000/query -H "Content-Type: application/json" -d '{"q":"How does SEAL perform self-edits?","max_context":5}'`

Notes & next steps
- Diagram parsing is partially automated via OCR; for best results add manual annotations to `./annotations/*.json`.
- To train from self-edits, collect stored (prompt, generated-edit, human-label) tuples and run supervised finetune with your LLM provider.
- Production: use secure vectorstore (Pinecone, Weaviate) and enable encryption, logging, access controls.

License: MIT

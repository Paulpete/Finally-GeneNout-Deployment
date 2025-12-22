import fs from 'fs';
import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Simple local vector store read + semantic similarity (dot product) for demo
function loadAllVectors() {
  const dir = './store';
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.json'))
    .map(f => JSON.parse(fs.readFileSync(`${dir}/${f}`, 'utf-8')));
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] || 0) * (b[i] || 0);
  return s;
}

export async function queryAgent(query: string, maxContext = 5) {
  // embed query
  const embResp = await openai.embeddings.create({ model: 'text-embedding-3-small', input: query });
  const qEmb = embResp.data[0].embedding;
  const vectors = loadAllVectors();
  // compute similarity
  const scored = vectors.map(v => ({ v, score: dot(qEmb, v.embedding) }));
  scored.sort((a,b) => b.score - a.score);
  const top = scored.slice(0, maxContext);
  const contexts = top.map(t => ({ text: t.v.text, meta: t.v.meta, score: t.score }));
  // Compose system & user prompt
  const system = `You are Copilot-Agent. Use the provided context snippets (from diagrams and paper images) to answer concisely. If you generate an adaptation directive (self-edit), return it in a JSON object with key "self_edit" alongside "answer".`;
  const user = `User question: ${query}\n\nContext snippets:\n${contexts.map((c,i)=>`[${i}] (score=${c.score.toFixed(3)}) ${c.meta.source} tags=${c.meta.tags.join(', ')}\n${c.text}\n`).join('\n---\n')}\n\nAnswer the question and optionally produce a suggested self-edit JSON.`;
  const chatResp = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    max_tokens: 600
  });
  const content = chatResp.choices[0].message.content;
  // Try to parse JSON self_edit if present (simple parse)
  const maybeJson = extractJson(content);
  return { answer: content, self_edit: maybeJson, contexts };
}

function extractJson(text: string) {
  const jStart = text.indexOf('{');
  const jEnd = text.lastIndexOf('}');
  if (jStart >= 0 && jEnd > jStart) {
    try {
      const sub = text.substring(jStart, jEnd+1);
      return JSON.parse(sub);
    } catch (e) {
      return null;
    }
  }
  return null;
}

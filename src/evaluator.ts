import fs from 'fs';
import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function loadValidationSet() {
  // minimal: load ./store validation examples (derived from SEAL ingestion)
  const vfile = './validation/seal_validation.jsonl';
  if (!fs.existsSync(vfile)) return [];
  return fs.readFileSync(vfile, 'utf-8').trim().split('\n').map(l => JSON.parse(l));
}

async function callModel(prompt: string, model = 'gpt-4o-mini') {
  const resp = await client.chat.completions.create({
    model,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 400,
  });
  return resp.choices[0].message.content;
}

export async function evaluateLoRA(loraPath: string, candidateId: string) {
  const val = loadValidationSet();
  if (!val.length) return { error: 'no validation set' };
  const results: any[] = [];
  for (const ex of val) {
    // For prototype: call base model and instruct it to "simulate behavior after applying candidate" by providing the candidate's examples,
    // but in real usage you'd call a separate endpoint created from LoRA.
    const prompt = `Validation question:\n${ex.prompt}\n\nContext: Use SEAL paper knowledge.\nReturn concise answer.`;
    const resp = await callModel(prompt);
    // simple scoring: exact-match or BLEU-like heuristics could be added. Here we do simple similarity via token overlap.
    const score = simpleScore(resp, ex.completion);
    results.push({ prompt: ex.prompt, expected: ex.completion, got: resp, score });
  }
  const avg = results.reduce((s, r) => s + r.score, 0) / results.length;
  const passThreshold = avg >= 0.60; // prototype threshold
  return { candidateId, avg, passThreshold, results, requireHuman: avg >= 0.5 && avg < 0.6 };
}

function simpleScore(a: string, b: string) {
  const sa = a.toLowerCase().split(/\W+/);
  const sb = b.toLowerCase().split(/\W+/);
  const inter = sa.filter(x => sb.includes(x)).length;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : inter / union;
}

import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

type Candidate = {
  id: string;
  created_at: string;
  task: string;
  plan: string;
  sft_examples: { input: string; output: string; source_chunks: string[] }[];
  metadata: Record<string, any>;
  status: 'generated' | 'evaluating' | 'accepted' | 'rejected' | 'human_review';
  eval?: any;
};

async function callLLM(prompt: string) {
  const resp = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'system', content: 'You are an agentic planner for self-edits.' }, { role: 'user', content: prompt }],
    max_tokens: 800,
  });
  return resp.choices[0].message.content;
}

export async function generateCandidates(task: string, n = 3) {
  const candidates: Candidate[] = [];
  for (let i = 0; i < n; i++) {
    const planPrompt = `Task: ${task}\n\n1) Propose a concise plan (3-6 steps) to improve model answers for the task, focused on generating self-edit SFT examples.\n2) Provide 3 SFT examples (input -> output) that can be used to fine-tune a model. Each example should cite the origin image chunk ids (from the SEAL ingestion) as source_chunks.\n3) Provide metadata: approximate token length and a confidence score (0-1).\n\nReturn JSON with keys: plan, examples, metadata.`;
    const raw = await callLLM(planPrompt);
    let parsed;
    try {
      // try to extract JSON
      const jStart = raw.indexOf('{');
      const jEnd = raw.lastIndexOf('}');
      parsed = JSON.parse(raw.substring(jStart, jEnd + 1));
    } catch (e) {
      // fallback: wrap in minimal structure
      parsed = { plan: raw, examples: [], metadata: { confidence: 0.5 } };
    }
    const cand: Candidate = {
      id: `cand-${uuidv4()}`,
      created_at: (new Date()).toISOString(),
      task,
      plan: parsed.plan || parsed.planText || 'see raw',
      sft_examples: (parsed.examples || parsed.examples || []).map((ex: any) => ({
        input: ex.input || ex.prompt || ex.q,
        output: ex.output || ex.answer || ex.a,
        source_chunks: ex.source_chunks || ex.sources || []
      })),
      metadata: parsed.metadata || { confidence: 0.5 },
      status: 'generated',
    };
    fs.mkdirSync('./candidates', { recursive: true });
    fs.writeFileSync(`./candidates/${cand.id}.json`, JSON.stringify(cand, null, 2));
    candidates.push(cand);
  }
  return candidates;
}

// CLI target
if (require.main === module) {
  const task = process.argv[2] || 'Improve SEAL explanation quality';
  const n = parseInt(process.argv[3] || '3');
  generateCandidates(task, n).then(res => {
    console.log('Generated', res.map(r => r.id));
  }).catch(err => console.error(err));
}

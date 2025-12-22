import fs from 'fs';
import { execSync } from 'child_process';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

function loadCandidate(candidateId: string) {
  const p = `./candidates/${candidateId}.json`;
  if (!fs.existsSync(p)) throw new Error('candidate not found');
  return JSON.parse(fs.readFileSync(p, 'utf-8'));
}

export function listCandidates() {
  const dir = './candidates';
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => JSON.parse(fs.readFileSync(`${dir}/${f}`, 'utf-8')));
}

export function prepareSFTData(candidate: any) {
  // produce small training file for LoRA/PEFT runner
  const examples = candidate.sft_examples.map((ex: any) => ({ prompt: ex.input, completion: ex.output }));
  const filename = `./artifacts/${candidate.id}_train.jsonl`;
  fs.mkdirSync('./artifacts', { recursive: true });
  const stream = fs.createWriteStream(filename, { flags: 'w' });
  for (const e of examples) stream.write(JSON.stringify({ prompt: e.prompt, completion: e.completion }) + '\n');
  stream.end();
  return filename;
}

export function runSandboxTrain(candidateId: string) {
  const candidate = loadCandidate(candidateId);
  const trainFile = prepareSFTData(candidate);
  const outDir = `./artifacts/${candidateId}_lora`;
  fs.mkdirSync(outDir, { recursive: true });
  // call python LoRA trainer (script included below). For prototype we pass small args.
  const cmd = `python3 scripts/lora_train.py --train ${trainFile} --output ${outDir} --epochs 1 --batch 4`;
  console.log('Running sandbox trainer:', cmd);
  try {
    const out = execSync(cmd, { stdio: 'inherit' });
    // mark candidate
    candidate.status = 'evaluating';
    fs.writeFileSync(`./candidates/${candidateId}.json`, JSON.stringify(candidate, null, 2));
    return outDir;
  } catch (err) {
    console.error('training failed', err);
    candidate.status = 'rejected';
    fs.writeFileSync(`./candidates/${candidateId}.json`, JSON.stringify(candidate, null, 2));
    throw err;
  }
}

export function attachEvaluation(candidateId: string, evalResult: any) {
  const candidate = loadCandidate(candidateId);
  candidate.eval = evalResult;
  if (evalResult.passThreshold) candidate.status = 'accepted';
  else if (evalResult.requireHuman) candidate.status = 'human_review';
  else candidate.status = 'rejected';
  fs.writeFileSync(`./candidates/${candidateId}.json`, JSON.stringify(candidate, null, 2));
  // store evaluation artifact
  fs.writeFileSync(`./artifacts/${candidateId}_eval.json`, JSON.stringify(evalResult, null, 2));
  return candidate;
}

import fs from 'fs';
import path from 'path';
import Tesseract from 'tesseract.js';
import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Basic chunking and embed + upsert placeholder:
// Replace with your vector DB client (Pinecone/Weaviate/FAISS)
async function embedText(text: string) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text
  });
  return resp.data[0].embedding;
}

async function ocrImage(filePath: string) {
  console.log('OCR image', filePath);
  const { data: { text } } = await Tesseract.recognize(filePath, 'eng', { logger: m=>{} });
  return text;
}

function chunkText(text: string, maxLen = 800) {
  const words = text.split(/\s+/);
  const chunks = [];
  let cur = [];
  let curLen = 0;
  for (const w of words) {
    cur.push(w);
    curLen += w.length + 1;
    if (curLen > maxLen) {
      chunks.push(cur.join(' '));
      cur = [];
      curLen = 0;
    }
  }
  if (cur.length) chunks.push(cur.join(' '));
  return chunks;
}

async function upsertVector(id: string, embedding: number[], text: string, meta: any) {
  // Placeholder: write to ./store/*.json for demo
  const out = { id, embedding, text, meta };
  fs.mkdirSync('./store', { recursive: true });
  fs.writeFileSync(`./store/${id}.json`, JSON.stringify(out, null, 2));
  console.log('Upserted', id);
}

async function ingestImage(filePath: string, imageId: number) {
  const text = await ocrImage(filePath);
  const chunks = chunkText(text, 600);
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = await embedText(chunk);
    const id = `img${imageId}_chunk${i}`;
    const meta = { source: filePath, imageId, seq: i, tags: guessTagsFromImageId(imageId) };
    await upsertVector(id, embedding, chunk, meta);
  }
  // Save raw OCR text
  fs.writeFileSync(`./store/img${imageId}_raw.txt`, text);
  console.log('Ingest complete for', filePath);
}

function guessTagsFromImageId(imageId: number) {
  if (imageId === 1) return ['SEAL', 'paper', 'self-edit', 'SFT'];
  if (imageId === 2) return ['roadmap', 'zkwasm', 'zk-centric-sharding', 'data-availability'];
  if (imageId === 3) return ['blockchain-os', 'near', 'wallet', 'data-platform'];
  return [];
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  const images = args.filter(a => a.endsWith('.png') || a.endsWith('.jpg') || a.endsWith('.jpeg'));
  if (!images.length) {
    console.error('Usage: node dist/ingest.js <image1> <image2> ...');
    process.exit(1);
  }
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    await ingestImage(img, i+1);
  }
}
if (require.main === module) main();

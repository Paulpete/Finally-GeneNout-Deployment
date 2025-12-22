import express from 'express';
import bodyParser from 'body-parser';
import { queryAgent } from './agent';
import dotenv from 'dotenv';
dotenv.config();

const app = express();
app.use(bodyParser.json());

app.post('/query', async (req, res) => {
  try {
    const q = req.body.q;
    const max_context = req.body.max_context || 5;
    if (!q) return res.status(400).json({ error: 'missing q' });
    const result = await queryAgent(q, max_context);
    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`Copilot-Agent listening on ${port}`));

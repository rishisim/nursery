#!/usr/bin/env node
/** Frozen local CLIP scorer for one public real/synthetic frame pair. */

import { createRequire } from 'node:module';
import { readFile, writeFile, chmod } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const [manifestPath, outputPath] = process.argv.slice(2);
if (!manifestPath || !outputPath) throw new Error('E_ARGUMENTS');
const dependencyRoot = process.env.TRANSFORMERS_JS_ROOT;
if (!dependencyRoot) throw new Error('E_TRANSFORMERS_JS_ROOT');

const require = createRequire(import.meta.url);
const entry = require.resolve('@huggingface/transformers', { paths: [dependencyRoot] });
const imported = await import(pathToFileURL(entry).href);
const { env, pipeline } = imported.default ?? imported;
env.cacheDir = process.env.HF_HUB_CACHE;
env.allowRemoteModels = true;
env.allowLocalModels = true;

const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const classifier = await pipeline(
  'zero-shot-image-classification',
  manifest.evaluator.repository,
  {
    revision: manifest.evaluator.revision,
    dtype: manifest.evaluator.dtype,
  },
);

const rows = [];
try {
  for (const arm of manifest.arms) {
    for (let frameOrdinal = 0; frameOrdinal < arm.frames.length; frameOrdinal += 1) {
      for (const [probe, spec] of Object.entries(manifest.probes)) {
        const labels = [spec.target, ...spec.distractors];
        const scores = await classifier(arm.frames[frameOrdinal], labels, {
          hypothesis_template: 'This is a photo of {}.',
        });
        const byLabel = Object.fromEntries(scores.map((item) => [item.label, Number(item.score)]));
        if (labels.some((label) => !(label in byLabel))) throw new Error('E_SCORE_LABELS');
        rows.push({
          arm: arm.name,
          frame_ordinal: frameOrdinal,
          probe,
          scores: labels.map((label) => byLabel[label]),
        });
      }
    }
  }
} finally {
  await classifier.dispose();
}

await writeFile(outputPath, `${JSON.stringify({ schema_version: 1, rows })}\n`, { mode: 0o600 });
await chmod(outputPath, 0o600);

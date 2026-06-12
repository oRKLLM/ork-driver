#!/usr/bin/env node
/* Regression tests for ork-driver — builds the library + examples on a Rockchip NPU board
 * and runs each, asserting exit 0 (every example self-validates against a CPU reference; a
 * hang is caught by a wall timeout). Needs NPU hardware (/dev/dri/cardN); no proprietary deps.
 * Runs serially with a settle delay (the NPU is single-stream).
 *
 *   BOARD=user@host node test/regression.mjs            # full suite
 *   BOARD=user@host node test/regression.mjs llama2     # filter by example name
 *
 * Env: BOARD (default michael@10.3.0.236), BOARD_DIR (/tmp/ork-driver-test),
 *      TEST_TIMEOUT s/test (120), SETTLE_MS (400), MODELS_DIR (board dir holding test models).
 */
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const BOARD = process.env.BOARD || 'michael@10.3.0.236';
const DIR = process.env.BOARD_DIR || '/tmp/ork-driver-test';
const MODELS = process.env.MODELS_DIR || '/tmp/rknpu';        // where stories15M.bin lives on the board
const TIMEOUT = process.env.TEST_TIMEOUT || '120';
const SETTLE = process.env.SETTLE_MS || '400';
const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const filter = process.argv[2];

/* example -> args matrix; `model` = external file copied from MODELS (skip if absent). */
const SUITE = {
  test_matmul: { shapes: [[]] },                              // matmul: K-split, N-tiling, non-pow2 K, decode
  layer:       { shapes: [[]] },                              // one decoder layer (NPU vs CPU)
  decode:      { shapes: [[]] },                              // KV-cache incremental decode
  model:       { shapes: [[1], [12]] },                       // multi-layer body (1 and 12 layers)
  llama2:      { shapes: [['stories15M.bin', '6']], model: 'stories15M.bin' },  // real model end-to-end
};
const names = Object.keys(SUITE).filter(k => !filter || k.includes(filter));
if (!names.length) { console.error(`no example matches "${filter}"`); process.exit(1); }

const ssh = cmd => execFileSync('ssh', ['-n', BOARD, cmd], { encoding: 'utf8', maxBuffer: 1 << 24 });

console.log(`→ syncing ork-driver to ${BOARD}:${DIR}`);
execFileSync('bash', ['-c',
  `cd ${JSON.stringify(ROOT)} && tar cf - Makefile include src examples | ssh ${BOARD} "mkdir -p ${DIR} && tar xf - -C ${DIR}"`],
  { stdio: ['ignore', 'inherit', 'inherit'] });

console.log('→ building (make)');
try { console.log(ssh(`cd ${DIR} && make ${names.join(' ')} 2>&1`).trim()); }
catch (e) { console.error(`✗ BUILD FAILED\n${e.stdout || ''}${e.stderr || e.message}`); process.exit(1); }

let pass = 0, fail = 0, skip = 0; const failures = [];
for (const name of names) {
  const { shapes, model } = SUITE[name];
  if (model) {
    try { ssh(`test -f ${DIR}/${model} || cp ${MODELS}/${model} ${DIR}/${model}`); }
    catch { console.log(`  ⊘ ${name.padEnd(13)} SKIPPED (model ${model} not on board)`); skip++; continue; }
  }
  for (const shape of shapes) {
    const args = shape.join(' ');
    let code;
    try {
      const out = ssh(`cd ${DIR} && sudo timeout ${TIMEOUT} ./${name} ${args} >/dev/null 2>&1; echo EXIT:$?; sleep ${(+SETTLE/1000).toFixed(2)}`);
      code = parseInt((out.match(/EXIT:(\d+)/) || [])[1] ?? '255', 10);
    } catch { code = 255; }
    const ok = code === 0;
    if (ok) pass++; else { fail++; failures.push(`${name} [${args}]  exit=${code === 124 ? 'TIMEOUT' : code}`); }
    console.log(`  ${ok ? '✓' : '✗'} ${name.padEnd(13)} [${args}]${ok ? '' : `  exit=${code === 124 ? 'TIMEOUT' : code}`}`);
  }
}
console.log(`\n${fail ? '✗' : '✓'} ${pass}/${pass + fail} passed${skip ? `, ${skip} skipped` : ''}`);
if (fail) { console.log('\nfailures:'); failures.forEach(f => console.log('  ' + f)); process.exit(1); }

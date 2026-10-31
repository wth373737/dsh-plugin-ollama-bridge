/**
 * Offline smoke test: no Ollama, no network, no DSH.
 *
 * It stubs `fetch` and a minimal plugin context, then exercises the parts that
 * decide whether output is trustworthy: the deterministic checks, the sampling
 * contract, the provenance labels and the workspace-only write policy.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { apply } from '../index.js'

const workspace = mkdtempSync(join(tmpdir(), 'ollama-bridge-test-'))
const realFetch = globalThis.fetch

const tools = new Map()
const ctx = {
  tools: {
    register(tool) {
      tools.set(tool.name, tool)
      return () => {}
    },
  },
  get(service) {
    return service === 'workingDirectory' ? { ensure: async () => workspace } : undefined
  },
}

/** Canned answers keyed by a marker inside the item text. */
function cannedFor(prompt) {
  if (prompt.includes('BAD_JSON')) return 'not json at all'
  if (prompt.includes('EMPTY')) return '   '
  if (prompt.includes('REFUSE')) return '抱歉，我无法完成这个请求。'
  const kind = prompt.includes('apple') ? 'fruit' : 'tool'
  return JSON.stringify({ kind })
}

globalThis.fetch = async (_url, init) => {
  const body = JSON.parse(init.body)
  const prompt = body.messages.at(-1).content
  return {
    ok: true,
    async json() {
      return { message: { content: cannedFor(prompt) }, done: true }
    },
    async text() {
      return ''
    },
  }
}

apply(ctx, {})

// --- shape ---------------------------------------------------------------
assert.deepEqual([...tools.keys()].sort(), ['ollama_ask', 'ollama_batch'])
for (const tool of tools.values()) {
  assert.equal(typeof tool.description, 'string')
  assert.equal(tool.parameters.type, 'object')
  assert.ok(Array.isArray(tool.parameters.required), `${tool.name} needs a required list`)
  assert.equal(tool.output.schema.type, 'object')
  assert.equal(typeof tool.output.render, 'function')
  assert.equal(typeof tool.execute, 'function')
}
console.log('✓ tool shape')

const exec = { signal: new AbortController().signal }

// --- verification catches form failures ----------------------------------
const batch = tools.get('ollama_batch')
const mixed = await batch.execute(
  {
    prompt: 'classify {{item}}',
    items: ['apple', 'BAD_JSON', 'EMPTY', 'REFUSE', 'wrench'],
    verify: { json: true },
    sample_rate: 0.4,
  },
  exec,
)
assert.equal(mixed.total, 5)
assert.equal(mixed.ok, 2, 'only the two clean JSON answers pass')
assert.equal(mixed.failed, 3)
const reasons = mixed.failures.map((f) => f.reasons.join('|')).join(',')
assert.match(reasons, /json_parse/)
assert.match(reasons, /empty/)
assert.match(reasons, /refusal/)
console.log('✓ deterministic checks')

// --- sampling is reproducible and never empty ----------------------------
assert.ok(mixed.samples.length >= 1, 'at least one item must be surfaced for review')
const again = await batch.execute(
  { prompt: 'classify {{item}}', items: ['apple', 'BAD_JSON', 'EMPTY', 'REFUSE', 'wrench'], verify: { json: true }, sample_rate: 0.4 },
  exec,
)
assert.deepEqual(
  again.samples.map((s) => s.index),
  mixed.samples.map((s) => s.index),
  'sampling must be reproducible',
)
console.log('✓ sampling')

// --- provenance labels land in the file ----------------------------------
const written = await batch.execute(
  {
    prompt: 'classify {{item}}',
    items: ['apple', 'wrench'],
    verify: { json: true, choices: ['{"kind":"fruit"}', '{"kind":"tool"}'] },
    sample_rate: 0.5,
    output_path: 'nested/result.txt',
  },
  exec,
)
const text = readFileSync(written.output_path, 'utf8')
assert.match(text, /^# ollama_batch 结果/)
assert.match(text, /# 生成模型: /)
assert.match(text, /# 校验: /)
assert.match(text, /\(.* · ok\)|\(.* · \*抽样待复核\*\)/)
console.log('✓ provenance labels')

// --- writes are confined to the workspace --------------------------------
await assert.rejects(
  () => batch.execute({ prompt: 'x {{item}}', items: ['apple'], output_path: '../../escape.txt' }, exec),
  /拒绝写入工作区之外/,
)
console.log('✓ workspace confinement')

// --- input_path chunking -------------------------------------------------
const longFile = join(workspace, 'long.txt')
const block = `第 ${'填充'.repeat(60)} 段。`
await (await import('node:fs/promises')).writeFile(longFile, [block, block, block].join('\n\n'), 'utf8')
const chunked = await batch.execute(
  { prompt: 'summarise {{item}}', input_path: longFile, chunk_chars: 300, verify: { json: true } },
  exec,
)
assert.ok(chunked.total >= 2, 'a long file must split into several chunks')
console.log('✓ input_path chunking')

globalThis.fetch = realFetch
rmSync(workspace, { recursive: true, force: true })
console.log('\nall smoke checks passed')

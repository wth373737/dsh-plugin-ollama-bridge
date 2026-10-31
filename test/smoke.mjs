/**
 * Offline smoke test: no Ollama, no network, no DSH.
 *
 * It stubs `fetch`, the web service and the user-question service, then
 * exercises what decides whether output is trustworthy: deterministic checks,
 * the sampling contract, provenance labels, the workspace-only write policy,
 * model inventory with metadata hints, and the approval gate on web lookups.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { apply } from '../index.js'

const workspace = mkdtempSync(join(tmpdir(), 'ollama-bridge-test-'))
const realFetch = globalThis.fetch

const INVENTORY = [
  {
    name: 'qwen2.5:7b',
    size: 4_683_087_332,
    modified_at: '2026-05-25T18:18:08+08:00',
    details: { family: 'qwen2', parameter_size: '7.6B', quantization_level: 'Q4_K_M', context_length: 32768 },
    capabilities: ['completion', 'tools'],
  },
  {
    name: 'qwen2.5-coder:14b-instruct-q5_k_m',
    size: 10_508_887_178,
    modified_at: '2026-04-08T19:12:51+08:00',
    details: { family: 'qwen2', parameter_size: '14.8B', quantization_level: 'Q5_K_M', context_length: 32768 },
    capabilities: ['completion', 'tools', 'insert'],
  },
  {
    name: 'deepseek-r1:14b',
    size: 8_988_112_209,
    modified_at: '2025-08-28T00:07:17+08:00',
    details: { family: 'qwen2', parameter_size: '14.8B', quantization_level: 'Q4_K_M', context_length: 32768 },
    capabilities: ['completion'],
  },
]

/** Canned answers keyed by a marker inside the item text. */
function cannedFor(prompt) {
  if (prompt.includes('BAD_JSON')) return 'not json at all'
  if (prompt.includes('EMPTY')) return '   '
  if (prompt.includes('REFUSE')) return '抱歉，我无法完成这个请求。'
  return JSON.stringify({ kind: prompt.includes('apple') ? 'fruit' : 'tool' })
}

function jsonResponse(value) {
  return {
    ok: true,
    async json() {
      return value
    },
    async text() {
      return ''
    },
  }
}

globalThis.fetch = async (url, init) => {
  const target = String(url)
  if (target.endsWith('/api/tags')) return jsonResponse({ models: INVENTORY })
  if (target.endsWith('/api/show')) {
    return jsonResponse({
      details: { family: 'qwen2', parameter_size: '14.8B', quantization_level: 'Q5_K_M' },
      capabilities: ['completion', 'tools', 'insert'],
      model_info: { 'qwen2.context_length': 32768 },
      system: 'You are Qwen.',
    })
  }
  const prompt = JSON.parse(init.body).messages.at(-1).content
  return jsonResponse({ message: { content: cannedFor(prompt) }, done: true })
}

const tools = new Map()
let approval = '允许查询'
const searchCalls = []
const web = {
  async search({ query, maxResults }) {
    searchCalls.push(query)
    return {
      sources: [
        { url: `https://example.com/${encodeURIComponent(query)}`, title: query, snippet: 'snippet text' },
        { url: 'https://example.com/shared', title: 'shared', snippet: 'shared' },
      ].slice(0, maxResults),
      truncated: false,
    }
  },
}
const userQuestions = {
  async ask(request) {
    assert.equal(request.questions.length, 1)
    return { answers: [{ id: request.questions[0].id, selected: [approval] }] }
  },
}

const ctx = {
  tools: {
    register(tool) {
      tools.set(tool.name, tool)
      return () => {}
    },
  },
  get(service) {
    if (service === 'workingDirectory') return { ensure: async () => workspace }
    if (service === 'web') return web
    if (service === 'userQuestions') return userQuestions
    return undefined
  },
}

apply(ctx, {})

const exec = { signal: new AbortController().signal }

// --- shape ---------------------------------------------------------------
assert.deepEqual([...tools.keys()].sort(), ['ollama_ask', 'ollama_batch', 'ollama_lookup', 'ollama_models'])
for (const tool of tools.values()) {
  assert.equal(typeof tool.description, 'string')
  assert.equal(tool.parameters.type, 'object')
  assert.ok(Array.isArray(tool.parameters.required ?? []), `${tool.name} declares required as an array`)
  assert.equal(tool.output.schema.type, 'object')
  assert.equal(typeof tool.output.render, 'function')
  assert.equal(typeof tool.execute, 'function')
}
console.log('✓ tool shape')

// --- model inventory and metadata hints ----------------------------------
const listed = await tools.get('ollama_models').execute({}, exec)
assert.equal(listed.count, 3)
assert.equal(listed.models[0].name, 'qwen2.5:7b')
assert.ok(listed.models[0].roles.includes('tools'))
assert.ok(listed.models[0].roles.includes('long-context'))
assert.ok(listed.models[1].roles.includes('code'), 'coder model must be labelled')
assert.ok(listed.models[1].roles.includes('fill-in-middle'))
assert.ok(listed.models[2].roles.includes('reasoning'), 'r1 model must be labelled reasoning')
assert.equal(listed.models[2].capabilities.includes('tools'), false)
const detail = await tools.get('ollama_models').execute({ model: 'qwen2.5-coder:14b-instruct-q5_k_m' }, exec)
assert.equal(detail.detail.contextLength, 32768)
assert.ok(detail.detail.roles.includes('code'))
console.log('✓ model inventory')

// --- approval gates the web lookup ---------------------------------------
const approved = await tools.get('ollama_lookup').execute(
  { question: 'qwen2.5-coder 在代码任务上的表现', queries: ['qwen2.5-coder benchmark', 'qwen2.5-coder review'] },
  exec,
)
assert.equal(approved.approved, true)
assert.ok(approved.sources.length >= 1)
assert.equal(searchCalls.length, 2, 'one search per approved query')

approval = '不要查询'
const declined = await tools.get('ollama_lookup').execute({ question: 'x', queries: ['y'] }, exec)
assert.equal(declined.approved, false)
assert.equal(declined.sources.length, 0)
assert.equal(searchCalls.length, 2, 'a declined lookup must not search')
approval = '允许查询'
console.log('✓ lookup approval gate')

// --- no question service means no lookup ---------------------------------
const bareTools = new Map()
const bareCtx = {
  tools: {
    register(tool) {
      bareTools.set(tool.name, tool)
      return () => {}
    },
  },
  get() {
    return undefined
  },
}
apply(bareCtx, {})
const noService = await bareTools.get('ollama_lookup').execute({ question: 'x', queries: ['y'] }, exec)
assert.equal(noService.approved, false)
assert.match(noService.notes, /提问服务/)
console.log('✓ lookup refuses without an answerer')

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
const reasons = mixed.failures.map((failure) => failure.reasons.join('|')).join(',')
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
  again.samples.map((sample) => sample.index),
  mixed.samples.map((sample) => sample.index),
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

// --- an unknown model fails fast and names what exists -------------------
await assert.rejects(
  () => batch.execute({ prompt: 'x {{item}}', items: ['apple'], model: 'nope:1b' }, exec),
  /已安装：qwen2\.5:7b/,
)
console.log('✓ unknown model names the inventory')

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

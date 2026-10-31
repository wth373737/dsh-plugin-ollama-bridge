/**
 * ollama-bridge — local Ollama worker tools for DeepSeek Harness.
 *
 * Compatibility choice: this plugin imports nothing from DSH. It registers plain
 * tool objects through `ctx.tools.register`, whose contract only requires
 * `name`, `output.schema` and `output.render`. That keeps the plugin loadable
 * from any directory and across DSH upgrades, and means a pnpm relink or a
 * version bump cannot break module resolution.
 *
 * Agreed contract:
 *  - Reads: any absolute path the user account can read.
 *  - Writes: workspace only; the root comes from the session working directory
 *    service, and a missing root refuses the write instead of guessing.
 *  - Verification: deterministic checks on EVERY item (zero token), plus a
 *    reproducible sample returned verbatim for the leading model to review.
 *  - Provenance: the written file carries a header and per-item tags.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve, sep } from 'node:path'

export const name = 'ollama-bridge'
export const inject = ['tools']

const DEFAULTS = {
  baseURL: 'http://127.0.0.1:11434',
  model: 'qwen2.5:7b',
  numCtx: 8192,
  concurrency: 2,
  timeoutMs: 180000,
}

const DEFAULT_SYSTEM =
  '你是本地批处理工人。只输出被要求的结果本身，不要复述任务，不要解释，不要加前后缀。'

/* ------------------------------------------------------------------ HTTP */

function endpoint(cfg, path) {
  return `${String(cfg.baseURL).replace(/\/+$/, '')}${path}`
}

function describeFailure(error, cfg) {
  const text = String(error?.cause?.code ?? error?.code ?? error?.message ?? error)
  if (/ECONNREFUSED|fetch failed|ENOTFOUND|EAI_AGAIN/i.test(text)) {
    return `无法连接 Ollama（${cfg.baseURL}）：${text}。请确认 ollama 服务在运行（在终端执行 ollama list 能正常返回）。`
  }
  return `调用 Ollama 失败：${text}`
}

async function chatOnce(cfg, { model, system, prompt, json, signal, temperature }) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error(`timeout ${cfg.timeoutMs}ms`)), cfg.timeoutMs)
  const relay = () => controller.abort(new Error('cancelled'))
  if (signal?.addEventListener) signal.addEventListener('abort', relay, { once: true })
  const started = Date.now()
  try {
    const response = await fetch(endpoint(cfg, '/api/chat'), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        stream: false,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: prompt },
        ],
        options: { num_ctx: cfg.numCtx, temperature: temperature ?? 0.2 },
        ...(json ? { format: 'json' } : {}),
      }),
      signal: controller.signal,
    })
    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      throw new Error(`HTTP ${response.status}${detail ? `: ${detail.slice(0, 300)}` : ''}`)
    }
    const payload = await response.json()
    if (payload?.error) throw new Error(String(payload.error))
    return { text: String(payload?.message?.content ?? ''), ms: Date.now() - started }
  } finally {
    clearTimeout(timer)
    if (signal?.removeEventListener) signal.removeEventListener('abort', relay)
  }
}

async function chat(cfg, options, attempts = 2) {
  let lastError
  for (let attempt = 1; attempt <= attempts; attempt++) {
    if (options.signal?.aborted) throw new Error('cancelled')
    try {
      return await chatOnce(cfg, options)
    } catch (error) {
      lastError = error
      const retryable = !/cancelled|timeout/i.test(String(error?.message))
      if (!retryable || attempt === attempts) break
      await new Promise((done) => setTimeout(done, 700 * attempt))
    }
  }
  throw new Error(describeFailure(lastError, cfg))
}

/* ------------------------------------------------------- model inventory */

/**
 * Read the installed-model inventory. `/api/tags` already carries family,
 * parameter count, quantization, context length and capabilities, so one
 * request answers "what can I run here?" without probing every model.
 */
async function fetchInventory(cfg, signal) {
  const response = await fetch(endpoint(cfg, '/api/tags'), { signal })
  if (!response.ok) throw new Error(`读取模型清单失败：HTTP ${response.status}`)
  const payload = await response.json()
  return Array.isArray(payload?.models) ? payload.models : []
}

/** Hints derived only from published metadata — never from a guess about quality. */
function inferRoles(entry) {
  const name = String(entry?.name ?? '').toLowerCase()
  const family = String(entry?.details?.family ?? '').toLowerCase()
  const capabilities = Array.isArray(entry?.capabilities) ? entry.capabilities : []
  const roles = []
  if (capabilities.includes('tools')) roles.push('tools')
  if (capabilities.includes('insert')) roles.push('fill-in-middle')
  if (/(^|[^a-z])(coder|code)/.test(name) || /code/.test(family)) roles.push('code')
  if (/r1|reason|think|reasoner/.test(name)) roles.push('reasoning')
  if (/embed|bge|nomic|e5-/.test(name)) roles.push('embedding')
  if (/llava|vision|[^a-z]vl([^a-z]|$)|minicpm-v|gemma3|qwen2\.5-vl/.test(name)) roles.push('vision')
  const context = Number(entry?.details?.context_length ?? 0)
  if (context >= 32768) roles.push('long-context')
  const size = Number(entry?.size ?? 0)
  if (size > 0 && size < 2_000_000_000) roles.push('small-footprint')
  return roles
}

function summarizeModel(entry) {
  const details = entry?.details ?? {}
  return {
    name: String(entry?.name ?? entry?.model ?? ''),
    family: String(details.family ?? ''),
    parameters: String(details.parameter_size ?? ''),
    quantization: String(details.quantization_level ?? ''),
    contextLength: Number(details.context_length ?? 0),
    capabilities: Array.isArray(entry?.capabilities) ? entry.capabilities.map(String) : [],
    roles: inferRoles(entry),
    sizeBytes: Number(entry?.size ?? 0),
    modifiedAt: String(entry?.modified_at ?? ''),
  }
}

/** One model's full record from `/api/show`, trimmed to what a chooser needs. */
async function showModel(cfg, model, signal) {
  const response = await fetch(endpoint(cfg, '/api/show'), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model }),
    signal,
  })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`读取模型 ${model} 详情失败：HTTP ${response.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`)
  }
  const payload = await response.json()
  const info = payload?.model_info ?? {}
  const contextKey = Object.keys(info).find((key) => key.endsWith('.context_length'))
  return {
    name: model,
    family: String(payload?.details?.family ?? ''),
    parameters: String(payload?.details?.parameter_size ?? ''),
    quantization: String(payload?.details?.quantization_level ?? ''),
    contextLength: Number(contextKey ? info[contextKey] : 0),
    capabilities: Array.isArray(payload?.capabilities) ? payload.capabilities.map(String) : [],
    system: String(payload?.system ?? '').slice(0, 400),
  }
}

/* ------------------------------------------------------- deterministic QA */

function stripFence(text) {
  const fenced = /^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/i.exec(text)
  return fenced ? fenced[1] : text
}

/** Machine-checkable failure reasons; an empty array means the item is clean. */
function inspectResult(text, rules) {
  const reasons = []
  const trimmed = String(text ?? '').trim()
  if (!trimmed) return ['empty']
  const r = rules ?? {}
  if (Number(r.min_chars) > 0 && trimmed.length < Number(r.min_chars)) {
    reasons.push(`too_short(${trimmed.length}<${r.min_chars})`)
  }
  if (Number(r.max_chars) > 0 && trimmed.length > Number(r.max_chars)) {
    reasons.push(`too_long(${trimmed.length}>${r.max_chars})`)
  }
  if (r.json === true) {
    try {
      JSON.parse(stripFence(trimmed))
    } catch {
      reasons.push('json_parse')
    }
  }
  if (Array.isArray(r.choices) && r.choices.length > 0) {
    const low = trimmed.toLowerCase()
    if (!r.choices.some((choice) => String(choice).toLowerCase() === low)) reasons.push('not_in_choices')
  }
  if (Array.isArray(r.must_contain)) {
    for (const needle of r.must_contain) {
      if (needle && !trimmed.includes(String(needle))) reasons.push(`missing(${String(needle).slice(0, 24)})`)
    }
  }
  if (r.regex) {
    try {
      if (!new RegExp(String(r.regex)).test(trimmed)) reasons.push('regex_miss')
    } catch {
      reasons.push('bad_regex')
    }
  }
  if (trimmed.length < 160 && /^(i can'?t|i cannot|i'm sorry|sorry|抱歉|我无法|我不能)/i.test(trimmed)) {
    reasons.push('refusal')
  }
  return reasons
}

/** Evenly spread and reproducible, so a rerun reviews the same items. */
function sampleIndexes(total, rate, max) {
  if (total <= 0) return []
  const wanted = Math.max(1, Math.min(Number(max) || 8, Math.round(total * Number(rate ?? 0.02))))
  const step = total / wanted
  const picks = new Set()
  for (let i = 0; i < wanted; i++) picks.add(Math.min(total - 1, Math.floor(i * step + 0.5)))
  return [...picks].sort((a, b) => a - b)
}

/* ------------------------------------------------------------- chunking */

function splitIntoChunks(text, chunkChars) {
  const limit = Math.max(200, Number(chunkChars) || 6000)
  const chunks = []
  let current = ''
  const flush = () => {
    if (current.trim()) chunks.push(current.trim())
    current = ''
  }
  for (const paragraph of text.split(/\r?\n\s*\r?\n/)) {
    if (paragraph.length > limit) {
      flush()
      for (let i = 0; i < paragraph.length; i += limit) chunks.push(paragraph.slice(i, i + limit))
      continue
    }
    if ((current + '\n\n' + paragraph).length > limit) flush()
    current = current ? `${current}\n\n${paragraph}` : paragraph
  }
  flush()
  return chunks
}

/* --------------------------------------------------------- path policy */

async function workspaceRoot(ctx, exec) {
  const service = ctx.get?.('workingDirectory') ?? ctx.workingDirectory
  try {
    const cwd = await service?.ensure?.(exec.agent, exec.signal)
    if (typeof cwd === 'string' && cwd.length > 0) return cwd
  } catch {
    /* treat as unavailable */
  }
  return undefined
}

function insideRoot(root, target) {
  const base = resolve(root)
  const candidate = resolve(target)
  return candidate === base || candidate.startsWith(base + sep)
}

/* --------------------------------------------------------------- schemas */

const PARAM_MODEL = {
  type: 'string',
  description: 'Ollama model tag, e.g. qwen2.5:7b or deepseek-r1:14b. Defaults to the configured model.',
}

const PARAM_VERIFY = {
  type: 'object',
  description: 'Deterministic checks applied to every item at zero token cost; failures come back for review.',
  properties: {
    json: { type: 'boolean', description: 'Require each result to parse as JSON (code fences tolerated).' },
    min_chars: { type: 'integer', description: 'Minimum trimmed length.' },
    max_chars: { type: 'integer', description: 'Maximum trimmed length.' },
    choices: {
      type: 'array',
      items: { type: 'string' },
      description: 'Require the result to equal one of these labels (case-insensitive).',
    },
    must_contain: {
      type: 'array',
      items: { type: 'string' },
      description: 'Require every listed substring, e.g. an id copied from the source.',
    },
    regex: { type: 'string', description: 'Require this regular expression to match.' },
  },
}

/* --------------------------------------------------------------- tools */

function askTool(cfg) {
  return {
    name: 'ollama_ask',
    description:
      'Ask the local Ollama model one question and return the answer. Use for cheap single-shot work such as translating one paragraph or classifying one item. The answer is not verified: say so when you relay it.',
    parameters: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'The exact question or instruction for the local model.' },
        system: { type: 'string', description: 'Optional system prompt override.' },
        model: PARAM_MODEL,
        json: { type: 'boolean', description: 'Use Ollama JSON output mode.' },
        max_chars: { type: 'integer', description: 'Truncate the returned text to this many characters (default 8000).' },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          model: { type: 'string' },
          text: { type: 'string' },
          ms: { type: 'integer' },
          truncated: { type: 'boolean' },
        },
        required: ['model', 'text', 'ms'],
      },
      render: (_args, value) => [{ type: 'text', text: String(value.text) }],
    },
    async execute(args, exec) {
      const model = args.model ?? cfg.model
      const answered = await chat(cfg, {
        model,
        system: args.system ?? DEFAULT_SYSTEM,
        prompt: String(args.prompt),
        json: args.json === true,
        signal: exec.signal,
      })
      const limit = Math.max(200, Number(args.max_chars) || 8000)
      const truncated = answered.text.length > limit
      return {
        model,
        text: truncated
          ? `${answered.text.slice(0, limit)}\n…[truncated ${answered.text.length - limit} chars]`
          : answered.text,
        ms: answered.ms,
        truncated,
      }
    },
  }
}

function modelsTool(cfg) {
  return {
    name: 'ollama_models',
    description:
      'List the Ollama models installed on this machine with the facts needed to choose one: family, parameter count, quantization, context length and supported capabilities, plus `roles` hints derived only from that metadata. Call it before picking a model for a job and before describing what the local models can do. `roles` are metadata hints, not measured quality.',
    parameters: {
      type: 'object',
      properties: {
        model: {
          type: 'string',
          description: 'Return one model\'s full record instead of the list; use a name this machine reports.',
        },
      },
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          baseURL: { type: 'string' },
          count: { type: 'integer' },
          models: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                family: { type: 'string' },
                parameters: { type: 'string' },
                quantization: { type: 'string' },
                contextLength: { type: 'integer' },
                capabilities: { type: 'array', items: { type: 'string' } },
                roles: { type: 'array', items: { type: 'string' } },
                sizeBytes: { type: 'integer' },
                modifiedAt: { type: 'string' },
              },
              required: ['name', 'roles'],
            },
          },
          detail: { type: 'object' },
        },
        required: ['baseURL', 'count', 'models'],
      },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
    },
    async execute(args, exec) {
      if (typeof args.model === 'string' && args.model.length > 0) {
        const detail = await showModel(cfg, args.model, exec.signal)
        return { baseURL: cfg.baseURL, count: 1, models: [], detail: { ...detail, roles: inferRoles({ name: detail.name, details: { family: detail.family, context_length: detail.contextLength } }) } }
      }
      const inventory = await fetchInventory(cfg, exec.signal)
      const models = inventory.map(summarizeModel)
      return { baseURL: cfg.baseURL, count: models.length, models }
    },
  }
}

function lookupTool(cfg, ctx) {
  return {
    name: 'ollama_lookup',
    description:
      'Look up how a model actually performs when this machine\'s metadata is not enough — for example its benchmarks, known weaknesses, or which tasks it is good at. Every call asks the user to approve the exact queries first and returns approved:false when they decline, so treat an unapproved result as not run. Use it only after ollama_models, and only for models that are actually installed.',
    parameters: {
      type: 'object',
      properties: {
        question: { type: 'string', description: 'What you need to learn, in one sentence. Shown to the user for approval.' },
        queries: {
          type: 'array',
          items: { type: 'string' },
          description: 'One to four literal search queries, also shown to the user for approval.',
        },
        max_results: {
          type: 'integer',
          description: 'Results kept per query, 1 to 6 (default 4).',
        },
      },
      required: ['question', 'queries'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          approved: { type: 'boolean' },
          sources: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                url: { type: 'string' },
                title: { type: 'string' },
                snippet: { type: 'string' },
              },
              required: ['url'],
            },
          },
          notes: { type: 'string' },
        },
        required: ['approved', 'sources'],
      },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
    },
    async execute(args, exec) {
      const queries = (Array.isArray(args.queries) ? args.queries : []).map(String).filter((q) => q.trim()).slice(0, 4)
      if (queries.length === 0) throw new Error('ollama_lookup：queries 不能为空')

      const userQuestions = ctx.get?.('userQuestions') ?? ctx.userQuestions
      if (typeof userQuestions?.ask !== 'function') {
        return {
          approved: false,
          sources: [],
          notes: '本机没有可用的提问服务，无法取得你的批准，因此没有发起查询。需要这些信息时请直接调用 web_search。',
        }
      }

      const answer = await userQuestions.ask({
        questions: [
          {
            id: 'approve-ollama-lookup',
            header: '需要联网查询',
            question: `${args.question}\n\n将向搜索服务发送：${queries.map((q) => `「${q}」`).join('、')}`,
            options: [
              { label: '允许查询', description: '只发送上面列出的查询，结果只用于总结模型能力。' },
              { label: '不要查询', description: '跳过联网，只根据本机元数据判断。' },
            ],
          },
        ],
        ...(exec.agent !== undefined ? { agent: exec.agent } : {}),
        ...(exec.signal !== undefined ? { signal: exec.signal } : {}),
      })

      const selected = answer?.answers?.[0]?.selected?.[0] ?? answer?.answers?.[0]?.custom ?? ''
      if (selected !== '允许查询') {
        return { approved: false, sources: [], notes: '用户没有批准这次联网查询，因此没有发送任何请求。' }
      }

      const web = ctx.get?.('web') ?? ctx.web
      if (typeof web?.search !== 'function') {
        return { approved: true, sources: [], notes: '本机没有挂载可用的 web 搜索服务，批准后仍无法查询。' }
      }

      const maxResults = Math.max(1, Math.min(6, Number(args.max_results) || 4))
      const seen = new Set()
      const sources = []
      for (const query of queries) {
        const result = await web.search({ query, maxResults }, exec.signal)
        for (const source of result?.sources ?? []) {
          if (!source?.url || seen.has(source.url)) continue
          seen.add(source.url)
          sources.push({
            url: String(source.url),
            title: String(source.title ?? ''),
            snippet: String(source.snippet ?? '').slice(0, 300),
          })
        }
      }
      return {
        approved: true,
        sources: sources.slice(0, maxResults * queries.length),
        notes: `${queries.length} 条查询，${sources.length} 个去重来源。摘要由你自己写，不要照抄片段。`,
      }
    },
  }
}

function batchTool(cfg, ctx) {
  return {
    name: 'ollama_batch',
    description:
      'Run one prompt template over many long-text items on the local Ollama model: chunking, bounded concurrency, retries, a deterministic check on every item, and a reproducible sample returned for review. Source text never enters this conversation — only counts, failures and the sample come back. Optionally writes a labelled result file inside the workspace.',
    parameters: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: 'Template applied per item. Substitutions: {{item}}, {{index}} (0-based), {{total}}.',
        },
        items: {
          type: 'array',
          items: { type: 'string' },
          description: 'Inline items. Use this or input_path.',
        },
        input_path: {
          type: 'string',
          description: 'Absolute path of a long text file; split on blank lines, then by chunk_chars.',
        },
        chunk_chars: { type: 'integer', description: 'Target characters per chunk for input_path (default 6000).' },
        model: PARAM_MODEL,
        concurrency: { type: 'integer', description: 'Parallel Ollama requests (default 2). Keep low on one machine.' },
        verify: PARAM_VERIFY,
        sample_rate: { type: 'number', description: 'Fraction of items returned for review (default 0.02, at least 1).' },
        max_sample: { type: 'integer', description: 'Cap on the review sample size (default 8).' },
        output_path: {
          type: 'string',
          description: 'Write the labelled result file here; must resolve inside the session workspace.',
        },
        write_labels: { type: 'boolean', description: 'Include the provenance header and per-item tags (default true).' },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          model: { type: 'string' },
          total: { type: 'integer' },
          ok: { type: 'integer' },
          failed: { type: 'integer' },
          elapsed_ms: { type: 'integer' },
          output_path: { type: 'string' },
          samples: {
            type: 'array',
            items: {
              type: 'object',
              properties: { index: { type: 'integer' }, text: { type: 'string' } },
              required: ['index', 'text'],
            },
          },
          failures: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                index: { type: 'integer' },
                reasons: { type: 'array', items: { type: 'string' } },
                snippet: { type: 'string' },
              },
              required: ['index', 'reasons'],
            },
          },
          reason_counts: { type: 'object' },
        },
        required: ['model', 'total', 'ok', 'failed'],
      },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
    },
    async execute(args, exec) {
      const model = args.model ?? cfg.model
      const started = Date.now()

      let items
      if (Array.isArray(args.items) && args.items.length > 0) {
        items = args.items.map((item) => String(item))
      } else if (typeof args.input_path === 'string' && args.input_path.length > 0) {
        items = splitIntoChunks(readFileSync(resolve(args.input_path), 'utf8'), args.chunk_chars)
      } else {
        throw new Error('ollama_batch: 需要 items[] 或 input_path 之一')
      }
      if (items.length === 0) throw new Error('ollama_batch: 没有可处理的条目')

      let outputTarget
      if (typeof args.output_path === 'string' && args.output_path.length > 0) {
        const root = await workspaceRoot(ctx, exec)
        if (!root) {
          throw new Error(
            'ollama_batch: 解析不到会话工作区，按约定拒绝写入。改用 items[] 只取返回值，或确认 workingDirectory 服务可用。',
          )
        }
        outputTarget = resolve(root, args.output_path)
        if (!insideRoot(root, outputTarget)) {
          throw new Error(
            `ollama_batch: 拒绝写入工作区之外：${outputTarget}（工作区 ${root}）。请把 output_path 指到工作区内。`,
          )
        }
      }

      // Fail before doing any work when the tag does not exist, and say what does.
      const installed = (await fetchInventory(cfg, exec.signal)).map((entry) => String(entry?.name ?? ''))
      if (!installed.includes(model)) {
        throw new Error(
          `ollama_batch: 本机没有模型 ${JSON.stringify(model)}。已安装：${installed.join(', ') || '(none)'}。先调用 ollama_models 选择模型。`,
        )
      }

      const concurrency = Math.max(1, Math.min(8, Number(args.concurrency) || cfg.concurrency || 2))
      const results = new Array(items.length)
      let cursor = 0

      async function worker() {
        while (cursor < items.length) {
          const index = cursor++
          if (exec.signal?.aborted) throw new Error('cancelled')
          const prompt = String(args.prompt)
            .replaceAll('{{item}}', items[index])
            .replaceAll('{{index}}', String(index))
            .replaceAll('{{total}}', String(items.length))
          try {
            const answered = await chat(cfg, {
              model,
              system: DEFAULT_SYSTEM,
              prompt,
              json: args.verify?.json === true,
              signal: exec.signal,
            })
            results[index] = { text: answered.text, reasons: inspectResult(answered.text, args.verify) }
          } catch (error) {
            results[index] = {
              text: '',
              reasons: [`call_failed(${String(error?.message ?? error).slice(0, 80)})`],
            }
          }
        }
      }

      await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()))

      const reasonCounts = {}
      const failures = []
      results.forEach((entry, index) => {
        if (!entry) {
          results[index] = { text: '', reasons: ['missing'] }
          entry = results[index]
        }
        if (entry.reasons.length === 0) return
        for (const reason of entry.reasons) reasonCounts[reason] = (reasonCounts[reason] ?? 0) + 1
        if (failures.length < 20) {
          failures.push({ index, reasons: entry.reasons, snippet: entry.text.trim().slice(0, 200) })
        }
      })

      const okCount = results.filter((entry) => entry.reasons.length === 0).length
      const picks = sampleIndexes(items.length, args.sample_rate ?? 0.02, args.max_sample ?? 8)
      const samples = picks.map((index) => ({ index, text: results[index].text.trim().slice(0, 600) }))

      let written
      if (outputTarget) {
        const labels = args.write_labels !== false
        const lines = []
        if (labels) {
          lines.push('# ollama_batch 结果')
          lines.push(`# 生成模型: ${model} (num_ctx=${cfg.numCtx})`)
          lines.push(`# 条数: ${items.length}  确定性校验通过: ${okCount}  失败: ${items.length - okCount}`)
          lines.push(
            `# 校验: 代码全量确定性校验${picks.length > 0 ? `；抽样待复核 ${picks.length} 条: ${picks.join(', ')}` : ''}`,
          )
          lines.push(`# 生成时间: ${new Date().toISOString()}`)
          lines.push('')
        }
        results.forEach((entry, index) => {
          const tag =
            entry.reasons.length > 0
              ? `!!校验失败(${entry.reasons.join('|')})`
              : picks.includes(index)
                ? '*抽样待复核*'
                : 'ok'
          lines.push(labels ? `[${index}] (${model} · ${tag}) ${entry.text.trim()}` : `[${index}] ${entry.text.trim()}`)
        })
        mkdirSync(dirname(outputTarget), { recursive: true })
        writeFileSync(outputTarget, `${lines.join('\n')}\n`, 'utf8')
        written = outputTarget
      }

      return {
        model,
        total: items.length,
        ok: okCount,
        failed: items.length - okCount,
        elapsed_ms: Date.now() - started,
        ...(written ? { output_path: written } : {}),
        samples,
        failures,
        reason_counts: reasonCounts,
      }
    },
  }
}

export function apply(ctx, config = {}) {
  const cfg = {
    baseURL: config.baseURL ?? DEFAULTS.baseURL,
    model: config.model ?? DEFAULTS.model,
    numCtx: config.numCtx ?? DEFAULTS.numCtx,
    concurrency: config.concurrency ?? DEFAULTS.concurrency,
    timeoutMs: config.timeoutMs ?? DEFAULTS.timeoutMs,
  }
  ctx.tools.register(modelsTool(cfg))
  ctx.tools.register(askTool(cfg))
  ctx.tools.register(batchTool(cfg, ctx))
  ctx.tools.register(lookupTool(cfg, ctx))
}

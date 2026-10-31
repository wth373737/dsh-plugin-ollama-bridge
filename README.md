# dsh-plugin-ollama-bridge

**English** | [简体中文](README.zh.md)

Hand the bulk low-difficulty long-text work to the Ollama models on this machine, and keep the leading model on scheduling and review.

> A DeepSeek Harness plugin that registers four tools covering the whole loop: see what is installed → try one call → run a batch over long text → fill metadata gaps over the web when needed. Long source text never enters the conversation: only counts, failure distribution and a review sample come back, so the saving is on main-model tokens.

## Tools

| Tool | What it does |
| --- | --- |
| `ollama_models` | Lists installed models with parameter count, quantization, context length, capabilities, and `roles` hints (`code` / `reasoning` / `vision` / `embedding` / `long-context` …) |
| `ollama_ask` | One question, **unverified answer** — for probes or a few high-value judgements |
| `ollama_batch` | Chunking + concurrency + retries + a deterministic check on every item + review sample + optional result file |
| `ollama_lookup` | Web lookup of model capability, **after the user approves the exact queries** |

## Recommended flow: look before you choose

1. Call `ollama_models` for the inventory and metadata hints.
2. Summarise what each model is good for. `roles` are metadata hints, not measured quality.
3. When metadata is not enough (benchmarks, known weaknesses), call `ollama_lookup` — it hands the exact queries to the user for approval, and returns `approved: false` without searching when they decline.
4. Run `ollama_batch` with the chosen tag. An unknown tag fails immediately and names the installed models.

`ollama_batch` returns only: item count, passed/failed, failure-reason counts, at most 20 failure snippets, and the 1–3% sample verbatim. The source text stays inside the plugin process.

## Install

**From GitHub (recommended, updatable)**

In DSH, open **Plugins → + Add plugin** and enter:

```
github:wth373737/dsh-plugin-ollama-bridge
```

**From a local directory**

```bash
pnpm add file:/path/to/dsh-plugin-ollama-bridge
```

Restart DSH after installing. The plugin appears under **Plugins → Installed plugins** as “本地 Ollama 桥”.

## Configuration

The `config` block of `cordis.patch.yml`:

```yaml
baseURL: http://127.0.0.1:11434
model: qwen2.5:7b
numCtx: 8192
concurrency: 2
timeoutMs: 180000
```

- `numCtx`: context available per chunk, and the main driver of memory use.
- `concurrency`: parallel requests. 1–2 is sensible on one machine; lower it for larger models.
- If Ollama keeps its models outside the default directory, set `OLLAMA_MODELS` to the model root.

## How far to trust the output

Every item carries one of three states, in the written file and in the returned sample:

| Label | Meaning |
| --- | --- |
| `ok` | Passed every deterministic check, but **not read one by one** |
| `*抽样待复核*` | Sampled and returned; the leading model read it |
| `!!校验失败(原因)` | Malformed: empty, too long, not JSON, outside the allowed choices, missing a required substring, regex miss, or refusing |

Deterministic checks verify **form**, never **meaning**. Report “all items are well-formed” separately from “the content is correct”, and raise `sample_rate` for small batches.

## Boundaries

- **Reads**: any absolute path the host process can read.
- **Writes**: only inside the session workspace. A path outside it is refused with the workspace root in the message; an unresolvable workspace refuses the write rather than guessing.
- **Network**: only the configured `baseURL`; `ollama_lookup` reaches the host's configured search provider only after the user approves.
- **Approval**: with no answerer available, `ollama_lookup` refuses to look up instead of silently skipping approval.

## Development

```bash
npm test     # offline smoke test: stubs fetch/web/userQuestions and checks
             # verification, sampling, labels, write confinement, approval gate
```

No Ollama needed, and nothing is written outside the workspace.

## License

MIT © 2026 wth373737

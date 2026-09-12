# TokenPolice — Node.js SDK

TokenPolice blocks expensive and runaway LLM requests **before** they reach the provider, using
the budget and anomaly rules you set in the dashboard. This SDK is the piece that runs in your app.

- **One install, working capture.** `npm install token-police` captures **OpenAI** and
  **Anthropic** calls out of the box — no extra packages to wire up.
- **Fails open, always.** If TokenPolice is unreachable or anything goes wrong inside the SDK, your
  LLM call proceeds normally. The firewall never becomes a hard dependency of your request path.
- **Private by design.** Token counts and session metadata are extracted in your own process. Only
  that metadata is sent to TokenPolice — never prompt or completion text.

Docs: [tokenpolice.ai/docs](https://tokenpolice.ai/docs) · Dashboard: [app.tokenpolice.ai](https://app.tokenpolice.ai)

## Installation

```bash
npm install token-police
```

That single package captures tokens for **OpenAI** (`openai`) and **Anthropic**
(`@anthropic-ai/sdk`). Cohere, AWS Bedrock, Google Gemini, OpenRouter, Cerebras, Together.ai, xAI,
the Vercel AI SDK and LlamaIndex are covered by the same install — see the
[support matrix](#provider--framework-support). **LangChain** needs one extra package:
`npm install token-police-langchain` (see [LangChain](#langchain)).

TokenPolice instruments the provider SDKs **your app already installs**. It never pulls or pins
those versions. If a provider SDK is missing or on a version the SDK doesn't fully support, token
capture for it is skipped and **your app keeps working**.

## Get an API key

Create a key in the dashboard at [app.tokenpolice.ai](https://app.tokenpolice.ai) → **API Keys**.
It looks like `tp_sk_…` and is shown **once**. Put it in the `TOKENPOLICE_API_KEY` environment
variable (recommended) or pass it to `init()`.

## Add it with your coding agent (recommended)

The fastest way to integrate is to let your coding agent do it. Install the TokenPolice skill
once — it works with Claude Code, Cursor, GitHub Copilot, Antigravity and Codex; the install
steps for each are at [Install the skill](https://tokenpolice.ai/docs/get-started/coding-agent/install).
Then ask the agent in plain language. You write no integration code yourself.

In your project:

> Integrate TokenPolice into this app.

The agent reads your code to find your LLM calls and where your user, plan and session values
live; asks you for your API key and confirms those fields; shows you the full plan and changes
nothing until you approve; wires the SDK in `dry_run` mode (nothing blocked yet); and after you
run the app once, checks that TokenPolice received the data. The edits are the same few lines
shown in the manual quick start below.
[Walkthrough](https://tokenpolice.ai/docs/get-started/coding-agent/walkthrough).

## Quick start (manual)

### 1. Initialize once, at startup

```typescript
import * as tp from 'token-police';
import OpenAI from 'openai';

tp.init({
  // apiKey: 'tp_sk_...',   // or set TOKENPOLICE_API_KEY
  firewall: 'enforce',      // 'dry_run' (default) evaluates rules without acting; 'enforce' acts on them; 'off' records usage only
});

const client = new OpenAI();

// Checked against your rules before it runs; token usage is recorded afterward.
const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello world!' }],
});
```

Call `tp.init()` before your first LLM request. The SDK finds the LLM libraries you have installed
and starts capturing their calls.

`enforce` acts on the rules you create in the dashboard — with no rules, nothing is blocked. Start
in `dry_run` to see what *would* happen, then switch to `enforce`.

> **ESM import ordering:** if you import your provider SDK *before* `tp.init()` runs and find calls
> aren't captured, pass the module explicitly via `instrumentModules` — e.g.
> `tp.init({ /* ... */, instrumentModules: { openAI: OpenAI } })`. This is rarely needed for CJS.
> Details: [ESM instrumentation order](https://tokenpolice.ai/docs/troubleshooting/esm-instrumentation-order).

### 2. Handle a blocked call

In `enforce` mode a blocked call throws `TokenPoliceBlockedError` instead of reaching the provider.
It is the **only** error the SDK ever throws into your code.

```typescript
try {
  await client.chat.completions.create({ /* ... */ });
} catch (e) {
  if (e instanceof tp.TokenPoliceBlockedError) {
    console.log(e.reason, e.ruleId, e.kind, e.traceId); // all optional
    return { error: 'Usage limit reached. Please try again later.' };
  }
  throw e;
}
```

### 3. Group by workflow & user

Attribute and budget usage per user, plan, feature or workflow.

```typescript
// Recommended for request handlers: session()  → an "agent" span by default
await tp.session(
  { name: 'customer_support', userId: 'user_123', paidPlan: 'pro', metadata: { tenant: 'acme' } },
  async () => {
    return client.chat.completions.create({ /* ... */ });
  },
);

// Recommended for reusable functions: workflow()  → a "chain" span by default
const runPipeline = tp.workflow(
  { name: 'rag_pipeline' },
  async (opts: { userId: string; query: string }) => {
    // userId is auto-extracted from the first object argument.
    return client.chat.completions.create({ /* ... */ });
  },
);
await runPipeline({ userId: 'user_123', query: 'How do I reset my password?' });
```

`userId`, `paidPlan`, `name` and `metadata` are what rules match on and what the dashboard groups
by. See [Identity](https://tokenpolice.ai/docs/concepts/identity).

#### Agent vs chain spans

The grouping span is one of two kinds:

- **`agent`** — a *dynamic, LLM-driven* loop (the model decides each next step).
  `tp.session(...)` and `tp.agent(...)` emit this.
- **`chain`** — a *static, developer-defined* sequence or glue code (step A → B → C),
  or a root entry point. `tp.chain(...)` and `tp.workflow(...)` emit this.

```typescript
await tp.chain({ name: 'rag_pipeline', userId: 'u1' }, async () => {
  const docs = await retrieve(q);                       // static step 1
  return client.chat.completions.create({ /* ... */ }); // static step 2
});

await tp.agent({ name: 'research_agent', userId: 'u1' }, async () => {
  // an autonomous loop that calls tools and re-plans
});
```

Pass `{ kind: 'agent' | 'chain' }` to `session()` / `workflow()` to override the default.
Grouping spans carry no cost of their own.

#### Thread a conversation with `sessionId`

A trace is one run (one turn). Pass the **same** `sessionId` across turns to group them into a
conversation, visible under **Conversations** in the dashboard:

```typescript
// Each turn is a separate request → separate session() call, same sessionId.
await tp.session({ name: 'support', userId: 'u1', sessionId: conversationId }, async () => {
  return client.chat.completions.create({ /* ... */ });
});
```

Omit it and each run gets its own id. `workflow()` can also pick it up from the first-arg object
(`{ sessionId }` or `{ session_id }`).

### 4. Track tool calls

Wrap tool/function executions so they appear as tool spans under the active session:

```typescript
// Wrap a function once, call it many times:
const searchDocs = tp.tool({ name: 'search_docs' }, async (query: string) => {
  return db.search(query);
});

// Or wrap a single execution inline (pass `args` so the tool's parameters are recorded):
const result = await tp.toolSpan({ name: 'search_docs', args: query }, () => db.search(query));
```

### 5. Serverless (AWS Lambda, Vercel, …)

Wrap your handler so pending telemetry is sent before the container freezes:

```typescript
export const handler = tp.serverless(async (event, context) => {
  return tp.session({ name: 'lambda_handler', userId: event.userId }, async () => {
    const response = await client.chat.completions.create({ /* ... */ });
    return { statusCode: 200, body: response.choices[0]?.message?.content };
  });
});
```

### 6. Name a span

```typescript
tp.setSpanName('nightly_summarizer'); // applies to the next LLM span in this context
```

### 7. Manual check / log & protecting unsupported SDKs

If you call an LLM through an SDK TokenPolice doesn't recognize, you can drive enforcement yourself
or register the method for automatic enforcement:

```typescript
const client = tp.getClient();

// Manual pre-flight check
const result = await client.check('user_123', 'pro', 'my_workflow');
if (result.status === 'blocked') return; // budget exceeded

// ... make your LLM call ...

// Manual log: (userId, paidPlan, workflow, sessionId, model, provider, inputTokens, outputTokens)
client.log('user_123', 'pro', 'my_workflow', '', 'gpt-4o', 'openai', 100, 50);

// Or register a method so TokenPolice enforces + logs it automatically:
tp.protect('my-llm-sdk', ['Client', 'prototype'], 'generate', /* isAsync */ true);
```

> **`isAsync` matters for enforcement.** Pass `true` for Promise-returning methods (the common
> case) to get full pre-flight enforcement. With `isAsync: false` the wrapper only records usage:
> a synchronous method can't wait for the check, so it is never blocked or rerouted. Register an
> async method with `isAsync: true` whenever you need enforcement.

### Clean shutdown (scripts, CLIs, batch jobs)

TokenPolice keeps a live connection open so rule changes reach your app immediately. In a
long-running **server** that's what you want. In a **short-lived script, CLI, or batch job** that
connection keeps Node's event loop alive, so the process won't exit on its own. Call
`await tp.shutdown()` when you're done — it sends any pending telemetry **and** closes the
connection:

```typescript
async function main() {
  await tp.session({ name: 'my_job', userId: 'user_123' }, async () => {
    await client.chat.completions.create({ /* ... */ });
  });
  await tp.shutdown(); // flush + close → the process exits cleanly
}
main();
```

> Use `tp.shutdown()` rather than a bare `tp.flush()` at the end of a script — `flush()` waits for
> pending logs but leaves the connection open, so the process hangs. Both also send the last call's
> telemetry, so a short script doesn't lose it. Long-running servers can skip this entirely.

### Fail-open guarantee

TokenPolice is designed to **never crash your application**. If the service is unreachable or
anything fails inside the SDK, the SDK fails open and lets your LLM call proceed. The only exception
is a deliberate block in `enforce` mode, which throws `TokenPoliceBlockedError`.
See [Fail-open](https://tokenpolice.ai/docs/concepts/fail-open).

### A note on the capture warning

In a constrained install environment, token capture for a provider can be skipped. If that happens
and you use that provider, TokenPolice logs **once**:

```
[TokenPolice Warning] token capture for openai is disabled — its bundled instrumentor
didn't load (optional dependencies may have been skipped at install); reinstall
token-police to enable it. Budgets are still enforced.
```

Budgets keep being enforced — only token *capture* for that provider is affected. To fix it,
reinstall `token-police`. You'll never see this warning for a provider you don't use.

## Provider & framework support

| Provider / framework | Coverage |
|---|---|
| OpenAI | Base install |
| Anthropic | Base install |
| Cohere (v2) | Base install |
| AWS Bedrock | Base install |
| Google Gemini (`@google/genai`, incl. 2.x) | Base install |
| OpenRouter | Base install |
| Cerebras | Base install |
| Together.ai | Base install |
| xAI (`@ai-sdk/xai`) | Base install |
| Voyage AI (`voyageai`) | Base install |
| Vercel AI SDK (`ai` — all providers) | Base install; community providers via `instrumentModules.aiSdkProviders` or `tokenPoliceAiSdkMiddleware()` |
| LlamaIndex | Base install |
| LangChain | `npm install token-police-langchain` |

Full matrix and tested versions: [Integrations](https://tokenpolice.ai/docs/integrations/matrix) ·
[Supported versions](https://tokenpolice.ai/docs/sdk/supported-versions).

### Vercel AI SDK

Every AI SDK model is covered: `generateText`, `streamText`, structured outputs, and each step of a
tool loop are checked before the call and their tokens recorded (v5 and v6 usage shapes, including
cache read/write and reasoning detail). `embed`/`embedMany`, `generateImage`,
`experimental_transcribe`, `experimental_generateSpeech` and `experimental_generateVideo` are
covered too, and modality-scoped rules apply to them.

- **First-party providers & AI Gateway** (plain `"vendor/model"` strings): zero config.
- **Community / custom providers** (they bundle their own nested `@ai-sdk/*` copy):
  `tp.init({ instrumentModules: { aiSdkProviders: [myProvider] } })`.
- **Middleware alternative**: `wrapLanguageModel({ model, middleware: tokenPoliceAiSdkMiddleware() })` —
  safe to use together with auto-detection.

With `firewall: 'enforce'`, a blocked `generateText` throws `TokenPoliceBlockedError` directly;
`streamText` surfaces it through the stream's error path (`onError` / rejected promises).

### LangChain

```bash
npm install token-police token-police-langchain
```

`token-police`'s `init()` detects the companion package automatically — there is no API to call. It
is a separate package so apps that don't use LangChain don't install LangChain's dependency tree.
See [LangChain](https://tokenpolice.ai/docs/integrations/langchain).

## Configuration

### `tp.init(options)`

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | `TOKENPOLICE_API_KEY` | Your TokenPolice API key (`tp_sk_...`) |
| `baseUrl` | `string` | `TOKENPOLICE_BASE_URL` or `https://collect.tokenpolice.ai` | TokenPolice endpoint. Leave unset unless told otherwise |
| `timeout` | `number` | `2.0` | Max seconds for a call to TokenPolice; on timeout the LLM call proceeds |
| `firewall` | `'enforce' \| 'dry_run' \| 'off'` | `'dry_run'` | `'enforce'` acts on your rules; `'dry_run'` evaluates them and records what would have happened without acting; `'off'` records usage only |
| `enforce` | `boolean` | — | **Deprecated** alias for `firewall` (`true` → `'enforce'`, `false` → `'off'`) |
| `logErrors` | `boolean` | `false` | Log SDK errors at WARNING level |
| `instrumentModules` | `object` | — | Explicit module refs (`{ openAI, anthropic, … }`) for ESM import-ordering edge cases |
| `deployment` | `'auto' \| 'daemon' \| 'serverless' \| 'edge'` | `'auto'` | Process shape hint; auto-detected. Setting it does not flush for you |
| `captureStreamUsage` | `boolean` | `true` | Ask the provider for the usage frame on streamed calls so they report token counts (`TP_CAPTURE_STREAM_USAGE=0` disables) |
| `errorDetail` | `'none' \| 'redacted' \| 'raw'` | `'redacted'` | How much of a failed provider call's error text leaves the process. `raw` is opt-in only — provider 400s can echo prompt content |
| `sseReconnectMaxIntervalSeconds` | `number` | `300` | Cap on the backoff between reconnect attempts of the live rule feed |
| `streamStaleGraceSeconds` | `number` | `60` | Clamped to 0–3600. How long the cached rule view is trusted after the live feed drops before a matching call is re-checked with the server |

### Environment variables

| Variable | Purpose |
|---|---|
| `TOKENPOLICE_API_KEY` | API key, used when `apiKey` is not passed |
| `TOKENPOLICE_BASE_URL` | Endpoint override, used when `baseUrl` is not passed |
| `TP_CAPTURE_STREAM_USAGE` | Set to `0` to stop requesting usage frames on streamed calls |

### Exports

| Export | Signature | Purpose |
|---|---|---|
| `init` | `init(options)` | Initialize the SDK. |
| `session` | `session(opts, callback)` | Run a callback inside a session context (`agent` span). |
| `agent` | `agent(opts, callback)` | Same as `session` — an explicit `agent` span. |
| `chain` | `chain(opts, callback)` | Run a callback inside a `chain` span. |
| `workflow` | `workflow(opts, fn)` | Wrap a function with a session context (auto-extracts `userId`). |
| `serverless` | `serverless(fn)` | Wrap a handler to auto-flush telemetry on return. |
| `tool` | `tool(opts, fn)` | Wrap a function so each call emits a tool span. |
| `toolSpan` | `toolSpan(opts, fn)` | Emit a tool span around a single execution. |
| `setSpanName` | `setSpanName(name)` | Name the next LLM span in the current context. |
| `getCurrentSession` | `getCurrentSession()` | The active `TPSession`, if any. |
| `TPSession` | class | The session object returned by `getCurrentSession()`. |
| `protect` | `protect(moduleName, objectPath, methodName, isAsync, options?)` | Register a custom SDK method for enforcement + logging. |
| `tokenPoliceAiSdkMiddleware` | `tokenPoliceAiSdkMiddleware()` | Vercel AI SDK middleware alternative to auto-detection. |
| `uninstrument` | `uninstrument()` | Remove all enforcement hooks. |
| `flush` | `flush()` | Async-wait for all pending log calls. |
| `flushSync` | `flushSync()` | **Diagnostic only — does not drain.** Logs how many sends are still pending. Use `await flush()` (or `await shutdown()`) to actually drain. |
| `shutdown` | `shutdown()` | Flush pending logs **and** close the live connection so a script/CLI process can exit. |
| `getClient` | `getClient()` | Get the underlying client for manual `check`/`log`. |
| `TokenPoliceBlockedError` | — | Thrown when an enforced call is blocked (`reason`, `ruleId`, `kind`, `traceId`). |

## What leaves your process

Prompt text and completion content never leave your process. TokenPolice receives token counts,
model and provider names, and the session metadata you attach (`userId`, `paidPlan`, `name`,
`sessionId`, `metadata`). Error text from a failed provider call is redacted by default
(`errorDetail`). See [Data privacy](https://tokenpolice.ai/docs/concepts/data-privacy) and the
[privacy policy](https://tokenpolice.ai/privacy).

## Requirements

- Node.js >= 20
- TypeScript >= 5.0 (optional, for TypeScript projects)

## Support

- Docs: [tokenpolice.ai/docs](https://tokenpolice.ai/docs)
- Issues: [github.com/tokenpolice/token-police-node/issues](https://github.com/tokenpolice/token-police-node/issues)
- Email: [support@tokenpolice.ai](mailto:support@tokenpolice.ai) · Security: [security@tokenpolice.ai](mailto:security@tokenpolice.ai)

## License

Apache-2.0

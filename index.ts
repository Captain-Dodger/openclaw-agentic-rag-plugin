import { spawn } from "node:child_process";

type PluginCfg = {
  enabled?: boolean;
  pythonBin?: string;
  workingDir?: string;
  bridgeScript?: string;
  retrievalMode?: "lexical" | "hybrid";
  embeddingEnabled?: boolean;
  embeddingBaseUrl?: string;
  embeddingModel?: string;
  embeddingTimeoutMs?: number;
  hybridLexicalWeight?: number;
  hybridMinLexicalScore?: number;
  corpusPath?: string;
  minRetrievalScore?: number;
  minConfidence?: number;
  topK?: number;
  maxContextChars?: number;
  abstainMessage?: string;
  arbiterEnabled?: boolean;
  arbiterSharedLabel?: string;
  arbiterMinEvidenceChars?: number;
  arbiterHighImpactMargin?: number;
  arbiterAllowRefine?: boolean;
  arbiterFailClosedOnConflict?: boolean;
  timeoutMs?: number;
};

type BridgePayload = {
  query: string;
  state?: Record<string, unknown>;
  pluginConfig?: PluginCfg;
};

type BridgeResult =
  | { ok: true; result: Record<string, unknown> }
  | { ok: false; error: string; traceback?: string };

function resolveConfig(api: any): Required<Pick<PluginCfg, "enabled" | "pythonBin" | "bridgeScript" | "timeoutMs">> &
  Omit<PluginCfg, "enabled" | "pythonBin" | "bridgeScript" | "timeoutMs"> {
  const cfg = ((api?.pluginConfig ?? {}) as PluginCfg) || {};
  return {
    enabled: cfg.enabled !== false,
    pythonBin: (cfg.pythonBin || "python3").trim(),
    bridgeScript: (cfg.bridgeScript || "bridge/run_agentic_rag_tool.py").trim(),
    timeoutMs:
      typeof cfg.timeoutMs === "number" && Number.isFinite(cfg.timeoutMs)
        ? Math.max(1000, Math.min(120000, Math.floor(cfg.timeoutMs)))
        : 15000,
    workingDir: cfg.workingDir,
    corpusPath: cfg.corpusPath,
    retrievalMode: cfg.retrievalMode,
    embeddingEnabled: cfg.embeddingEnabled,
    embeddingBaseUrl: cfg.embeddingBaseUrl,
    embeddingModel: cfg.embeddingModel,
    embeddingTimeoutMs: cfg.embeddingTimeoutMs,
    hybridLexicalWeight: cfg.hybridLexicalWeight,
    hybridMinLexicalScore: cfg.hybridMinLexicalScore,
    minRetrievalScore: cfg.minRetrievalScore,
    minConfidence: cfg.minConfidence,
    topK: cfg.topK,
    maxContextChars: cfg.maxContextChars,
    abstainMessage: cfg.abstainMessage,
    arbiterEnabled: cfg.arbiterEnabled,
    arbiterSharedLabel: cfg.arbiterSharedLabel,
    arbiterMinEvidenceChars: cfg.arbiterMinEvidenceChars,
    arbiterHighImpactMargin: cfg.arbiterHighImpactMargin,
    arbiterAllowRefine: cfg.arbiterAllowRefine,
    arbiterFailClosedOnConflict: cfg.arbiterFailClosedOnConflict,
  };
}

function runBridge(params: {
  pythonBin: string;
  scriptPath: string;
  cwd?: string;
  timeoutMs: number;
  payload: BridgePayload;
}): Promise<BridgeResult> {
  return new Promise((resolve) => {
    const child = spawn(params.pythonBin, [params.scriptPath], {
      cwd: params.cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    let done = false;

    const finish = (result: BridgeResult) => {
      if (done) {
        return;
      }
      done = true;
      resolve(result);
    };

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({
        ok: false,
        error: `bridge timeout after ${params.timeoutMs} ms`,
      });
    }, params.timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      finish({ ok: false, error: `bridge spawn failed: ${err.message}` });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      const trimmed = stdout.trim();
      if (!trimmed) {
        finish({
          ok: false,
          error: `bridge returned empty output (exit=${code ?? "unknown"}): ${stderr.trim()}`,
        });
        return;
      }
      try {
        const parsed = JSON.parse(trimmed) as BridgeResult;
        if (parsed && typeof parsed === "object" && "ok" in parsed) {
          finish(parsed);
          return;
        }
        finish({
          ok: false,
          error: "bridge output is not a valid response envelope",
        });
      } catch (err) {
        finish({
          ok: false,
          error: `bridge output is not valid JSON: ${(err as Error).message}`,
          traceback: trimmed.slice(0, 4000),
        });
      }
    });

    child.stdin.end(JSON.stringify(params.payload));
  });
}

const plugin = {
  id: "agentic-rag",
  name: "Agentic RAG",
  description: "Confidence-gated retrieval tool backed by Python bridge.",
  register(api: any) {
    api.registerTool(
      {
        name: "agentic_rag",
        description:
          "Retrieve grounded evidence from local corpus. Returns answer when grounded, abstains when evidence is weak.",
        parameters: {
          type: "object",
          additionalProperties: false,
          properties: {
            query: {
              type: "string",
              description: "Question or task prompt for retrieval-grounded answer.",
            },
            state: {
              type: "object",
              additionalProperties: true,
              description: "Optional session state hints forwarded to Python handler.",
            },
          },
          required: ["query"],
        },
        async execute(_id: string, params: Record<string, unknown>) {
          const cfg = resolveConfig(api);
          if (!cfg.enabled) {
            return {
              content: [{ type: "text", text: "agentic_rag plugin is disabled by config." }],
              details: { mode: "disabled" },
            };
          }

          const query = typeof params.query === "string" ? params.query.trim() : "";
          if (!query) {
            return {
              content: [{ type: "text", text: "query required" }],
              details: { mode: "input_error" },
            };
          }

          const bridgePayload: BridgePayload = {
            query,
            pluginConfig: {
              corpusPath: cfg.corpusPath,
              retrievalMode: cfg.retrievalMode,
              embeddingEnabled: cfg.embeddingEnabled,
              embeddingBaseUrl: cfg.embeddingBaseUrl,
              embeddingModel: cfg.embeddingModel,
              embeddingTimeoutMs: cfg.embeddingTimeoutMs,
              hybridLexicalWeight: cfg.hybridLexicalWeight,
              hybridMinLexicalScore: cfg.hybridMinLexicalScore,
              minRetrievalScore: cfg.minRetrievalScore,
              minConfidence: cfg.minConfidence,
              topK: cfg.topK,
              maxContextChars: cfg.maxContextChars,
              abstainMessage: cfg.abstainMessage,
              arbiterEnabled: cfg.arbiterEnabled,
              arbiterSharedLabel: cfg.arbiterSharedLabel,
              arbiterMinEvidenceChars: cfg.arbiterMinEvidenceChars,
              arbiterHighImpactMargin: cfg.arbiterHighImpactMargin,
              arbiterAllowRefine: cfg.arbiterAllowRefine,
              arbiterFailClosedOnConflict: cfg.arbiterFailClosedOnConflict,
            },
          };
          if (params.state && typeof params.state === "object" && !Array.isArray(params.state)) {
            bridgePayload.state = params.state as Record<string, unknown>;
          }

          const bridge = await runBridge({
            pythonBin: cfg.pythonBin,
            scriptPath: api.resolvePath(cfg.bridgeScript),
            cwd: cfg.workingDir ? api.resolvePath(cfg.workingDir) : undefined,
            timeoutMs: cfg.timeoutMs,
            payload: bridgePayload,
          });

          if (!bridge.ok) {
            const msg = `agentic_rag bridge error: ${bridge.error}`;
            return {
              content: [{ type: "text", text: msg }],
              details: { mode: "bridge_error", error: bridge.error, traceback: bridge.traceback },
            };
          }

          const details = bridge.result;
          const mode = String(details.mode ?? "unknown");
          const confidence = details.confidence;
          const answer =
            typeof details.answer === "string" && details.answer.trim()
              ? details.answer
              : "No answer payload returned.";
          const rationale =
            typeof details.rationale === "string" && details.rationale.trim()
              ? details.rationale
              : "No rationale provided.";
          const text = [
            `mode=${mode}`,
            `confidence=${confidence ?? "n/a"}`,
            "",
            answer,
            "",
            `rationale: ${rationale}`,
          ].join("\n");

          return {
            content: [{ type: "text", text }],
            details,
          };
        },
      },
      { optional: true },
    );
  },
};

export default plugin;

"use client";
import React, { createContext, useContext, useState, useRef, useEffect } from "react";
import { api, auth, keyStore, decryptApiKey } from "./api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export type ColabStep = "idle" | "creating" | "gist" | "done" | "error";

interface PersistentStateContextType {
  // Colab Launcher state
  colabStep: ColabStep;
  setColabStep: (s: ColabStep) => void;
  colabUrl: string;
  setColabUrl: (u: string) => void;
  colabErrMsg: string;
  setColabErrMsg: (m: string) => void;
  colabLoading: boolean;
  setColabLoading: (l: boolean) => void;
  colabJobId: string;
  setColabJobId: (id: string) => void;
  launcherConfig: {
    datasetId: string;
    model: string;
    taskType: string;
    epochs: number;
    loraR: number;
    hfRepo: string;
  };
  setLauncherConfig: React.Dispatch<React.SetStateAction<any>>;
  handleColabLaunch: (githubTokenEncrypted: string, userId: string, onJobsRefresh?: () => void) => Promise<void>;

  // Instruct Generator state
  genForm: any;
  setGenForm: React.Dispatch<React.SetStateAction<any>>;
  genPreview: any[];
  setGenPreview: React.Dispatch<React.SetStateAction<any[]>>;
  genResult: any;
  setGenResult: React.Dispatch<React.SetStateAction<any>>;
  genStep: "form" | "preview" | "result";
  setGenStep: (s: "form" | "preview" | "result") => void;
  genError: string;
  setGenError: (e: string) => void;
  genLoading: boolean;
  setGenLoading: (l: boolean) => void;
  genPreviewing: boolean;
  setGenPreviewing: (p: boolean) => void;
  genIsStreaming: boolean;
  genStreamProgress: any;
  startGenerateTextDataset: () => Promise<void>;
  connectTextStream: (id: string) => Promise<void>;

  // Structured Generator state
  structForm: any;
  setStructForm: React.Dispatch<React.SetStateAction<any>>;
  structColumns: any[];
  setStructColumns: React.Dispatch<React.SetStateAction<any[]>>;
  structResult: any;
  setStructResult: React.Dispatch<React.SetStateAction<any>>;
  structIsStreaming: boolean;
  structStreamProgress: any;
  structError: string;
  setStructError: (e: string) => void;
  structLoading: boolean;
  setStructLoading: (l: boolean) => void;
  structStep: "form" | "schema" | "result";
  setStructStep: (s: "form" | "schema" | "result") => void;
  startGenerateStructuredDataset: () => Promise<void>;
  connectStructuredStream: (id: string) => Promise<void>;

  // Scraper Ingest state
  scraperForm: any;
  setScraperForm: React.Dispatch<React.SetStateAction<any>>;
  scraperIsStreaming: boolean;
  scraperLogs: string[];
  scraperResult: any;
  scraperError: string;
  setScraperError: (e: string) => void;
  scraperLoading: boolean;
  setScraperLoading: (l: boolean) => void;
  scraperProgress: { pages: number; chunks: number; pairs: number };
  startScraperIngest: () => Promise<void>;
  connectScraperStream: (id: string) => Promise<void>;
}

const PersistentStateContext = createContext<PersistentStateContextType | undefined>(undefined);

export function PersistentStateProvider({ children }: { children: React.ReactNode }) {
  // ── 1. Colab Launcher State ──────────────────────────────────────────────
  const [colabStep, setColabStep] = useState<ColabStep>("idle");
  const [colabUrl, setColabUrl] = useState("");
  const [colabErrMsg, setColabErrMsg] = useState("");
  const [colabLoading, setColabLoading] = useState(false);
  const [colabJobId, setColabJobId] = useState("");
  const [launcherConfig, setLauncherConfig] = useState({
    datasetId: "",
    model: "microsoft/phi-3-mini-4k-instruct",
    taskType: "chat",
    epochs: 3,
    loraR: 16,
    hfRepo: "",
  });

  const handleColabLaunch = async (githubTokenEncrypted: string, userId: string, onJobsRefresh?: () => void) => {
    setColabLoading(true);
    setColabErrMsg("");
    try {
      setColabStep("gist");
      const githubToken = await decryptApiKey(githubTokenEncrypted, userId);
      const nb = await api.generateColabNotebook({
        dataset_id: launcherConfig.datasetId,
        base_model: launcherConfig.model || "microsoft/phi-3-mini-4k-instruct",
        task_type: launcherConfig.taskType,
        epochs: launcherConfig.epochs,
        lora_r: launcherConfig.loraR,
        lora_alpha: launcherConfig.loraR * 2,
        lora_dropout: 0.05,
        max_seq_length: 2048,
        warmup_steps: 10,
        learning_rate: 0.0002,
        github_token: githubToken,
        hf_repo_id: launcherConfig.hfRepo,
      });

      setColabUrl(nb.colab_url);
      setColabStep("done");
      if (onJobsRefresh) onJobsRefresh();
    } catch (e: any) {
      setColabErrMsg(e.message || "Something went wrong.");
      setColabStep("error");
    } finally {
      setColabLoading(false);
    }
  };

  // ── 2. Instruct Generator State ──────────────────────────────────────────
  const [genForm, setGenForm] = useState({
    topic: "",
    task_type: "chat",
    style: "casual",
    total_rows: 50,
    domain_context: "",
    dataset_name: "",
    multi_turn: false,
  });
  const [genPreview, setGenPreview] = useState<any[]>([]);
  const [genResult, setGenResult] = useState<any>(null);
  const [genStep, setGenStep] = useState<"form" | "preview" | "result">("form");
  const [genError, setGenError] = useState("");
  const [genLoading, setGenLoading] = useState(false);
  const [genPreviewing, setGenPreviewing] = useState(false);
  const [genIsStreaming, setGenIsStreaming] = useState(false);
  const [genStreamProgress, setGenStreamProgress] = useState<{
    coveragePlan: Record<string, number> | null;
    currentBatch: number;
    totalBatches: number;
    currentSubdomain: string;
    currentPersona: string;
    currentDifficulty: string;
    currentMode: string;
    currentDepth: string;
    generatedSoFar: number;
    logs: string[];
  }>({
    coveragePlan: null,
    currentBatch: 0,
    totalBatches: 0,
    currentSubdomain: "",
    currentPersona: "",
    currentDifficulty: "",
    currentMode: "",
    currentDepth: "",
    generatedSoFar: 0,
    logs: [],
  });

  const connectTextStream = async (datasetId: string) => {};

  const startGenerateTextDataset = async () => {
    setGenLoading(true);
    setGenIsStreaming(true);
    setGenError("");
    setGenStep("form");

    setGenStreamProgress({
      coveragePlan: null,
      currentBatch: 0,
      totalBatches: 0,
      currentSubdomain: "",
      currentPersona: "",
      currentDifficulty: "",
      currentMode: "",
      currentDepth: "",
      generatedSoFar: 0,
      logs: ["Initializing connection to dataset compiler..."],
    });

    try {
      const token = auth.getToken();
      const userId = auth.getUserId();
      let openaiKey = "";
      let deepseekKey = "";
      let groqKey = "";
      let hasKeys = false;
      if (userId) {
        const encOpenAI = keyStore.get("openai");
        if (encOpenAI) {
          try {
            openaiKey = await decryptApiKey(encOpenAI, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt OpenAI key", e);
          }
        }
        const encDeepseek = keyStore.get("deepseek");
        if (encDeepseek) {
          try {
            deepseekKey = await decryptApiKey(encDeepseek, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt DeepSeek key", e);
          }
        }
        const encGroq = keyStore.get("groq");
        if (encGroq) {
          try {
            groqKey = await decryptApiKey(encGroq, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt Groq key", e);
          }
        }
      }

      const response = await fetch(`${API_BASE}/generate/dataset/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": token ? `Bearer ${token}` : "",
          ...(hasKeys ? { "X-Key-Configured": "true" } : {}),
        },
        body: JSON.stringify({
          ...genForm,
          save_to_account: true,
          openai_api_key: openaiKey || undefined,
          deepseek_api_key: deepseekKey || undefined,
          groq_api_key: groqKey || undefined,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Compilation request failed" }));
        throw new Error(err.detail || "Compilation request failed");
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("Network stream unavailable");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const dataStr = trimmed.slice(6);
            try {
              const event = JSON.parse(dataStr);
              if (event.event === "start") {
                setGenStreamProgress((prev) => ({
                  ...prev,
                  coveragePlan: event.coverage_plan,
                  totalBatches: event.batches,
                  logs: [
                    ...prev.logs,
                    `✓ Connection established. Profiled "${event.policy.dataset_name}" (Factuality risk: ${event.policy.factuality_risk}).`,
                    `✓ Decomposed topic into subdomains & allocated quotas.`,
                  ],
                }));
              } else if (event.event === "batch_start") {
                setGenStreamProgress((prev) => ({
                  ...prev,
                  currentBatch: event.batch,
                  currentSubdomain: event.subdomain,
                  currentPersona: event.persona,
                  currentDifficulty: event.difficulty,
                  currentMode: event.mode,
                  currentDepth: event.depth,
                  logs: [
                    ...prev.logs,
                    `⚡ Synthesizing batch ${event.batch}/${prev.totalBatches} for subdomain [${event.subdomain.toUpperCase()}] using ${event.persona.toUpperCase()} (mode: ${event.mode}, depth: ${event.depth})...`,
                  ],
                }));
              } else if (event.event === "batch_done") {
                const reasons: string[] = [];
                if (event.filter_reasons) {
                  const fr = event.filter_reasons;
                  if (fr.empty_io) reasons.push(`${fr.empty_io} empty`);
                  if (fr.too_short) reasons.push(`${fr.too_short} too short`);
                  if (fr.generic_phrase) reasons.push(`${fr.generic_phrase} generic`);
                  if (fr.local_duplicate) reasons.push(`${fr.local_duplicate} local duplicate`);
                  if (fr.banned_starter) reasons.push(`${fr.banned_starter} banned starter`);
                  if (fr.semantic_duplicate) reasons.push(`${fr.semantic_duplicate} duplicate`);
                  if (fr.bigram_cap_exceeded) reasons.push(`${fr.bigram_cap_exceeded} repetition`);
                }
                const reasonStr = reasons.length > 0 ? ` (Filtered: ${reasons.join(", ")})` : "";
                setGenStreamProgress((prev) => ({
                  ...prev,
                  generatedSoFar: event.total_so_far,
                  logs: [
                    ...prev.logs,
                    `✓ Batch ${event.batch} → ${event.total_so_far} rows total. (${event.generated} generated, ${event.filtered_out} filtered${reasonStr}) [Took ${event.time_taken}s | ETA: ${event.eta}s]`,
                  ],
                }));
              } else if (event.event === "filter_adapted") {
                setGenStreamProgress((prev) => ({
                  ...prev,
                  logs: [
                    ...prev.logs,
                    `🔧 Adaptive filter: yield=${(event.yield_rate * 100).toFixed(0)}% — relaxed sim→${event.new_sim_threshold}, bigram_cap→${event.new_bigram_cap} for domain dataset.`,
                  ],
                }));
              } else if (event.event === "batch_error") {
                setGenStreamProgress((prev) => ({
                  ...prev,
                  logs: [...prev.logs, `⚠ Error in batch ${event.batch}: ${event.error}`],
                }));
              } else if (event.event === "complete") {
                setGenStreamProgress((prev) => ({
                  ...prev,
                  logs: [
                    ...prev.logs,
                    `✓ Generation complete! Running advanced semantic and logical audit...`,
                    `✓ Contradiction audits: ${event.fingerprint?.contradictions_found || 0} contradictions found.`,
                    `✓ Shannon starting-word entropy: ${event.fingerprint?.starting_word_entropy || 0}`,
                    `✓ Shannon category entropy: ${event.fingerprint?.cluster_balance || 0}`,
                    `✓ Dataset saved to database.`,
                  ],
                }));
                setGenResult({
                  id: event.id,
                  name: event.name,
                  row_count: event.row_count,
                  quality_score: event.quality_score ?? event.audit?.quality_score ?? 0,
                  audit: event.audit,
                  fingerprint: event.fingerprint,
                  coverage_plan: event.coverage_plan,
                  sample_rows: event.sample_rows,
                  rows: event.rows,
                });
                setGenStep("result");
                setGenIsStreaming(false);
              } else if (event.event === "error") {
                throw new Error(event.message || "Compilation failed");
              }
            } catch (err: any) {
              if (err?.message && !err.message.includes("JSON")) throw err;
            }
          }
        }
      }
    } catch (e: any) {
      setGenError(e.message || "Text dataset generation failed");
      setGenIsStreaming(false);
    } finally {
      setGenLoading(false);
    }
  };

  // ── 3. Structured Generator State ────────────────────────────────────────
  const [structForm, setStructForm] = useState({
    description: "",
    num_columns: 8,
    total_rows: 200,
    domain_context: "",
    dataset_name: "",
    missingness_rate: 0.0,
    label_noise_rate: 0.0,
  });
  const [structColumns, setStructColumns] = useState<any[]>([]);
  const [structResult, setStructResult] = useState<any>(null);
  const [structIsStreaming, setStructIsStreaming] = useState(false);
  const [structError, setStructError] = useState("");
  const [structLoading, setStructLoading] = useState(false);
  const [structStep, setStructStep] = useState<"form" | "schema" | "result">("form");
  const [structStreamProgress, setStructStreamProgress] = useState<{
    currentBatch: number;
    totalBatches: number;
    generatedSoFar: number;
    logs: string[];
  }>({
    currentBatch: 0,
    totalBatches: 0,
    generatedSoFar: 0,
    logs: [],
  });

  const connectStructuredStream = async (datasetId: string) => {};

  const startGenerateStructuredDataset = async () => {
    setStructLoading(true);
    setStructIsStreaming(true);
    setStructError("");
    setStructStreamProgress({
      currentBatch: 0,
      totalBatches: 0,
      generatedSoFar: 0,
      logs: ["Initializing connection to tabular synthesizer..."],
    });

    try {
      const token = auth.getToken();
      const userId = auth.getUserId();
      let openaiKey = "";
      let deepseekKey = "";
      let groqKey = "";
      let hasKeys = false;
      if (userId) {
        const encOpenAI = keyStore.get("openai");
        if (encOpenAI) {
          try {
            openaiKey = await decryptApiKey(encOpenAI, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt OpenAI key", e);
          }
        }
        const encDeepseek = keyStore.get("deepseek");
        if (encDeepseek) {
          try {
            deepseekKey = await decryptApiKey(encDeepseek, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt DeepSeek key", e);
          }
        }
        const encGroq = keyStore.get("groq");
        if (encGroq) {
          try {
            groqKey = await decryptApiKey(encGroq, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt Groq key", e);
          }
        }
      }

      const payload = {
        description: structForm.description,
        total_rows: structForm.total_rows,
        dataset_name: structForm.dataset_name || "Tabular Dataset",
        domain_context: structForm.domain_context,
        missingness_rate: structForm.missingness_rate,
        label_noise_rate: structForm.label_noise_rate,
        columns: structColumns.map((c) => ({
          name: c.name,
          type: c.type || "string",
          description: c.description || "",
          range: c.range || null,
          values: c.values || null,
          distribution_prior: c.distribution_prior || null,
        })),
        openai_api_key: openaiKey || undefined,
        deepseek_api_key: deepseekKey || undefined,
        groq_api_key: groqKey || undefined,
      };

      const response = await fetch(`${API_BASE}/structured/generate/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": token ? `Bearer ${token}` : "",
          ...(hasKeys ? { "X-Key-Configured": "true" } : {}),
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Tabular compilation failed" }));
        throw new Error(err.detail || "Tabular compilation failed");
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("Network stream unavailable");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const dataStr = trimmed.slice(6);
            try {
              const event = JSON.parse(dataStr);
              if (event.event === "start") {
                setStructStreamProgress((prev) => ({
                  ...prev,
                  totalBatches: event.batches,
                  logs: [
                    ...prev.logs,
                    `✓ Connection established. Generating ${event.total_requested} rows across ${event.batches} parallel batches.`,
                  ],
                }));
              } else if (event.event === "batch_start") {
                setStructStreamProgress((prev) => ({
                  ...prev,
                  currentBatch: event.batch,
                  logs: [
                    ...prev.logs,
                    `⚡ Dispatching LLM batch completion for Batch ${event.batch}/${prev.totalBatches} (${event.generating} rows)...`,
                  ],
                }));
              } else if (event.event === "batch_done") {
                const reasons: string[] = [];
                if (event.filter_reasons) {
                  const fr = event.filter_reasons;
                  if (fr.missing_cols) reasons.push(`${fr.missing_cols} missing columns`);
                  if (fr.all_null) reasons.push(`${fr.all_null} empty/null`);
                  if (fr.duplicate) reasons.push(`${fr.duplicate} duplicate`);
                }
                const reasonStr = reasons.length > 0 ? ` (Filtered: ${reasons.join(", ")})` : "";
                setStructStreamProgress((prev) => ({
                  ...prev,
                  generatedSoFar: event.total_so_far,
                  logs: [
                    ...prev.logs,
                    `✓ Batch ${event.batch} finished. Generated ${event.generated} rows. (${event.filtered_out} filtered${reasonStr}) [Took ${event.time_taken}s | ETA: ${event.eta}s]`,
                  ],
                }));
              } else if (event.event === "batch_error") {
                setStructStreamProgress((prev) => ({
                  ...prev,
                  logs: [...prev.logs, `⚠ Error in batch ${event.batch}: ${event.error}`],
                }));
              } else if (event.event === "complete") {
                setStructStreamProgress((prev) => ({
                  ...prev,
                  logs: [
                    ...prev.logs,
                    `✓ Generation complete! Running tabular quality audit...`,
                    `✓ Realism Score: ${event.audit?.synthetic_realism_score || 0}%`,
                    `✓ Target leakage check: ${event.audit?.target_leakage_found ? "LEAKAGE DETECTED" : "Clean"}`,
                    `✓ Saved to database.`,
                  ],
                }));
                setStructResult({
                  ...event,
                  quality_score: event.quality_score ?? event.audit?.synthetic_realism_score ?? 0
                });
                setStructStep("result");
                setStructIsStreaming(false);
              } else if (event.event === "error") {
                throw new Error(event.message || "Structured generation failed");
              }
            } catch (err: any) {
              if (err?.message && !err.message.includes("JSON")) throw err;
            }
          }
        }
      }
    } catch (e: any) {
      setStructError(e.message || "Structured dataset generation failed");
      setStructIsStreaming(false);
    } finally {
      setStructLoading(false);
    }
  };

  // ── 4. Scraper Ingest State ──────────────────────────────────────────────
  const [scraperForm, setScraperForm] = useState({
    url: "",
    task_type: "qa",
    style: "formal",
    max_pages: 5,
    pairs_per_chunk: 3,
    dataset_name: "",
    dataset_type: "conversational", // conversational | tabular
    columns: [
      { name: "name", type: "string", description: "Name of the entity" }
    ],
  });
  const [scraperIsStreaming, setScraperIsStreaming] = useState(false);
  const [scraperLogs, setScraperLogs] = useState<string[]>([]);
  const [scraperResult, setScraperResult] = useState<any>(null);
  const [scraperError, setScraperError] = useState("");
  const [scraperLoading, setScraperLoading] = useState(false);
  const [scraperProgress, setScraperProgress] = useState({ pages: 0, chunks: 0, pairs: 0 });

  const connectScraperStream = async (datasetId: string) => {};

  const startScraperIngest = async () => {
    setScraperLoading(true);
    setScraperIsStreaming(true);
    setScraperError("");
    setScraperResult(null);
    setScraperLogs([`🌐 Connecting to ${scraperForm.url}...`]);
    setScraperProgress({ pages: 0, chunks: 0, pairs: 0 });

    try {
      const token = auth.getToken();
      const userId = auth.getUserId();
      let openaiKey = "";
      let deepseekKey = "";
      let groqKey = "";
      let hasKeys = false;
      if (userId) {
        const encOpenAI = keyStore.get("openai");
        if (encOpenAI) {
          try {
            openaiKey = await decryptApiKey(encOpenAI, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt OpenAI key", e);
          }
        }
        const encDeepseek = keyStore.get("deepseek");
        if (encDeepseek) {
          try {
            deepseekKey = await decryptApiKey(encDeepseek, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt DeepSeek key", e);
          }
        }
        const encGroq = keyStore.get("groq");
        if (encGroq) {
          try {
            groqKey = await decryptApiKey(encGroq, userId);
            hasKeys = true;
          } catch (e) {
            console.error("Failed to decrypt Groq key", e);
          }
        }
      }

      const isTabular = scraperForm.dataset_type === "tabular";
      const endpoint = isTabular ? "ingest/scrape/tabular" : "ingest/scrape";
      
      const payload = isTabular 
        ? {
            url: scraperForm.url,
            max_pages: scraperForm.max_pages,
            dataset_name: scraperForm.dataset_name || undefined,
            columns: scraperForm.columns.map((c: any) => ({
              name: c.name,
              type: c.type || "string",
              description: c.description || "",
            })),
            openai_api_key: openaiKey || undefined,
            deepseek_api_key: deepseekKey || undefined,
            groq_api_key: groqKey || undefined,
          }
        : {
            url: scraperForm.url,
            task_type: scraperForm.task_type,
            style: scraperForm.style,
            max_pages: scraperForm.max_pages,
            pairs_per_chunk: scraperForm.pairs_per_chunk,
            dataset_name: scraperForm.dataset_name || undefined,
          };

      const response = await fetch(`${API_BASE}/${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": token ? `Bearer ${token}` : "",
          ...(hasKeys ? { "X-Key-Configured": "true" } : {}),
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Scrape request failed" }));
        throw new Error(err.detail || "Scrape request failed");
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("Stream unavailable");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(trimmed.slice(6));
            if (event.event === "page_scraped") {
              setScraperProgress((prev) => ({
                ...prev,
                pages: event.pages_done,
                chunks: event.total_chunks,
              }));
              setScraperLogs((prev) => [
                ...prev,
                `✓ Scraped ${event.url.slice(0, 60)} → ${event.chunks} chunks`,
              ]);
            } else if (event.event === "scrape_complete") {
              setScraperLogs((prev) => [
                ...prev,
                `✓ Scraping complete. ${event.total_chunks} text chunks extracted from ${event.pages_scraped} pages. Converting to ${isTabular ? "rows" : "pairs"}...`,
              ]);
            } else if (event.event === "restructuring") {
              setScraperLogs((prev) => [...prev, `⚡ ${event.message}`]);
            } else if (event.event === "restructure_progress") {
              const elapsed = event.elapsed || 1;
              const completed = event.completed_chunks || 1;
              const total = event.total_chunks || 1;
              const speed = completed / elapsed;
              const remaining = total - completed;
              const etaSeconds = Math.max(0, Math.ceil(remaining / speed));
              const etaString = etaSeconds > 0 ? `${etaSeconds}s remaining` : "wrapping up...";

              setScraperProgress((prev) => ({ ...prev, pairs: event.total_pairs }));
              setScraperLogs((prev) => [
                ...prev,
                `⚡ [Chunk ${completed}/${total}] Generated ${event.pairs_generated} pairs (Total: ${event.total_pairs}). ETA: ${etaString}`,
                `   └─ reasoning: "${event.reasoning}"`,
              ]);
            } else if (event.event === "extract_progress") {
              const elapsed = event.elapsed || 1;
              const completed = event.completed_chunks || 1;
              const total = event.total_chunks || 1;
              const speed = completed / elapsed;
              const remaining = total - completed;
              const etaSeconds = Math.max(0, Math.ceil(remaining / speed));
              const etaString = etaSeconds > 0 ? `${etaSeconds}s remaining` : "wrapping up...";

              setScraperProgress((prev) => ({ ...prev, pairs: event.total_rows }));
              setScraperLogs((prev) => [
                ...prev,
                `⚡ [Chunk ${completed}/${total}] Extracted ${event.rows_extracted} rows (Total: ${event.total_rows}). ETA: ${etaString}`,
              ]);
            } else if (event.event === "auditing") {
              setScraperLogs((prev) => [...prev, `⚡ ${event.message}`]);
            } else if (event.event === "complete") {
              setScraperProgress((prev) => ({ ...prev, pairs: event.row_count }));
              setScraperLogs((prev) => [...prev, `✓ Done! ${event.row_count} rows saved to dataset.`]);
              setScraperResult(event);
              setScraperIsStreaming(false);
            } else if (event.event === "error") {
              throw new Error(event.message);
            }
          } catch (parseErr: any) {
            if (parseErr?.message && !parseErr.message.includes("JSON")) throw parseErr;
          }
        }
      }
    } catch (e: any) {
      setScraperError(e.message || "Scrape failed");
      setScraperIsStreaming(false);
    } finally {
      setScraperLoading(false);
    }
  };

  return (
    <PersistentStateContext.Provider
      value={{
        colabStep,
        setColabStep,
        colabUrl,
        setColabUrl,
        colabErrMsg,
        setColabErrMsg,
        colabLoading,
        setColabLoading,
        colabJobId,
        setColabJobId,
        launcherConfig,
        setLauncherConfig,
        handleColabLaunch,

        genForm,
        setGenForm,
        genPreview,
        setGenPreview,
        genResult,
        setGenResult,
        genStep,
        setGenStep,
        genError,
        setGenError,
        genLoading,
        setGenLoading,
        genPreviewing,
        setGenPreviewing,
        genIsStreaming,
        genStreamProgress,
        startGenerateTextDataset,
        connectTextStream,

        structForm,
        setStructForm,
        structColumns,
        setStructColumns,
        structResult,
        setStructResult,
        structIsStreaming,
        structStreamProgress,
        structError,
        setStructError,
        structLoading,
        setStructLoading,
        structStep,
        setStructStep,
        startGenerateStructuredDataset,
        connectStructuredStream,

        scraperForm,
        setScraperForm,
        scraperIsStreaming,
        scraperLogs,
        scraperResult,
        scraperError,
        setScraperError,
        scraperLoading,
        setScraperLoading,
        scraperProgress,
        startScraperIngest,
        connectScraperStream,
      }}
    >
      {children}
    </PersistentStateContext.Provider>
  );
}

export function usePersistentState() {
  const context = useContext(PersistentStateContext);
  if (!context) {
    throw new Error("usePersistentState must be used within a PersistentStateProvider");
  }
  return context;
}

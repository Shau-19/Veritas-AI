"use client";
import { useState, useEffect, useCallback, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, auth, keyStore, decryptApiKey } from "@/lib/api";
import Script from "next/script";

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-zinc-800 text-zinc-400 border-zinc-700",
  running: "bg-cyan-500 text-black border-black animate-pulse",
  completed: "bg-emerald-500 text-black border-black",
  failed: "bg-red-500 text-white border-black",
};

interface MetricCardProps {
  label: string;
  value: number | string;
  sub: string;
  colorClass: string;
}

function MetricCard({ label, value, sub, colorClass }: MetricCardProps) {
  return (
    <div className="bg-[#12131c] border border-zinc-800 rounded-xl p-4 shadow-[2px_2px_0px_0px_#000]">
      <p className="text-[10px] text-zinc-500 font-mono font-bold uppercase tracking-wider">{label}</p>
      <div className="flex items-baseline gap-2 mt-1">
        <p className={`text-2xl font-display font-black ${colorClass}`}>{value}</p>
      </div>
      <p className="text-[10px] text-zinc-650 font-mono mt-1 font-semibold">{sub}</p>
    </div>
  );
}

interface NewEvalModalProps {
  jobs: any[];
  datasets: any[];
  onClose: () => void;
  onCreated: () => void;
}

function NewEvaluationModal({ jobs, datasets, onClose, onCreated }: NewEvalModalProps) {
  const [trainingJobId, setTrainingJobId] = useState(jobs[0]?.id || "");
  const [datasetId, setDatasetId] = useState(datasets[0]?.id || "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!trainingJobId || !datasetId) {
      setError("Please select both a training job and a test dataset.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await api.createEvaluation({ training_job_id: trainingJobId, dataset_id: datasetId });
      onCreated();
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to start evaluation job");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/85 backdrop-blur-md z-50 flex items-center justify-center px-4">
      <div className="bg-[#10111a] w-full max-w-md border-2 border-zinc-800 rounded-xl overflow-hidden shadow-[0px_0px_30px_0px_rgba(34,211,238,0.1)]">
        {/* Header */}
        <div className="bg-[#141622] px-4 py-3 border-b border-zinc-800 flex items-center justify-between font-mono font-bold text-white text-sm">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
              <div className="w-2.5 h-2.5 rounded-full bg-amber-500" />
              <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
            </div>
            <span className="ml-1 text-zinc-300">Run Model Evaluation</span>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-white text-xl font-bold font-body">&times;</button>
        </div>

        {/* Warning Icon/Info */}
        <div className="bg-cyan-500/10 p-4 border-b border-zinc-800 flex items-center gap-4">
          <div className="w-10 h-10 bg-cyan-500 rounded-lg flex items-center justify-center font-bold text-lg text-black shadow-[0_0_10px_0_rgba(6,182,212,0.2)]">
            📊
          </div>
          <div>
            <p className="text-xs text-white font-bold">Side-by-Side Model Comparison</p>
            <p className="text-[10px] text-zinc-400 font-mono">This compares your fine-tuned model checkpoint vs the base LLM on test dataset prompts.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="text-xs text-zinc-300 font-bold block mb-1.5">Fine-Tuned Model Checkpoint</label>
            <select
              value={trainingJobId}
              onChange={(e) => setTrainingJobId(e.target.value)}
              className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-sm text-white font-body focus:border-cyan-500 focus:outline-none appearance-none"
            >
              <option value="" disabled>Select a completed job</option>
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.base_model?.split("/").pop()} (Job ID: {j.id.slice(0, 8)}...)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs text-zinc-300 font-bold block mb-1.5">Evaluation / Test Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-sm text-white font-body focus:border-cyan-500 focus:outline-none appearance-none"
            >
              <option value="" disabled>Select a dataset</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.row_count} rows)
                </option>
              ))}
            </select>
          </div>

          {error && (
            <p className="text-xs text-red-400 bg-red-950/20 border border-red-900/30 rounded-lg px-3 py-2 font-mono font-bold">
              {error}
            </p>
          )}

          <div className="flex gap-4 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="w-1/2 py-2.5 text-sm font-bold border border-zinc-800 hover:bg-zinc-900 text-zinc-400 hover:text-white rounded-lg transition-all"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !trainingJobId || !datasetId}
              className="w-1/2 py-2.5 text-sm font-bold bg-cyan-500 hover:bg-cyan-400 text-black rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed font-display shadow-[0_0_15px_0_rgba(6,182,212,0.2)]"
            >
              {loading ? "☕ Starting eval..." : "Run Evaluation"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

const SUGGESTED_PROMPTS: Record<string, string[]> = {
  chat: [
    "I'm experiencing a runny nose and itchy eyes. Could this be allergies?",
    "What could cause a sudden sore throat without any other symptoms?",
    "I have a mild fever of 100.5°F and body aches. Is this a mild flu?",
    "Can seasonal changes trigger a cough without any fever?",
    "Why would I get a cold and then develop throat pain a week later?"
  ],
  summarization: [
    "Summarize this text in 2 sentences: 'The Genome Aggregation Database (gnomAD) is a coalition of geneticists seeking to aggregate and harmonize exome and genome sequencing data from a wide variety of large-scale sequencing projects.'",
    "Summarize the main events of World War I in three bullet points."
  ],
  classification: [
    "Classify the sentiment of this review: 'The new design is acceptable but the performance degrades significantly under load.'",
    "Categorize this support ticket: 'My billing cycle started yesterday but I haven't received my receipt yet.'"
  ],
  default: [
    "I'm experiencing a runny nose and itchy eyes. Could this be allergies?",
    "What could cause a sudden sore throat without any other symptoms?",
    "I have a mild fever of 100.5°F and body aches. Is this a mild flu?",
    "Can seasonal changes trigger a cough without any fever?",
    "Why would I get a cold and then develop throat pain a week later?"
  ]
};

function estimateTokensPerSec(text: string, latencyMs: number) {
  if (!text || latencyMs <= 0) return 0;
  const words = text.trim().split(/\s+/).length;
  const estimatedTokens = Math.max(1, Math.round(words * 1.33));
  const seconds = latencyMs / 1000;
  return Math.round(estimatedTokens / seconds);
}

function EvaluatePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const depIdParam = searchParams.get("dep_id");

  const [activeTab, setActiveTab] = useState<"playground" | "batch">("playground");
  const [evaluations, setEvaluations] = useState<any[]>([]);
  const [completedJobs, setCompletedJobs] = useState<any[]>([]);
  const [readyDatasets, setReadyDatasets] = useState<any[]>([]);
  const [selectedEvalId, setSelectedEvalId] = useState<string | null>(null);
  const [selectedEval, setSelectedEval] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);

  // SageMaker / Inference mode states
  const [deployments, setDeployments] = useState<any[]>([]);
  const [selectedDeploymentId, setSelectedDeploymentId] = useState<string>("");
  const [inferenceMode, setInferenceMode] = useState<"simulated" | "sagemaker">("simulated");

  // Playground state
  const [selectedPlaygroundJobId, setSelectedPlaygroundJobId] = useState<string>("");
  const [playgroundPrompt, setPlaygroundPrompt] = useState<string>("");
  const [playgroundLoading, setPlaygroundLoading] = useState<boolean>(false);
  const [playgroundResult, setPlaygroundResult] = useState<{
    base_response: string;
    fine_tuned_response: string;
    base_latency_ms: number;
    ft_latency_ms: number;
  } | null>(null);
  const [playgroundError, setPlaygroundError] = useState<string>("");

  // Filters for Comparison Playground (Batch Tab)
  const [filterType, setFilterType] = useState<"all" | "poor" | "slow">("all");
  const [searchQuery, setSearchQuery] = useState("");

  const fetchActiveDeployments = useCallback(async () => {
    try {
      const token = auth.getToken();
      const res = await fetch("http://localhost:8000/api/v1/deploy/", {
        headers: {
          "Authorization": token ? `Bearer ${token}` : "",
          "Content-Type": "application/json",
        },
      });
      if (res.ok) {
        const data = await res.json();
        const active = data.filter((d: any) => d.status === "active");
        setDeployments(active);
        if (active.length > 0) {
          setSelectedDeploymentId((prev) => prev || active[0].id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch active deployments", err);
    }
  }, []);

  useEffect(() => {
    fetchActiveDeployments();
  }, [fetchActiveDeployments]);

  useEffect(() => {
    if (depIdParam) {
      setInferenceMode("sagemaker");
      setSelectedDeploymentId(depIdParam);
    }
  }, [depIdParam]);

  const fetchData = useCallback(async () => {
    try {
      const [evals, jobs, datasets] = await Promise.all([
        api.listEvaluations(),
        api.listJobs(),
        api.listDatasets(),
      ]);
      setEvaluations(evals);

      // Filter out non-supported models (gated models like Llama 3.2, Gemma 2B)
      // Keep Microsoft Phi-3, Mistral 7B, Qwen 2.5, and unsloth un-gated models
      const finishedJobs = jobs.filter((j) => {
        if (j.status !== "completed") return false;
        const modelLower = (j.base_model || "").toLowerCase();
        return (
          modelLower.includes("phi-3") ||
          modelLower.includes("mistral") ||
          modelLower.includes("qwen") ||
          modelLower.includes("unsloth")
        );
      });
      setCompletedJobs(finishedJobs);

      const readyD = datasets.filter((d) => d.status === "ready");
      setReadyDatasets(readyD);

      if (evals.length > 0 && !selectedEvalId) {
        setSelectedEvalId(evals[0].id);
      }
    } catch (err) {
      console.error("Error fetching evaluations data:", err);
    } finally {
      setLoading(false);
    }
  }, [selectedEvalId]);

  useEffect(() => {
    if (!auth.isLoggedIn()) {
      router.push("/auth");
      return;
    }
    fetchData();
  }, [fetchData, router]);

  // Set default playground job
  useEffect(() => {
    if (completedJobs.length > 0 && !selectedPlaygroundJobId) {
      setSelectedPlaygroundJobId(completedJobs[0].id);
    }
  }, [completedJobs, selectedPlaygroundJobId]);

  // Poll active evaluations
  useEffect(() => {
    const activeStates = ["queued", "running"];
    const hasActive = evaluations.some((e) => activeStates.includes(e.status));

    if (hasActive) {
      const interval = setInterval(() => {
        pollData();
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [evaluations]);

  // Fetch full details when selected evaluation changes or updates
  useEffect(() => {
    if (selectedEvalId) {
      fetchDetail(selectedEvalId);
    } else {
      setSelectedEval(null);
    }
  }, [selectedEvalId]);

  const pollData = async () => {
    try {
      const evals = await api.listEvaluations();
      setEvaluations(evals);

      if (selectedEvalId) {
        const matching = evals.find((e) => e.id === selectedEvalId);
        if (matching && (matching.status === "running" || selectedEval?.status !== matching.status)) {
          const det = await api.getEvaluation(selectedEvalId);
          setSelectedEval(det);
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  };

  const fetchDetail = async (id: string) => {
    setDetailLoading(true);
    try {
      const det = await api.getEvaluation(id);
      setSelectedEval(det);
    } catch (err) {
      console.error("Error fetching evaluation details:", err);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleCreated = async () => {
    const evals = await api.listEvaluations();
    setEvaluations(evals);
    if (evals.length > 0) {
      setSelectedEvalId(evals[0].id);
    }
  };

  const handlePlaygroundRun = async () => {
    if (inferenceMode === "simulated" && !selectedPlaygroundJobId) return;
    if (inferenceMode === "sagemaker" && !selectedDeploymentId) return;
    if (!playgroundPrompt.trim()) return;

    setPlaygroundLoading(true);
    setPlaygroundError("");
    setPlaygroundResult(null);
    try {
      const userId = typeof window !== "undefined" ? sessionStorage.getItem("user_id") || "" : "";
      let hfToken = "";
      const encryptedHf = keyStore.get("huggingface");
      if (encryptedHf) {
        try {
          hfToken = await decryptApiKey(encryptedHf, userId);
        } catch (e) {
          console.error("Failed to decrypt Hugging Face token", e);
        }
      }

      if (inferenceMode === "simulated") {
        const res = await api.generatePlaygroundResponse(selectedPlaygroundJobId, playgroundPrompt, hfToken || undefined);
        setPlaygroundResult(res);
      } else {
        // AWS SageMaker mode
        // 1. Get base response via mock generator fallback using first completed job if available
        let baseResponse = "Raw base LLM output placeholder";
        let baseLatency = 120;
        try {
          const fallbackJobId = selectedPlaygroundJobId || (completedJobs.length > 0 ? completedJobs[0].id : null);
          if (fallbackJobId) {
            const baseRes = await api.generatePlaygroundResponse(fallbackJobId, playgroundPrompt, hfToken || undefined);
            baseResponse = baseRes.base_response;
            baseLatency = baseRes.base_latency_ms;
          }
        } catch (e) {
          console.warn("Could not get mock base response", e);
        }

        // 2. Fetch live SageMaker response
        let accessKey = "";
        let secretKey = "";
        const encryptedAws = keyStore.get("aws");
        if (encryptedAws) {
          const rawKeys = await decryptApiKey(encryptedAws, userId);
          const parts = rawKeys.trim().split(/\s+/);
          if (parts.length >= 2) {
            accessKey = parts[0];
            secretKey = parts[1];
          }
        }

        const token = auth.getToken();
        const start = performance.now();
        const res = await fetch(`http://localhost:8000/api/v1/deploy/${selectedDeploymentId}/invoke`, {
          method: "POST",
          headers: {
            "Authorization": token ? `Bearer ${token}` : "",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: playgroundPrompt,
            aws_access_key: accessKey || undefined,
            aws_secret_key: secretKey || undefined,
          }),
        });
        const end = performance.now();
        const ftLatency = Math.round(end - start);

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "SageMaker invocation failed" }));
          throw new Error(err.detail || "SageMaker invocation failed");
        }
        const data = await res.json();
        
        setPlaygroundResult({
          base_response: baseResponse,
          fine_tuned_response: data.response,
          base_latency_ms: baseLatency,
          ft_latency_ms: ftLatency,
        });
      }
    } catch (err: any) {
      setPlaygroundError(err.message || "Failed to generate playground response");
    } finally {
      setPlaygroundLoading(false);
    }
  };

  const selectedJob = completedJobs.find((j) => j.id === selectedPlaygroundJobId);

  // Filter comparison data (Batch Tab)
  const comparisonList = selectedEval?.comparison_data || [];
  const filteredComparisons = comparisonList.filter((row: any) => {
    if (searchQuery) {
      const textToSearch = `${row.input} ${row.target} ${row.base_response} ${row.fine_tuned_response}`.toLowerCase();
      if (!textToSearch.includes(searchQuery.toLowerCase())) return false;
    }
    if (filterType === "poor") {
      const similarity = row.metrics?.semantic_similarity ?? 1;
      if (similarity >= 0.75) return false;
    } else if (filterType === "slow") {
      const latency = row.ft_latency_ms ?? 0;
      if (latency <= 500) return false;
    }
    return true;
  });

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Page Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 border-b-2 border-black pb-6 gap-4">
        <div>
          <h1 className="font-display font-black text-3xl text-text uppercase tracking-tight">Model Evaluation Studio</h1>
          <p className="text-sm text-subtle font-body mt-1">
            Compare model generation accuracy, semantic consistency, and latency side-by-side
          </p>
        </div>
        {activeTab === "batch" && (
          <button
            onClick={() => setShowModal(true)}
            disabled={completedJobs.length === 0 || readyDatasets.length === 0}
            className="px-5 py-3 rounded-lg text-xs font-display font-bold bg-cyan-500 hover:bg-cyan-400 text-black border-2 border-black shadow-[4px_4px_0px_0px_#000] hover:shadow-[2px_2px_0px_0px_#000] transition-all disabled:opacity-50 disabled:cursor-not-allowed active:translate-x-0.5 active:translate-y-0.5"
          >
            + Launch Evaluation
          </button>
        )}
      </div>

      {/* Tabs Selector */}
      <div className="flex gap-6 border-b border-zinc-800 pb-px mb-8 font-mono">
        <button
          onClick={() => setActiveTab("playground")}
          className={`pb-4 px-2 text-sm font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-2 ${
            activeTab === "playground"
              ? "border-cyan-400 text-white"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <span>🧪</span> Playground Studio
        </button>
        <button
          onClick={() => setActiveTab("batch")}
          className={`pb-4 px-2 text-sm font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-2 ${
            activeTab === "batch"
              ? "border-cyan-400 text-white"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <span>📊</span> Batch Evaluations
        </button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="space-y-3 lg:col-span-1">
            {Array(3).fill(0).map((_, i) => (
              <div key={i} className="shimmer h-20 rounded-xl" />
            ))}
          </div>
          <div className="lg:col-span-2 space-y-6">
            <div className="grid grid-cols-3 gap-4">
              {Array(3).fill(0).map((_, i) => (
                <div key={i} className="shimmer h-24 rounded-xl" />
              ))}
            </div>
            <div className="shimmer h-80 rounded-xl" />
          </div>
        </div>
      ) : activeTab === "playground" ? (
        // -------------------------------------------------------------
        // TAB A: INTERACTIVE PLAYGROUND STUDIO
        // -------------------------------------------------------------
        (completedJobs.length === 0 && deployments.length === 0) ? (
          <div className="glass rounded-2xl p-16 border border-zinc-800 text-center bg-[#111219]/30">
            <p className="text-5xl mb-4">🧪</p>
            <p className="font-display font-bold text-xl text-text mb-2">No models available to test</p>
            <p className="text-sm text-subtle font-body mb-6">
              You need to finish training a model checkpoint or deploy an AWS SageMaker endpoint to access the Interactive Playground.
            </p>
            <div className="flex gap-4 justify-center">
              <a href="/jobs" className="px-6 py-2.5 rounded-lg text-sm font-display font-bold bg-[#1e202f] hover:bg-zinc-800 border border-zinc-700 text-white transition-all text-center">
                Train a Model
              </a>
              <a href="/deploy" className="px-6 py-2.5 rounded-lg text-sm font-display font-bold bg-cyan-500 hover:bg-cyan-400 text-black border-2 border-black shadow-[3px_3px_0px_0px_#000] transition-all text-center">
                Deploy to SageMaker
              </a>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
            {/* Left Control Panel */}
            <div className="lg:col-span-1 space-y-6">
              <div className="bg-[#12131c] border border-zinc-800 rounded-xl p-5 shadow-[4px_4px_0px_0px_#000] space-y-4">
                <h3 className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest">
                  Model Playground Config
                </h3>

                {/* Selector for Inference Mode */}
                <div className="flex gap-2 p-1 bg-[#0f0f12] rounded-lg border-2 border-black mb-4">
                  <button
                    onClick={() => {
                      setInferenceMode("simulated");
                      setPlaygroundResult(null);
                    }}
                    disabled={completedJobs.length === 0}
                    className={`flex-1 py-1.5 rounded text-[11px] font-mono font-bold transition-all disabled:opacity-40 ${
                      inferenceMode === "simulated" ? "bg-cyan-500 text-black" : "text-subtle hover:text-text"
                    }`}
                  >
                    Simulated
                  </button>
                  <button
                    onClick={() => {
                      setInferenceMode("sagemaker");
                      setPlaygroundResult(null);
                      fetchActiveDeployments();
                    }}
                    className={`flex-1 py-1.5 rounded text-[11px] font-mono font-bold transition-all ${
                      inferenceMode === "sagemaker" ? "bg-cyan-500 text-black" : "text-subtle hover:text-text"
                    }`}
                  >
                    Live SageMaker
                  </button>
                </div>

                {inferenceMode === "simulated" ? (
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono font-bold block mb-1.5 uppercase">
                      Select Fine-Tuned Model
                    </label>
                    <select
                      value={selectedPlaygroundJobId}
                      onChange={(e) => {
                        setSelectedPlaygroundJobId(e.target.value);
                        setPlaygroundResult(null);
                      }}
                      className="w-full px-3 py-2.5 bg-zinc-950 border border-zinc-800 rounded-lg text-xs text-white font-mono focus:border-cyan-500 focus:outline-none appearance-none"
                    >
                      {completedJobs.map((j) => (
                        <option key={j.id} value={j.id}>
                          {j.base_model?.split("/").pop()} ({j.id.slice(0, 8)})
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono font-bold block mb-1.5 uppercase">
                      Select SageMaker Endpoint
                    </label>
                    {deployments.length === 0 ? (
                      <div className="text-xs text-amber-500 font-mono py-2.5 bg-amber-950/20 px-3 border border-amber-900/40 rounded-lg">
                        No active deployments found.{" "}
                        <a href="/deploy" className="underline text-cyan-400 hover:text-cyan-300 font-bold">
                          Deploy a model first ↗
                        </a>
                      </div>
                    ) : (
                      <select
                        value={selectedDeploymentId}
                        onChange={(e) => {
                          setSelectedDeploymentId(e.target.value);
                          setPlaygroundResult(null);
                        }}
                        className="w-full px-3 py-2.5 bg-zinc-950 border border-zinc-800 rounded-lg text-xs text-white font-mono focus:border-cyan-500 focus:outline-none appearance-none"
                      >
                        {deployments.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.model_id.split("/").pop()} ({d.endpoint_name.slice(0, 15)}...)
                          </option>
                        ))}
                      </select>
                    )}
                  </div>
                )}

                {inferenceMode === "simulated" && selectedJob && (
                  <div className="text-[10px] font-mono text-subtle space-y-1 bg-zinc-950/40 p-3 rounded-lg border border-zinc-900">
                    <div>
                      <span className="text-zinc-550">Base Model:</span> {selectedJob.base_model}
                    </div>
                    <div>
                      <span className="text-zinc-550">Task Type:</span> {selectedJob.task_type}
                    </div>
                    {selectedJob.hf_model_id && (
                      <div className="truncate">
                        <span className="text-zinc-550">HuggingFace:</span>{" "}
                        <a
                          href={`https://huggingface.co/${selectedJob.hf_model_id}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-cyan-400 hover:underline"
                        >
                          {selectedJob.hf_model_id}
                        </a>
                      </div>
                    )}
                  </div>
                )}

                {inferenceMode === "sagemaker" && selectedDeploymentId && (
                  (() => {
                    const dep = deployments.find((d) => d.id === selectedDeploymentId);
                    if (!dep) return null;
                    return (
                      <div className="text-[10px] font-mono text-subtle space-y-1 bg-zinc-950/40 p-3 rounded-lg border border-zinc-900 animate-fadeIn">
                        <div className="truncate" title={dep.model_id}>
                          <span className="text-zinc-550">Model ID:</span> {dep.model_id}
                        </div>
                        <div className="truncate" title={dep.endpoint_name}>
                          <span className="text-zinc-550">Endpoint:</span> {dep.endpoint_name}
                        </div>
                        <div>
                          <span className="text-zinc-550">Instance:</span> {dep.instance_type}
                        </div>
                        <div>
                          <span className="text-zinc-550">Region:</span> {dep.region}
                        </div>
                      </div>
                    );
                  })()
                )}


                {/* Main Prompt Input */}
                <div className="space-y-2">
                  <label className="text-[10px] text-zinc-400 font-mono font-bold block uppercase">
                    Prompt Query
                  </label>
                  <textarea
                    rows={4}
                    value={playgroundPrompt}
                    onChange={(e) => setPlaygroundPrompt(e.target.value)}
                    placeholder="Type in a custom prompt or click a suggestion above..."
                    className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-cyan-500 font-mono resize-none leading-relaxed"
                  />
                </div>

                {/* Submit button */}
                <button
                  onClick={handlePlaygroundRun}
                  disabled={playgroundLoading || !playgroundPrompt.trim()}
                  className="w-full py-3 rounded-lg text-xs font-display font-bold bg-cyan-500 hover:bg-cyan-400 text-black border-2 border-black shadow-[3px_3px_0px_0px_#000] hover:shadow-[1px_1px_0px_0px_#000] active:translate-x-0.5 active:translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none disabled:translate-y-0 disabled:translate-x-0"
                >
                  {playgroundLoading ? "Running Inference..." : "Test Model Checkpoint ⚡"}
                </button>

                {playgroundError && (
                  <p className="text-[10px] font-mono font-bold text-red-400 bg-red-950/20 border border-red-900/30 rounded p-2">
                    {playgroundError}
                  </p>
                )}
              </div>
            </div>

            {/* Right Output Panel */}
            <div className="lg:col-span-2 space-y-6">
              {!playgroundResult && !playgroundLoading ? (
                <div className="bg-[#12131c]/60 border border-zinc-800 rounded-xl p-16 text-center shadow-[4px_4px_0px_0px_#000] flex flex-col items-center justify-center min-h-[400px]">
                  <span className="text-5xl mb-4">🧪</span>
                  <h3 className="font-display font-black text-white text-lg uppercase tracking-tight">
                    Interactive Playground Studio
                  </h3>
                  <p className="text-xs text-subtle font-body max-w-md mt-2 leading-relaxed">
                    Compare responses side-by-side. Our framework tests both the unoptimized raw Base LLM and your custom Fine-Tuned checkpoint in real-time.
                  </p>
                  <p className="text-[10px] text-zinc-550 font-mono mt-4 uppercase font-semibold">
                    Select a checkpoint, select or type a prompt, and click Run.
                  </p>
                </div>
              ) : playgroundLoading ? (
                <div className="bg-[#12131c] border border-zinc-800 rounded-xl p-8 shadow-[4px_4px_0px_0px_#000] min-h-[400px] flex flex-col gap-6 justify-center items-center">
                  <div className="space-y-2 text-center">
                    <div style={{ width: 140, height: 140, margin: "0 auto" }}>
                      {/* @ts-ignore */}
                      <lottie-player
                        src="/cooking.json"
                        background="transparent"
                        speed="1.2"
                        style={{ width: "100%", height: "100%" }}
                        loop
                        autoplay
                      />
                    </div>
                    <p className="text-sm text-text font-display font-bold uppercase tracking-wider animate-pulse">
                      Running Inference Comparison...
                    </p>
                    <p className="text-xs text-subtle font-mono">
                      Querying model checkpoints & calculating latency differences
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="shimmer h-32 rounded-xl" />
                    <div className="shimmer h-32 rounded-xl" />
                  </div>
                </div>
              ) : (
                playgroundResult && (
                  <div className="space-y-6 animate-fadeIn">
                    {/* Latency Comparison Card */}
                    <div className="bg-[#12131c] border border-zinc-800 rounded-xl p-5 shadow-[4px_4px_0px_0px_#000]">
                      <div className="flex items-center justify-between mb-4 pb-2 border-b border-zinc-900">
                        <h3 className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest">
                          Playground Latency & Performance Details
                        </h3>
                        {playgroundResult.base_latency_ms > playgroundResult.ft_latency_ms && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-950 border border-emerald-900 text-emerald-400 font-bold uppercase tracking-wide">
                            ⚡ {(playgroundResult.base_latency_ms / playgroundResult.ft_latency_ms).toFixed(1)}x Speedup
                          </span>
                        )}
                      </div>

                      <div className="space-y-4 font-mono text-xs">
                        {/* Base model speed */}
                        <div>
                          <div className="flex justify-between text-subtle mb-1.5">
                            <span className="text-zinc-400 font-bold">Base Model Latency</span>
                            <span>
                              {playgroundResult.base_latency_ms} ms (
                              {estimateTokensPerSec(playgroundResult.base_response, playgroundResult.base_latency_ms)} tok/sec)
                            </span>
                          </div>
                          <div className="h-2.5 bg-zinc-950 border border-zinc-850 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-zinc-650"
                              style={{
                                width: `${Math.min(
                                  100,
                                  (playgroundResult.base_latency_ms /
                                    Math.max(playgroundResult.base_latency_ms, playgroundResult.ft_latency_ms)) *
                                    100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>

                        {/* Fine-Tuned speed */}
                        <div>
                          <div className="flex justify-between text-subtle mb-1.5">
                            <span className="text-cyan-400 font-bold">Fine-Tuned Model (Optimized)</span>
                            <span className="text-cyan-400 font-bold">
                              {playgroundResult.ft_latency_ms} ms (
                              {estimateTokensPerSec(playgroundResult.fine_tuned_response, playgroundResult.ft_latency_ms)} tok/sec)
                            </span>
                          </div>
                          <div className="h-2.5 bg-zinc-950 border border-zinc-850 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-cyan-400"
                              style={{
                                width: `${Math.min(
                                  100,
                                  (playgroundResult.ft_latency_ms /
                                    Math.max(playgroundResult.base_latency_ms, playgroundResult.ft_latency_ms)) *
                                    100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Side-by-Side Response Box */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                      {/* Base Model Response Panel */}
                      <div className="bg-[#12131c] border border-zinc-850 rounded-xl p-5 flex flex-col justify-between shadow-[2px_2px_0px_0px_#000]">
                        <div>
                          <div className="flex justify-between items-center mb-3 pb-2 border-b border-zinc-900">
                            <h4 className="text-xs font-mono font-bold text-zinc-400 uppercase tracking-wide">
                              Raw Base LLM Output
                            </h4>
                            <span className="text-[9px] font-mono bg-zinc-950 border border-zinc-900 text-zinc-500 px-2 py-0.5 rounded">
                              {playgroundResult.base_latency_ms}ms
                            </span>
                          </div>
                          <p className="text-xs font-mono text-subtle leading-relaxed whitespace-pre-wrap">
                            {playgroundResult.base_response}
                          </p>
                        </div>
                        <div className="text-[9px] font-mono text-zinc-650 mt-6 border-t border-zinc-950 pt-2 uppercase font-semibold">
                          Verbose & unaligned structure
                        </div>
                      </div>

                      {/* Fine-Tuned Model Response Panel */}
                      <div className="bg-[#131722] border border-cyan-950/70 rounded-xl p-5 flex flex-col justify-between shadow-[2px_2px_0px_0px_#000]">
                        <div>
                          <div className="flex justify-between items-center mb-3 pb-2 border-b border-cyan-950/40">
                            <h4 className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wide">
                              Fine-Tuned Checkpoint Output
                            </h4>
                            <span className="text-[9px] font-mono bg-cyan-950/40 border border-cyan-900/30 text-cyan-400 px-2 py-0.5 rounded">
                              {playgroundResult.ft_latency_ms}ms
                            </span>
                          </div>
                          <p className="text-xs font-mono text-text leading-relaxed whitespace-pre-wrap">
                            {playgroundResult.fine_tuned_response}
                          </p>
                        </div>
                        <div className="text-[9px] font-mono text-cyan-500/70 mt-6 border-t border-cyan-950/20 pt-2 uppercase font-semibold">
                          Optimized context & response format
                        </div>
                      </div>
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )
      ) : (
        // -------------------------------------------------------------
        // TAB B: HISTORICAL BATCH EVALUATIONS (RETAINED FROM BACKUP)
        // -------------------------------------------------------------
        evaluations.length === 0 ? (
          <div className="glass rounded-2xl p-16 border border-zinc-800 text-center bg-[#111219]/30">
            <p className="text-5xl mb-4">📊</p>
            <p className="font-display font-bold text-xl text-text mb-2">No evaluations completed yet</p>
            <p className="text-sm text-subtle font-body mb-6">
              Launch a batch evaluation to benchmark your fine-tuned model checkpoint against the base LLM on a full test dataset.
            </p>
            <button
              onClick={() => setShowModal(true)}
              disabled={completedJobs.length === 0 || readyDatasets.length === 0}
              className="px-6 py-2.5 rounded-lg text-sm font-display font-bold bg-[#1e202f] hover:bg-zinc-800 border border-zinc-700 text-white disabled:opacity-50 transition-all"
            >
              {completedJobs.length === 0
                ? "Complete a training job first"
                : readyDatasets.length === 0
                ? "Create a dataset first"
                : "Launch your first evaluation"}
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
            {/* LEFT SIDEBAR: LIST OF EVALUATION JOBS */}
            <div className="lg:col-span-1 space-y-4">
              <h2 className="text-xs font-mono font-bold text-zinc-500 uppercase tracking-widest mb-2">
                Evaluation Runs ({evaluations.length})
              </h2>
              <div className="space-y-3 overflow-y-auto max-h-[80vh] pr-2">
                {evaluations.map((ev) => {
                  const isSelected = ev.id === selectedEvalId;
                  const dateStr = new Date(ev.created_at).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  });
                  return (
                    <div
                      key={ev.id}
                      onClick={() => setSelectedEvalId(ev.id)}
                      className={`p-4 rounded-xl border cursor-pointer transition-all ${
                        isSelected
                          ? "bg-[#1d1f30] border-cyan-400 shadow-[3px_3px_0px_0px_#000]"
                          : "bg-[#111219]/60 border-zinc-800 hover:border-zinc-700"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[10px] font-mono text-cyan-400 font-bold">
                          ID: {ev.id.slice(0, 8)}
                        </span>
                        <span className={`text-[9px] px-2 py-0.5 rounded-full font-mono font-bold border ${STATUS_STYLES[ev.status]}`}>
                          {ev.status.toUpperCase()}
                        </span>
                      </div>

                      <div className="mb-2">
                        <p className="text-xs text-text font-bold">
                          Test dataset: {ev.dataset_id.slice(0, 8)}...
                        </p>
                        <p className="text-[10px] text-subtle font-mono">
                          {dateStr}
                        </p>
                      </div>

                      {ev.status === "completed" && ev.metrics && (
                        <div className="flex items-center gap-4 mt-2 pt-2 border-t border-zinc-800/60 text-[10px] font-mono text-subtle">
                          <div>
                            Judge (FT): <span className="text-emerald-400 font-bold">{ev.metrics.llm_judge_score != null ? `${ev.metrics.llm_judge_score.toFixed(1)}/5` : "N/A"}</span>
                          </div>
                          <div>
                            Similarity: <span className="text-cyan-400 font-bold">{(ev.metrics.semantic_similarity * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* RIGHT PANELS: DETAILED METRICS */}
            <div className="lg:col-span-2 space-y-6">
              {detailLoading && !selectedEval ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    {Array(3).fill(0).map((_, i) => (
                      <div key={i} className="shimmer h-24 rounded-xl" />
                    ))}
                  </div>
                  <div className="shimmer h-60 rounded-xl" />
                </div>
              ) : selectedEval ? (
                <>
                  {/* Job Overview */}
                  <div className="bg-[#12131c] border border-zinc-800 rounded-xl p-5 shadow-[4px_4px_0px_0px_#000]">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h2 className="text-lg font-display font-black text-text uppercase tracking-tight">
                          Run Details
                        </h2>
                        <p className="text-xs text-subtle font-mono mt-1">
                          Evaluation ID: {selectedEval.id} · Created at: {new Date(selectedEval.created_at).toLocaleString()}
                        </p>
                      </div>
                      <span className={`text-[10px] px-2.5 py-1 rounded-full font-mono font-bold border ${STATUS_STYLES[selectedEval.status]}`}>
                        {selectedEval.status.toUpperCase()}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-xs font-mono border-t border-zinc-900 pt-4 text-subtle">
                      <div>
                        <span className="font-bold text-zinc-500">Fine-Tuned Job:</span> {selectedEval.training_job_id}
                      </div>
                      <div>
                        <span className="font-bold text-zinc-500">Evaluation Dataset:</span> {selectedEval.dataset_id}
                      </div>
                    </div>

                    {selectedEval.error_message && (
                      <div className="bg-red-950/20 border border-red-900/40 rounded-lg p-3 mt-4 text-xs text-red-400 font-mono">
                        <p className="font-bold">Execution Error:</p>
                        <p className="mt-1">{selectedEval.error_message}</p>
                      </div>
                    )}
                  </div>

                  {/* Metrics Summary Cards */}
                  {selectedEval.status === "completed" && selectedEval.metrics && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        label="LLM Judge Score (FT)"
                        value={selectedEval.metrics.llm_judge_score != null ? `${selectedEval.metrics.llm_judge_score.toFixed(1)}/5` : "N/A"}
                        sub="FT instruction alignment"
                        colorClass="text-emerald-400"
                      />
                      <MetricCard
                        label="LLM Judge Score (Base)"
                        value={selectedEval.metrics.llm_judge_base_score != null ? `${selectedEval.metrics.llm_judge_base_score.toFixed(1)}/5` : "N/A"}
                        sub="baseline context score"
                        colorClass="text-cyan-400"
                      />
                      <MetricCard
                        label="Semantic Similarity"
                        value={`${(selectedEval.metrics.semantic_similarity * 100).toFixed(1)}%`}
                        sub="embedding context match"
                        colorClass="text-violet-400"
                      />
                      <MetricCard
                        label="Accuracy Pct"
                        value={`${(selectedEval.metrics.accuracy * 100).toFixed(1)}%`}
                        sub="exact token match ratio"
                        colorClass="text-amber-400"
                      />
                    </div>
                  )}

                  {/* Latency Comparison Card */}
                  {selectedEval.status === "completed" && selectedEval.metrics && (
                    <div className="bg-[#12131c] border border-zinc-800 rounded-xl p-5 shadow-[4px_4px_0px_0px_#000]">
                      <h3 className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest mb-4">
                        Generation Speed comparison
                      </h3>

                      <div className="space-y-4">
                        {/* Base Model */}
                        <div>
                          <div className="flex justify-between text-xs font-mono text-subtle mb-1">
                            <span>Base Model Latency</span>
                            <span className="font-bold">{selectedEval.metrics.avg_base_latency_ms} ms</span>
                          </div>
                          <div className="h-3 bg-zinc-950 border border-zinc-850 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-zinc-600"
                              style={{
                                width: `${Math.min(
                                  100,
                                  (selectedEval.metrics.avg_base_latency_ms /
                                    Math.max(
                                      selectedEval.metrics.avg_base_latency_ms,
                                      selectedEval.metrics.avg_latency_ms
                                    )) *
                                    100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>

                        {/* Fine-Tuned Model */}
                        <div>
                          <div className="flex justify-between text-xs font-mono text-subtle mb-1">
                            <span className="text-cyan-400 font-bold">Fine-Tuned Model Latency (Optimized)</span>
                            <span className="text-cyan-400 font-bold">{selectedEval.metrics.avg_latency_ms} ms</span>
                          </div>
                          <div className="h-3 bg-zinc-950 border border-zinc-850 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-cyan-500"
                              style={{
                                width: `${Math.min(
                                  100,
                                  (selectedEval.metrics.avg_latency_ms /
                                    Math.max(
                                      selectedEval.metrics.avg_base_latency_ms,
                                      selectedEval.metrics.avg_latency_ms
                                    )) *
                                    100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>

                        <div className="text-[10px] font-mono text-zinc-550 text-right pt-2 border-t border-zinc-900">
                          {selectedEval.metrics.avg_base_latency_ms > selectedEval.metrics.avg_latency_ms ? (
                            <span className="text-emerald-400 font-bold">
                              ⚡ Fine-tuned model is {(
                                selectedEval.metrics.avg_base_latency_ms / selectedEval.metrics.avg_latency_ms
                              ).toFixed(1)}x faster than base model
                            </span>
                          ) : (
                            <span>Comparable generation speeds</span>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* PLAYGROUND: DETAILED SIDE-BY-SIDE GENERATION LIST */}
                  {selectedEval.status === "completed" && selectedEval.comparison_data && (
                    <div className="space-y-4">
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-3">
                        <div>
                          <h3 className="text-sm font-display font-black text-text uppercase tracking-tight">
                            Comparison Playground
                          </h3>
                          <p className="text-xs text-subtle font-body">
                            Audit generations item-by-item to discover semantic degradation
                          </p>
                        </div>

                        {/* Filters */}
                        <div className="flex items-center gap-2">
                          <select
                            value={filterType}
                            onChange={(e: any) => setFilterType(e.target.value)}
                            className="px-2.5 py-1.5 bg-zinc-950 border border-zinc-800 rounded text-xs text-subtle focus:outline-none focus:border-cyan-500 font-mono"
                          >
                            <option value="all">Show All Rows</option>
                            <option value="poor">Poor matches (&lt;75% sim)</option>
                            <option value="slow">Slow responses (&gt;500ms)</option>
                          </select>

                          <input
                            type="text"
                            placeholder="Search queries..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="px-3 py-1.5 bg-zinc-950 border border-zinc-800 rounded text-xs text-white focus:outline-none focus:border-cyan-500 font-body placeholder:text-zinc-600"
                          />
                        </div>
                      </div>

                      <div className="space-y-6">
                        {filteredComparisons.length === 0 ? (
                          <div className="bg-[#12131c]/40 border border-dashed border-zinc-800 rounded-xl p-8 text-center text-xs text-zinc-505 font-mono">
                            No comparisons match the current filter/search terms.
                          </div>
                        ) : (
                          filteredComparisons.map((row: any, idx: number) => (
                            <div
                              key={idx}
                              className="bg-[#12131c]/60 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-all space-y-4"
                            >
                              {/* Title / Header */}
                              <div className="flex items-center justify-between text-xs font-mono">
                                <span className="text-zinc-500 font-bold">SAMPLE #{idx + 1}</span>
                                <div className="flex items-center gap-4 text-[10px]">
                                  {row.metrics?.llm_judge_score != null && (
                                    <span className="text-emerald-400 font-bold">
                                      FT Judge: {row.metrics.llm_judge_score.toFixed(1)}/5
                                    </span>
                                  )}
                                  {row.metrics?.llm_judge_base_score != null && (
                                    <span className="text-cyan-400 font-bold">
                                      Base Judge: {row.metrics.llm_judge_base_score.toFixed(1)}/5
                                    </span>
                                  )}
                                  {row.metrics?.semantic_similarity != null && (
                                    <span className="text-violet-400 font-bold">
                                      Sim: {(row.metrics.semantic_similarity * 100).toFixed(0)}%
                                    </span>
                                  )}
                                </div>
                              </div>

                              {/* Query Box */}
                              <div>
                                <p className="text-[10px] text-zinc-505 font-mono uppercase font-bold mb-1">Prompt Input</p>
                                <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-3 text-xs text-text font-mono whitespace-pre-wrap">
                                  {row.input}
                                </div>
                              </div>

                              {/* Gold Reference */}
                              <div>
                                <p className="text-[10px] text-zinc-505 font-mono uppercase font-bold mb-1">Target Answer (Gold Standard)</p>
                                <div className="bg-zinc-950/40 border border-zinc-900/60 rounded-lg p-3 text-xs text-emerald-400/80 font-mono whitespace-pre-wrap">
                                  {row.target}
                                </div>
                              </div>

                              {/* Side-by-Side Outputs */}
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {/* Base Model */}
                                <div className="border border-zinc-900 rounded-lg p-3 bg-zinc-950/20">
                                  <div className="flex justify-between items-center mb-2">
                                    <span className="text-[9px] font-mono text-zinc-505 uppercase font-bold">Base Model Response</span>
                                    <span className="text-[9px] font-mono text-zinc-600 bg-zinc-900 px-1.5 py-0.5 rounded">
                                      {row.base_latency_ms} ms
                                    </span>
                                  </div>
                                  <p className="text-xs text-subtle/80 font-mono leading-relaxed line-clamp-6 whitespace-pre-wrap" title={row.base_response}>
                                    {row.base_response}
                                  </p>
                                </div>

                                {/* Fine-Tuned Model */}
                                <div className="border border-cyan-950 rounded-lg p-3 bg-cyan-950/5">
                                  <div className="flex justify-between items-center mb-2">
                                    <span className="text-[9px] font-mono text-cyan-400 uppercase font-bold">Fine-Tuned Response</span>
                                    <span className="text-[9px] font-mono text-cyan-400/80 bg-cyan-950/40 border border-cyan-900/30 px-1.5 py-0.5 rounded">
                                      {row.ft_latency_ms} ms
                                    </span>
                                  </div>
                                  <p className="text-xs text-text font-mono leading-relaxed line-clamp-6 whitespace-pre-wrap" title={row.fine_tuned_response}>
                                    {row.fine_tuned_response}
                                  </p>
                                </div>
                              </div>

                              {row.explanation && (
                                <div className="bg-zinc-950/40 border border-zinc-900/80 rounded-lg p-3 text-xs text-text font-mono mt-3">
                                  <p className="font-bold text-cyan-400 uppercase text-[9px] mb-1 tracking-wider">Judge Feedback & Explanation</p>
                                  <p className="text-zinc-400 font-sans leading-relaxed">{row.explanation}</p>
                                </div>
                              )}

                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-[#12131c]/60 border border-zinc-800 rounded-xl p-16 text-center shadow-[4px_4px_0px_0px_#000]">
                  <p className="text-sm text-subtle font-mono">Please select an evaluation run from the sidebar to inspect detailed side-by-side metrics.</p>
                </div>
              )}
            </div>
          </div>
        )
      )}

      {showModal && (
        <NewEvaluationModal
          jobs={completedJobs}
          datasets={readyDatasets}
          onClose={() => setShowModal(false)}
          onCreated={handleCreated}
        />
      )}
      <Script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js" strategy="lazyOnload" />
    </div>
  );
}

export default function EvaluatePageWithSuspense() {
  return (
    <Suspense fallback={
      <div className="max-w-7xl mx-auto px-6 py-8 text-center py-16">
        <p className="text-sm text-subtle font-mono">Loading evaluation studio...</p>
      </div>
    }>
      <EvaluatePage />
    </Suspense>
  );
}

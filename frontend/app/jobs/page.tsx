"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { auth, api, keyStore, decryptApiKey } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const MODELS = [
  { id: "microsoft/phi-3-mini-4k-instruct", label: "Phi-3 Mini · 3.8B", badge: "Efficient" },
  { id: "Qwen/Qwen2.5-7B-Instruct",         label: "Qwen 2.5 · 7B",      badge: "SOTA Open" },
  { id: "Qwen/Qwen2.5-Coder-1.5B-Instruct", label: "Qwen 2.5 Coder · 1.5B", badge: "Fast Code" },
  { id: "unsloth/Llama-3.2-3B-Instruct",    label: "Llama 3.2 · 3B (Ungated)", badge: "Un-gated" },
  { id: "unsloth/Llama-3.2-1B-Instruct",    label: "Llama 3.2 · 1B (Ungated)", badge: "Lightweight" },
  { id: "mistralai/Mistral-7B-v0.1",        label: "Mistral · 7B",       badge: "Powerful" },
];

const TASK_TYPES = ["chat", "completion", "classification", "summarization"];

type Dataset = { id: string; name: string; row_count: number; status: string };

const STATUS_STYLES: Record<string, string> = {
  queued:       "bg-zinc-800 text-white border-black",
  provisioning: "bg-amber-500 text-black border-black",
  training:     "bg-cyan-500 text-black border-black",
  uploading:    "bg-violet-500 text-white border-black",
  evaluating:   "bg-indigo-500 text-white border-black",
  completed:    "bg-emerald-500 text-black border-black",
  failed:       "bg-red-500 text-white border-black",
  cancelled:    "bg-zinc-700 text-zinc-300 border-black",
};

function LossCurve({ history }: { history: any[] }) {
  if (!history || history.length < 2) {
    return (
      <div className="h-24 bg-black/40 border border-zinc-800 rounded-lg flex items-center justify-center text-[10px] text-zinc-500 font-mono font-bold mt-3 mb-4">
        ⚡ Plotting live loss curve once steps complete...
      </div>
    );
  }

  const steps = history.map(h => h.step);
  const losses = history.map(h => h.loss);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);

  const width = 450;
  const height = 110;
  const paddingX = 40;
  const paddingY = 15;

  const getX = (step: number) => {
    if (maxStep === minStep) return paddingX;
    return paddingX + ((step - minStep) / (maxStep - minStep)) * (width - 2 * paddingX);
  };

  const getY = (loss: number) => {
    if (maxLoss === minLoss) return height / 2;
    return height - paddingY - ((loss - minLoss) / (maxLoss - minLoss)) * (height - 2 * paddingY);
  };

  const points = history.map(h => `${getX(h.step).toFixed(1)},${getY(h.loss).toFixed(1)}`).join(" ");

  return (
    <div className="bg-black/40 border border-zinc-800 rounded-lg p-3 mt-3 mb-4">
      <p className="text-[9px] text-cyan-400 font-mono font-bold mb-2 uppercase tracking-widest">
        Live Training Loss Curve
      </p>
      <div className="relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          {/* Grid lines */}
          <line x1={paddingX} y1={paddingY} x2={width - paddingX} y2={paddingY} stroke="#27272a" strokeDasharray="3" strokeWidth="1" />
          <line x1={paddingX} y1={height - paddingY} x2={width - paddingX} y2={height - paddingY} stroke="#27272a" strokeDasharray="3" strokeWidth="1" />
          
          {/* Axis Labels */}
          <text x={paddingX - 8} y={paddingY + 3} fill="#71717a" fontSize="8" fontFamily="monospace" textAnchor="end">{maxLoss.toFixed(3)}</text>
          <text x={paddingX - 8} y={height - paddingY + 3} fill="#71717a" fontSize="8" fontFamily="monospace" textAnchor="end">{minLoss.toFixed(3)}</text>
          
          <text x={paddingX} y={height - 2} fill="#52525b" fontSize="8" fontFamily="monospace">step {minStep}</text>
          <text x={width - paddingX} y={height - 2} fill="#52525b" fontSize="8" fontFamily="monospace" textAnchor="end">step {maxStep}</text>

          {/* Polyline */}
          <polyline
            fill="none"
            stroke="#22d3ee" // Cyan accent
            strokeWidth="2"
            points={points}
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Dot on last step */}
          <circle
            cx={getX(history[history.length - 1].step)}
            cy={getY(history[history.length - 1].loss)}
            r="3.5"
            fill="#22d3ee"
            stroke="#000"
            strokeWidth="1.5"
          />
        </svg>
      </div>
    </div>
  );
}

function getETAEstimate(job: any) {
  if (job.status !== "training" || !job.current_step || !job.total_steps || !job.started_at) {
    return null;
  }
  const started = new Date(job.started_at).getTime();
  const elapsedMs = Date.now() - started;
  if (elapsedMs <= 0) return null;
  
  const stepsDone = job.current_step;
  const totalSteps = job.total_steps;
  const stepsRemaining = totalSteps - stepsDone;
  
  if (stepsRemaining <= 0) return "Finishing up...";
  
  const avgTimePerStepMs = elapsedMs / stepsDone;
  const remainingMs = stepsRemaining * avgTimePerStepMs;
  const remainingSeconds = Math.round(remainingMs / 1000);
  
  if (remainingSeconds < 60) {
    return `ETA: ~${remainingSeconds}s`;
  }
  const remainingMinutes = Math.floor(remainingSeconds / 60);
  const remSec = remainingSeconds % 60;
  return `ETA: ~${remainingMinutes}m ${remSec}s`;
}

function JobCard({ job, onRefresh, onDelete }: { job: any; onRefresh: () => void; onDelete: (id: string) => void }) {
  const [cancelling, setCancelling] = useState(false);
  const pct = job.total_steps ? Math.round((job.current_step / job.total_steps) * 100) : 0;
  const eta = getETAEstimate(job);

  const handleCancel = async () => {
    if (!window.confirm("Are you sure you want to cancel this training job?")) return;
    setCancelling(true);
    try {
      await api.cancelJob(job.id);
      onRefresh();
    } catch (err: any) {
      alert(err.message || "Failed to cancel job");
    } finally {
      setCancelling(false);
    }
  };

  const isActive = ["queued", "provisioning", "training", "uploading", "evaluating"].includes(job.status);

  const baseClean = job.base_model?.split("/").pop()?.replace(/\./g, "-").replace(/_/g, "-").toLowerCase() || "";
  const fineTunedName = job.hf_model_id || `veritas-${baseClean}-${job.id.slice(0, 8)}`;

  return (
    <div className="glass rounded-xl border border-zinc-800 p-5 hover:border-zinc-700 transition-all bg-[#111219]/60 mb-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-sm font-mono font-bold text-white mb-1">
            {fineTunedName}
          </p>
          <p className="text-xs text-zinc-500 font-mono font-bold">
            base: {job.base_model?.split("/").pop()} · {job.task_type} · {job.training_config?.epochs} epochs
          </p>
        </div>
        <span className={`text-[10px] px-2.5 py-1 rounded-full font-mono font-bold border ${STATUS_STYLES[job.status] || STATUS_STYLES.queued} shadow-[2px_2px_0px_0px_#000]`}>
          {job.status.toUpperCase()}
        </span>
      </div>

      {/* Progress Section */}
      {isActive && (
        <div className="mb-4 mt-3">
          <div className="flex justify-between text-xs font-mono text-zinc-500 mb-1 font-bold">
            <span>
              {job.status === "provisioning" ? "Provisioning GPU..." : `Step ${job.current_step}/${job.total_steps}`}
            </span>
            {job.status === "training" && <span>{pct}%</span>}
          </div>
          <div className="h-2 bg-zinc-950 border border-zinc-850 rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-500 ${job.status === "provisioning" ? "bg-amber-500 w-1/3 animate-pulse" : "bg-cyan-500"}`} 
              style={{ width: job.status === "provisioning" ? "30%" : `${pct}%` }} 
            />
          </div>
          <div className="flex justify-between items-center text-xs font-mono text-zinc-500 mt-1.5 font-bold">
            {job.train_loss ? (
              <span>loss: {job.train_loss.toFixed(4)}</span>
            ) : <span />}
            {eta && <span className="text-cyan-400">{eta}</span>}
          </div>
        </div>
      )}

      {/* Metrics graph */}
      {(job.status === "training" || job.status === "completed" || (job.metrics_history && job.metrics_history.length > 0)) && (
        <LossCurve history={job.metrics_history} />
      )}

      <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-900">
        <div className="flex gap-4">
          {job.estimated_cost_usd != null && job.estimated_cost_usd > 0 && (
            <p className="text-xs text-zinc-500 font-mono font-bold">est. cost: ${job.estimated_cost_usd.toFixed(4)}</p>
          )}
          {job.actual_cost_usd != null && job.status === "completed" && (
            <p className="text-xs text-emerald-400 font-mono font-bold">actual: ${job.actual_cost_usd.toFixed(4)}</p>
          )}
          {job.gpu_seconds_used > 0 && (
            <p className="text-xs text-zinc-500 font-mono font-bold">time: {job.gpu_seconds_used}s</p>
          )}
        </div>
        
        <div className="flex items-center gap-3">
          {job.hf_model_id && (
            <a href={`https://huggingface.co/${job.hf_model_id}`} target="_blank" rel="noreferrer"
              className="text-xs text-cyan-400 hover:text-cyan-300 font-mono font-bold border border-cyan-400/20 px-2 py-1 rounded bg-cyan-950/20 hover:bg-cyan-950/40 transition-all">
              HF Hub ↗
            </a>
          )}
          
          {isActive ? (
            <button
              onClick={handleCancel}
              disabled={cancelling}
              className="text-xs text-red-400 hover:text-red-350 font-mono font-bold border border-red-500/20 px-2 py-1 rounded bg-red-950/10 hover:bg-red-950/30 transition-all disabled:opacity-50"
            >
              {cancelling ? "Stopping..." : "Stop"}
            </button>
          ) : (
            <button
              onClick={() => {
                if (window.confirm("Are you sure you want to permanently delete this training job and all associated deployments/evaluations?")) {
                  onDelete(job.id);
                }
              }}
              className="text-xs text-zinc-400 hover:text-white font-mono font-bold border border-zinc-750 px-2 py-1 rounded bg-zinc-950/20 hover:bg-zinc-900 transition-all"
            >
              Delete 🗑️
            </button>
          )}
        </div>
      </div>

      {job.error_message && (
        <p className="text-xs text-red-400 bg-red-950/20 border border-red-900/40 rounded-lg px-3 py-2 mt-3 font-mono font-bold break-all">
          Error: {job.error_message}
        </p>
      )}
    </div>
  );
}

import { usePersistentState } from "@/lib/PersistentStateContext";

export default function JobsPage() {
  const router = useRouter();
  const [datasets, setDatasets]   = useState<Dataset[]>([]);
  const [jobs, setJobs]           = useState<any[]>([]);
  const [tunnel, setTunnel]       = useState<{ url: string; mode: string } | null>(null);
  
  const {
    colabStep: step,
    setColabStep: setStep,
    colabUrl,
    setColabUrl,
    colabErrMsg: errMsg,
    setColabErrMsg: setErrMsg,
    colabLoading: loading,
    setColabLoading: setLoading,
    launcherConfig,
    setLauncherConfig,
    handleColabLaunch,
  } = usePersistentState();

  const datasetId = launcherConfig.datasetId;
  const setDatasetId = (val: string) => setLauncherConfig((prev: any) => ({ ...prev, datasetId: val }));
  const model = launcherConfig.model || MODELS[0].id;
  const setModel = (val: string) => setLauncherConfig((prev: any) => ({ ...prev, model: val }));
  const taskType = launcherConfig.taskType;
  const setTaskType = (val: string) => setLauncherConfig((prev: any) => ({ ...prev, taskType: val }));
  const epochs = launcherConfig.epochs;
  const setEpochs = (val: number) => setLauncherConfig((prev: any) => ({ ...prev, epochs: val }));
  const loraR = launcherConfig.loraR;
  const setLoraR = (val: number) => setLauncherConfig((prev: any) => ({ ...prev, loraR: val }));
  const hfRepo = launcherConfig.hfRepo;
  const setHfRepo = (val: string) => setLauncherConfig((prev: any) => ({ ...prev, hfRepo: val }));

  const userId = typeof window !== "undefined" ? sessionStorage.getItem("user_id") || "" : "";

  const fetchJobs = async () => {
    try {
      const j = await api.listJobs();
      setJobs(j);
    } catch {}
  };

  const handleDeleteJob = async (id: string) => {
    try {
      const token = auth.getToken();
      const res = await fetch(`${API_BASE}/jobs/${id}`, {
        method: "DELETE",
        headers: {
          "Authorization": token ? `Bearer ${token}` : "",
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) throw new Error("Failed to delete training job");
      fetchJobs();
    } catch (err: any) {
      alert(err.message || "Failed to delete job");
    }
  };

  useEffect(() => {
    if (!auth.isLoggedIn()) { router.push("/auth"); return; }
    api.listDatasets().then((ds: any[]) => {
      const ready = ds.filter(d => d.status === "ready");
      setDatasets(ready);
      if (ready.length) setDatasetId(ready[0].id);
    }).catch(() => {});
    // Fetch tunnel status to show in UI
    api.getTunnelStatus().then(t => setTunnel({ url: t.url, mode: t.mode })).catch(() => {});
    fetchJobs();
  }, []);

  // Poll active training jobs
  useEffect(() => {
    const activeStates = ["queued", "provisioning", "training", "uploading", "evaluating"];
    const hasActiveJob = jobs.some(j => activeStates.includes(j.status));
    
    if (hasActiveJob) {
      const interval = setInterval(() => {
        fetchJobs();
      }, 4000);
      return () => clearInterval(interval);
    }
  }, [jobs]);

  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    setStep("idle");
    setColabUrl("");
    setErrMsg("");
  }, [datasetId, model, taskType, epochs, loraR, hfRepo]);

  async function handleColab() {
    if (!datasetId) { setErrMsg("Please select a dataset first."); return; }
    const encGithub = keyStore.get("github");
    if (!encGithub) {
      setErrMsg("No GitHub token found. Add it in Settings → API Keys.");
      return;
    }
    await handleColabLaunch(encGithub, userId, fetchJobs);
  }

  const accent = "#ff5c36";

  return (
    <div style={{ minHeight: "calc(100vh - 56px)", background: "#0f0f12", padding: "32px 20px" }}>
      <div style={{ maxWidth: 820, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
            <span style={{ fontSize: 28 }}>🧠</span>
            <h1 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 26, fontWeight: 900, color: "#fff", margin: 0 }}>
              Fine-Tuning Launcher
            </h1>
          </div>
          <p style={{ fontFamily: "'Inter', sans-serif", fontSize: 13, color: "#7a7f9a", margin: 0 }}>
            Configure your training job and launch on your preferred compute provider.
          </p>
        </div>

        {/* Tunnel status bar */}
        {tunnel && (
          <div style={{
            background: "#161722", border: "2px solid #000", boxShadow: "3px 3px 0 #000",
            borderRadius: 8, padding: "10px 16px", marginBottom: 24,
            display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", border: "1px solid #000", display: "inline-block" }} />
              <span style={{ fontFamily: "monospace", fontSize: 11, color: "#a0a5c0" }}>
                <span style={{ color: "#fff", fontWeight: 700 }}>Tunnel Active</span> — Colab can reach your backend
              </span>
            </div>
            <span style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a" }}>{tunnel.url}</span>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 32 }}>
          {/* Left — Config */}
          <div style={{ background: "#161722", border: "3px solid #000", boxShadow: "5px 5px 0 #000", borderRadius: 10, padding: 24 }}>
            <h2 style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 700, color: accent, margin: "0 0 20px", textTransform: "uppercase", letterSpacing: ".08em" }}>
              Job Config
            </h2>

            {/* Dataset */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a", display: "block", marginBottom: 6, fontWeight: 700 }}>
                DATASET
              </label>
              {datasets.length === 0 ? (
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#ff5c36", padding: "10px 0" }}>
                  No ready datasets — generate one first.
                </div>
              ) : (
                <select value={datasetId} onChange={e => setDatasetId(e.target.value)} style={{
                  width: "100%", background: "#0f0f12", border: "2px solid #000", borderRadius: 6,
                  color: "#fff", fontFamily: "monospace", fontSize: 12, padding: "8px 10px", outline: "none",
                }}>
                  {datasets.map(d => (
                    <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
                  ))}
                </select>
              )}
            </div>

            {/* Model */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a", display: "block", marginBottom: 6, fontWeight: 700 }}>
                BASE MODEL
              </label>
              <select value={model} onChange={e => setModel(e.target.value)} style={{
                width: "100%", background: "#0f0f12", border: "2px solid #000", borderRadius: 6,
                color: "#fff", fontFamily: "monospace", fontSize: 12, padding: "8px 10px", outline: "none",
              }}>
                {MODELS.map(m => (
                  <option key={m.id} value={m.id}>{m.label}</option>
                ))}
              </select>
            </div>

            {/* Task Type */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a", display: "block", marginBottom: 6, fontWeight: 700 }}>
                TASK TYPE
              </label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {TASK_TYPES.map(t => (
                  <button key={t} onClick={() => setTaskType(t)} style={{
                    fontFamily: "monospace", fontSize: 11, fontWeight: 700,
                    padding: "5px 12px", borderRadius: 5, cursor: "pointer",
                    border: "2px solid #000",
                    background: taskType === t ? accent : "#0f0f12",
                    color: taskType === t ? "#000" : "#7a7f9a",
                    boxShadow: taskType === t ? "2px 2px 0 #000" : "none",
                    textTransform: "capitalize",
                  }}>{t}</button>
                ))}
              </div>
            </div>

            {/* Epochs */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a", display: "block", marginBottom: 6, fontWeight: 700 }}>
                EPOCHS — {epochs}
              </label>
              <input type="range" min={1} max={10} value={epochs} onChange={e => setEpochs(+e.target.value)}
                style={{ width: "100%", accentColor: accent }} />
            </div>

            {/* LoRA R */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a", display: "block", marginBottom: 6, fontWeight: 700 }}>
                LORA RANK — {loraR}
              </label>
              <input type="range" min={4} max={64} step={4} value={loraR} onChange={e => setLoraR(+e.target.value)}
                style={{ width: "100%", accentColor: accent }} />
            </div>

            {/* HF Repo */}
            <div>
              <label style={{ fontFamily: "monospace", fontSize: 11, color: "#7a7f9a", display: "block", marginBottom: 6, fontWeight: 700 }}>
                HF REPO ID <span style={{ fontWeight: 400, color: "#555" }}>(optional)</span>
              </label>
              <input
                type="text" value={hfRepo} onChange={e => setHfRepo(e.target.value)}
                placeholder="username/my-model"
                style={{
                  width: "100%", background: "#0f0f12", border: "2px solid #000", borderRadius: 6,
                  color: "#fff", fontFamily: "monospace", fontSize: 12, padding: "8px 10px", outline: "none",
                  boxSizing: "border-box",
                }}
              />
            </div>
          </div>

          {/* Right — Provider Cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Colab Card */}
            <div style={{
              background: "#161722", border: "3px solid #000", boxShadow: "5px 5px 0 #000",
              borderRadius: 10, padding: 24, flex: 1, position: "relative",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{ fontSize: 28 }}>📓</span>
                <div>
                  <div style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 900, color: "#fff" }}>Google Colab</div>
                  <div style={{ fontFamily: "monospace", fontSize: 10, color: "#22c55e", fontWeight: 700 }}>FREE · T4 GPU</div>
                </div>
              </div>
              <p style={{ fontFamily: "'Inter', sans-serif", fontSize: 12, color: "#7a7f9a", lineHeight: 1.6, marginBottom: 16 }}>
                Generates a pre-filled notebook published to your GitHub Gists.
                One click opens it directly in Colab — just run all cells.
              </p>

              {/* Steps progress */}
              {step !== "idle" && step !== "error" && (
                <div style={{ marginBottom: 14 }}>
                  {[
                    { key: "gist",     label: "Publishing to GitHub Gist..." },
                    { key: "done",     label: "Notebook ready!" },
                  ].map((s, i) => {
                    const states = ["gist", "done"];
                    const cur = states.indexOf(step);
                    const done = states.indexOf(s.key) < cur || step === "done";
                    const active = s.key === step;
                    return (
                      <div key={s.key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <span style={{ fontSize: 12, color: done ? "#22c55e" : active ? accent : "#555" }}>
                          {done ? "✓" : active ? "⟳" : "○"}
                        </span>
                        <span style={{ fontFamily: "monospace", fontSize: 11, color: done ? "#22c55e" : active ? "#fff" : "#555" }}>
                          {s.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {step === "done" && colabUrl ? (
                <div>
                  <a href={colabUrl} target="_blank" rel="noreferrer" style={{
                    display: "block", width: "100%", textAlign: "center",
                    padding: "11px 0", borderRadius: 7, fontSize: 13, fontWeight: 900,
                    fontFamily: "monospace", textDecoration: "none", cursor: "pointer",
                    background: "#f9ab00", color: "#000", border: "3px solid #000",
                    boxShadow: "3px 3px 0 #000", animation: "pulse 1.5s infinite",
                  }}>
                    🚀 Open in Colab ↗
                  </a>
                  <button
                    onClick={() => {
                      setStep("idle");
                      setColabUrl("");
                      setErrMsg("");
                    }}
                    style={{
                      marginTop: 10, width: "100%", padding: "6px 0", borderRadius: 6,
                      fontSize: 11, fontWeight: 700, fontFamily: "monospace", cursor: "pointer",
                      background: "#1e1e2e", color: "#a0a5c0", border: "2px solid #333",
                    }}
                  >
                    🔄 Configure New Job
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleColab}
                  disabled={loading || datasets.length === 0}
                  style={{
                    width: "100%", padding: "11px 0", borderRadius: 7, fontSize: 13,
                    fontWeight: 900, fontFamily: "monospace", cursor: loading ? "wait" : "pointer",
                    background: loading ? "#333" : accent, color: loading ? "#777" : "#000",
                    border: "3px solid #000", boxShadow: "3px 3px 0 #000", opacity: datasets.length === 0 ? 0.4 : 1,
                  }}
                >
                  {loading ? "Generating..." : "📓 Open in Colab"}
                </button>
              )}

              {errMsg && (
                <p style={{ fontFamily: "monospace", fontSize: 11, color: "#f87171", marginTop: 10, lineHeight: 1.5 }}>
                  ⚠ {errMsg}
                </p>
              )}
            </div>

            {/* RunPod Card — Coming Soon */}
            <div style={{
              background: "#161722", border: "3px solid #000", boxShadow: "5px 5px 0 #000",
              borderRadius: 10, padding: 24, flex: 1, position: "relative", opacity: 0.65,
            }}>
              <div style={{
                position: "absolute", top: -13, right: 16,
                background: "#f9ab00", color: "#000", fontFamily: "monospace",
                fontSize: 9, fontWeight: 900, padding: "3px 10px",
                border: "2px solid #000", borderRadius: 4,
                boxShadow: "2px 2px 0 #000", letterSpacing: ".06em",
              }}>COMING SOON</div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{ fontSize: 28 }}>⚡</span>
                <div>
                  <div style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 900, color: "#fff" }}>RunPod Serverless</div>
                  <div style={{ fontFamily: "monospace", fontSize: 10, color: "#f9ab00", fontWeight: 700 }}>RTX 4090 · $0.22/hr</div>
                </div>
              </div>
              <p style={{ fontFamily: "'Inter', sans-serif", fontSize: 12, color: "#7a7f9a", lineHeight: 1.6, marginBottom: 16 }}>
                One-click dispatch to a serverless GPU pod. Live loss curve streams directly into this dashboard.
              </p>
              <button disabled style={{
                width: "100%", padding: "11px 0", borderRadius: 7, fontSize: 13,
                fontWeight: 900, fontFamily: "monospace", cursor: "not-allowed",
                background: "#1e1e2e", color: "#555", border: "3px solid #333",
              }}>
                ⚡ Launch on RunPod
              </button>
            </div>
          </div>
        </div>

        {/* Your Training Jobs List Section */}
        <div style={{ marginTop: 40 }}>
          <h2 style={{
            fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
            fontSize: "20px",
            fontWeight: 900,
            color: "#ffffff",
            marginBottom: 16
          }}>
            Your Training Jobs
          </h2>
          {jobs.length === 0 ? (
            <div className="glass rounded-xl border border-zinc-800 p-8 text-center bg-[#111219]/30">
              <p style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, color: "#a0a5c0" }}>
                No active or completed training jobs yet. Use the launcher above to start one!
              </p>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {jobs.map((j) => (
                <JobCard key={j.id} job={j} onRefresh={fetchJobs} onDelete={handleDeleteJob} />
              ))}
            </div>
          )}
        </div>

        {/* Info footer */}
        <div style={{ fontFamily: "monospace", fontSize: 11, color: "#555", textAlign: "center", marginTop: 40 }}>
          Need to add keys? Go to{" "}
          <a href="/settings" style={{ color: accent, textDecoration: "none" }}>Settings → API Keys</a>
          {" "}to add your GitHub and Hugging Face tokens.
        </div>
      </div>
    </div>
  );
}

"use client";
import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import Script from "next/script";
import { api, auth } from "@/lib/api";
import { usePersistentState } from "@/lib/PersistentStateContext";

const TASK_TYPES = ["chat", "instruction", "qa", "classification", "summarization"];
const STYLES = ["casual", "formal", "technical", "simple"];
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

function downloadCSV(rows: any[], name: string) {
  if (!rows?.length) return;
  const keys = Object.keys(rows[0]);
  const header = keys.join(",");
  const body = rows.map(r => keys.map(k => `"${String(r[k]).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([header + "\n" + body], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = `${name}.csv`; a.click();
  URL.revokeObjectURL(url);
}

function downloadJSONL(rows: any[], name: string) {
  if (!rows?.length) return;
  const content = rows.map(r => JSON.stringify(r)).join("\n");
  const blob = new Blob([content], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = `${name}.jsonl`; a.click();
  URL.revokeObjectURL(url);
}

function InteractiveDatasetEditor({
  datasetId,
  initialRows,
  datasetName,
  accent
}: {
  datasetId: string;
  initialRows: any[];
  datasetName: string;
  accent: string;
}) {
  const [rows, setRows] = useState<any[]>([]);
  const [name, setName] = useState("");
  const [isEditMode, setIsEditMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    if (initialRows) {
      setRows(JSON.parse(JSON.stringify(initialRows)));
    }
  }, [initialRows]);

  useEffect(() => {
    if (datasetName) {
      setName(datasetName);
    }
  }, [datasetName]);

  const handleRowChange = (index: number, field: "input" | "output", val: string) => {
    const updated = [...rows];
    updated[index] = { ...updated[index], [field]: val };
    setRows(updated);
  };

  const handleSaveChanges = async () => {
    setSaving(true);
    setError("");
    setSuccess("");
    try {
      const token = auth.getToken();
      const res = await fetch(`${API_BASE}/datasets/${datasetId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "Authorization": token ? `Bearer ${token}` : ""
        },
        body: JSON.stringify({
          name: name.trim() || undefined,
          rows: rows
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to save dataset edits");
      setSuccess("💾 Changes saved successfully!");
      setTimeout(() => setSuccess(""), 3500);
    } catch (e: any) {
      setError(e.message || "Failed to save dataset edits");
    } finally {
      setSaving(false);
    }
  };

  const filteredRows = rows.map((row, idx) => ({ ...row, originalIndex: idx })).filter(row => {
    const q = searchQuery.toLowerCase();
    return (
      (row.input || "").toLowerCase().includes(q) ||
      (row.output || "").toLowerCase().includes(q)
    );
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
          {isEditMode ? (
            <div>
              <label style={{ display: "block", fontSize: 10, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginBottom: 2 }}>DATASET NAME</label>
              <input
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                className="vt-input"
                style={{ width: "100%", padding: "6px 10px", fontSize: 12 }}
              />
            </div>
          ) : (
            <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#ffffff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              📋 {name} ({rows.length} rows)
            </span>
          )}
        </div>
        <button
          onClick={() => {
            setIsEditMode(!isEditMode);
            setError("");
            setSuccess("");
          }}
          className="tab-btn tab-btn-inactive"
          style={{
            padding: "4px 10px",
            fontSize: 9,
            borderRadius: 6,
            background: isEditMode ? accent : "#12131c",
            color: isEditMode ? "#000" : "#a0a5c0",
            border: "2px solid #000",
            boxShadow: "1px 1px 0px #000",
            flexShrink: 0
          }}
        >
          {isEditMode ? "👁️ Preview Mode" : "✍️ Edit Mode"}
        </button>
      </div>

      {/* Search Input */}
      <div>
        <input
          type="text"
          placeholder="🔍 Search rows..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          className="vt-input"
          style={{ width: "100%", padding: "8px 12px", fontSize: 11, background: "#0c0d12" }}
        />
      </div>

      {/* Scrollable list */}
      <div style={{ maxHeight: 220, overflowY: "auto", display: "flex", flexDirection: "column", gap: 10, paddingRight: 4 }}>
        {filteredRows.length === 0 ? (
          <div style={{ padding: 20, textAlign: "center", fontSize: 11, color: "#6c7293", fontFamily: "'JetBrains Mono', monospace", border: "1px dashed #27293d", borderRadius: 8 }}>
            No rows match your search query.
          </div>
        ) : (
          filteredRows.map(row => (
            <div key={row.originalIndex} style={{ background: "#0c0d12", border: "2px solid #000000", borderRadius: 8, padding: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, fontWeight: "bold", color: isEditMode ? accent : "#a0a5c0" }}>
                  ROW #{row.originalIndex + 1}
                </span>
              </div>
              {isEditMode ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: "bold", textTransform: "uppercase", color: accent, marginBottom: 2, display: "block" }}>input</span>
                    <textarea
                      value={row.input || ""}
                      onChange={e => handleRowChange(row.originalIndex, "input", e.target.value)}
                      rows={2}
                      className="vt-input"
                      style={{ width: "100%", padding: "6px 8px", fontSize: 11, outline: "none", resize: "vertical", background: "#06070a", fontFamily: "inherit" }}
                    />
                  </div>
                  <div>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: "bold", textTransform: "uppercase", color: "#a0a5c0", marginBottom: 2, display: "block" }}>output</span>
                    <textarea
                      value={row.output || ""}
                      onChange={e => handleRowChange(row.originalIndex, "output", e.target.value)}
                      rows={2}
                      className="vt-input"
                      style={{ width: "100%", padding: "6px 8px", fontSize: 11, outline: "none", resize: "vertical", background: "#06070a", fontFamily: "inherit" }}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: "bold", textTransform: "uppercase", color: accent, marginBottom: 2, display: "block" }}>input</span>
                  <div style={{ fontSize: 11, color: "#ffffff", marginBottom: 6, lineHeight: 1.5, whiteSpace: "pre-wrap" }}>{row.input}</div>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: "bold", textTransform: "uppercase", color: "#a0a5c0", marginBottom: 2, display: "block" }}>output</span>
                  <div style={{ fontSize: 11, color: "#a0a5c0", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>{row.output}</div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {error && (
        <div style={{ fontSize: 11, color: "#ffffff", background: "#f43f5e", border: "2px solid #000000", padding: "6px 10px", borderRadius: "6px", fontFamily: "'JetBrains Mono', monospace" }}>
          ⚠ {error}
        </div>
      )}

      {success && (
        <div style={{ fontSize: 11, color: "#000000", background: "#10b981", border: "2px solid #000000", padding: "6px 10px", borderRadius: "6px", fontFamily: "'JetBrains Mono', monospace", fontWeight: "bold" }}>
          ✓ {success}
        </div>
      )}

      <div style={{ display: "flex", gap: 12 }}>
        <button onClick={() => downloadCSV(rows, name || "dataset")} className="btn-primary" style={{ flex: 1, padding: "10px", fontSize: 11 }}>
          ↓ Download CSV
        </button>
        <button onClick={() => downloadJSONL(rows, name || "dataset")} className="btn-ghost" style={{ flex: 1, padding: "10px", fontSize: 11 }}>
          ↓ Download JSONL
        </button>
        {isEditMode && (
          <button
            onClick={handleSaveChanges}
            disabled={saving}
            className="btn-primary"
            style={{
              flex: 1,
              padding: "10px",
              fontSize: 11,
              background: "#10b981",
              color: "#000",
              boxShadow: "2px 2px 0px #000",
              cursor: "pointer"
            }}
          >
            {saving ? "Saving..." : "💾 Save Changes"}
          </button>
        )}
      </div>
    </div>
  );
}


function GenerateTab({ accent }: { accent: string }) {
  const {
    genForm: form,
    setGenForm: setForm,
    genPreview: preview,
    setGenPreview: setPreview,
    genResult: result,
    setGenResult: setResult,
    genStep: step,
    setGenStep: setStep,
    genError: error,
    setGenError: setError,
    genLoading: loading,
    setGenLoading: setLoading,
    genPreviewing: previewing,
    setGenPreviewing: setPreviewing,
    genIsStreaming: isStreaming,
    genStreamProgress: streamProgress,
    startGenerateTextDataset,
  } = usePersistentState();

  const [showDeepSeekTip, setShowDeepSeekTip] = useState(false);
  useEffect(() => {
    if (typeof window !== "undefined") {
      const hasOpenAI = !!sessionStorage.getItem("ek_openai");
      const hasDeepSeek = !!sessionStorage.getItem("ek_deepseek");
      setShowDeepSeekTip(hasOpenAI && !hasDeepSeek);
    }
  }, [isStreaming]);

  const handlePreview = async () => {
    if (!form.topic.trim()) { setError("Enter a topic first"); return; }
    setPreviewing(true); setError("");
    try {
      const res = await api.previewDataset({ topic: form.topic, task_type: form.task_type, style: form.style, domain_context: form.domain_context });
      setPreview(res.preview_rows || []); setStep("preview");
    } catch (e: any) { setError(e.message); }
    finally { setPreviewing(false); }
  };

  const handleGenerate = async () => {
    await startGenerateTextDataset();
  };

  const scoreColor = result?.quality_score >= 90 ? "#10b981" : result?.quality_score >= 70 ? "#f59e0b" : "#f43f5e";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
      {/* Left — form */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
            Describe your use case <span style={{ color: accent }}>*</span>
          </label>
          <textarea value={form.topic} onChange={e => setForm({ ...form, topic: e.target.value })}
            placeholder="e.g. Customer support bot for a food delivery app that handles order tracking, cancellations, and refunds"
            rows={4}
            disabled={loading || previewing || isStreaming}
            style={{ width: "100%", padding: "12px 16px", outline: "none", resize: "none" }}
            className="vt-input" />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
              Task Type
            </label>
            <select value={form.task_type} onChange={e => setForm({ ...form, task_type: e.target.value })}
              disabled={loading || previewing || isStreaming}
              style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }}
              className="vt-input">
              {TASK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
              Style
            </label>
            <select value={form.style} onChange={e => setForm({ ...form, style: e.target.value })}
              disabled={loading || previewing || isStreaming}
              style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }}
              className="vt-input">
              {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
            Number of Rows: <span style={{ color: accent }}>{form.total_rows}</span>
          </label>
          <input type="range" min={10} max={5000} step={1} value={form.total_rows}
            disabled={loading || previewing || isStreaming}
            onChange={e => setForm({ ...form, total_rows: parseInt(e.target.value) })}
            style={{ width: "100%", accentColor: accent }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}>
            <span>10</span><span>5000</span>
          </div>

        </div>

        {/* Phase 2 UI addition: Checkbox to toggle multi-turn conversation generation */}
        <div>
          <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", userSelect: "none" }}>
            <input type="checkbox" checked={form.multi_turn}
              disabled={loading || previewing || isStreaming}
              onChange={e => setForm({ ...form, multi_turn: e.target.checked })}
              style={{ width: 18, height: 18, accentColor: accent, cursor: "pointer" }} />
            Enable Multi-Turn Conversations (2 to 4 turns)
          </label>
          <span style={{ display: "block", fontSize: "11px", color: "#a0a5c0", marginTop: 4, marginLeft: 28, lineHeight: 1.4 }}>
            Generates nested OpenAI-compatible messages along with flat input/output keys for training compatibility.
          </span>
        </div>

        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
            Domain Context <span style={{ color: "#6c7293", fontSize: "12px" }}>(optional)</span>
          </label>
          <input type="text" value={form.domain_context} onChange={e => setForm({ ...form, domain_context: e.target.value })}
            disabled={loading || previewing || isStreaming}
            placeholder="Extra context about your domain, constraints, terminology..."
            style={{ width: "100%", padding: "12px 16px", outline: "none" }}
            className="vt-input" />
        </div>

        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
            Dataset Name
          </label>
          <input type="text" value={form.dataset_name} onChange={e => setForm({ ...form, dataset_name: e.target.value })}
            disabled={loading || previewing || isStreaming}
            placeholder={form.topic ? `${form.topic.slice(0, 30)}...` : "My dataset"}
            style={{ width: "100%", padding: "12px 16px", outline: "none" }}
            className="vt-input" />
        </div>

        {error && (
          <div style={{
            fontSize: 12,
            color: "#ffffff",
            background: "#f43f5e",
            border: "3px solid #000000",
            boxShadow: "2px 2px 0px 0px #000000",
            padding: "10px 14px",
            borderRadius: "8px",
            fontFamily: "'JetBrains Mono', monospace"
          }}>
            {error}
          </div>
        )}

        <div style={{ display: "flex", gap: 12, paddingTop: 4 }}>
          <button onClick={handlePreview} disabled={previewing || loading || isStreaming} className="btn-ghost" style={{ flex: 1, padding: "12px 24px" }}>
            {previewing ? "Generating..." : "Preview 5 rows"}
          </button>
          <button onClick={handleGenerate} disabled={loading || previewing || isStreaming || !form.topic.trim()} className="btn-primary" style={{ flex: 1, padding: "12px 24px" }}>
            {loading ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
              <span style={{ width: 12, height: 12, border: "1.5px solid #000", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
              Generating...
            </span> : `Generate ${form.total_rows} rows →`}
          </button>
        </div>
      </div>

      {/* Right — preview / result */}
      <div>
        {step === "form" && !isStreaming && (
          <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: "40px" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>⚗️</div>
            <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 20, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Dataset Studio</div>
            <div style={{ fontSize: 13, color: "#a0a5c0", maxWidth: 280, lineHeight: 1.7 }}>
              Preview 5 rows first to verify quality, then generate the full dataset. Our anti-noise filters run automatically.
            </div>
          </div>
        )}

        {isStreaming && (
          <div className="vt-card" style={{ display: "flex", flexDirection: "column", gap: 20, padding: 20 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: accent, border: "1px solid #000", animation: "pulse 1.5s infinite", display: "inline-block" }} />
                <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>
                  Dataset Compiler Active
                </span>
              </div>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: accent, fontWeight: "bold" }}>
                {streamProgress.generatedSoFar} / {form.total_rows} rows
              </span>
            </div>

            {/* ☕ Coffee brewing banner */}
            <div style={{ background: "rgba(245,158,11,0.08)", border: "2px solid #f59e0b", borderRadius: 10, padding: "10px 14px", display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ width: 120, height: 120, flexShrink: 0 }}>
                {/* @ts-ignore */}
                <lottie-player
                  src="/brewing_2.json"
                  background="transparent"
                  speed="1.2"
                  style={{ width: "100%", height: "100%" }}
                  loop
                  autoplay
                />
              </div>
              <div>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#fcd34d" }}>Grab a coffee — the dataset is brewing!</div>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#a0a5c0", marginTop: 2 }}>Large datasets take 2–5 minutes. Anti-noise filters + embedding dedup run on every batch.</div>
              </div>
            </div>

            {showDeepSeekTip && (
              <div style={{
                background: "rgba(59, 130, 246, 0.08)",
                border: "2px dashed #3b82f6",
                borderRadius: 10,
                padding: "10px 14px",
                display: "flex",
                flexDirection: "column",
                gap: 4
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 14 }}>🚀</span>
                  <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#60a5fa" }}>
                    Pro-tip: DeepSeek is 5x faster & cheaper!
                  </span>
                </div>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#a0a5c0" }}>
                  You are currently generating using OpenAI. Adding a DeepSeek API key in Settings will speed up dataset generation and significantly reduce cost.
                </div>
              </div>
            )}

            {/* Total progress bar */}
            <div>
              <div style={{ height: 12, background: "#000000", border: "2px solid #000000", borderRadius: "10px", overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  background: accent,
                  width: `${(streamProgress.generatedSoFar / form.total_rows) * 100}%`,
                  transition: "width 0.4s ease"
                }} />
              </div>
            </div>

            {/* Inferred subdomains & quotas */}
            {streamProgress.coveragePlan && (
              <div style={{ background: "#0c0d12", border: "2px solid #000000", borderRadius: "8px", padding: 12, boxShadow: "2px 2px 0px 0px #000000" }}>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 12, fontWeight: "bold", color: "#ffffff", marginBottom: 8, textTransform: "uppercase" }}>
                  Inferred Subdomain Coverage & Quotas:
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {Object.entries(streamProgress.coveragePlan).map(([sub, quota]) => {
                    const isCurrent = streamProgress.currentSubdomain === sub;
                    return (
                      <div key={sub} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 8px", background: isCurrent ? "rgba(255, 92, 54, 0.1)" : "transparent", border: isCurrent ? `1px solid ${accent}` : "1px solid transparent", borderRadius: "4px" }}>
                        <span style={{ fontSize: 12, color: isCurrent ? "#ffffff" : "#a0a5c0", fontWeight: isCurrent ? "bold" : "normal" }}>
                          {isCurrent ? "➔ " : ""} {sub}
                        </span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: isCurrent ? accent : "#ffffff", fontWeight: "bold" }}>
                          {String(quota)} rows
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Current generation parameters */}
            {streamProgress.currentSubdomain && (
              <div style={{ background: "#0c0d12", border: "2px solid #000000", borderRadius: "8px", padding: 12, boxShadow: "2px 2px 0px 0px #000000", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div>
                  <span style={{ display: "block", fontSize: 9, textTransform: "uppercase", color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace" }}>Persona</span>
                  <span style={{ fontSize: 11, color: "#ffffff", fontWeight: "bold" }}>{streamProgress.currentPersona}</span>
                </div>
                <div>
                  <span style={{ display: "block", fontSize: 9, textTransform: "uppercase", color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace" }}>Difficulty</span>
                  <span style={{ fontSize: 11, color: "#ffffff", fontWeight: "bold" }}>{streamProgress.currentDifficulty}</span>
                </div>
                <div>
                  <span style={{ display: "block", fontSize: 9, textTransform: "uppercase", color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace" }}>Mode</span>
                  <span style={{ fontSize: 11, color: "#ffffff", fontWeight: "bold" }}>{streamProgress.currentMode}</span>
                </div>
                <div>
                  <span style={{ display: "block", fontSize: 9, textTransform: "uppercase", color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace" }}>Depth</span>
                  <span style={{ fontSize: 11, color: "#ffffff", fontWeight: "bold" }}>{streamProgress.currentDepth}</span>
                </div>
              </div>
            )}

            {/* Telemetry logs */}
            <div>
              <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 12, fontWeight: "bold", color: "#ffffff", marginBottom: 6, textTransform: "uppercase" }}>
                Telemetry Log
              </div>
              <div style={{
                background: "#050508",
                border: "2px solid #000000",
                boxShadow: "2px 2px 0px 0px #000000",
                borderRadius: "8px",
                padding: "10px",
                height: "150px",
                overflowY: "auto",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: "10px",
                color: "#10b981",
                display: "flex",
                flexDirection: "column",
                gap: 4
              }}>
                {streamProgress.logs.map((log: string, idx: number) => (
                  <div key={idx} style={{ lineHeight: 1.4 }}>
                    {log}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {step === "preview" && preview.length > 0 && (
          <div className="vt-card" style={{ display: "flex", flexDirection: "column", height: "100%", padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "12px 16px", borderBottom: "3px solid #000000", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: "bold", textTransform: "uppercase", color: "var(--accent)" }}>
                Preview — {preview.length} sample rows
              </span>
              <button onClick={() => setStep("form")} className="btn-ghost" style={{ padding: "4px 10px", fontSize: 10, borderRadius: "6px" }}>
                ← Edit
              </button>
            </div>
            <div style={{ padding: 16, flex: 1, maxHeight: 420, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
              {preview.map((row, i) => (
                <div key={i} style={{ background: "#0c0d12", border: "2px solid #000000", borderRadius: "8px", padding: "12px", boxShadow: "2px 2px 0px 0px #000000" }}>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: "bold", textTransform: "uppercase", color: "var(--accent)", marginBottom: 4, display: "block" }}>Input</span>
                  <div style={{ fontSize: 12, color: "#ffffff", marginBottom: 8, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{row.input}</div>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, fontWeight: "bold", textTransform: "uppercase", color: "#a0a5c0", marginBottom: 4, display: "block" }}>Output</span>
                  <div style={{ fontSize: 12, color: "#a0a5c0", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{row.output}</div>
                </div>
              ))}
            </div>
            <div style={{ padding: "16px", borderTop: "3px solid #000000", background: "rgba(0,0,0,0.2)" }}>
              <div style={{ fontSize: 12, color: "#a0a5c0", marginBottom: 12, fontFamily: "'Comic Neue', cursive", fontWeight: "bold" }}>Looks good? Generate the full dataset.</div>
              <button onClick={handleGenerate} disabled={loading} className="btn-primary" style={{ width: "100%", padding: "12px" }}>
                {loading ? "Generating..." : `Generate ${form.total_rows} rows →`}
              </button>
            </div>
          </div>
        )}

        {step === "result" && result && (
          <div className="vt-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#10b981", border: "1px solid #000", display: "inline-block" }} />
              <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#10b981" }}>Dataset Generated Successfully</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Stats */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                {[
                  ["rows", result.row_count, "#ffffff"],
                  ["quality", `${result.quality_score}%`, scoreColor],
                  ["dupes", result.audit?.duplicate_count ?? 0, result.audit?.duplicate_count > 0 ? "#f59e0b" : "#10b981"]
                ].map(([l, v, c]) => (
                  <div key={String(l)} style={{ background: "#0c0d12", border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", borderRadius: "8px", padding: "12px 8px", textAlign: "center" }}>
                    <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 22, fontWeight: 900, color: String(c) }}>{v}</div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0", marginTop: 4 }}>{l}</div>
                  </div>
                ))}
              </div>

              {/* Quality bar */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#a0a5c0", marginBottom: 6, fontWeight: "bold" }}>
                  <span>Audit Quality Score</span>
                  <span style={{ color: scoreColor }}>{result.quality_score}%</span>
                </div>
                <div style={{ height: 12, background: "#000000", border: "2px solid #000000", borderRadius: "10px", overflow: "hidden" }}>
                  <div style={{ height: "100%", background: scoreColor, width: `${result.quality_score}%`, transition: "width 1s" }} />
                </div>
              </div>

              {/* Phase 2: Fine-Tuning Readiness Grade Badge & Status (A to F) */}
              {result.audit?.readiness_grade && (
                <div style={{
                  background: "#0c0d12",
                  border: "3px solid #000000",
                  boxShadow: `4px 4px 0px 0px ${scoreColor}`,
                  borderRadius: "12px",
                  padding: "16px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 16
                }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0" }}>
                      Fine-Tuning Readiness Grade
                    </span>
                    <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>
                      {result.audit.readiness_label}
                    </span>
                    <span style={{ fontSize: 11, color: "#a0a5c0" }}>
                      Audit Score: {result.audit.readiness_score}% (Shannon word entropy: {result.fingerprint?.starting_word_entropy ?? result.audit?.starting_word_entropy ?? "N/A"})
                    </span>
                  </div>
                  <div style={{
                    width: 64,
                    height: 64,
                    borderRadius: "8px",
                    background: scoreColor,
                    border: "3px solid #000000",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: "2px 2px 0px 0px #000000"
                  }}>
                    <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 36, fontWeight: 900, color: "#000000" }}>
                      {result.audit.readiness_grade}
                    </span>
                  </div>
                </div>
              )}

              {/* Phase 2: Actionable Repair Recommendations for Text/QA */}
              {result.audit?.issues?.length > 0 && (
                <div style={{
                  background: "#0c0d12",
                  border: "3px solid #000000",
                  borderRadius: "12px",
                  padding: "16px",
                  boxShadow: "3px 3px 0px 0px #000000"
                }}>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#ffffff", marginBottom: 10, textTransform: "uppercase", display: "flex", alignItems: "center", gap: 6 }}>
                    🔧 Actionable Repair Suggestions
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {result.audit.issues.map((issue: any, i: number) => {
                      let recommendation = issue.fix || "Regenerate dataset with updated parameters or higher temperature to improve variety.";
                      return (
                        <div key={i} style={{ fontSize: 12, background: "rgba(0,0,0,0.2)", border: "1px solid #27293d", borderRadius: "8px", padding: 10 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                            <span style={{ color: issue.severity === "high" || issue.severity === "critical" ? "#f43f5e" : "#f59e0b", fontWeight: "bold", fontSize: 14 }}>●</span>
                            <span style={{ color: "#ffffff", fontWeight: "bold", fontSize: 11, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase" }}>{issue.type.replace(/_/g, ' ')}</span>
                          </div>
                          <div style={{ color: "#a0a5c0", marginBottom: 6 }}>{issue.message}</div>
                          <div style={{ fontSize: 11, color: accent, fontFamily: "'Comic Neue', cursive", fontWeight: "bold" }}>
                            💡 Suggestion: {recommendation}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Dataset Fingerprint Panel */}
              {result.fingerprint && (
                <div style={{
                  background: "#0c0d12",
                  border: "3px solid #000000",
                  borderRadius: "12px",
                  padding: "16px",
                  boxShadow: "4px 4px 0px 0px #000000",
                  display: "flex",
                  flexDirection: "column",
                  gap: 16
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, borderBottom: "2px solid #000000", paddingBottom: 8 }}>
                    <span style={{ fontSize: 18 }}>🧬</span>
                    <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Dataset Fingerprint
                    </span>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    <div style={{ background: "rgba(0,0,0,0.2)", border: "1px solid #27293d", borderRadius: "8px", padding: 8 }}>
                      <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase" }}>Duplicate Rate</span>
                      <span style={{ fontSize: 13, color: result.fingerprint.duplicate_rate > 0.1 ? "#f59e0b" : "#10b981", fontWeight: "bold", fontFamily: "'JetBrains Mono', monospace" }}>
                        {(result.fingerprint.duplicate_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ background: "rgba(0,0,0,0.2)", border: "1px solid #27293d", borderRadius: "8px", padding: 8 }}>
                      <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase" }}>Cluster Balance</span>
                      <span style={{ fontSize: 13, color: result.fingerprint.cluster_balance > 0.7 ? "#10b981" : result.fingerprint.cluster_balance > 0.4 ? "#f59e0b" : "#f43f5e", fontWeight: "bold", fontFamily: "'JetBrains Mono', monospace" }}>
                        {result.fingerprint.cluster_balance} <span style={{ fontSize: 10, color: "#a0a5c0", fontWeight: "normal" }}>/ 1.0</span>
                      </span>
                    </div>
                    <div style={{ background: "rgba(0,0,0,0.2)", border: "1px solid #27293d", borderRadius: "8px", padding: 8 }}>
                      <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase" }}>Contradictions</span>
                      <span style={{ fontSize: 13, color: result.fingerprint.contradictions_found > 0 ? "#f43f5e" : "#10b981", fontWeight: "bold", fontFamily: "'JetBrains Mono', monospace" }}>
                        {result.fingerprint.contradictions_found}
                      </span>
                    </div>
                    <div style={{ background: "rgba(0,0,0,0.2)", border: "1px solid #27293d", borderRadius: "8px", padding: 8 }}>
                      <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase" }}>Lexical Entropy</span>
                      <span style={{ fontSize: 13, color: result.fingerprint.starting_word_entropy > 3.0 ? "#10b981" : result.fingerprint.starting_word_entropy > 2.0 ? "#f59e0b" : "#f43f5e", fontWeight: "bold", fontFamily: "'JetBrains Mono', monospace" }}>
                        {result.fingerprint.starting_word_entropy}
                      </span>
                    </div>
                  </div>

                  {/* Subdomain representation */}
                  {result.fingerprint.subdomain_distribution && (
                    <div>
                      <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", marginBottom: 6 }}>Subdomain Distribution</span>
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        {Object.entries(result.fingerprint.subdomain_distribution).map(([sub, count]: [string, any]) => {
                          const percent = Math.round((count / result.row_count) * 100);
                          return (
                            <div key={sub}>
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                                <span style={{ color: "#ffffff" }}>{sub}</span>
                                <span style={{ color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace" }}>{count} rows ({percent}%)</span>
                              </div>
                              <div style={{ height: 6, background: "#000000", border: "1px solid #27293d", borderRadius: "3px", overflow: "hidden" }}>
                                <div style={{ height: "100%", background: accent, width: `${percent}%` }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Modes & Depths */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    {result.fingerprint.mode_distribution && (
                      <div>
                        <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", marginBottom: 6 }}>Generation Modes</span>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {Object.entries(result.fingerprint.mode_distribution).map(([m, c]: [string, any]) => (
                            <div key={m} style={{ fontSize: 9, background: "#1b1c2a", border: "1px solid #27293d", padding: "2px 6px", borderRadius: "4px", color: "#ffffff", fontFamily: "'JetBrains Mono', monospace" }}>
                              {m}: {c}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {result.fingerprint.reasoning_depth_distribution && (
                      <div>
                        <span style={{ display: "block", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", marginBottom: 6 }}>Reasoning Depths</span>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {Object.entries(result.fingerprint.reasoning_depth_distribution).map(([d, c]: [string, any]) => (
                            <div key={d} style={{ fontSize: 9, background: "#1b1c2a", border: "1px solid #27293d", padding: "2px 6px", borderRadius: "4px", color: "#ffffff", fontFamily: "'JetBrains Mono', monospace" }}>
                              {d}: {c}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <InteractiveDatasetEditor
                datasetId={result.id}
                initialRows={result.rows || result.sample_rows || []}
                datasetName={result.name}
                accent={accent}
              />

              <Link href="/dashboard" className="btn-primary" style={{ textDecoration: "none", textAlign: "center", display: "block", width: "100%", padding: "10px", fontSize: 11, marginBottom: 12 }}>
                View on Dashboard →
              </Link>
              <button onClick={() => { setStep("form"); setResult(null); setPreview([]); }}
                className="btn-ghost" style={{ width: "100%", padding: "10px", fontSize: 11 }}>
                Generate another dataset
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StructuredTab({ accent }: { accent: string }) {
  const {
    structForm: form,
    setStructForm: setForm,
    structColumns: columns,
    setStructColumns: setColumns,
    structResult: result,
    setStructResult: setResult,
    structIsStreaming: isStreaming,
    structStreamProgress: streamProgress,
    structError: error,
    setStructError: setError,
    structLoading: loading,
    setStructLoading: setLoading,
    structStep: step,
    setStructStep: setStep,
    startGenerateStructuredDataset,
  } = usePersistentState();

  const handleSuggestSchema = async () => {
    if (!form.description.trim()) { setError("Describe your dataset first"); return; }
    setLoading(true); setError("");
    try {
      const res = await api.suggestSchema(form.description, form.num_columns);
      // Phase 2 Schema Mapping: Initialize distribution_prior for all numeric columns
      setColumns((res.columns || []).map((c: any) => ({
        name: c.name,
        type: c.type || "string",
        description: c.description || "",
        range: c.range || null,
        values: c.values || null,
        distribution_prior: c.distribution_prior || ""
      })));
      setForm((prev: any) => ({
        ...prev,
        dataset_name: res.dataset_name || prev.dataset_name || "Structured Dataset"
      }));
      setStep("schema");
    } catch (e: any) {
      setError(e.message || "Failed to suggest schema");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (columns.length === 0) { setError("Schema must have at least 1 column"); return; }
    if (columns.some(c => !c.name.trim())) { setError("All columns must have a name"); return; }
    await startGenerateStructuredDataset();
  };


  const updateColumn = (index: number, key: string, value: any) => {
    const updated = [...columns];
    updated[index] = { ...updated[index], [key]: value };
    setColumns(updated);
  };

  const addColumn = () => {
    // Phase 2: Add column schema initialization with empty distribution_prior
    setColumns([...columns, { name: "", type: "string", description: "", range: "", values: [], distribution_prior: "" }]);
  };

  const removeColumn = (index: number) => {
    setColumns(columns.filter((_, i) => i !== index));
  };

  return (
    <div style={{ minHeight: 400 }}>
      {step === "form" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          {/* Left Form */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
              <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                Describe your dataset ML goal <span style={{ color: accent }}>*</span>
              </label>
              <textarea value={form.description} onChange={e => setForm({ ...form, description: e.target.value })}
                placeholder="e.g. house price prediction dataset for Indian real estate"
                rows={4}
                style={{ width: "100%", padding: "12px 16px", outline: "none", resize: "none" }}
                className="vt-input" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div>
                <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                  Suggest Columns: <span style={{ color: accent }}>{form.num_columns}</span>
                </label>
                <input type="range" min={2} max={15} step={1} value={form.num_columns}
                  onChange={e => setForm({ ...form, num_columns: parseInt(e.target.value) })}
                  style={{ width: "100%", accentColor: accent }} />
              </div>
              <div>
                <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                  Target Rows: <span style={{ color: accent }}>{form.total_rows}</span>
                </label>
                <input type="range" min={10} max={5000} step={1} value={form.total_rows}
                  onChange={e => setForm({ ...form, total_rows: parseInt(e.target.value) })}
                  style={{ width: "100%", accentColor: accent }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}>
                  <span>10</span><span>5000</span>
                </div>

              </div>
            </div>

            {/* Phase 2: Sliders for missingness rate and label noise rate */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div>
                <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                  Simulated Missingness (Nulls): <span style={{ color: accent }}>{(form.missingness_rate * 100).toFixed(0)}%</span>
                </label>
                <input type="range" min={0.0} max={0.20} step={0.01} value={form.missingness_rate}
                  onChange={e => setForm({ ...form, missingness_rate: parseFloat(e.target.value) })}
                  style={{ width: "100%", accentColor: accent }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}>
                  <span>0%</span><span>20%</span>
                </div>
              </div>
              <div>
                <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                  Simulated Label Noise: <span style={{ color: accent }}>{(form.label_noise_rate * 100).toFixed(0)}%</span>
                </label>
                <input type="range" min={0.0} max={0.15} step={0.01} value={form.label_noise_rate}
                  onChange={e => setForm({ ...form, label_noise_rate: parseFloat(e.target.value) })}
                  style={{ width: "100%", accentColor: accent }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}>
                  <span>0%</span><span>15%</span>
                </div>
              </div>
            </div>

            <div>
              <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                Domain Context <span style={{ color: "#6c7293", fontSize: "12px" }}>(highly recommended)</span>
              </label>
              <textarea value={form.domain_context} onChange={e => setForm({ ...form, domain_context: e.target.value })}
                placeholder="e.g. Indian real estate market. Mumbai and Bangalore are most expensive. Prices must correlate with area, bedrooms, location..."
                rows={4}
                style={{ width: "100%", padding: "12px 16px", outline: "none", resize: "none" }}
                className="vt-input" />
            </div>

            {error && (
              <div style={{
                fontSize: 12, color: "#ffffff", background: "#f43f5e", border: "3px solid #000000",
                boxShadow: "2px 2px 0px 0px #000000", padding: "10px 14px", borderRadius: "8px", fontFamily: "'JetBrains Mono', monospace"
              }}>
                {error}
              </div>
            )}

            <button onClick={handleSuggestSchema} disabled={loading || !form.description.trim()} className="btn-primary" style={{ padding: "12px 24px", width: "100%" }}>
              {loading ? (
                <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <span style={{ width: 12, height: 12, border: "1.5px solid #000", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                  Suggesting schema...
                </span>
              ) : "Suggest Schema ✨"}
            </button>
          </div>

          {/* Phase 2: Right Box / Stream Progress telemetry card */}
          {isStreaming ? (
            <div className="vt-card" style={{ display: "flex", flexDirection: "column", gap: 20, padding: 20 }}>
              <div style={{ display: "flex", alignItems: "center", justifyItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: accent, border: "1px solid #000", animation: "pulse 1.5s infinite", display: "inline-block" }} />
                  <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>
                    Tabular Synthesizer Active
                  </span>
                </div>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: accent, fontWeight: "bold" }}>
                  {streamProgress.generatedSoFar} / {form.total_rows} rows
                </span>
              </div>

              {/* Progress bar */}
              <div>
                <div style={{ height: 12, background: "#000000", border: "2px solid #000000", borderRadius: "10px", overflow: "hidden" }}>
                  <div style={{
                    height: "100%",
                    background: accent,
                    width: `${(streamProgress.generatedSoFar / form.total_rows) * 100}%`,
                    transition: "width 0.4s ease"
                  }} />
                </div>
              </div>

              {/* Telemetry logs */}
              <div>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 12, fontWeight: "bold", color: "#ffffff", marginBottom: 6, textTransform: "uppercase" }}>
                  Telemetry Log
                </div>
                <div style={{
                  background: "#050508",
                  border: "2px solid #000000",
                  boxShadow: "2px 2px 0px 0px #000000",
                  borderRadius: "8px",
                  padding: "10px",
                  height: "180px",
                  overflowY: "auto",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "10px",
                  color: "#10b981",
                  display: "flex",
                  flexDirection: "column",
                  gap: 4
                }}>
                  {streamProgress.logs.map((log: string, idx: number) => (
                    <div key={idx} style={{ lineHeight: 1.4 }}>
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: "40px" }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
              <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 20, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Structured Dataset Studio</div>
              <div style={{ fontSize: 13, color: "#a0a5c0", maxWidth: 320, lineHeight: 1.7 }}>
                Create custom tables for ML tasks like classification, regression, or clustering. Describe your goal, let the AI suggest a schema, tweak the columns, and synthesize realistic records.
              </div>
            </div>
          )}
        </div>
      )}

      {step === "schema" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {isStreaming ? (
            <div className="vt-card" style={{ display: "flex", flexDirection: "column", gap: 20, padding: 20, maxWidth: 600, margin: "0 auto", width: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", justifyItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: accent, border: "1px solid #000", animation: "pulse 1.5s infinite", display: "inline-block" }} />
                  <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>
                    Tabular Synthesizer Active
                  </span>
                </div>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: accent, fontWeight: "bold" }}>
                  {streamProgress.generatedSoFar} / {form.total_rows} rows
                </span>
              </div>

              {/* ☕ Coffee brewing banner */}
              <div style={{ background: "rgba(245,158,11,0.08)", border: "2px solid #f59e0b", borderRadius: 10, padding: "10px 14px", display: "flex", alignItems: "center", gap: 14 }}>
                <div style={{ width: 120, height: 120, flexShrink: 0 }}>
                  {/* @ts-ignore */}
                  <lottie-player
                    src="/brewing_2.json"
                    background="transparent"
                    speed="1.2"
                    style={{ width: "100%", height: "100%" }}
                    loop
                    autoplay
                  />
                </div>
                <div>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#fcd34d" }}>Grab a coffee — the structured dataset is brewing!</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#a0a5c0", marginTop: 2 }}>Synthesizing realistic table rows with normal, log-normal, and skewed distributions, missingness patterns, and target class imbalance.</div>
                </div>
              </div>

              {/* Progress bar */}
              <div>
                <div style={{ height: 12, background: "#000000", border: "2px solid #000000", borderRadius: "10px", overflow: "hidden" }}>
                  <div style={{
                    height: "100%",
                    background: accent,
                    width: `${(streamProgress.generatedSoFar / form.total_rows) * 100}%`,
                    transition: "width 0.4s ease"
                  }} />
                </div>
              </div>

              {/* Telemetry logs */}
              <div>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 12, fontWeight: "bold", color: "#ffffff", marginBottom: 6, textTransform: "uppercase" }}>
                  Telemetry Log
                </div>
                <div style={{
                  background: "#050508",
                  border: "2px solid #000000",
                  boxShadow: "2px 2px 0px 0px #000000",
                  borderRadius: "8px",
                  padding: "10px",
                  height: "240px",
                  overflowY: "auto",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "10px",
                  color: "#10b981",
                  display: "flex",
                  flexDirection: "column",
                  gap: 4
                }}>
                  {streamProgress.logs.map((log: string, idx: number) => (
                    <div key={idx} style={{ lineHeight: 1.4 }}>
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="vt-card" style={{ padding: 24 }}>
                <div style={{ marginBottom: 20 }}>
                  <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: "14px", fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>
                    Dataset Name
                  </label>
                  <input type="text" value={form.dataset_name} onChange={e => setForm({ ...form, dataset_name: e.target.value })}
                    style={{ width: "100%", maxWidth: 400, padding: "10px 14px", outline: "none" }}
                    className="vt-input" />
                </div>

                <h3 style={{ fontFamily: "'Comic Neue', cursive", fontSize: "16px", fontWeight: "bold", color: "#ffffff", marginBottom: 16 }}>
                  Customize Columns Schema ({columns.length} columns)
                </h3>

                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'Comic Neue', cursive", fontSize: "13px" }}>
                    <thead>
                      <tr style={{ borderBottom: "3px solid #000000" }}>
                        <th style={{ textAlign: "left", padding: "8px", color: "#a0a5c0" }}>Name</th>
                        <th style={{ textAlign: "left", padding: "8px", color: "#a0a5c0", width: 120 }}>Type</th>
                        <th style={{ textAlign: "left", padding: "8px", color: "#a0a5c0" }}>Description</th>
                        <th style={{ textAlign: "left", padding: "8px", color: "#a0a5c0", width: 140 }}>Distribution Prior</th>
                        <th style={{ textAlign: "left", padding: "8px", color: "#a0a5c0", width: 180 }}>Constraints (Range/Values)</th>
                        <th style={{ textAlign: "center", padding: "8px", color: "#a0a5c0", width: 50 }}>Delete</th>
                      </tr>
                    </thead>
                    <tbody>
                      {columns.map((col, idx) => (
                        <tr key={idx} style={{ borderBottom: "1px solid #27293d" }}>
                          <td style={{ padding: "8px" }}>
                            <input type="text" value={col.name} onChange={e => updateColumn(idx, "name", e.target.value)}
                              placeholder="e.g. house_price" style={{ width: "90%", padding: "6px 10px", outline: "none" }} className="vt-input" />
                          </td>
                          <td style={{ padding: "8px" }}>
                            <select value={col.type} onChange={e => updateColumn(idx, "type", e.target.value)}
                              style={{ width: "100%", padding: "6px 10px", outline: "none" }} className="vt-input">
                              {["string", "int", "float", "boolean", "category"].map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                          </td>
                          <td style={{ padding: "8px" }}>
                            <input type="text" value={col.description || ""} onChange={e => updateColumn(idx, "description", e.target.value)}
                              placeholder="Column description" style={{ width: "95%", padding: "6px 10px", outline: "none" }} className="vt-input" />
                          </td>
                          <td style={{ padding: "8px" }}>
                            {/* Phase 2: Select distribution prior (Gaussian normal, log-normal, poisson, exponential) for numeric variables */}
                            {(col.type === "int" || col.type === "float") ? (
                              <select value={col.distribution_prior || ""} onChange={e => updateColumn(idx, "distribution_prior", e.target.value || null)}
                                style={{ width: "100%", padding: "6px 10px", outline: "none" }} className="vt-input">
                                <option value="">default</option>
                                <option value="normal">normal</option>
                                <option value="log_normal">log_normal</option>
                                <option value="poisson">poisson</option>
                                <option value="exponential">exponential</option>
                                <option value="uniform">uniform</option>
                              </select>
                            ) : (
                              <span style={{ color: "#4f526b", fontSize: "11px", fontFamily: "'JetBrains Mono', monospace" }}>Not numeric</span>
                            )}
                          </td>
                          <td style={{ padding: "8px" }}>
                            {(col.type === "int" || col.type === "float") ? (
                              <input type="text" value={col.range || ""} onChange={e => updateColumn(idx, "range", e.target.value)}
                                placeholder="e.g. 1-10 or 300-5000" style={{ width: "90%", padding: "6px 10px", outline: "none" }} className="vt-input" />
                            ) : col.type === "category" ? (
                              <input type="text" value={Array.isArray(col.values) ? col.values.join(",") : (col.values || "")}
                                onChange={e => updateColumn(idx, "values", e.target.value.split(","))}
                                placeholder="e.g. Mumbai,Delhi,Pune" style={{ width: "90%", padding: "6px 10px", outline: "none" }} className="vt-input" />
                            ) : (
                              <span style={{ color: "#4f526b", fontSize: "11px", fontFamily: "'JetBrains Mono', monospace" }}>No parameters needed</span>
                            )}
                          </td>
                          <td style={{ padding: "8px", textAlign: "center" }}>
                            <button onClick={() => removeColumn(idx)} className="btn-ghost" style={{ padding: "4px 8px", borderRadius: "6px" }} title="Remove Column">
                              ❌
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <button onClick={addColumn} className="btn-ghost" style={{ marginTop: 16, fontSize: "11px", padding: "6px 14px" }}>
                  ➕ Add Column
                </button>
              </div>

              {error && (
                <div style={{
                  fontSize: 12, color: "#ffffff", background: "#f43f5e", border: "3px solid #000000",
                  boxShadow: "2px 2px 0px 0px #000000", padding: "10px 14px", borderRadius: "8px", fontFamily: "'JetBrains Mono', monospace"
                }}>
                  {error}
                </div>
              )}

              <div style={{ display: "flex", gap: 16 }}>
                <button onClick={() => setStep("form")} className="btn-ghost" style={{ padding: "12px 24px", flex: 1 }}>
                  ← Back to Prompt
                </button>
                <button onClick={handleGenerate} disabled={loading} className="btn-primary" style={{ padding: "12px 24px", flex: 2 }}>
                  {loading ? (
                    <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                      <span style={{ width: 12, height: 12, border: "1.5px solid #000", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                      Generating {form.total_rows} rows...
                    </span>
                  ) : `Generate ${form.total_rows} rows 🚀`}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {step === "result" && result && (() => {
        // Calculate neo-brutalist letter grades for structured dataset based on realism score
        const structuredScore = result.quality_score ?? 0;
        let structuredGrade = "F";
        let structuredLabel = "Critical issues";
        if (structuredScore >= 90) {
          structuredGrade = "A";
          structuredLabel = "Model Ready (Excellent)";
        } else if (structuredScore >= 80) {
          structuredGrade = "B";
          structuredLabel = "Good (Minor issues)";
        } else if (structuredScore >= 70) {
          structuredGrade = "C";
          structuredLabel = "Needs Polish (Moderate warnings)";
        } else if (structuredScore >= 60) {
          structuredGrade = "D";
          structuredLabel = "Weak (Repair recommended)";
        }
        const structuredColor = structuredScore >= 90 ? "#10b981" : structuredScore >= 70 ? "#f59e0b" : "#f43f5e";

        return (
          <div className="vt-card" style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#10b981", border: "1px solid #000", display: "inline-block" }} />
              <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#10b981" }}>
                Structured Dataset Synthesized successfully
              </span>
            </div>

            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
              {[
                ["rows", result.row_count, "#ffffff"],
                ["columns", result.columns?.length || 0, "var(--accent)"],
                ["quality score", `${result.quality_score}%`, structuredColor]
              ].map(([l, v, c]) => (
                <div key={String(l)} style={{ background: "#0c0d12", border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", borderRadius: "8px", padding: "16px 8px", textAlign: "center" }}>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 26, fontWeight: 900, color: String(c) }}>{v}</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0", marginTop: 4 }}>{l}</div>
                </div>
              ))}
            </div>

            {/* Quality Score Bar */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#a0a5c0", marginBottom: 6, fontWeight: "bold" }}>
                <span>Synthesizer Audit Quality Score</span>
                <span style={{ color: structuredColor }}>
                  {result.quality_score}%
                </span>
              </div>
              <div style={{ height: 12, background: "#000000", border: "2px solid #000000", borderRadius: "10px", overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  background: structuredColor,
                  width: `${result.quality_score}%`, transition: "width 1s"
                }} />
              </div>
            </div>

            {/* Phase 2: Tabular Realism & Readiness Grade Card */}
            <div style={{
              background: "#0c0d12",
              border: `3px solid #000000`,
              boxShadow: `4px 4px 0px 0px ${structuredColor}`,
              borderRadius: "12px",
              padding: "16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 16
            }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0" }}>
                  Tabular Realism & Readiness
                </span>
                <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>
                  {structuredLabel}
                </span>
                <span style={{ fontSize: 11, color: "#a0a5c0" }}>
                  Realism Score: {structuredScore}% (Missingness: {(form.missingness_rate * 100).toFixed(0)}%, Label Noise: {(form.label_noise_rate * 100).toFixed(0)}%)
                </span>
              </div>
              <div style={{
                width: 64,
                height: 64,
                borderRadius: "8px",
                background: structuredColor,
                border: "3px solid #000000",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "2px 2px 0px 0px #000000"
              }}>
                <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 36, fontWeight: 900, color: "#000000" }}>
                  {structuredGrade}
                </span>
              </div>
            </div>

            {/* Phase 2: Tabular Actionable Repair Suggestions */}
            {result.audit?.issues?.length > 0 && (
              <div style={{
                background: "#0c0d12",
                border: "3px solid #000000",
                borderRadius: "12px",
                padding: "16px",
                boxShadow: "3px 3px 0px 0px #000000"
              }}>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#ffffff", marginBottom: 10, textTransform: "uppercase", display: "flex", alignItems: "center", gap: 6 }}>
                  🔧 Tabular Actionable Repair Suggestions
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {result.audit.issues.map((issue: any, i: number) => {
                    let recommendation = issue.fix || "Check features and column specifications. Ensure target column is not leaked.";
                    if (issue.type === "target_leakage") {
                      recommendation = "Remove the leaking feature or aggregate it to prevent the model from learning a trivial mapping.";
                    } else if (issue.type === "perfect_correlation") {
                      recommendation = "Add non-linear relation specifications or inject noise to reduce exact linear dependency.";
                    } else if (issue.type === "low_variance") {
                      recommendation = "Increase standard deviation/variance or edit range constraints for this column.";
                    } else if (issue.type === "collapsed_category") {
                      recommendation = "Provide more balanced target examples or enforce uniform priors to broaden category spread.";
                    }
                    return (
                      <div key={i} style={{ fontSize: 12, background: "rgba(0,0,0,0.2)", border: "1px solid #27293d", borderRadius: "8px", padding: 10 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                          <span style={{ color: issue.severity === "critical" || issue.severity === "high" ? "#f43f5e" : "#f59e0b", fontWeight: "bold", fontSize: 14 }}>●</span>
                          <span style={{ color: "#ffffff", fontWeight: "bold", fontSize: 11, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase" }}>{issue.type.replace(/_/g, ' ')}</span>
                        </div>
                        <div style={{ color: "#a0a5c0", marginBottom: 6 }}>{issue.message}</div>
                        <div style={{ fontSize: 11, color: accent, fontFamily: "'Comic Neue', cursive", fontWeight: "bold" }}>
                          💡 Suggestion: {recommendation}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Preview Table */}
            {result.sample_rows && result.sample_rows.length > 0 && (
              <div>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#ffffff", marginBottom: 8, textTransform: "uppercase" }}>
                  Tabular Preview (5 sample rows):
                </div>
                <div style={{ overflowX: "auto", border: "2px solid #000000", borderRadius: "8px", boxShadow: "3px 3px 0px 0px #000000" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'Comic Neue', cursive", fontSize: "12px", background: "#0c0d12" }}>
                    <thead>
                      <tr style={{ borderBottom: "2px solid #000000", background: "rgba(255,255,255,0.02)" }}>
                        {result.columns.map((c: string) => (
                          <th key={c} style={{ padding: "8px 12px", textAlign: "left", color: "#ffffff", fontWeight: "bold", borderRight: "1px solid #27293d" }}>
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.sample_rows.map((row: any, rIdx: number) => (
                        <tr key={rIdx} style={{ borderBottom: rIdx === result.sample_rows.length - 1 ? "none" : "1px solid #27293d" }}>
                          {result.columns.map((c: string) => {
                            const val = row[c];
                            // Highlight simulated missing values (None / Null / Empty strings / "null")
                            const isNull = val === null || val === undefined || String(val).trim() === "" || String(val) === "null";
                            return (
                              <td key={c} style={{
                                padding: "8px 12px",
                                color: isNull ? "#f43f5e" : "#a0a5c0",
                                borderRight: "1px solid #27293d",
                                whiteSpace: "nowrap",
                                background: isNull ? "rgba(244, 63, 94, 0.08)" : "transparent"
                              }}>
                                {isNull ? (
                                  <span style={{
                                    fontFamily: "'JetBrains Mono', monospace",
                                    fontSize: 9,
                                    background: "#f43f5e",
                                    color: "#ffffff",
                                    padding: "2px 6px",
                                    borderRadius: "4px",
                                    fontWeight: "bold",
                                    border: "1.5px solid #000000",
                                    boxShadow: "1px 1px 0px #000000"
                                  }}>
                                    NaN / Null
                                  </span>
                                ) : String(val)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
              <div style={{ display: "flex", gap: 12 }}>
                <button onClick={() => downloadCSV(result.rows || result.sample_rows, result.name || "dataset")}
                  className="btn-primary" style={{ flex: 1, padding: "12px", fontSize: 12 }}>
                  ↓ Download CSV
                </button>
                <Link href="/dashboard" className="btn-primary" style={{ textDecoration: "none", textAlign: "center", display: "block", flex: 1, padding: "12px", fontSize: 12 }}>
                  View on Dashboard →
                </Link>
              </div>
              <button onClick={() => { setStep("form"); setResult(null); setColumns([]); setError(""); }}
                className="btn-ghost" style={{ width: "100%", padding: "10px", fontSize: 11 }}>
                Generate another structured dataset
              </button>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

function UploadTab({ accent }: { accent: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true); setError("");
    try { const res = await api.uploadDataset(file, name || file.name); setResult(res); }
    catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ maxWidth: 500, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
      <div onClick={() => document.getElementById("file-input")?.click()}
        style={{
          border: `3px dashed ${file ? "var(--accent)" : "#000000"}`,
          background: file ? "rgba(255, 92, 54, 0.04)" : "transparent",
          padding: 48,
          textAlign: "center",
          cursor: "pointer",
          transition: "all .2s",
          borderRadius: "16px",
          boxShadow: file ? "3px 3px 0px 0px var(--accent)" : "none"
        }}>
        <input id="file-input" type="file" accept=".csv,.jsonl" style={{ display: "none" }} onChange={e => setFile(e.target.files?.[0] || null)} />
        <div style={{ fontSize: 32, marginBottom: 12 }}>{file ? "📄" : "📁"}</div>
        <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff", marginBottom: 4 }}>
          {file ? file.name : "Click to upload dataset"}
        </div>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#a0a5c0" }}>.csv or .jsonl · max 100MB</div>
        {file && <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: accent, marginTop: 8, fontWeight: "bold" }}>{(file.size / 1024).toFixed(1)} KB</div>}
      </div>

      <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Dataset name"
        style={{ width: "100%", padding: "12px 16px", outline: "none" }}
        className="vt-input" />

      {error && (
        <div style={{
          fontSize: 12,
          color: "#ffffff",
          background: "#f43f5e",
          border: "3px solid #000000",
          boxShadow: "2px 2px 0px 0px #000000",
          padding: "10px 14px",
          borderRadius: "8px",
          fontFamily: "'JetBrains Mono', monospace"
        }}>
          {error}
        </div>
      )}

      <button onClick={handleUpload} disabled={!file || loading} className="btn-primary" style={{ padding: "12px", opacity: !file || loading ? .4 : 1 }}>
        {loading ? "Auditing dataset..." : "Upload & audit →"}
      </button>

      {result && (
        <div className="vt-card" style={{ padding: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
            <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 12, fontWeight: "bold", color: "#10b981" }}>Uploaded successfully</span>
          </div>
          <div style={{ fontSize: 14, color: "#ffffff", fontWeight: "bold", marginBottom: 4, fontFamily: "'Comic Neue', cursive" }}>{result.name}</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#a0a5c0", marginBottom: 16 }}>
            {result.audit_results?.row_count} rows · quality: <span style={{ color: "#10b981", fontWeight: "bold" }}>{result.quality_score}%</span>
          </div>
          <Link href="/dashboard" className="btn-primary" style={{ textDecoration: "none", textAlign: "center", display: "block", width: "100%", padding: "10px", fontSize: 11 }}>
            View on Dashboard →
          </Link>
        </div>
      )}
    </div>
  );
}

// ─── SCRAPE TAB ──────────────────────────────────────────────────────────────
function ScrapeTab({ accent }: { accent: string }) {
  const {
    scraperForm: form,
    setScraperForm: setForm,
    scraperIsStreaming: isStreaming,
    scraperLogs: logs,
    scraperResult: result,
    scraperError: error,
    setScraperError: setError,
    scraperLoading: loading,
    setScraperLoading: setLoading,
    scraperProgress: progress,
    startScraperIngest,
  } = usePersistentState();

  const handleScrape = async () => {
    if (!form.url.trim()) { setError("Enter a URL first"); return; }
    await startScraperIngest();
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
      {/* Left — form */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div style={{ background: "rgba(59,130,246,0.08)", border: "2px solid #3b82f6", borderRadius: 10, padding: "10px 14px", fontSize: 12, color: "#93c5fd", lineHeight: 1.6 }}>
          📡 Enter any webpage URL or sitemap URL. We'll extract the text, then use AI to convert it into clean {"{"}input, output{"}"}  training pairs.
        </div>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>URL(s) to scrape <span style={{ color: accent }}>*</span></label>
          <textarea
            value={form.url}
            onChange={e => setForm({ ...form, url: e.target.value })}
            placeholder="Enter URL(s), one per line or comma-separated:&#10;https://docs.yourproduct.com&#10;https://yoursite.com/sitemap.xml"
            disabled={loading}
            rows={4}
            style={{ width: "100%", padding: "12px 16px", outline: "none", resize: "vertical", minHeight: "100px", fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}
            className="vt-input"
          />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Task Type</label>
            <select value={form.task_type} onChange={e => setForm({ ...form, task_type: e.target.value })} disabled={loading} style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }} className="vt-input">
              {TASK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Style</label>
            <select value={form.style} onChange={e => setForm({ ...form, style: e.target.value })} disabled={loading} style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }} className="vt-input">
              {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Max Pages: <span style={{ color: accent }}>{form.max_pages}</span></label>
            <input type="range" min={1} max={50} value={form.max_pages} disabled={loading} onChange={e => setForm({ ...form, max_pages: parseInt(e.target.value) })} style={{ width: "100%", accentColor: accent }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}><span>1</span><span>50</span></div>
          </div>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Pairs per chunk: <span style={{ color: accent }}>{form.pairs_per_chunk}</span></label>
            <input type="range" min={1} max={10} value={form.pairs_per_chunk} disabled={loading} onChange={e => setForm({ ...form, pairs_per_chunk: parseInt(e.target.value) })} style={{ width: "100%", accentColor: accent }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}><span>1</span><span>10</span></div>
          </div>
        </div>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Dataset Name</label>
          <input type="text" value={form.dataset_name} onChange={e => setForm({ ...form, dataset_name: e.target.value })} disabled={loading} placeholder={form.url ? `Scraped: ${form.url.slice(0, 30)}...` : "My scraped dataset"} style={{ width: "100%", padding: "12px 16px", outline: "none" }} className="vt-input" />
        </div>
        {error && (<div style={{ fontSize: 12, color: "#ffffff", background: "#f43f5e", border: "3px solid #000000", boxShadow: "2px 2px 0px 0px #000000", padding: "10px 14px", borderRadius: "8px", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>)}
        <button onClick={handleScrape} disabled={loading || !form.url.trim()} className="btn-primary" style={{ padding: "12px 24px" }}>
          {loading ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}><span style={{ width: 12, height: 12, border: "1.5px solid #000", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} /> Scraping...</span> : "🌐 Scrape & Generate →"}
        </button>
      </div>

      {/* Right — progress + result */}
      <div>
        {!isStreaming && !result && (
          <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 40 }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🌐</div>
            <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 20, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Web Scraper</div>
            <div style={{ fontSize: 13, color: "#a0a5c0", maxWidth: 280, lineHeight: 1.7 }}>Point us at any docs site, blog, or knowledge base. We extract the text and convert it to training pairs grounded in your content.</div>
          </div>
        )}
        {(isStreaming || (logs.length > 0 && !result)) && (
          <div className="vt-card" style={{ display: "flex", flexDirection: "column", gap: 16, padding: 20 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: accent, border: "1px solid #000", animation: "pulse 1.5s infinite", display: "inline-block" }} />
                <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>Scraper Active</span>
              </div>
              <div style={{ display: "flex", gap: 16, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                <span style={{ color: accent }}>{progress.pages} pages</span>
                <span style={{ color: "#a0a5c0" }}>{progress.chunks} chunks</span>
                {progress.pairs > 0 && <span style={{ color: "#10b981" }}>{progress.pairs} pairs</span>}
              </div>
            </div>

            {/* ☕ Coffee brewing banner */}
            <div style={{ background: "rgba(245,158,11,0.08)", border: "2px solid #f59e0b", borderRadius: 10, padding: "10px 14px", display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ width: 120, height: 120, flexShrink: 0 }}>
                {/* @ts-ignore */}
                <lottie-player
                  src="/brewing_2.json"
                  background="transparent"
                  speed="1.2"
                  style={{ width: "100%", height: "100%" }}
                  loop
                  autoplay
                />
              </div>
              <div>
                <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#fcd34d" }}>Grab a coffee — the scraped dataset is brewing!</div>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#a0a5c0", marginTop: 2 }}>Crawling pages, extracting main content, and running concurrent LLM restructurers.</div>
              </div>
            </div>

            <div style={{ background: "#050508", border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", borderRadius: 8, padding: 10, height: 200, overflowY: "auto", fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#10b981", display: "flex", flexDirection: "column", gap: 4 }}>
              {logs.map((log, idx) => <div key={idx} style={{ lineHeight: 1.4 }}>{log}</div>)}
            </div>
          </div>
        )}
        {result && (
          <div className="vt-card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
              <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#10b981" }}>Scrape Complete</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
              {[["rows", result.row_count, "#ffffff"], ["pages", progress.pages, accent], ["chunks", progress.chunks, "#a0a5c0"]].map(([l, v, c]) => (
                <div key={String(l)} style={{ background: "#0c0d12", border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", borderRadius: 8, padding: "12px 8px", textAlign: "center" }}>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 22, fontWeight: 900, color: String(c) }}>{v}</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0", marginTop: 4 }}>{l}</div>
                </div>
              ))}
            </div>
            <InteractiveDatasetEditor
              datasetId={result.id}
              initialRows={result.rows || result.sample_rows || []}
              datasetName={result.name}
              accent={accent}
            />
            <Link href="/dashboard" className="btn-primary" style={{ textDecoration: "none", textAlign: "center", display: "block", width: "100%", padding: "10px", fontSize: 11 }}>View on Dashboard →</Link>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── PDF TAB ─────────────────────────────────────────────────────────────────
function PDFTab({ accent }: { accent: string }) {
  const [form, setForm] = useState({ task_type: "qa", style: "formal", pairs_per_chunk: 3, dataset_name: "" });
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<any>(null);
  const [dragging, setDragging] = useState(false);

  const handleFileSelect = (f: File) => {
    if (!f.name.toLowerCase().endsWith(".pdf")) { setError("Only PDF files supported"); return; }
    if (f.size > 20 * 1024 * 1024) { setError("PDF must be under 20MB"); return; }
    setFile(f); setError("");
  };

  const handleUpload = async () => {
    if (!file) { setError("Select a PDF first"); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const token = auth.getToken();
      const fd = new FormData();
      fd.append("pdf_file", file);
      fd.append("task_type", form.task_type);
      fd.append("style", form.style);
      fd.append("pairs_per_chunk", String(form.pairs_per_chunk));
      fd.append("dataset_name", form.dataset_name || `PDF: ${file.name}`);
      const response = await fetch(`${API_BASE}/ingest/pdf`, {
        method: "POST",
        headers: { "Authorization": token ? `Bearer ${token}` : "" },
        body: fd
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "PDF upload failed");
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
      {/* Left */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div style={{ background: "rgba(139,92,246,0.08)", border: "2px solid #8b5cf6", borderRadius: 10, padding: "10px 14px", fontSize: 12, color: "#c4b5fd", lineHeight: 1.6 }}>
          📄 Upload a PDF — manuals, research papers, legal docs, internal wikis. We extract the text page by page and convert it to training pairs.
        </div>
        {/* Drop zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFileSelect(f); }}
          onClick={() => { const inp = document.getElementById("pdf-file-input") as HTMLInputElement; inp?.click(); }}
          style={{ border: `3px dashed ${dragging ? accent : "#27293d"}`, borderRadius: 12, padding: "32px 20px", textAlign: "center", cursor: "pointer", background: dragging ? "rgba(255,92,54,0.05)" : "#0c0d12", transition: "all 0.2s" }}
        >
          <input id="pdf-file-input" type="file" accept=".pdf" style={{ display: "none" }} onChange={e => { const f = e.target.files?.[0]; if (f) handleFileSelect(f); }} />
          {file ? (
            <div>
              <div style={{ fontSize: 28, marginBottom: 8 }}>📄</div>
              <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 4 }}>{file.name}</div>
              <div style={{ fontSize: 11, color: "#a0a5c0", fontFamily: "'JetBrains Mono', monospace" }}>{(file.size / 1024).toFixed(1)} KB</div>
              <div style={{ fontSize: 11, color: accent, marginTop: 8 }}>Click to change file</div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 40, marginBottom: 12 }}>📤</div>
              <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff", marginBottom: 6 }}>Drop PDF here or click to browse</div>
              <div style={{ fontSize: 11, color: "#a0a5c0" }}>Max 20MB · PDF only</div>
            </div>
          )}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Task Type</label>
            <select value={form.task_type} onChange={e => setForm({ ...form, task_type: e.target.value })} disabled={loading} style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }} className="vt-input">
              {TASK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Style</label>
            <select value={form.style} onChange={e => setForm({ ...form, style: e.target.value })} disabled={loading} style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }} className="vt-input">
              {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Pairs per page chunk: <span style={{ color: accent }}>{form.pairs_per_chunk}</span></label>
          <input type="range" min={1} max={10} value={form.pairs_per_chunk} disabled={loading} onChange={e => setForm({ ...form, pairs_per_chunk: parseInt(e.target.value) })} style={{ width: "100%", accentColor: accent }} />
        </div>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Dataset Name</label>
          <input type="text" value={form.dataset_name} onChange={e => setForm({ ...form, dataset_name: e.target.value })} disabled={loading} placeholder={file ? `PDF: ${file.name}` : "My PDF dataset"} style={{ width: "100%", padding: "12px 16px", outline: "none" }} className="vt-input" />
        </div>
        {error && (<div style={{ fontSize: 12, color: "#ffffff", background: "#f43f5e", border: "3px solid #000000", boxShadow: "2px 2px 0px 0px #000000", padding: "10px 14px", borderRadius: "8px", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>)}
        <button onClick={handleUpload} disabled={loading || !file} className="btn-primary" style={{ padding: "12px 24px" }}>
          {loading ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}><span style={{ width: 12, height: 12, border: "1.5px solid #000", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} /> Processing PDF...</span> : "📄 Extract & Generate →"}
        </button>
      </div>

      {/* Right */}
      <div>
        {!result && !loading && (
          <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 40 }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>📄</div>
            <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 20, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>PDF Ingestion</div>
            <div style={{ fontSize: 13, color: "#a0a5c0", maxWidth: 280, lineHeight: 1.7 }}>Upload documentation, research papers, or any PDF. We extract text page by page and generate grounded training pairs — no hallucination.</div>
          </div>
        )}
        {loading && (
          <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 40, gap: 16 }}>
            <div style={{ width: 180, height: 180 }}>
              {/* @ts-ignore */}
              <lottie-player
                src="/brewing_2.json"
                background="transparent"
                speed="1.2"
                style={{ width: "100%", height: "100%" }}
                loop
                autoplay
              />
            </div>
            <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>Extracting PDF & generating pairs...</div>
            <div style={{ fontSize: 12, color: "#a0a5c0" }}>This may take 30–90 seconds depending on PDF size.</div>
          </div>
        )}
        {result && (
          <div className="vt-card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
              <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#10b981" }}>PDF Extracted Successfully</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
              {[["pairs", result.row_count, "#ffffff"], ["pages", result.extracted_pages, accent], ["skipped", result.skipped_pages, "#a0a5c0"]].map(([l, v, c]) => (
                <div key={String(l)} style={{ background: "#0c0d12", border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", borderRadius: 8, padding: "12px 8px", textAlign: "center" }}>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 22, fontWeight: 900, color: String(c) }}>{v}</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0", marginTop: 4 }}>{l}</div>
                </div>
              ))}
            </div>
            <InteractiveDatasetEditor
              datasetId={result.id}
              initialRows={result.rows || result.sample_rows || []}
              datasetName={result.name}
              accent={accent}
            />
            <Link href="/dashboard" className="btn-primary" style={{ textDecoration: "none", textAlign: "center", display: "block", width: "100%", padding: "10px", fontSize: 11 }}>View on Dashboard →</Link>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── MCP TAB ─────────────────────────────────────────────────────────────────
function MCPTab({ accent }: { accent: string }) {
  const [serverUrl, setServerUrl] = useState("");
  const [authToken, setAuthToken] = useState("");
  const [form, setForm] = useState({ task_type: "qa", style: "formal", pairs_per_chunk: 3, dataset_name: "" });
  const [resources, setResources] = useState<any[]>([]);
  const [selectedUris, setSelectedUris] = useState<Set<string>>(new Set());
  const [browsing, setBrowsing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<any>(null);
  const [hoverIngest, setHoverIngest] = useState(false);

  const handleBrowse = async () => {
    if (!serverUrl.trim()) { setError("Enter an MCP server URL first"); return; }
    setBrowsing(true); setError(""); setResources([]); setSelectedUris(new Set());
    try {
      const token = auth.getToken();
      const res = await fetch(`${API_BASE}/ingest/mcp/resources`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": token ? `Bearer ${token}` : "" },
        body: JSON.stringify({ server_url: serverUrl, auth_token: authToken || null })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not connect to MCP server");
      setResources(data.resources || []);
      if (!data.resources?.length) setError("No resources found on this MCP server.");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBrowsing(false);
    }
  };

  const toggleResource = (uri: string) => {
    setSelectedUris(prev => { const n = new Set(prev); n.has(uri) ? n.delete(uri) : n.add(uri); return n; });
  };

  const handleIngest = async () => {
    if (selectedUris.size === 0) { setError("Select at least one resource"); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const token = auth.getToken();
      const res = await fetch(`${API_BASE}/ingest/mcp/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": token ? `Bearer ${token}` : "" },
        body: JSON.stringify({ server_url: serverUrl, auth_token: authToken || null, resource_uris: [...selectedUris], ...form })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "MCP ingest failed");
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
      {/* Left */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div style={{ background: "rgba(16,185,129,0.08)", border: "2px solid #10b981", borderRadius: 10, padding: "10px 14px", fontSize: 12, color: "#6ee7b7", lineHeight: 1.6 }}>
          🔌 Connect any MCP server (Notion, GitHub, Postgres, Slack, Google Drive). Browse its resources and ingest selected ones as training data.
        </div>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>MCP Server URL <span style={{ color: accent }}>*</span></label>
          <input type="url" value={serverUrl} onChange={e => setServerUrl(e.target.value)}
            placeholder="http://localhost:3001/mcp or https://your-mcp.server.com/mcp"
            disabled={loading || browsing} style={{ width: "100%", padding: "12px 16px", outline: "none" }} className="vt-input" />
        </div>
        <div>
          <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Auth Token <span style={{ color: "#6c7293", fontSize: 12 }}>(optional)</span></label>
          <input type="password" value={authToken} onChange={e => setAuthToken(e.target.value)}
            placeholder="Bearer token if your MCP server requires auth"
            disabled={loading || browsing} style={{ width: "100%", padding: "12px 16px", outline: "none" }} className="vt-input" />
        </div>
        <button onClick={handleBrowse} disabled={browsing || loading || !serverUrl.trim()} className="btn-ghost" style={{ padding: "10px 24px" }}>
          {browsing ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}><span style={{ width: 12, height: 12, border: "1.5px solid #fff", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} /> Connecting...</span> : "🔍 Browse Resources"}
        </button>

        {resources.length > 0 && (
          <div style={{ background: "#06070a", border: "2px solid #27293d", borderRadius: 8, padding: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 13, fontWeight: "bold", color: "#ffffff", display: "flex", alignItems: "center", gap: 6 }}>
                📁 Selected Resources ({selectedUris.size} / {resources.length})
              </span>
              <button type="button" onClick={() => setSelectedUris(selectedUris.size === resources.length ? new Set() : new Set(resources.map((r: any) => r.uri)))} style={{ fontSize: 11, color: accent, background: "none", border: "none", cursor: "pointer", fontFamily: "'Comic Neue', cursive", fontWeight: "bold" }}>
                {selectedUris.size === resources.length ? "Deselect all" : "Select all"}
              </button>
            </div>
            <div style={{ maxHeight: 200, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6, paddingRight: 4 }}>
              {resources.map((r: any) => (
                <div key={r.uri} onClick={() => toggleResource(r.uri)}
                  style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "8px 10px", background: selectedUris.has(r.uri) ? "rgba(255,92,54,0.1)" : "#0c0d12", border: `2px solid ${selectedUris.has(r.uri) ? accent : "#27293d"}`, borderRadius: 8, cursor: "pointer", transition: "all 0.15s" }}>
                  <span style={{ width: 16, height: 16, border: `2px solid ${selectedUris.has(r.uri) ? accent : "#555"}`, borderRadius: 4, background: selectedUris.has(r.uri) ? accent : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>
                    {selectedUris.has(r.uri) && <span style={{ color: "#000", fontSize: 10, fontWeight: "bold" }}>✓</span>}
                  </span>
                  <div>
                    <div style={{ fontSize: 12, color: "#ffffff", fontWeight: "bold", fontFamily: "'Comic Neue', cursive" }}>{r.name}</div>
                    {r.description && <div style={{ fontSize: 11, color: "#a0a5c0", marginTop: 2 }}>{r.description}</div>}
                    <div style={{ fontSize: 9, color: "#555", fontFamily: "'JetBrains Mono', monospace", marginTop: 2 }}>{r.uri}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedUris.size > 0 && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div>
              <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Task Type</label>
              <select value={form.task_type} onChange={e => setForm({ ...form, task_type: e.target.value })} disabled={loading} style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }} className="vt-input">
                {TASK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div>
              <label style={{ display: "block", fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>Style</label>
              <select value={form.style} onChange={e => setForm({ ...form, style: e.target.value })} disabled={loading} style={{ width: "100%", padding: "12px 16px", outline: "none", cursor: "pointer" }} className="vt-input">
                {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>
        )}

        {error && (<div style={{ fontSize: 12, color: "#ffffff", background: "#f43f5e", border: "3px solid #000000", boxShadow: "2px 2px 0px 0px #000000", padding: "10px 14px", borderRadius: "8px", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>)}

        {selectedUris.size > 0 && (
          <div style={{ position: "relative", display: "inline-block", alignSelf: "flex-start" }}>
            <button
              onClick={handleIngest}
              disabled={loading}
              onMouseEnter={() => setHoverIngest(true)}
              onMouseLeave={() => setHoverIngest(false)}
              className="btn-primary"
              style={{ padding: "8px 16px", fontSize: "11px" }}
            >
              {loading ? (
                <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <span style={{ width: 12, height: 12, border: "1.5px solid #000", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                  Ingesting...
                </span>
              ) : (
                `🔌 Ingest ${selectedUris.size} resource${selectedUris.size > 1 ? "s" : ""} →`
              )}
            </button>
            {hoverIngest && (
              <div style={{
                position: "absolute",
                bottom: "calc(100% + 8px)",
                left: 0,
                zIndex: 50,
                background: "#08080c",
                border: "2px solid var(--accent)",
                borderRadius: 8,
                padding: "8px 12px",
                width: 220,
                fontSize: 10,
                color: "#a0a5c0",
                lineHeight: 1.4,
                boxShadow: "4px 4px 0px 0px #000000",
                pointerEvents: "none"
              }}>
                Starts AI ingestion of the selected MCP server resources. We'll parse their text and convert them into clean fine-tuning pairs.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right */}
      <div>
        {!result && !loading && resources.length === 0 && (
          <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 40 }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🔌</div>
            <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 20, fontWeight: "bold", color: "#ffffff", marginBottom: 8 }}>MCP Source</div>
            <div style={{ fontSize: 13, color: "#a0a5c0", maxWidth: 280, lineHeight: 1.7 }}>Connect any MCP-compatible server — Notion, GitHub, Postgres, Slack — and convert its data into fine-tuning pairs.</div>
            <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 8, width: "100%", maxWidth: 260 }}>
              {["Notion pages → chatbot training", "GitHub files → code assistant", "Postgres records → domain model", "Slack logs → conversation data"].map(ex => (
                <div key={ex} style={{ fontSize: 11, color: "#a0a5c0", background: "#0c0d12", border: "1px solid #27293d", borderRadius: 6, padding: "6px 10px", textAlign: "left" }}>→ {ex}</div>
              ))}
            </div>
          </div>
        )}
        {loading && (
          <div className="vt-card" style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 40, gap: 16 }}>
            <div style={{ width: 180, height: 180 }}>
              {/* @ts-ignore */}
              <lottie-player
                src="/brewing_2.json"
                background="transparent"
                speed="1.2"
                style={{ width: "100%", height: "100%" }}
                loop
                autoplay
              />
            </div>
            <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>Reading MCP resources...</div>
            <div style={{ fontSize: 12, color: "#a0a5c0" }}>Converting {selectedUris.size} resource{selectedUris.size > 1 ? "s" : ""} to training pairs.</div>
          </div>
        )}
        {result && (
          <div className="vt-card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
              <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#10b981" }}>MCP Ingest Complete</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
              {[["pairs", result.row_count, "#ffffff"], ["resources", result.resources_read, accent], ["segments", result.text_segments, "#a0a5c0"]].map(([l, v, c]) => (
                <div key={String(l)} style={{ background: "#0c0d12", border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", borderRadius: 8, padding: "12px 8px", textAlign: "center" }}>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 22, fontWeight: 900, color: String(c) }}>{v}</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, letterSpacing: ".1em", textTransform: "uppercase", color: "#a0a5c0", marginTop: 4 }}>{l}</div>
                </div>
              ))}
            </div>
            <InteractiveDatasetEditor
              datasetId={result.id}
              initialRows={result.rows || result.sample_rows || []}
              datasetName={result.name}
              accent={accent}
            />
            <Link href="/dashboard" className="btn-primary" style={{ textDecoration: "none", textAlign: "center", display: "block", width: "100%", padding: "10px", fontSize: 11 }}>View on Dashboard →</Link>
          </div>
        )}
      </div>
    </div>
  );
}

function GeneratePageInner() {
  const router = useRouter();
  const params = useSearchParams();
  const {
    setGenForm,
    setStructForm,
    setStructColumns,
    setScraperForm,
  } = usePersistentState();

  const [tab, setTab] = useState(
    params.get("tab") === "upload"
      ? "upload"
      : params.get("tab") === "structured"
      ? "structured"
      : "generate"
  );
  const [accent, setAccent] = useState("#ff5c36");

  useEffect(() => {
    if (!auth.isLoggedIn()) { router.push("/auth"); return; }
    const a = document.documentElement.style.getPropertyValue("--accent");
    if (a) setAccent(a);
  }, []);

  return (
    <div className="sketchy-grid" style={{ minHeight: "calc(100vh - 56px)", background: "#0f0f12" }}>
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        select option{background:#0c0d12}
        input:focus,textarea:focus,select:focus{border-color:var(--accent) !important;box-shadow: 2px 2px 0px 0px #000000 !important;}
        .tab-btn {
          font-family: 'Comic Neue', 'Comic Sans MS', cursive, sans-serif !important;
          font-weight: 900 !important;
          text-transform: uppercase !important;
          letter-spacing: .05em !important;
          transition: transform 0.1s ease, box-shadow 0.1s ease, border-radius 0.1s ease, background-color 0.15s ease, color 0.15s ease !important;
          cursor: pointer !important;
          padding: 6px 12px !important;
          font-size: 10px !important;
          border-radius: 120px 15px 100px 15px/15px 100px 15px 120px !important;
        }
        .tab-btn-active {
          background: var(--accent) !important;
          color: #000000 !important;
          border: 2px solid #000000 !important;
          box-shadow: 2px 2px 0px 0px #000000 !important;
        }
        .tab-btn-inactive {
          background: #12131c !important;
          color: #a0a5c0 !important;
          border: 2px solid #000000 !important;
          box-shadow: 1px 1px 0px 0px #000000 !important;
        }
        .tab-btn:hover {
          transform: translate(-1px, -1px) !important;
          border-radius: 15px 100px 15px 120px/120px 15px 100px 15px !important;
        }
        .tab-btn-active:hover {
          box-shadow: 3px 3px 0px 0px #000000 !important;
          background: #ff7554 !important;
        }
        .tab-btn-inactive:hover {
          box-shadow: 2px 2px 0px 0px #000000 !important;
          background: #1c1d29 !important;
          color: #ffffff !important;
        }
        .tab-btn:active {
          transform: translate(1px, 1px) !important;
          box-shadow: 0px 0px 0px 0px #000000 !important;
        }
      ` }} />
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "36px 52px" }}>
        
        {/* Title Section */}
        <div style={{ marginBottom: 24 }}>
          <h1 className="misaligned-text" data-text="Dataset Studio" style={{
            fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
            fontSize: "36px",
            fontWeight: 900,
            color: "#ffffff",
            marginBottom: 6
          }}>
            Dataset Studio
          </h1>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "11px",
            letterSpacing: ".05em",
            textTransform: "uppercase",
            color: "#a0a5c0"
          }}>
            Generate clean training data or upload your own
          </div>
        </div>

        {/* Tab Selector */}
        <div style={{ display: "flex", gap: 12, marginBottom: 28, flexWrap: "wrap" }}>
          {[
            { id: "generate", label: "⚗️ Text/QA" },
            { id: "structured", label: "📊 Structured" },
            { id: "upload", label: "📁 Upload" },
            { id: "scrape", label: "🌐 Scrape URL" },
            { id: "pdf", label: "📄 PDF" },
            { id: "mcp", label: "🔌 MCP Source" },
          ].map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`tab-btn ${tab === t.id ? "tab-btn-active" : "tab-btn-inactive"}`}>
              {t.label}
            </button>
          ))}
        </div>

        {tab === "generate" ? (
          <GenerateTab accent={accent} />
        ) : tab === "structured" ? (
          <StructuredTab accent={accent} />
        ) : tab === "scrape" ? (
          <ScrapeTab accent={accent} />
        ) : tab === "pdf" ? (
          <PDFTab accent={accent} />
        ) : tab === "mcp" ? (
          <MCPTab accent={accent} />
        ) : (
          <UploadTab accent={accent} />
        )}
      </div>
      <Script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js" strategy="lazyOnload" />
    </div>
  );
}

export default function GeneratePage() {
  return <Suspense><GeneratePageInner /></Suspense>;
}

"use client";
import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, auth } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

function scoreColor(s: number) { return s >= 90 ? "#10b981" : s >= 70 ? "#f59e0b" : "#f43f5e"; }

function Badge({ status }: { status: string }) {
  const map: Record<string, string> = {
    ready:       "#10b981",
    completed:   "#10b981",
    training:    "#ff5c36",
    queued:      "#f59e0b",
    failed:      "#f43f5e",
    uploading:   "#06b6d4",
  };
  const color = map[status] || "#a0a5c0";
  return (
    <span style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: "9px",
      fontWeight: "bold",
      textTransform: "uppercase",
      color: "#000000",
      background: color,
      border: "2px solid #000000",
      boxShadow: "1px 1px 0px 0px #000000",
      padding: "2px 8px",
      borderRadius: "4px"
    }}>
      {status}
    </span>
  );
}

function downloadCSV(datasetId: string, name: string, token: string) {
  fetch(`${API_BASE}/generate/download/${datasetId}?format=csv`, {
    headers: { Authorization: `Bearer ${token}` }
  }).then(r => r.blob()).then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${name}.csv`; a.click();
    URL.revokeObjectURL(url);
  });
}

export default function Dashboard() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<any[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [evaluations, setEvaluations] = useState<any[]>([]);
  const [dashboardTab, setDashboardTab] = useState<"overview" | "lineage" | "admin">("overview");
  const [adminStats, setAdminStats] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [accent, setAccent] = useState("#ff5c36");
  
  const [showAllModal, setShowAllModal] = useState(false);
  const [previewDataset, setPreviewDataset] = useState<any | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const [previewJob, setPreviewJob] = useState<any | null>(null);
  const [previewJobLoading, setPreviewJobLoading] = useState(false);

  const handlePreviewJob = async (id: string) => {
    setPreviewJobLoading(true);
    try {
      const details = await api.getJob(id);
      setPreviewJob(details);
    } catch (err: any) {
      alert(err.message || "Failed to load job details for preview");
    } finally {
      setPreviewJobLoading(false);
    }
  };

  const handleDeleteDataset = async (id: string) => {
    if (!confirm("Are you sure you want to permanently delete this dataset? This will remove it from the database.")) return;
    try {
      await api.deleteDataset(id);
      await fetchData();
      if (previewDataset?.id === id) {
        setPreviewDataset(null);
      }
    } catch (err: any) {
      alert(err.message || "Failed to delete dataset");
    }
  };

  const handleDeleteJob = async (id: string) => {
    if (!confirm("Are you sure you want to permanently delete this training job? This will also remove any associated deployment and evaluation records from the database.")) return;
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
      await fetchData();
      if (previewJob?.id === id) {
        setPreviewJob(null);
      }
    } catch (err: any) {
      alert(err.message || "Failed to delete training job");
    }
  };

  const handlePreviewDataset = async (id: string) => {
    setPreviewLoading(true);
    try {
      const details = await api.getDataset(id);
      setPreviewDataset(details);
    } catch (err: any) {
      alert(err.message || "Failed to load dataset details for preview");
    } finally {
      setPreviewLoading(false);
    }
  };

  const fetchData = useCallback(async () => {
    setError("");
    try {
      const [d, j, e] = await Promise.all([
        api.listDatasets(),
        api.listJobs(),
        api.listEvaluations().catch(() => [])
      ]);
      setDatasets(d);
      setJobs(j);
      setEvaluations(e);

      if (auth.isAdmin()) {
        try {
          const aStats = await api.getAdminStats();
          setAdminStats(aStats);
        } catch (err) {
          console.error("Failed to load admin stats:", err);
        }
      }
    } catch (err: any) {
      setError(err.message || "Failed to load dashboard data");
    }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    if (!auth.isLoggedIn()) { router.push("/auth"); return; }
    const a = document.documentElement.style.getPropertyValue("--accent");
    if (a) setAccent(a);
    fetchData();
  }, []);

  useEffect(() => {
    const onFocus = () => { if (auth.isLoggedIn()) fetchData(); };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [fetchData]);

  const token = auth.getToken() || "";

  return (
    <div className="sketchy-grid" style={{ minHeight: "calc(100vh - 56px)", background: "#0f0f12" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "36px 52px" }}>
        
        {/* Title Section */}
        <div style={{ marginBottom: 32 }}>
          <h1 className="misaligned-text" data-text="Dashboard" style={{
            fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
            fontSize: "36px",
            fontWeight: 900,
            color: "#ffffff",
            marginBottom: 6
          }}>
            Dashboard
          </h1>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "11px",
            letterSpacing: ".05em",
            textTransform: "uppercase",
            color: "#a0a5c0"
          }}>
            Your fine-tuning workspace
          </div>
        </div>

        {/* Tab Switcher */}
        <div style={{ display: "flex", gap: 16, borderBottom: "2px solid #27293d", marginBottom: 24, paddingBottom: 8 }}>
          <button
            onClick={() => setDashboardTab("overview")}
            style={{
              background: "none",
              border: "none",
              fontFamily: "'Comic Neue', cursive",
              fontSize: "14px",
              fontWeight: "bold",
              color: dashboardTab === "overview" ? accent : "#7a7f9a",
              borderBottom: dashboardTab === "overview" ? `3px solid ${accent}` : "3px solid transparent",
              padding: "4px 12px",
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            📊 Activity Overview
          </button>
          <button
            onClick={() => setDashboardTab("lineage")}
            style={{
              background: "none",
              border: "none",
              fontFamily: "'Comic Neue', cursive",
              fontSize: "14px",
              fontWeight: "bold",
              color: dashboardTab === "lineage" ? accent : "#7a7f9a",
              borderBottom: dashboardTab === "lineage" ? `3px solid ${accent}` : "3px solid transparent",
              padding: "4px 12px",
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            🔗 Traceability Lineage Map
          </button>
          {auth.isAdmin() && (
            <button
              onClick={() => setDashboardTab("admin")}
              style={{
                background: "none",
                border: "none",
                fontFamily: "'Comic Neue', cursive",
                fontSize: "14px",
                fontWeight: "bold",
                color: dashboardTab === "admin" ? accent : "#7a7f9a",
                borderBottom: dashboardTab === "admin" ? `3px solid ${accent}` : "3px solid transparent",
                padding: "4px 12px",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
            >
              🔒 Admin Console
            </button>
          )}
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
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 24
          }}>
            {error}
          </div>
        )}

        {dashboardTab === "overview" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 32, marginBottom: 32 }}>
            
            {/* Datasets Column */}
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <h2 style={{
                  fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                  fontSize: "20px",
                  fontWeight: 900,
                  color: "#ffffff"
                }}>
                  Datasets
                </h2>
                <div style={{ display: "flex", gap: 8 }}>
                  {datasets.length > 0 && (
                    <button onClick={() => setShowAllModal(true)} className="btn-ghost" style={{
                      padding: "4px 12px",
                      fontSize: "10px",
                    }}>
                      View all ({datasets.length})
                    </button>
                  )}
                  <Link href="/generate" className="btn-ghost" style={{
                    padding: "4px 12px",
                    fontSize: "10px",
                    textDecoration: "none"
                  }}>
                    + Generate
                  </Link>
                </div>
              </div>

              {loading ? Array(3).fill(0).map((_, i) => (
                <div key={i} className="shimmer" style={{ height: 68, marginBottom: 12, border: "3px solid #000000", borderRadius: "8px" }} />
              )) : datasets.length === 0 ? (
                <div className="vt-card" style={{ padding: 40, textAlign: "center" }}>
                  <div style={{ fontSize: 32, marginBottom: 12 }}>📂</div>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#a0a5c0", marginBottom: 14 }}>No datasets yet</div>
                  <Link href="/generate" className="btn-primary" style={{
                    textDecoration: "none",
                    padding: "10px 20px",
                    display: "inline-block",
                    fontSize: 12
                  }}>
                    Create dataset →
                  </Link>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {datasets.slice(0, 3).map(d => (
                    <div key={d.id} className="vt-card" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "12px 16px" }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: "14px", color: "#ffffff", fontFamily: "'Comic Neue', cursive", fontWeight: "bold", marginBottom: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={d.name}>
                          {d.name}
                        </div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0" }}>
                          {d.row_count?.toLocaleString()} rows · {d.file_format?.toUpperCase()}
                          {d.quality_score && (
                            <span style={{ color: scoreColor(d.quality_score), marginLeft: 8, fontWeight: "bold" }}>
                              · {d.quality_score}% quality
                            </span>
                          )}
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                        <Badge status={d.status} />
                        <button onClick={() => handlePreviewDataset(d.id)}
                          title="Preview"
                          className="action-btn">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a0a5c0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                          </svg>
                        </button>
                        <button onClick={() => downloadCSV(d.id, d.name, token)}
                          title="Download"
                          className="action-btn">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a0a5c0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                          </svg>
                        </button>
                        <button onClick={() => handleDeleteDataset(d.id)}
                          title="Delete"
                          className="action-btn action-btn-danger">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Jobs Column */}
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <h2 style={{
                  fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                  fontSize: "20px",
                  fontWeight: 900,
                  color: "#ffffff"
                }}>
                  Training Jobs
                </h2>
                <div style={{ display: "flex", gap: 8 }}>
                  <Link href="/jobs" className="btn-ghost" style={{
                    padding: "4px 12px",
                    fontSize: "10px",
                    textDecoration: "none"
                  }}>
                    + Launch job
                  </Link>
                </div>
              </div>

              {loading ? Array(3).fill(0).map((_, i) => (
                <div key={i} className="shimmer" style={{ height: 68, marginBottom: 12, border: "3px solid #000000", borderRadius: "8px" }} />
              )) : jobs.length === 0 ? (
                <div className="vt-card" style={{ padding: 40, textAlign: "center" }}>
                  <div style={{ fontSize: 32, marginBottom: 12 }}>🧠</div>
                  <div style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#a0a5c0", marginBottom: 14 }}>No training jobs yet</div>
                  <Link href="/jobs" className="btn-primary" style={{
                    textDecoration: "none",
                    padding: "10px 20px",
                    display: "inline-block",
                    fontSize: 12
                  }}>
                    Configure training →
                  </Link>
                </div>
              ) : (
                <>
                  {jobs.slice(0, 3).map(j => (
                    <div key={j.id} className="vt-card" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 12, padding: "12px 16px" }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          fontSize: "14px",
                          color: "#ffffff",
                          fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                          fontWeight: "bold",
                          marginBottom: 4,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap"
                        }} title={j.base_model}>
                          {j.base_model?.split("/").pop()}
                        </div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0" }}>
                          Job ID: {j.id.slice(0, 8)} · Epochs: {j.training_config?.epochs}
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                        <Badge status={j.status} />
                        <button onClick={() => handlePreviewJob(j.id)}
                          title="Inspect Details"
                          className="action-btn">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a0a5c0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                          </svg>
                        </button>
                        {j.status !== "failed" && j.status !== "completed" && j.status !== "cancelled" && (
                          <button onClick={async () => {
                            if (confirm("Cancel this training run?")) {
                              try { await api.cancelJob(j.id); fetchData(); } catch (e: any) { alert(e.message); }
                            }
                          }}
                            title="Cancel Job"
                            className="action-btn action-btn-danger">
                            <svg xmlns="http://www.w3.org/2005/Atom" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="12" cy="12" r="10"></circle>
                              <line x1="15" y1="9" x2="9" y2="15"></line>
                              <line x1="9" y1="9" x2="15" y2="15"></line>
                            </svg>
                          </button>
                        )}
                        {(j.status === "completed" || j.status === "failed") && (
                          <button onClick={() => handleDeleteJob(j.id)}
                            title="Delete"
                            className="action-btn action-btn-danger">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="3 6 5 6 21 6"></polyline>
                              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                              <line x1="10" y1="11" x2="10" y2="17"></line>
                              <line x1="14" y1="11" x2="14" y2="17"></line>
                            </svg>
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                  {jobs.length > 3 && (
                    <Link href="/jobs" className="btn-ghost" style={{ display: "block", textAlign: "center", padding: "10px", marginTop: 12, fontSize: 11, textDecoration: "none" }}>
                      View all {jobs.length} training jobs →
                    </Link>
                  )}
                </>
              )}
            </div>

          </div>
        )}

        {dashboardTab === "lineage" && (
          /* Traceability Lineage Map */
          <div style={{ display: "flex", flexDirection: "column", gap: 24, marginBottom: 32 }} className="animate-fadeIn">
            {datasets.length === 0 ? (
              <div className="vt-card" style={{ padding: 48, textAlign: "center" }}>
                <span style={{ fontSize: 40 }}>🔗</span>
                <h3 style={{ fontFamily: "'Comic Neue', cursive", fontSize: 18, fontWeight: "bold", color: "#ffffff", marginTop: 12 }}>No Datasets Found</h3>
                <p style={{ fontSize: 12, color: "#a0a5c0", marginTop: 4 }}>You need to generate or upload a dataset first to see the lineage.</p>
              </div>
            ) : (
              datasets.map((dataset) => {
                const datasetJobs = jobs.filter((j) => j.dataset_id === dataset.id || j.dataset_id === dataset.name);
                return (
                  <div key={dataset.id} className="vt-card" style={{ padding: 24, border: "3px solid #000000", background: "#13141f" }}>
                    {/* Dataset Header */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "2px solid #27293d", paddingBottom: 12, marginBottom: 16 }}>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{ fontSize: 16 }}>📁</span>
                          <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 16, fontWeight: "bold", color: "#ffffff" }}>
                            {dataset.name}
                          </span>
                          <Badge status={dataset.status} />
                        </div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0", marginTop: 4 }}>
                          ID: {dataset.id} · {dataset.row_count} rows · {dataset.file_format?.toUpperCase()}
                        </div>
                      </div>
                      {dataset.quality_score && (
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
                          <span style={{ fontSize: "10px", fontFamily: "'JetBrains Mono', monospace", color: "#7a7f9a" }}>Quality Score</span>
                          <span style={{ fontSize: 16, fontFamily: "'JetBrains Mono', monospace", fontWeight: "bold", color: scoreColor(dataset.quality_score) }}>
                            {dataset.quality_score}%
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Children: Training Jobs */}
                    <div style={{ paddingLeft: 20, borderLeft: "2px dashed #27293d", marginLeft: 10, display: "flex", flexDirection: "column", gap: 16 }}>
                      {datasetJobs.length === 0 ? (
                        <div style={{ padding: "12px 16px", background: "rgba(255,255,255,0.01)", border: "1px dashed #27293d", borderRadius: "8px", color: "#7a7f9a", fontSize: 11, fontFamily: "'Comic Neue', cursive" }}>
                          No models fine-tuned from this dataset yet.
                        </div>
                      ) : (
                        datasetJobs.map((job) => {
                          const jobEvals = evaluations.filter((ev) => ev.training_job_id === job.id);
                          return (
                            <div key={job.id} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                              {/* Job Node Card */}
                              <div className="vt-card" style={{ padding: 16, border: "2px solid #000000", background: "#0c0d12", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <div>
                                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <span style={{ fontSize: 14 }}>🧠</span>
                                    <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 14, fontWeight: "bold", color: "#ffffff" }}>
                                      {job.base_model?.split("/").pop()} Checkpoint
                                    </span>
                                    <Badge status={job.status} />
                                  </div>
                                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#7a7f9a", marginTop: 4 }}>
                                    Job: {job.id.slice(0, 8)} · Epochs: {job.training_config?.epochs} · Rank: {job.training_config?.lora_r} · LR: {job.training_config?.learning_rate}
                                  </div>
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                                  {job.hf_model_id && (
                                    <span style={{ fontSize: "10px", fontFamily: "'JetBrains Mono', monospace", color: accent, border: `1px solid ${accent}`, padding: "2px 6px", borderRadius: "4px" }}>
                                      HF Hubbed
                                    </span>
                                  )}
                                  <button onClick={() => handlePreviewJob(job.id)} className="btn-ghost" style={{ padding: "4px 8px", fontSize: 10 }}>
                                    Inspect Details
                                  </button>
                                </div>
                              </div>

                              {/* Nested Children: Evaluations */}
                              <div style={{ paddingLeft: 24, borderLeft: `2px dashed ${accent}`, marginLeft: 16, display: "flex", flexDirection: "column", gap: 8 }}>
                                {jobEvals.length === 0 ? (
                                  <div style={{ padding: "8px 12px", background: "rgba(255,255,255,0.01)", border: "1px dashed #27293d", borderRadius: "6px", color: "#5d617a", fontSize: 10, fontFamily: "'Comic Neue', cursive" }}>
                                    No evaluations run on this checkpoint yet.
                                  </div>
                                ) : (
                                  jobEvals.map((ev) => (
                                    <div key={ev.id} className="vt-card" style={{ padding: 12, border: "2px solid #000000", background: "#111219", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                      <div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                          <span style={{ fontSize: 12 }}>🧪</span>
                                          <span style={{ fontFamily: "'Comic Neue', cursive", fontSize: 12, fontWeight: "bold", color: "#ffffff" }}>
                                            Benchmarked vs {datasets.find((d) => d.id === ev.dataset_id)?.name || "Test Set"}
                                          </span>
                                          <span style={{ fontSize: "9px", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", background: "#06b6d4", color: "#000", padding: "1px 6px", borderRadius: "3px", fontWeight: "bold" }}>
                                            {ev.status}
                                          </span>
                                        </div>
                                        {ev.metrics && (
                                          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0", marginTop: 4 }}>
                                            {Object.entries(ev.metrics).slice(0, 3).map(([k, v]: any) => (
                                              <span key={k} style={{ marginRight: 8 }}>
                                                {k}: <strong style={{ color: "#22d3ee" }}>{typeof v === "number" ? v.toFixed(2) : String(v)}</strong>
                                              </span>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                      <Link href={`/evaluate?job_id=${job.id}`} className="btn-ghost" style={{ padding: "4px 8px", fontSize: 9, textDecoration: "none" }}>
                                        View Metrics ↗
                                      </Link>
                                    </div>
                                  ))
                                )}
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {dashboardTab === "admin" && adminStats && (
          <div style={{ display: "flex", flexDirection: "column", gap: 32, marginBottom: 32 }} className="animate-fadeIn">
            {/* Stats Row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 20 }}>
              {[
                { label: "Total Users", val: adminStats.summary.total_users, color: "#a0a5c0" },
                { label: "Total Datasets", val: adminStats.summary.total_datasets, color: "#10b981" },
                { label: "Total Jobs Run", val: adminStats.summary.total_jobs, color: accent },
                { label: "Active Jobs", val: adminStats.summary.active_jobs_count, color: "#38bdf8" },
                { label: "GPU Hours", val: `${adminStats.summary.total_gpu_hours}h`, color: "#a855f7" },
                { label: "Total Cost", val: `$${adminStats.summary.total_cost_usd.toFixed(2)}`, color: "#eab308" },
              ].map(stat => (
                <div key={stat.label} className="vt-card" style={{ padding: 20, textAlign: "center", border: "2px solid #000" }}>
                  <div style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: "#7a7f9a", textTransform: "uppercase", marginBottom: 6 }}>
                    {stat.label}
                  </div>
                  <div style={{ fontSize: 22, fontFamily: "'JetBrains Mono', monospace", fontWeight: "bold", color: stat.color }}>
                    {stat.val}
                  </div>
                </div>
              ))}
            </div>

            {/* Audit Table */}
            <div className="vt-card" style={{ padding: 24, border: "3px solid #000" }}>
              <h3 style={{ fontFamily: "'Comic Neue', cursive", fontSize: 18, fontWeight: "bold", color: "#ffffff", marginBottom: 16 }}>
                User Resource Consumption Audit
              </h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'Comic Neue', cursive", fontSize: "13px" }}>
                  <thead>
                    <tr style={{ borderBottom: "2px solid #27293d", textAlign: "left", background: "rgba(255,255,255,0.02)" }}>
                      <th style={{ padding: 12, color: "#ffffff" }}>User / Email</th>
                      <th style={{ padding: 12, color: "#ffffff" }}>Joined</th>
                      <th style={{ padding: 12, color: "#ffffff", textAlign: "center" }}>Role</th>
                      <th style={{ padding: 12, color: "#ffffff", textAlign: "center" }}>Datasets</th>
                      <th style={{ padding: 12, color: "#ffffff", textAlign: "center" }}>Jobs Run</th>
                      <th style={{ padding: 12, color: "#ffffff", textAlign: "center" }}>GPU Hours</th>
                      <th style={{ padding: 12, color: "#ffffff", textAlign: "right" }}>Total Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {adminStats.users.map((user: any) => (
                      <tr key={user.id} style={{ borderBottom: "1px solid #27293d" }}>
                        <td style={{ padding: 12 }}>
                          <div style={{ fontWeight: "bold", color: "#ffffff" }}>{user.full_name || "No name"}</div>
                          <div style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: "#7a7f9a" }}>{user.email}</div>
                        </td>
                        <td style={{ padding: 12, color: "#a0a5c0", fontFamily: "monospace" }}>
                          {user.created_at ? new Date(user.created_at).toLocaleDateString() : "-"}
                        </td>
                        <td style={{ padding: 12, textAlign: "center" }}>
                          <span style={{
                            fontSize: 9,
                            fontFamily: "monospace",
                            background: user.is_admin ? "rgba(255,92,54,0.15)" : "rgba(255,255,255,0.05)",
                            color: user.is_admin ? accent : "#7a7f9a",
                            border: `1px solid ${user.is_admin ? accent : "#27293d"}`,
                            padding: "2px 6px",
                            borderRadius: 4
                          }}>
                            {user.is_admin ? "ADMIN" : "USER"}
                          </span>
                        </td>
                        <td style={{ padding: 12, textAlign: "center", color: "#ffffff", fontWeight: "bold" }}>
                          {user.datasets_count}
                        </td>
                        <td style={{ padding: 12, textAlign: "center", color: "#ffffff", fontWeight: "bold" }}>
                          {user.jobs_count}
                        </td>
                        <td style={{ padding: 12, textAlign: "center", color: "#a855f7", fontFamily: "monospace", fontWeight: "bold" }}>
                          {user.total_gpu_hours}h
                        </td>
                        <td style={{ padding: 12, textAlign: "right", color: "#eab308", fontFamily: "monospace", fontWeight: "bold" }}>
                          ${user.total_spent_usd.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions Grid */}
        <div style={{ marginTop: 40 }}>
          <h2 style={{
            fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
            fontSize: "20px",
            fontWeight: 900,
            color: "#ffffff",
            marginBottom: 16
          }}>
            Quick actions
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
            {[
              { icon: "⚗️", label: "Generate dataset", href: "/generate" },
              { icon: "📁", label: "Upload dataset", href: "/generate?tab=upload" },
              { icon: "🔑", label: "Add API keys", href: "/settings" },
            ].map(a => (
              <Link key={a.label} href={a.href} className="vt-card" style={{
                padding: "24px 16px",
                textAlign: "center",
                textDecoration: "none",
                display: "block"
              }}>
                <div style={{ fontSize: "32px", marginBottom: 12 }}>{a.icon}</div>
                <div style={{
                  fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                  fontSize: "13px",
                  fontWeight: "bold",
                  color: "#ffffff"
                }}>
                  {a.label}
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>


      {/* View All Datasets Modal */}
      {showAllModal && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0, 0, 0, 0.85)", zIndex: 900,
          display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
          backdropFilter: "blur(4px)"
        }}>
          <div className="vt-card" style={{
            width: "100%", maxWidth: 700, maxHeight: "85vh",
            display: "flex", flexDirection: "column", background: "#161722", padding: 24,
            border: "4px solid #000000", boxShadow: "8px 8px 0px 0px #000000"
          }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "3px solid #000000", paddingBottom: 12, marginBottom: 16 }}>
              <h3 style={{ fontFamily: "'Comic Neue', cursive", fontSize: 20, fontWeight: "bold", color: "#ffffff" }}>
                All Datasets ({datasets.length})
              </h3>
              <button onClick={() => setShowAllModal(false)} className="btn-ghost" style={{ padding: "6px 12px", borderRadius: "8px" }}>
                ✕ Close
              </button>
            </div>

            {/* List */}
            <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, paddingRight: 4 }}>
              {datasets.map(d => (
                <div key={d.id} className="vt-card" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "12px 16px" }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: "14px", color: "#ffffff", fontFamily: "'Comic Neue', cursive", fontWeight: "bold", marginBottom: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {d.name}
                    </div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0" }}>
                      {d.row_count?.toLocaleString()} rows · {d.file_format?.toUpperCase()}
                      {d.quality_score && (
                        <span style={{ color: scoreColor(d.quality_score), marginLeft: 8, fontWeight: "bold" }}>
                          · {d.quality_score}% quality
                        </span>
                      )}
                    </div>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                    <Badge status={d.status} />
                    <button onClick={() => handlePreviewDataset(d.id)}
                      title="Preview"
                      className="action-btn">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a0a5c0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                      </svg>
                    </button>
                    <button onClick={() => downloadCSV(d.id, d.name, token)}
                      title="Download"
                      className="action-btn">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a0a5c0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                      </svg>
                    </button>
                    <button onClick={() => handleDeleteDataset(d.id)}
                      title="Delete"
                      className="action-btn action-btn-danger">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Dataset Preview Modal */}
      {(previewDataset || previewLoading) && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0, 0, 0, 0.85)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
          backdropFilter: "blur(4px)"
        }}>
          <div className="vt-card" style={{
            width: "100%", maxWidth: 800, maxHeight: "85vh",
            display: "flex", flexDirection: "column", background: "#161722", padding: 24,
            border: "4px solid #000000", boxShadow: "8px 8px 0px 0px #000000"
          }}>
            {previewLoading ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 48, gap: 16 }}>
                <span style={{ width: 24, height: 24, border: "2px solid var(--accent)", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                <div style={{ fontFamily: "'Comic Neue', cursive", color: "#ffffff", fontSize: 14 }}>Loading preview rows...</div>
              </div>
            ) : (
              <>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "3px solid #000000", paddingBottom: 12, marginBottom: 16 }}>
                  <div style={{ minWidth: 0, flex: 1, marginRight: 12 }}>
                    <h3 style={{ fontFamily: "'Comic Neue', cursive", fontSize: 18, fontWeight: "bold", color: "#ffffff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      🔍 Preview: {previewDataset.name}
                    </h3>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0", marginTop: 4 }}>
                      {previewDataset.row_count} rows · {previewDataset.file_format?.toUpperCase()} · {previewDataset.quality_score}% quality
                    </div>
                  </div>
                  <button onClick={() => setPreviewDataset(null)} className="btn-ghost" style={{ padding: "6px 12px", borderRadius: "8px" }}>
                    ✕ Close
                  </button>
                </div>

                {/* Table Content */}
                <div style={{ flex: 1, overflowY: "auto", overflowX: "auto", marginBottom: 16 }}>
                  {previewDataset.audit_results?.rows && previewDataset.audit_results.rows.length > 0 ? (
                    <div style={{ border: "2px solid #000000", borderRadius: "8px", boxShadow: "3px 3px 0px 0px #000000" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'Comic Neue', cursive", fontSize: "12px", background: "#0c0d12" }}>
                        <thead>
                          <tr style={{ borderBottom: "2px solid #000000", background: "rgba(255,255,255,0.02)" }}>
                            {Object.keys(previewDataset.audit_results.rows[0]).map((c: string) => (
                              <th key={c} style={{ padding: "8px 12px", textAlign: "left", color: "#ffffff", fontWeight: "bold", borderRight: "1px solid #27293d" }}>
                                {c}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {previewDataset.audit_results.rows.slice(0, 5).map((row: any, rIdx: number) => (
                            <tr key={rIdx} style={{ borderBottom: rIdx === Math.min(5, previewDataset.audit_results.rows.length) - 1 ? "none" : "1px solid #27293d" }}>
                              {Object.keys(previewDataset.audit_results.rows[0]).map((c: string) => (
                                <td key={c} style={{ padding: "8px 12px", color: "#a0a5c0", borderRight: "1px solid #27293d", whiteSpace: "nowrap" }}>
                                  {row[c] !== undefined ? String(row[c]) : ""}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div style={{ padding: 40, textAlign: "center", background: "#0c0d12", border: "2px dashed #4f526b", borderRadius: "8px" }}>
                      <div style={{ color: "#a0a5c0", fontSize: 13, fontFamily: "'Comic Neue', cursive" }}>
                        No preview rows stored for this dataset.
                      </div>
                    </div>
                  )}
                  {previewDataset.audit_results?.rows && previewDataset.audit_results.rows.length > 5 && (
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0", marginTop: 8, textAlign: "right" }}>
                      * Showing first 5 of {previewDataset.row_count} rows
                    </div>
                  )}
                </div>

                {/* Footer buttons */}
                <div style={{ display: "flex", justifyContent: "flex-end", gap: 12 }}>
                  <button onClick={() => downloadCSV(previewDataset.id, previewDataset.name, token)} className="btn-primary" style={{ padding: "8px 16px", fontSize: 11 }}>
                    ↓ Download CSV
                  </button>
                  <button onClick={() => handleDeleteDataset(previewDataset.id)} className="btn-ghost" style={{ padding: "8px 16px", fontSize: 11, color: "#f43f5e" }}>
                    🗑️ Delete
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Model/Job Preview Modal */}
      {(previewJob || previewJobLoading) && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0, 0, 0, 0.85)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
          backdropFilter: "blur(4px)"
        }}>
          <div className="vt-card" style={{
            width: "100%", maxWidth: 650, maxHeight: "90vh",
            display: "flex", flexDirection: "column", background: "#161722", padding: 24,
            border: "4px solid #000000", boxShadow: "8px 8px 0px 0px #000000"
          }}>
            {previewJobLoading ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 48, gap: 16 }}>
                <span style={{ width: 24, height: 24, border: "2px solid var(--accent)", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                <div style={{ fontFamily: "'Comic Neue', cursive", color: "#ffffff", fontSize: 14 }}>Loading job details...</div>
              </div>
            ) : (
              <>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "3px solid #000000", paddingBottom: 12, marginBottom: 16 }}>
                  <div style={{ minWidth: 0, flex: 1, marginRight: 12 }}>
                    <h3 style={{ fontFamily: "'Comic Neue', cursive", fontSize: 18, fontWeight: "bold", color: "#ffffff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      🧠 Model Preview: {previewJob.base_model?.split("/").pop()}
                    </h3>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "10px", color: "#a0a5c0", marginTop: 4 }}>
                      Job ID: {previewJob.id} · Status: <span style={{ textTransform: "uppercase", fontWeight: "bold" }}>{previewJob.status}</span>
                    </div>
                  </div>
                  <button onClick={() => setPreviewJob(null)} className="btn-ghost" style={{ padding: "6px 12px", borderRadius: "8px" }}>
                    ✕ Close
                  </button>
                </div>

                {/* Content */}
                <div style={{ flex: 1, overflowY: "auto", marginBottom: 16, paddingRight: 4 }}>
                  {/* Info tables */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginBottom: 20 }}>
                    <div style={{ background: "#0c0d12", border: "2px solid #000000", borderRadius: "8px", padding: 12, boxShadow: "2px 2px 0px 0px #000000" }}>
                      <p style={{ fontFamily: "'Comic Neue', cursive", fontSize: "12px", fontWeight: "bold", color: "#ffffff", borderBottom: "1px solid #27293d", paddingBottom: 4, marginBottom: 6 }}>
                        Hyperparameters
                      </p>
                      <table style={{ width: "100%", fontFamily: "monospace", fontSize: "11px", color: "#a0a5c0" }}>
                        <tbody>
                          <tr><td style={{ color: "#7a7f9a" }}>Epochs:</td><td>{previewJob.training_config?.epochs}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>LoRA Rank:</td><td>{previewJob.training_config?.lora_r}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>Max Seq Len:</td><td>{previewJob.training_config?.max_seq_length}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>Batch Size:</td><td>{previewJob.training_config?.batch_size}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>LR:</td><td>{previewJob.training_config?.learning_rate}</td></tr>
                        </tbody>
                      </table>
                    </div>

                    <div style={{ background: "#0c0d12", border: "2px solid #000000", borderRadius: "8px", padding: 12, boxShadow: "2px 2px 0px 0px #000000" }}>
                      <p style={{ fontFamily: "'Comic Neue', cursive", fontSize: "12px", fontWeight: "bold", color: "#ffffff", borderBottom: "1px solid #27293d", paddingBottom: 4, marginBottom: 6 }}>
                        Run Statistics
                      </p>
                      <table style={{ width: "100%", fontFamily: "monospace", fontSize: "11px", color: "#a0a5c0" }}>
                        <tbody>
                          <tr><td style={{ color: "#7a7f9a" }}>Task Type:</td><td>{previewJob.task_type}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>Started At:</td><td>{previewJob.started_at ? new Date(previewJob.started_at).toLocaleTimeString() : "-"}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>Completed At:</td><td>{previewJob.completed_at ? new Date(previewJob.completed_at).toLocaleTimeString() : "-"}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>GPU Time:</td><td>{previewJob.gpu_seconds_used ? `${previewJob.gpu_seconds_used}s` : "-"}</td></tr>
                          <tr><td style={{ color: "#7a7f9a" }}>Actual Cost:</td><td>{previewJob.actual_cost_usd ? `$${previewJob.actual_cost_usd.toFixed(4)}` : "-"}</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Loss Curve */}
                  {(previewJob.status === "completed" || previewJob.status === "training" || (previewJob.metrics_history && previewJob.metrics_history.length > 0)) && (
                    <LossCurve history={previewJob.metrics_history} />
                  )}

                  {/* Output details */}
                  {previewJob.hf_model_id && (
                    <div style={{ background: "rgba(34,211,238,0.05)", border: "2px solid #06b6d4", borderRadius: "8px", padding: 12 }}>
                      <p style={{ fontFamily: "'Comic Neue', cursive", fontSize: "12px", fontWeight: "bold", color: "#22d3ee", marginBottom: 4 }}>
                        Model Repository
                      </p>
                      <p style={{ fontFamily: "monospace", fontSize: "12px", color: "#ffffff", wordBreak: "break-all" }}>
                        huggingface.co/{previewJob.hf_model_id}
                      </p>
                    </div>
                  )}

                  {previewJob.error_message && (
                    <div style={{ background: "rgba(244,63,94,0.05)", border: "2px solid #f43f5e", borderRadius: "8px", padding: 12 }}>
                      <p style={{ fontFamily: "'Comic Neue', cursive", fontSize: "12px", fontWeight: "bold", color: "#f43f5e", marginBottom: 4 }}>
                        Error
                      </p>
                      <p style={{ fontFamily: "monospace", fontSize: "11px", color: "#ffffff" }}>
                        {previewJob.error_message}
                      </p>
                    </div>
                  )}
                </div>

                {/* Footer buttons */}
                <div style={{ display: "flex", justifyContent: "flex-end", gap: 12, borderTop: "2px solid #000000", paddingTop: 16 }}>
                  {previewJob.hf_model_id && (
                    <a href={`https://huggingface.co/${previewJob.hf_model_id}`} target="_blank" rel="noreferrer" className="btn-primary" style={{ padding: "8px 16px", fontSize: 11, textDecoration: "none" }}>
                      HF Hub ↗
                    </a>
                  )}
                  <button onClick={() => setPreviewJob(null)} className="btn-ghost" style={{ padding: "8px 16px", fontSize: 11 }}>
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

    </div>
  );
}

function LossCurve({ history }: { history: any[] }) {
  if (!history || history.length < 2) {
    return (
      <div style={{
        height: "80px",
        background: "rgba(0,0,0,0.4)",
        border: "1px solid #27272a",
        borderRadius: "8px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "10px",
        color: "#52525b",
        fontFamily: "monospace",
        fontWeight: "bold",
        margin: "12px 0"
      }}>
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
    <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid #27272a", borderRadius: "8px", padding: "12px", margin: "12px 0" }}>
      <p style={{ fontSize: "9px", color: "#22d3ee", fontFamily: "monospace", fontWeight: "bold", margin: "0 0 8px 0", textTransform: "uppercase", letterSpacing: "0.1em" }}>
        Training Loss Curve
      </p>
      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", overflow: "visible" }}>
          <line x1={paddingX} y1={paddingY} x2={width - paddingX} y2={paddingY} stroke="#27272a" strokeDasharray="3" strokeWidth="1" />
          <line x1={paddingX} y1={height - paddingY} x2={width - paddingX} y2={height - paddingY} stroke="#27272a" strokeDasharray="3" strokeWidth="1" />
          <text x={paddingX - 8} y={paddingY + 3} fill="#71717a" fontSize="8" fontFamily="monospace" textAnchor="end">{maxLoss.toFixed(3)}</text>
          <text x={paddingX - 8} y={height - paddingY + 3} fill="#71717a" fontSize="8" fontFamily="monospace" textAnchor="end">{minLoss.toFixed(3)}</text>
          <text x={paddingX} y={height - 2} fill="#52525b" fontSize="8" fontFamily="monospace">step {minStep}</text>
          <text x={width - paddingX} y={height - 2} fill="#52525b" fontSize="8" fontFamily="monospace" textAnchor="end">step {maxStep}</text>
          <polyline fill="none" stroke="#22d3ee" strokeWidth="2" points={points} strokeLinecap="round" strokeLinejoin="round" />
          <circle cx={getX(history[history.length - 1].step)} cy={getY(history[history.length - 1].loss)} r="3.5" fill="#22d3ee" stroke="#000" strokeWidth="1.5" />
        </svg>
      </div>
    </div>
  );
}

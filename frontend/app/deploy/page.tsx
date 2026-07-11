"use client";
import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { auth, api, keyStore, decryptApiKey } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

type Deployment = {
  id: string;
  job_id?: string;
  model_id: string;
  endpoint_name: string;
  instance_type: string;
  region: string;
  status: string;
  error_message?: string;
  created_at: string;
};

type CompletedJob = {
  id: string;
  base_model: string;
  task_type: string;
  hf_model_id?: string;
};

const INSTANCES = [
  { id: "ml.g5.xlarge", label: "ml.g5.xlarge (1x A10G GPU, 24GB VRAM)", cost: "$1.40/hr" },
  { id: "ml.g5.2xlarge", label: "ml.g5.2xlarge (1x A10G GPU, 24GB VRAM)", cost: "$1.51/hr" },
  { id: "ml.g4dn.xlarge", label: "ml.g4dn.xlarge (1x T4 GPU, 16GB VRAM)", cost: "$0.73/hr" },
];

const REGIONS = [
  { id: "us-east-1", label: "us-east-1 (N. Virginia)" },
  { id: "us-east-2", label: "us-east-2 (Ohio)" },
  { id: "us-west-2", label: "us-west-2 (Oregon)" },
  { id: "ap-southeast-2", label: "ap-southeast-2 (Sydney)" },
  { id: "eu-west-1", label: "eu-west-1 (Ireland)" },
];

const STATUS_STYLES: Record<string, string> = {
  deploying: "bg-amber-500 text-black border-black",
  active: "bg-emerald-500 text-black border-black",
  failed: "bg-red-500 text-white border-black",
  terminated: "bg-zinc-700 text-zinc-300 border-black",
};

export default function DeployPage() {
  const router = useRouter();
  const [completedJobs, setCompletedJobs] = useState<CompletedJob[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);

  // Form states
  const [deployType, setDeployType] = useState<"job" | "custom">("job");
  const [selectedJobId, setSelectedJobId] = useState("");
  const [customModelId, setCustomModelId] = useState("microsoft/phi-3-mini-4k-instruct");
  const [selectedInstance, setSelectedInstance] = useState("ml.g5.xlarge");
  const [selectedRegion, setSelectedRegion] = useState("us-east-1");
  const [roleArnOverride, setRoleArnOverride] = useState("");
  
  const [deploying, setDeploying] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  const userId = typeof window !== "undefined" ? sessionStorage.getItem("user_id") || "" : "";

  // Fetch deployments & completed jobs
  const fetchData = useCallback(async () => {
    try {
      const [jobs, deps] = await Promise.all([
        api.listJobs(),
        // We will add helper methods on client api.ts or use raw fetch
        fetchDeployments(),
      ]);

      // Keep only completed jobs that have a Hugging Face model repository ID
      const completed = jobs
        .filter((j: any) => j.status === "completed" && j.hf_model_id)
        .map((j: any) => ({
          id: j.id,
          base_model: j.base_model,
          task_type: j.task_type,
          hf_model_id: j.hf_model_id,
        }));

      setCompletedJobs(completed);
      if (completed.length > 0 && !selectedJobId) {
        setSelectedJobId(completed[0].id);
      }
      setDeployments(deps);
    } catch (err: any) {
      console.error("Error loading deployment data:", err);
    } finally {
      setLoading(false);
    }
  }, [selectedJobId]);

  // Raw API helpers for Deployments (since they are new backend endpoints)
  const fetchDeployments = async (): Promise<Deployment[]> => {
    const token = auth.getToken();
    const res = await fetch(`${API_BASE}/deploy/`, {
      headers: {
        "Authorization": token ? `Bearer ${token}` : "",
        "Content-Type": "application/json",
      },
    });
    if (!res.ok) throw new Error("Failed to fetch deployments");
    return res.json();
  };

  const triggerDeploy = async (body: any): Promise<Deployment> => {
    const token = auth.getToken();
    const res = await fetch(`${API_BASE}/deploy/`, {
      method: "POST",
      headers: {
        "Authorization": token ? `Bearer ${token}` : "",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Deployment failed" }));
      throw new Error(err.detail || "Deployment failed");
    }
    return res.json();
  };

  const terminateDeployment = async (depId: string, body: any) => {
    const token = auth.getToken();
    const res = await fetch(`${API_BASE}/deploy/${depId}/terminate`, {
      method: "POST",
      headers: {
        "Authorization": token ? `Bearer ${token}` : "",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Teardown failed" }));
      throw new Error(err.detail || "Teardown failed");
    }
    return res.json();
  };

  useEffect(() => {
    if (!auth.isLoggedIn()) {
      router.push("/auth");
      return;
    }
    fetchData();
  }, []);

  // Poll status if there are active "deploying" endpoints
  useEffect(() => {
    const hasDeploying = deployments.some((d) => d.status === "deploying");
    if (hasDeploying) {
      const t = setInterval(() => {
        fetchDeployments()
          .then((deps) => setDeployments(deps))
          .catch((err) => console.error(err));
      }, 5000);
      return () => clearInterval(t);
    }
  }, [deployments]);

  // Handle Deploy
  const handleDeploy = async () => {
    setErrorMsg("");
    setSuccessMsg("");
    setDeploying(true);

    try {
      // 1. Decrypt AWS keys if stored in settings
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

      const body: any = {
        instance_type: selectedInstance,
        region: selectedRegion,
        aws_access_key: accessKey || undefined,
        aws_secret_key: secretKey || undefined,
        aws_role_arn: roleArnOverride.trim() || undefined,
      };

      if (deployType === "job") {
        if (!selectedJobId) {
          throw new Error("No completed training job selected.");
        }
        body.job_id = selectedJobId;
      } else {
        if (!customModelId.trim()) {
          throw new Error("Please enter a custom Hugging Face model repository ID.");
        }
        body.model_id = customModelId.trim();
      }

      const dep = await triggerDeploy(body);
      setSuccessMsg(`🚀 Deployment created! Endpoint: ${dep.endpoint_name}. Provisioning GPU instance in background...`);
      fetchData();
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to deploy model to SageMaker");
    } finally {
      setDeploying(false);
    }
  };

  // Handle Terminate
  const handleTerminate = async (depId: string) => {
    if (!window.confirm("⚠️ WARNING: Are you sure you want to shut down and delete this AWS SageMaker endpoint? This will immediately terminate all billing for this instance.")) {
      return;
    }

    try {
      // Decrypt AWS keys for authentication
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

      await terminateDeployment(depId, {
        aws_access_key: accessKey || undefined,
        aws_secret_key: secretKey || undefined,
      });

      alert("SageMaker endpoint successfully deleted and billing terminated.");
      fetchData();
    } catch (err: any) {
      alert(err.message || "Teardown failed");
    }
  };

  const handleDelete = async (depId: string) => {
    if (!window.confirm("Are you sure you want to remove this deployment from history?")) {
      return;
    }
    try {
      const token = auth.getToken();
      const res = await fetch(`${API_BASE}/deploy/${depId}`, {
        method: "DELETE",
        headers: {
          "Authorization": token ? `Bearer ${token}` : "",
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) throw new Error("Failed to delete deployment");
      fetchData();
    } catch (err: any) {
      alert(err.message || "Deletion failed");
    }
  };

  const accent = "#ff5c36";

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8 animate-fade-up opacity-0">
        <h1 className="font-display font-bold text-3xl text-text mb-1">AWS SageMaker Deployment</h1>
        <p className="text-sm text-subtle font-body">Provision dedicated GPU instances to host fine-tuned checkpoints and base models.</p>
      </div>

      {loading ? (
        <div className="text-center py-16">
          <p className="text-sm text-subtle font-mono">Loading deployments...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          {/* Left: Deployment Manager Config */}
          <div className="lg:col-span-1 space-y-6">
            <div className="glass rounded-xl border-3 border-black p-6 bg-[#161722] shadow-[5px_5px_0px_0px_#000]">
              <h2 className="text-xs font-mono font-bold text-accent uppercase tracking-widest mb-4">
                Deploy Endpoint
              </h2>

              {/* Selector for deployment type */}
              <div className="flex gap-2 p-1 bg-[#0f0f12] rounded-lg border-2 border-black mb-6">
                <button
                  onClick={() => setDeployType("job")}
                  className={`flex-1 py-1.5 rounded text-xs font-mono font-bold transition-all ${
                    deployType === "job" ? "bg-accent text-black" : "text-subtle hover:text-text"
                  }`}
                >
                  Fine-tuned Job
                </button>
                <button
                  onClick={() => setDeployType("custom")}
                  className={`flex-1 py-1.5 rounded text-xs font-mono font-bold transition-all ${
                    deployType === "custom" ? "bg-accent text-black" : "text-subtle hover:text-text"
                  }`}
                >
                  Base / Custom Model
                </button>
              </div>

              {/* Model selection based on type */}
              {deployType === "job" ? (
                <div className="mb-4">
                  <label className="block text-[10px] font-mono font-bold text-subtle uppercase mb-2">
                    SELECT COMPLETED JOB
                  </label>
                  {completedJobs.length === 0 ? (
                    <p className="text-xs text-red-400 font-mono py-2 bg-red-950/20 px-3 border border-red-900/40 rounded-lg">
                      No completed training jobs with HuggingFace repos found.
                    </p>
                  ) : (
                    <select
                      value={selectedJobId}
                      onChange={(e) => setSelectedJobId(e.target.value)}
                      className="w-full bg-[#0f0f12] border-2 border-black rounded-lg text-xs text-text font-mono p-2.5 outline-none"
                    >
                      {completedJobs.map((j) => (
                        <option key={j.id} value={j.id}>
                          {j.hf_model_id} (job: {j.id.slice(0, 8)})
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              ) : (
                <div className="mb-4">
                  <label className="block text-[10px] font-mono font-bold text-subtle uppercase mb-2">
                    HUGGINGFACE MODEL ID
                  </label>
                  <input
                    type="text"
                    value={customModelId}
                    onChange={(e) => setCustomModelId(e.target.value)}
                    placeholder="e.g. microsoft/phi-3-mini-4k-instruct"
                    className="w-full bg-[#0f0f12] border-2 border-black rounded-lg text-xs text-text font-mono p-2.5 outline-none"
                  />
                </div>
              )}

              {/* GPU Instance Type */}
              <div className="mb-4">
                <label className="block text-[10px] font-mono font-bold text-subtle uppercase mb-2">
                  GPU INSTANCE TYPE
                </label>
                <select
                  value={selectedInstance}
                  onChange={(e) => setSelectedInstance(e.target.value)}
                  className="w-full bg-[#0f0f12] border-2 border-black rounded-lg text-xs text-text font-mono p-2.5 outline-none"
                >
                  {INSTANCES.map((inst) => (
                    <option key={inst.id} value={inst.id}>
                      {inst.label} — {inst.cost}
                    </option>
                  ))}
                </select>
              </div>

              {/* Region */}
              <div className="mb-4">
                <label className="block text-[10px] font-mono font-bold text-subtle uppercase mb-2">
                  AWS REGION
                </label>
                <select
                  value={selectedRegion}
                  onChange={(e) => setSelectedRegion(e.target.value)}
                  className="w-full bg-[#0f0f12] border-2 border-black rounded-lg text-xs text-text font-mono p-2.5 outline-none"
                >
                  {REGIONS.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Role ARN Override */}
              <div className="mb-6">
                <label className="block text-[10px] font-mono font-bold text-subtle uppercase mb-1">
                  ROLE ARN OVERRIDE <span className="text-[#555] font-normal">(optional)</span>
                </label>
                <input
                  type="text"
                  value={roleArnOverride}
                  onChange={(e) => setRoleArnOverride(e.target.value)}
                  placeholder="arn:aws:iam::1234567890:role/..."
                  className="w-full bg-[#0f0f12] border-2 border-black rounded-lg text-xs text-text font-mono p-2.5 outline-none"
                />
              </div>

              {/* Action button */}
              <button
                onClick={handleDeploy}
                disabled={deploying || (deployType === "job" && completedJobs.length === 0)}
                className="w-full btn-primary py-3 rounded-lg text-xs font-display font-bold flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {deploying ? "Deploying (3-5 mins)..." : "🚀 Launch SageMaker Endpoint"}
              </button>

              {/* Warnings and errors */}
              {errorMsg && (
                <p className="text-xs text-red-400 bg-red-950/20 border border-red-900/40 rounded-lg px-3 py-2 mt-4 font-mono font-bold">
                  ⚠️ {errorMsg}
                </p>
              )}
              {successMsg && (
                <p className="text-xs text-emerald-400 bg-emerald-950/20 border border-emerald-900/40 rounded-lg px-3 py-2 mt-4 font-mono font-bold">
                  {successMsg}
                </p>
              )}
            </div>
          </div>

          {/* Right: Active Deployments List */}
          <div className="lg:col-span-2 space-y-6">
            <h2 className="font-display font-bold text-text text-lg mb-4">Active Endpoints</h2>
            {deployments.length === 0 ? (
              <div className="glass rounded-xl border border-zinc-800 p-8 text-center bg-[#111219]/30">
                <p className="text-3xl mb-3">☁️</p>
                <p className="text-sm text-subtle font-body font-bold">No active SageMaker endpoints found.</p>
                <p className="text-xs text-zinc-500 font-mono mt-1">Configure parameters and launch your first instance on the left panel.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {deployments.map((dep) => {
                  const isActive = dep.status === "active";
                  const isDeploying = dep.status === "deploying";
                  const inst = INSTANCES.find((i) => i.id === dep.instance_type);
                  const cost = inst ? inst.cost : "$1.40/hr";
                  
                  return (
                    <div
                      key={dep.id}
                      className="glass rounded-xl border border-zinc-850 p-5 hover:border-zinc-700 transition-all bg-[#111219]/60 flex flex-col md:flex-row justify-between md:items-center gap-4"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2.5 flex-wrap">
                          <p className="text-sm font-mono font-bold text-white leading-none">
                            {dep.model_id}
                          </p>
                          <span
                            className={`text-[9px] px-2 py-0.5 rounded font-mono font-bold border ${
                              STATUS_STYLES[dep.status] || STATUS_STYLES.deploying
                            } shadow-[1.5px_1.5px_0px_0px_#000]`}
                          >
                            {dep.status.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-xs text-zinc-500 font-mono">
                          Endpoint: {dep.endpoint_name}
                        </p>
                        <p className="text-xs text-zinc-500 font-mono font-bold">
                          {dep.instance_type} · {dep.region} · <span className="text-emerald-400 font-bold">{cost}</span>
                        </p>
                        
                        {isDeploying && (
                          <div className="flex items-center gap-2 text-[10px] text-amber-500 font-mono font-bold pt-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-ping border border-black" />
                            Provisioning node... this takes 3-5 minutes.
                          </div>
                        )}
                        
                        {dep.error_message && (
                          <p className="text-xs text-red-400 bg-red-950/20 border border-red-900/40 rounded-lg px-2.5 py-1.5 mt-2 font-mono font-bold break-all">
                            Error: {dep.error_message}
                          </p>
                        )}
                      </div>

                      <div className="flex items-center gap-4 shrink-0 justify-end">
                        {isActive && (
                          <button
                            onClick={() => router.push(`/evaluate?dep_id=${dep.id}`)}
                            className="text-xs text-cyan-400 hover:text-cyan-300 font-mono font-bold border border-cyan-400/20 px-3 py-1.5 rounded bg-cyan-950/20 hover:bg-cyan-950/40 transition-all"
                          >
                            Test in Playground ↗
                          </button>
                        )}
                        
                        {dep.status !== "terminated" && dep.status !== "failed" ? (
                          <button
                            onClick={() => handleTerminate(dep.id)}
                            className="text-xs text-red-400 hover:text-red-350 font-mono font-bold border border-red-500/20 px-3 py-1.5 rounded bg-red-950/10 hover:bg-red-950/30 transition-all"
                          >
                            Tear Down 🛑
                          </button>
                        ) : (
                          <button
                            onClick={() => handleDelete(dep.id)}
                            className="text-xs text-zinc-400 hover:text-white font-mono font-bold border border-zinc-700/40 px-3 py-1.5 rounded bg-zinc-950/20 hover:bg-zinc-900 transition-all"
                          >
                            Remove Card 🗑️
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

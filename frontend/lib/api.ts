const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// Persistent auth — survives refresh (dynamically retrieved to prevent server/client cache mismatch in Next.js)
export const auth = {
  setToken: (token: string, userId: string, isAdmin?: boolean) => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem("auth_token", token);
      sessionStorage.setItem("user_id", userId);
      sessionStorage.setItem("is_admin", String(!!isAdmin));
    }
  },
  getToken: () => typeof window !== "undefined" ? sessionStorage.getItem("auth_token") : null,
  getUserId: () => typeof window !== "undefined" ? sessionStorage.getItem("user_id") : null,
  isAdmin: () => typeof window !== "undefined" ? sessionStorage.getItem("is_admin") === "true" : false,
  clear: () => {
    if (typeof window !== "undefined") {
      sessionStorage.removeItem("auth_token");
      sessionStorage.removeItem("user_id");
      sessionStorage.removeItem("is_admin");
    }
  },
  isLoggedIn: () => typeof window !== "undefined" ? !!sessionStorage.getItem("auth_token") : false,
};

// AES-GCM client-side encryption for API keys
export async function encryptApiKey(rawKey: string, userId: string): Promise<string> {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey("raw", enc.encode(userId + "veritas-salt"), { name: "PBKDF2" }, false, ["deriveKey"]);
  const derivedKey = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: enc.encode(userId), iterations: 100000, hash: "SHA-256" },
    keyMaterial, { name: "AES-GCM", length: 256 }, false, ["encrypt"]
  );
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, derivedKey, enc.encode(rawKey));
  const combined = new Uint8Array(iv.byteLength + encrypted.byteLength);
  combined.set(iv, 0); combined.set(new Uint8Array(encrypted), iv.byteLength);
  return btoa(String.fromCharCode(...combined));
}

export const keyStore = {
  save: (provider: string, encrypted: string) => sessionStorage.setItem(`ek_${provider}`, encrypted),
  get: (provider: string) => sessionStorage.getItem(`ek_${provider}`),
  remove: (provider: string) => sessionStorage.removeItem(`ek_${provider}`),
  clear: () => ["openai","runpod","huggingface","github","deepseek","aws"].forEach(k => sessionStorage.removeItem(`ek_${k}`)),
};

// Decrypt a stored encrypted key back to plaintext
export async function decryptApiKey(encrypted: string, userId: string): Promise<string> {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey("raw", enc.encode(userId + "veritas-salt"), { name: "PBKDF2" }, false, ["deriveKey"]);
  const derivedKey = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: enc.encode(userId), iterations: 100000, hash: "SHA-256" },
    keyMaterial, { name: "AES-GCM", length: 256 }, false, ["decrypt"]
  );
  const bytes = Uint8Array.from(atob(encrypted), c => c.charCodeAt(0));
  const iv = bytes.slice(0, 12);
  const data = bytes.slice(12);
  const decrypted = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, derivedKey, data);
  return new TextDecoder().decode(decrypted);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json", ...(options.headers as any || {}) };
  const token = auth.getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const hasKeys = ["openai", "deepseek", "huggingface", "github", "runpod", "aws"].some(
    k => typeof window !== "undefined" && !!sessionStorage.getItem(`ek_${k}`)
  );
  if (hasKeys) headers["X-Key-Configured"] = "true";

  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...options, headers });
  if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail || "Request failed"); }
  return res.json();
}

async function upload(path: string, formData: FormData): Promise<any> {
  const headers: Record<string, string> = {};
  const token = auth.getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const hasKeys = ["openai", "deepseek", "huggingface", "github", "runpod", "aws"].some(
    k => typeof window !== "undefined" && !!sessionStorage.getItem(`ek_${k}`)
  );
  if (hasKeys) headers["X-Key-Configured"] = "true";

  const res = await fetch(`${API_BASE}${path}`, { method: "POST", cache: "no-store", headers, body: formData });
  if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail || "Upload failed"); }
  return res.json();
}

export const api = {
  register: (email: string, password: string, fullName: string) =>
    request<any>("/auth/register", { method: "POST", body: JSON.stringify({ email, password, full_name: fullName }) }),
  login: (email: string, password: string) =>
    request<any>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  uploadDataset: (file: File, name: string) => { const fd = new FormData(); fd.append("file", file); fd.append("name", name); return upload("/datasets/upload", fd); },
  listDatasets: () => request<any[]>("/datasets/"),
  getDataset: (id: string) => request<any>(`/datasets/${id}`),
  deleteDataset: (id: string) => request<any>(`/datasets/${id}`, { method: "DELETE" }),
  previewDataset: (body: object) => request<any>("/generate/preview", { method: "POST", body: JSON.stringify(body) }),
  generateDataset: (body: object) => request<any>("/generate/dataset", { method: "POST", body: JSON.stringify(body) }),
  suggestSchema: (description: string, numColumns: number) =>
    request<any>("/structured/schema", { method: "POST", body: JSON.stringify({ description, num_columns: numColumns }) }),
  generateStructured: (body: object) => request<any>("/structured/generate", { method: "POST", body: JSON.stringify(body) }),
  repairDataset: (body: object) => request<any>("/structured/repair", { method: "POST", body: JSON.stringify(body) }),
  createJob: (body: object) => request<any>("/jobs/", { method: "POST", body: JSON.stringify(body) }),
  listJobs: () => request<any[]>("/jobs/"),
  getJob: (id: string) => request<any>(`/jobs/${id}`),
  cancelJob: (id: string) => request<any>(`/jobs/${id}/cancel`, { method: "POST" }),
  getNotebookLink: (jobId: string, body: { github_token: string; hf_repo_id: string }) =>
    request<{ colab_url: string; gist_id: string; notebook_name: string }>(`/jobs/${jobId}/notebook`, { method: "POST", body: JSON.stringify(body) }),
  generateColabNotebook: (body: object) =>
    request<{ colab_url: string; gist_id: string; notebook_name: string }>("/jobs/colab/generate-notebook", { method: "POST", body: JSON.stringify(body) }),
  getTunnelStatus: () => request<{ active: boolean; url: string; is_ngrok: boolean; mode: string }>("/tunnel/status"),
  generatePlaygroundResponse: (jobId: string, prompt: string, hfToken?: string) =>
    request<{ base_response: string; fine_tuned_response: string; base_latency_ms: number; ft_latency_ms: number }>(`/jobs/${jobId}/generate`, { method: "POST", body: JSON.stringify({ prompt, hf_token: hfToken }) }),
  listEvaluations: () => request<any[]>("/evaluations/"),
  getEvaluation: (id: string) => request<any>(`/evaluations/${id}`),
  createEvaluation: (body: object) => request<any>("/evaluations/", { method: "POST", body: JSON.stringify(body) }),
  getAdminStats: () => request<any>("/admin/stats"),
};

"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { auth, encryptApiKey, keyStore } from "@/lib/api";

const PROVIDERS: {
  id: string;
  name: string;
  logoUrl: string;
  desc: string;
  placeholder: string;
  docsUrl: string;
  required: boolean;
  isFree?: boolean;
}[] = [
  {
    id: "openai",
    name: "OpenAI",
    logoUrl: "/OpenAI_logo.webp",
    desc: "Used for synthetic dataset generation",
    placeholder: "sk-...",
    docsUrl: "https://platform.openai.com/api-keys",
    required: true,
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    logoUrl: "/Deepseek_logo.webp",
    desc: "Used for high-throughput, low-cost dataset generation",
    placeholder: "sk-...",
    docsUrl: "https://platform.deepseek.com",
    required: false,
  },
  {
    id: "groq",
    name: "Groq",
    logoUrl: "/groq_logo.png",
    desc: "Free-tier LLM inference — fastest alternative to OpenAI (llama-3.3-70b)",
    placeholder: "gsk_...",
    docsUrl: "https://console.groq.com/keys",
    required: false,
    isFree: true,
  },
  {
    id: "huggingface",
    name: "Hugging Face",
    logoUrl: "/hugging_face_logo.webp",
    desc: "Used to push trained models to HF Hub",
    placeholder: "hf_...",
    docsUrl: "https://huggingface.co/settings/tokens",
    required: false,
  },
  {
    id: "github",
    name: "GitHub",
    logoUrl: "/GitHub_logo.webp",
    desc: "Used to publish Colab notebooks as secret Gists",
    placeholder: "ghp_...",
    docsUrl: "https://github.com/settings/tokens",
    required: false,
  },
  {
    id: "runpod",
    name: "RunPod",
    logoUrl: "/runpod_logo.webp",
    desc: "Used for GPU training jobs (coming soon)",
    placeholder: "rp_...",
    docsUrl: "https://runpod.io",
    required: false,
  },
  {
    id: "aws",
    name: "AWS SageMaker",
    logoUrl: "/AWS_LOGO.webp",
    desc: "Used to deploy live inference endpoints on SageMaker",
    placeholder: "ACCESS_KEY_ID SECRET_ACCESS_KEY (space-separated)",
    docsUrl: "https://console.aws.amazon.com",
    required: false,
  },
];


function KeyRow({ provider, userId }: { provider: typeof PROVIDERS[0]; userId: string }) {
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(false);
  const [show, setShow] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const existing = keyStore.get(provider.id);
    setSaved(!!existing);
  }, [provider.id]);

  const handleSave = async () => {
    if (!value.trim()) return;
    setSaving(true);
    try {
      const sessionKey = sessionStorage.getItem("session_key") || userId;
      const encrypted = await encryptApiKey(value, sessionKey);
      keyStore.save(provider.id, encrypted);
      setSaved(true);
      setValue("");
    } catch (e) {
      console.error("Encryption failed", e);
    } finally {
      setSaving(false);
    }
  };

  const handleClear = () => {
    keyStore.remove(provider.id);
    setSaved(false);
    setValue("");
  };

  return (
    <div className="glass rounded-xl border border-border p-5 hover:border-accent/20 transition-all">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 flex items-center justify-center bg-transparent rounded-lg overflow-hidden shrink-0">
            <img src={provider.logoUrl} alt={provider.name} className="max-w-full max-h-full object-contain" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <p className="font-display font-bold text-text text-sm">{provider.name}</p>
              {provider.required && (
                <span className="text-[10px] font-mono font-bold text-black bg-accent border border-black px-1.5 py-0.5 rounded-full shadow-[1px_1px_0px_0px_#000]">
                  required
                </span>
              )}
            </div>
            <p className="text-xs text-subtle font-body">{provider.desc}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {provider.isFree && (
            <span className="text-[10px] font-mono font-bold text-black bg-green border border-black px-1.5 py-0.5 rounded-full shadow-[1px_1px_0px_0px_#000]">
              free
            </span>
          )}
          {saved && (
            <span className="flex items-center gap-1.5 text-xs text-green font-mono font-bold">
              <span className="w-2 h-2 rounded-full bg-green border border-black" />
              Saved
            </span>
          )}
          <a href={provider.docsUrl} target="_blank" rel="noreferrer"
            className="text-xs text-accent hover:underline font-mono font-bold">
            Get Key ↗
          </a>
        </div>
      </div>

      {saved ? (
        <div className="flex items-center gap-3">
          <div className="flex-1 input-dark px-3 py-2.5 rounded-lg text-sm font-mono text-subtle border-2 border-black">
            ••••••••••••••••••••••••••••••••
          </div>
          <button onClick={handleClear}
            className="btn-ghost px-4 py-2.5 rounded-lg text-xs font-display font-bold whitespace-nowrap">
            Remove
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <input
              type={show ? "text" : "password"}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder={provider.placeholder}
              className="input-dark w-full px-3 py-2.5 rounded-lg text-sm font-mono pr-10 border-2 border-black"
            />
            <button type="button" onClick={() => setShow(!show)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-subtle hover:text-text font-bold text-xs font-mono">
              {show ? "hide" : "show"}
            </button>
          </div>
          <button onClick={handleSave} disabled={!value.trim() || saving}
            className="btn-primary px-4 py-2.5 rounded-lg text-xs font-display font-bold disabled:opacity-50 whitespace-nowrap">
            {saving ? "Saving..." : "Save Key"}
          </button>
        </div>
      )}
    </div>
  );
}

export default function SettingsPage() {
  const router = useRouter();
  const [userId, setUserId] = useState("");

  useEffect(() => {
    if (!auth.isLoggedIn()) { router.push("/auth"); return; }
    setUserId(sessionStorage.getItem("user_id") || auth.getUserId() || "");
  }, []);

  return (
    <div className="max-w-2xl mx-auto px-6 py-8">
      <div className="mb-8 animate-fade-up opacity-0">
        <h1 className="font-display font-bold text-3xl text-text mb-1">Preferences & Keys</h1>
        <p className="text-sm text-subtle font-body">Manage external API integrations and keys.</p>
      </div>

      {/* Security notice */}
      <div className="glass rounded-xl p-5 mb-8 animate-fade-up opacity-0 delay-100 border-2 border-black">
        <div className="flex items-start gap-4">
          <span className="text-2xl">🔐</span>
          <div>
            <p className="text-sm font-display font-bold text-text mb-1">Client-Side Encryption Active</p>
            <p className="text-xs text-subtle font-body leading-relaxed">
              Your API keys are encrypted in your browser using <span className="text-white font-mono font-bold">AES-GCM-256</span> before
              being stored. The encryption key is derived locally and never leaves your device.
              Even our backend database cannot see your raw keys.
            </p>
          </div>
        </div>
      </div>

      {/* API Keys */}
      <div className="animate-fade-up opacity-0 delay-200">
        <h2 className="font-display font-bold text-text text-lg mb-4">API keys</h2>
        <div className="space-y-4">
          {PROVIDERS.map((p) => (
            <KeyRow key={p.id} provider={p} userId={userId} />
          ))}
        </div>
      </div>

      {/* Account */}
      <div className="mt-12 animate-fade-up opacity-0 delay-300">
        <h2 className="font-display font-bold text-text text-lg mb-4">Account Preferences</h2>
        <div className="glass rounded-xl border border-border p-8 text-center">
          <p className="text-3xl mb-2">🔧</p>
          <p className="text-sm text-subtle font-body font-bold">Billing & Usage details coming soon</p>
          <p className="text-xs text-subtle font-mono mt-1">Profile management and team access controls.</p>
        </div>
      </div>
    </div>
  );
}

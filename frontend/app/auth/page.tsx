"use client";
import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { api, auth } from "@/lib/api";

function AuthForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [mode, setMode] = useState<"login" | "register">(
    params.get("mode") === "register" ? "register" : "login"
  );
  const [form, setForm] = useState({ email: "", password: "", name: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      let res;
      if (mode === "register") {
        res = await api.register(form.email, form.password, form.name);
      } else {
        res = await api.login(form.email, form.password);
      }
      auth.setToken(res.access_token, res.user_id);
      sessionStorage.setItem("user_email", res.email);
      sessionStorage.setItem("user_id", res.user_id);
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center px-4 relative overflow-hidden">
      {/* Background grid/noise */}
      <div className="noise" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full pointer-events-none"
        style={{ background: "radial-gradient(ellipse, rgba(255,92,54,0.06) 0%, transparent 70%)" }} />

      <div className="w-full max-w-sm relative">
        {/* Logo */}
        <div className="flex items-center justify-center mb-8">
          <Link href="/" style={{ textDecoration: "none" }}>
            <img src="/logo.png" alt="VeritasAI Logo" style={{ height: "48px", display: "block", mixBlendMode: "screen" }} />
          </Link>
        </div>

        {/* Card */}
        <div className="glass rounded-xl p-8 border-4 border-black shadow-[8px_8px_0px_0px_var(--accent)] bg-[#181922]">
          {/* Tabs */}
          <div className="flex gap-2 mb-8 bg-black border-2 border-black rounded-lg p-1.5 shadow-[2px_2px_0px_0px_rgba(0,0,0,0.5)]">
            {(["login", "register"] as const).map((m) => {
              const active = mode === m;
              return (
                <button key={m} onClick={() => setMode(m)}
                  className={`flex-1 py-2 text-xs font-display font-bold rounded-md transition-all capitalize border-2
                    ${active ? "bg-accent border-black text-black shadow-[2px_2px_0px_0px_#000]" : "border-transparent text-subtle hover:text-text"}`}>
                  {m === "login" ? "Sign in" : "Register"}
                </button>
              );
            })}
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <div>
                <label className="text-xs text-text font-bold block mb-1.5">Full Name</label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="Enter your name"
                  className="input-dark w-full px-3 py-2.5 rounded-lg text-sm font-body border-2 border-black"
                  required
                />
              </div>
            )}
            <div>
              <label className="text-xs text-text font-bold block mb-1.5">Email</label>
              <input
                type="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                placeholder="you@example.com"
                className="input-dark w-full px-3 py-2.5 rounded-lg text-sm font-body border-2 border-black"
                required
              />
            </div>
            <div>
              <label className="text-xs text-text font-bold block mb-1.5">Password</label>
              <input
                type="password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                placeholder="••••••••"
                className="input-dark w-full px-3 py-2.5 rounded-lg text-sm font-body border-2 border-black"
                required
              />
              {mode === "register" && (
                <p className="text-[10px] text-subtle mt-1.5 font-mono font-bold">
                  Your password is used to encrypt your API keys client-side. We never see it.
                </p>
              )}
            </div>

            {error && (
              <div className="text-xs text-red bg-red/10 border-2 border-black rounded-lg px-3 py-2.5 font-bold">
                {error}
              </div>
            )}

            <button type="submit" disabled={loading}
              className="btn-primary w-full py-2.5 rounded-lg text-sm font-display font-bold disabled:opacity-50 disabled:cursor-not-allowed mt-2">
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3 h-3 border border-black/30 border-t-black rounded-full animate-spin" />
                  {mode === "login" ? "Signing in..." : "Registering..."}
                </span>
              ) : (
                mode === "login" ? "Sign in →" : "Create account →"
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-subtle mt-6 font-bold">
          <Link href="/" className="hover:text-accent transition-colors">← Back to home</Link>
        </p>
      </div>
    </div>
  );
}

export default function AuthPage() {
  return (
    <Suspense>
      <AuthForm />
    </Suspense>
  );
}

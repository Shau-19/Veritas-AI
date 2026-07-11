"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";

const THEMES = [
  { name: "Orange", accent: "#ff5c36", glow: "rgba(255,92,54,", dim: "#ba3d1e" },
  { name: "Emerald", accent: "#10b981", glow: "rgba(16,185,129,",  dim: "#065f46" },
  { name: "Rose",    accent: "#f43f5e", glow: "rgba(244,63,94,",   dim: "#9f1239" },
  { name: "Amber",   accent: "#f59e0b", glow: "rgba(245,158,11,",  dim: "#92400e" },
  { name: "Cyan",    accent: "#06b6d4", glow: "rgba(6,182,212,",   dim: "#164e63" },
];

const FEATURES = [
  { icon: `<path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"/>`, title: "Synthetic Data Studio", desc: "Describe your use case. Generate hundreds of clean, diverse, noise-filtered training examples in seconds.", live: true },
  { icon: `<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>`, title: "Quality Intelligence", desc: "Every dataset audited for noise, duplicates, and generic outputs before a single GPU second is spent.", live: true },
  { icon: `<path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>`, title: "Dataset Repair Engine", desc: "Detects weak rows automatically. Regenerates only those. Iterates until quality threshold is met.", live: true },
  { icon: `<path d="M13 10V3L4 14h7v7l9-11h-7z"/>`, title: "GPU Fine-tuning", desc: "LoRA training on RTX 4090s via RunPod. Pay per second, not per month. Llama 3, Mistral, Phi-3.", live: false },
  { icon: `<path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>`, title: "Training Observatory", desc: "Watch loss curves, token costs, and sample outputs in real time as your model trains.", live: false },
  { icon: `<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>`, title: "Key Vault", desc: "API keys encrypted client-side with AES-GCM-256. We never see your raw credentials. Ever.", live: true },
];

const DEMO_ROWS = [
  { input: "Why do houses near good schools cost more?", output: "Proximity to high-rated schools increases demand from families, creating a supply-demand imbalance. Studies show each school rating point correlates with a 1.8% price premium within a 2km radius." },
  { input: "How does a house's age affect its resale value?", output: "Older homes depreciate structurally but may appreciate as historical properties. A 10-year-old home in good condition typically retains 85% of new-build value, while pre-1950 homes in desirable areas often command premiums." },
  { input: "What makes Mumbai real estate more expensive than Pune?", output: "Mumbai's land scarcity, financial hub status, and superior infrastructure drive prices 3-4× higher than Pune. The city's limited geography forces vertical development, compressing supply while demand from finance and entertainment sectors remains high." },
];

export default function LandingPage() {
  const [themeIdx, setThemeIdx] = useState(0);
  const [mouse, setMouse] = useState({ x: 50, y: 50 });
  const [demoIdx, setDemoIdx] = useState(0);
  const [demoFade, setDemoFade] = useState(true);
  const heroRef = useRef<HTMLDivElement>(null);
  const theme = THEMES[themeIdx];

  // Theme cycle
  useEffect(() => {
    const iv = setInterval(() => {
      setThemeIdx(i => (i + 1) % THEMES.length);
    }, 8000);
    return () => clearInterval(iv);
  }, []);

  // Apply theme CSS vars
  useEffect(() => {
    document.documentElement.style.setProperty("--accent", theme.accent);
    document.documentElement.style.setProperty("--accent-dim", theme.dim);
  }, [theme]);

  // Demo rotation
  useEffect(() => {
    const iv = setInterval(() => {
      setDemoFade(false);
      setTimeout(() => { setDemoIdx(i => (i + 1) % DEMO_ROWS.length); setDemoFade(true); }, 400);
    }, 5000);
    return () => clearInterval(iv);
  }, []);

  // Scroll reveal
  useEffect(() => {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { (e.target as HTMLElement).classList.add("in"); obs.unobserve(e.target); }});
    }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });
    document.querySelectorAll(".vt-reveal").forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  // Mouse tracking
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!heroRef.current) return;
      const r = heroRef.current.getBoundingClientRect();
      setMouse({ x: ((e.clientX - r.left) / r.width) * 100, y: ((e.clientY - r.top) / r.height) * 100 });
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  const row = DEMO_ROWS[demoIdx];
  return (
    <>
      <div className="noise" />

      {/* Theme dots */}
      <div className="theme-indicator">
        {THEMES.map((t, i) => (
          <div key={t.name} className={`theme-dot${i === themeIdx ? " active" : ""}`}
            style={{ background: t.accent }} onClick={() => setThemeIdx(i)} title={t.name} />
        ))}
      </div>

      {/* MAIN */}
      <div id="main" className="sketchy-grid">
        {/* NAV */}
        <nav>
          <Link href="/" className="logo">
            <img src="/logo.png" alt="VeritasAI Logo" style={{ height: "40px", display: "block", mixBlendMode: "screen" }} />
          </Link>
          <div className="nav-links">
            {[["How it works","#how"], ["Features","#features"], ["Pricing","#pricing"]].map(([l, h]) => (
              <a key={l} href={h} className="nav-a">{l}</a>
            ))}
            <Link href="/auth" className="nav-a">Sign in</Link>
            <Link href="/auth?mode=register" className="nav-btn" style={{ background: theme.accent }}>
              Start free →
            </Link>
          </div>
        </nav>

        {/* HERO */}
        <div ref={heroRef} className="hero">
          <div className="hero-gl" />
          {/* Mouse glow */}
          <div style={{ position: "absolute", width: 700, height: 700, borderRadius: "50%", background: `radial-gradient(circle, ${theme.glow}0.05) 0%, transparent 70%`, left: `${mouse.x}%`, top: `${mouse.y}%`, transform: "translate(-50%,-50%)", pointerEvents: "none", transition: "left .8s ease, top .8s ease, background .5s", zIndex: 0 }} />
          {/* Static glow */}
          <div style={{ position: "absolute", top: "50%", left: "30%", transform: "translate(-50%,-50%)", width: 800, height: 400, borderRadius: "50%", background: `radial-gradient(ellipse, ${theme.glow}0.05) 0%, transparent 70%`, pointerEvents: "none", transition: "background .5s" }} />
          <div style={{ position: "relative", zIndex: 1 }} className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center w-full max-w-6xl mx-auto">
            <div className="lg:col-span-7">
              <div className="hero-badge" style={{ background: theme.accent }}>
                <div className="bdot" />
                <span className="btxt">Dataset generation · live · beta</span>
              </div>
              
              <h1 className="hero-h1">
                Train <span className="marker-highlight">Smarter.</span><br />
                Not on <span className="marker-highlight">noise.</span>
              </h1>

              <p className="hero-p">
                It's time to fine-tune your model. You need <strong>clean, diverse, audited training data</strong> — not generic rows that teach your model bad habits.<br /><br />
                Veritas AI generates it from a sentence, audits every row for quality, repairs weak examples automatically, and fires a LoRA training job for under a dollar.<br /><br />
                <strong>Prompt → clean dataset → fine-tuned model. That's the entire workflow.</strong>
              </p>

              <div className="hero-actions" style={{ display: "flex", alignItems: "center", gap: 24, position: "relative" }}>
                <div style={{ position: "relative" }}>
                  <Link href="/auth?mode=register" className="btn-hero" style={{ background: theme.accent, display: "inline-block" }}>
                    Start fine-tuning! →
                  </Link>
                  
                  {/* Hand-drawn scribbled SVG arrow */}
                  <svg className="absolute hidden md:block" style={{ width: 140, height: 80, left: "100%", top: "50%", transform: "translate(10px, -50%)", pointerEvents: "none", zIndex: 10 }} viewBox="0 0 140 80">
                    <path d="M120,15 C90,12 40,22 5,40 M5,40 L18,30 M5,40 L18,50" stroke={theme.accent} strokeWidth="3.5" fill="none" strokeLinecap="round" />
                    <text x="35" y="28" fill={theme.accent} fontSize="14" fontWeight="bold" fontFamily="'Comic Neue', cursive">click here!</text>
                  </svg>
                </div>

                <a href="#how" className="btn-ghost2" style={{ marginLeft: 160 }}>See how it works ↓</a>
              </div>
            </div>

            <div className="lg:col-span-5 flex justify-center w-full vt-reveal d3">
              {/* Desktop interactive wobbly stickers collage */}
              <div className="hidden lg:block relative w-full max-w-[420px] h-[720px] select-none overflow-visible">
                {/* Background glow behind collage */}
                <div style={{
                  position: "absolute",
                  inset: "-40px",
                  background: `radial-gradient(circle, ${theme.glow}0.08) 0%, transparent 70%`,
                  zIndex: 0,
                  pointerEvents: "none"
                }} />

                {/* Speech bubbles & decorative drawings (leaves/sparks) to match mockup */}
                <div className="absolute top-[100px] right-[-10px] bg-[#1e1f29] text-white border-2 border-black px-3 py-1 text-xs rounded-full font-bold shadow-[2px_2px_0px_0px_#000] rotate-[8deg] z-40 animate-bounce" style={{ animationDuration: '3s' }}>
                  Certainly!
                </div>
                <div className="absolute top-[280px] left-[-30px] bg-[#1e1f29] text-white border-2 border-black px-3 py-1 text-xs rounded-full font-bold shadow-[2px_2px_0px_0px_#000] rotate-[-6deg] z-40 animate-pulse">
                  Clean & Ready!
                </div>
                <div className="absolute top-[480px] right-[10px] bg-[#1e1f29] text-white border-2 border-black px-3 py-1 text-xs rounded-full font-bold shadow-[2px_2px_0px_0px_#000] rotate-[4deg] z-40 animate-bounce" style={{ animationDuration: '4s' }}>
                  Sure!
                </div>

                {/* Hand-drawn stars and sparks (SVGs) */}
                <svg className="absolute top-[20px] left-[160px] w-6 h-6 text-yellow-500 fill-current opacity-70 animate-pulse" viewBox="0 0 24 24">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
                <svg className="absolute top-[420px] left-[10px] w-5 h-5 text-yellow-500 fill-current opacity-60 animate-ping" viewBox="0 0 24 24" style={{ animationDuration: '3s' }}>
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
                <svg className="absolute bottom-[100px] left-[150px] w-6 h-6 text-yellow-500 fill-current opacity-75 animate-bounce" viewBox="0 0 24 24">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>

                {/* Staggered floating Collage stickers */}
                <div className="collage-item animate-float-1 w-[130px] z-20 top-[0px] left-[20px]">
                  <div className="glass p-1.5 rotate-[-6deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_1.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-2 w-[140px] z-10 top-[30px] right-[20px]">
                  <div className="glass p-1.5 rotate-[3deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_2.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-3 w-[125px] z-30 top-[140px] left-[80px]">
                  <div className="glass p-1.5 rotate-[-3deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_3.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-4 w-[135px] z-20 top-[220px] right-[40px]">
                  <div className="glass p-1.5 rotate-[6deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_4.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-5 w-[145px] z-10 top-[300px] left-[10px]">
                  <div className="glass p-1.5 rotate-[-12deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_5.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-6 w-[135px] z-30 top-[380px] right-[30px]">
                  <div className="glass p-1.5 rotate-[3deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_6.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-7 w-[125px] z-20 top-[460px] left-[60px]">
                  <div className="glass p-1.5 rotate-[6deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_7.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>

                <div className="collage-item animate-float-8 w-[145px] z-10 top-[500px] right-[20px]">
                  <div className="glass p-1.5 rotate-[-6deg] hover:rotate-0 hover:scale-110 transition-all duration-300">
                    <img src="/collage_8.png" alt="VeritasAI Character" className="w-full h-auto rounded" />
                  </div>
                </div>
              </div>

              {/* Mobile static wobbly card collage fallback */}
              <div className="lg:hidden glass p-2 w-full max-w-md overflow-hidden hover:scale-[1.02] transition-all duration-300">
                <img 
                  src="/landing_illustration.jpg" 
                  alt="VeritasAI Workflow Illustration" 
                  className="w-full h-auto rounded-lg"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="d2l" />

        {/* HOW IT WORKS */}
        <section className="how-sec" id="how">
          <div style={{ maxWidth: 1100, margin: "0 auto" }}>
            <div className="how-label-row vt-reveal">
              <div>
                <div className="sec-lbl dark" style={{ marginBottom: 10 }}>The workflow</div>
                <h2 className="how-h">From sentence to <span className="marker-highlight">fine-tuned model.</span></h2>
              </div>
              <p className="how-sub">Four steps. No ML expertise. No GPU setup. No data science team.</p>
            </div>
            <div className="steps">
              {[
                { n: "01", title: "Describe your use case", desc: "Type what you want your model to do. One sentence is enough.", outs: ["Topic + task type", "Style preference", "Domain context"], icon: `<path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z" stroke-linecap="round" stroke-linejoin="round" /><path d="M8 10h5M8 14h8" stroke-linecap="round" />` },
                { n: "02", title: "Generate clean dataset", desc: "The platform generates 100–1000 diverse examples, filters noise, audits quality, and repairs weak rows automatically.", outs: ["Zero duplicates", "No filler outputs", "Repair engine"], icon: `<path d="M4 21V3 M12 21V3 M20 21V3" stroke-linecap="round" /><circle cx="4" cy="9" r="2.5" /><circle cx="12" cy="15" r="2.5" /><circle cx="20" cy="8" r="2.5" />` },
                { n: "03", title: "Fine-tune on GPU", desc: "Pick a base model. Click train. An RTX 4090 spins up, trains with LoRA, shuts down. You pay ~$0.50.", outs: ["Llama 3 · Mistral · Phi-3", "Pay per second", "Auto-destroy after"], icon: `<circle cx="12" cy="12" r="8" fill="none" /><path d="M8.5 11.5 Q9.5 9 10.5 11.5 M13.5 11.5 Q14 9.5 15.5 11.5" stroke-linecap="round" fill="none" /><path d="M9 14.5c.8 1.5 2.2 2 3 2s2.2-.5 3-2" stroke-linecap="round" fill="none" /><path d="M2.5 15.5c.5-1 1.5-2.5 2.5-1.5s0 2.5-.5 3 M21.5 15.5c-.5-1-1.5-2.5-2.5-1.5s0 2.5.5 3" stroke-linecap="round" fill="none" />` },
                { n: "04", title: "Deploy to HuggingFace", desc: "Your fine-tuned model is pushed to your Hugging Face Hub. Ready via API instantly.", outs: ["Auto-push to HF Hub", "Eval scores attached", "API-ready instantly"], icon: `<path d="M4 9h16v10a2 2 0 01-2 2H6a2 2 0 01-2-2V9zM2 6h20v3H2z" stroke-linecap="round" stroke-linejoin="round" /><path d="M9 13v.01 M15 13v.01 M9 16 Q12 18.5 15 16" stroke-linecap="round" fill="none" /><path d="M12 1v5 M9 3.5l3-2.5 3 2.5" stroke-linecap="round" stroke-linejoin="round" />` },
              ].map((s, i) => (
                <div key={s.n} className={`step vt-reveal d${i + 1}`}>
                  <div className="step-num" style={{ color: theme.accent }}>{s.n}</div>
                  <div className="step-icon">
                    <svg viewBox="0 0 24 24" dangerouslySetInnerHTML={{ __html: s.icon }} />
                  </div>
                  <div className="step-title">{s.title}</div>
                  <div className="step-desc">{s.desc}</div>
                  <div className="step-outs">
                    {s.outs.map(o => <div key={o} className="step-out">
                      <span style={{ color: theme.accent }}>→</span>{o}
                    </div>)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* LIVE DEMO */}
        <section className="demo-sec">
          <div className="demo-inner vt-reveal">
            <div>
              <div className="sec-lbl" style={{ color: theme.accent, marginBottom: 18 }}>
                Quality Intelligence · Live audit
              </div>
              <h2 className="demo-copy-h">Not just "did it generate?" —<br /><span className="marker-highlight">is it actually good?</span></h2>
              <p className="demo-copy-p">Every dataset runs through our audit engine before you see it. We detect semantic duplicates, short outputs, generic filler phrases, low variance, and contradiction patterns — then repair them automatically.</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {[["01", "Semantic deduplication", "Near-identical inputs are detected and replaced. Models overfit on duplicate patterns."],
                  ["02", "Generic filler detection", "Outputs starting with 'Certainly!', 'Sure!', 'Great question' are filtered and regenerated."],
                  ["03", "Entropy scoring", "Lexical diversity, bigram variance, and structural patterns are measured and scored."],
                  ["04", "Repair iterations", "Weak rows are detected, regenerated in-place, and re-audited until the threshold is met."],
                ].map(([n, title, desc]) => (
                  <div key={n} style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                    <span style={{ fontSize: 13, fontWeight: "bold", color: theme.accent, width: 18, flexShrink: 0, marginTop: 2 }}>{n}</span>
                    <div style={{ fontSize: 14, color: "#a0a5c0", lineHeight: 1.65 }}>
                      <strong style={{ color: "#ffffff", fontWeight: "bold" }}>{title}</strong> — {desc}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            {/* Live audit card */}
            <div className="audit-card vt-reveal d2">
              <div className="ac-bar" style={{ background: theme.accent }} />
              <div className="ac-head">
                <div className="ac-title" style={{ opacity: demoFade ? 1 : 0 }}>{row.input}</div>
                <div className="ac-score">
                  <div className="ac-score-num">PASS</div>
                  <div className="ac-score-lbl">Status</div>
                </div>
              </div>
              <div className="ac-status">
                <div className="ac-badge">Clean & Ready</div>
                <div className="ac-status-text">0 issues detected · Repair not needed</div>
              </div>
              <div className="ac-rows">
                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, fontWeight: "bold", color: "var(--accent)", marginBottom: 6 }}>Sample row · rotating</div>
                <div className="ac-row-item" style={{ opacity: demoFade ? 1 : 0 }}>
                  <div className="row-in">Input</div>
                  <div className="row-out" style={{ fontWeight: "bold" }}>{row.input}</div>
                </div>
                <div className="ac-row-item" style={{ opacity: demoFade ? 1 : 0, transitionDelay: demoFade ? ".1s" : "0s" }}>
                  <div className="row-in">Output</div>
                  <div className="row-out" style={{ color: "#a0a5c0" }}>{row.output}</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* FEATURES */}
        <section className="feat-sec" id="features">
          <div style={{ maxWidth: 1100, margin: "0 auto" }}>
            <div className="sec-lbl dark vt-reveal" style={{ marginBottom: 16 }}>Platform capabilities</div>
            <h2 className="feat-h vt-reveal">Everything you need to build<br />a fine-tuned model from scratch.</h2>
            <div className="feat-grid">
              {FEATURES.map((f, i) => (
                <div key={f.title} className={`feat vt-reveal d${(i % 4) + 1}`}>
                  {!f.live && <span className="feat-soon">Coming soon</span>}
                  <div className="feat-icon">
                    <svg viewBox="0 0 24 24" dangerouslySetInnerHTML={{ __html: f.icon }} />
                  </div>
                  <div className="feat-title">{f.title}</div>
                  <div className="feat-desc">{f.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* TECH STACK LIST SECTION */}
        <section className="tech-sec">
          <div className="tech-row">
            {["FastAPI backend", "Celery + Redis queue", "OpenAI gpt-4o-mini", "Unsloth LoRA training", "RunPod GPU pods", "HuggingFace Hub"].map(t => (
              <div key={t} className="tech-p">{t}</div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className="cta-sec" id="pricing">
          <div style={{ position: "relative" }}>
            <div style={{ fontSize: 12, fontWeight: "bold", textTransform: "uppercase", color: "var(--accent)", marginBottom: 18 }}>No credit card · Free to start</div>
            <h2 className="cta-h vt-reveal">
              Your first fine-tuned model,<br /><span className="marker-highlight">built tonight.</span>
            </h2>
            <p className="cta-sub vt-reveal">Generate clean datasets free. GPU training costs ~$0.50. No subscriptions, no minimums.</p>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 14, flexWrap: "wrap" }} className="vt-reveal">
              <Link href="/auth?mode=register" className="btn-cta" style={{ background: theme.accent }}>
                Start building free →
              </Link>
              <Link href="/auth" className="btn-ghost2">Sign in</Link>
            </div>
            <div className="cta-note vt-reveal">Free dataset generation · GPU pay-as-you-go · AES-256 key encryption</div>
          </div>
        </section>

        {/* FOOTER */}
        <footer className="footer">
          <Link href="/" className="logo">
            <img src="/logo.png" alt="VeritasAI Logo" style={{ height: "32px", display: "block", mixBlendMode: "screen" }} />
          </Link>
          <div className="foot-links">
            <a href="#how" className="foot-link">How it works</a>
            <a href="#features" className="foot-link">Features</a>
            <Link href="/auth" className="foot-link">Sign in</Link>
          </div>
          <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: "#4f526b" }}>© 2026 Veritas AI</div>
        </footer>
      </div>
    </>
  );
}

"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { auth } from "@/lib/api";

const NAV = [
  { label: "Dashboard", href: "/dashboard", live: true },
  { label: "Generate",  href: "/generate",  live: true },
  { label: "Jobs",      href: "/jobs",      live: true },
  { label: "Evaluate",  href: "/evaluate",  live: true },
  { label: "Deploy",    href: "/deploy",    live: true },
  { label: "Settings",  href: "/settings",  live: true },
];


const THEMES = ["#ff5c36", "#10b981", "#f43f5e", "#f59e0b", "#06b6d4"];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [accent, setAccent] = useState("#ff5c36");
  const [scrolled, setScrolled] = useState(false);
  const [themeIdx, setThemeIdx] = useState(0);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    // Sync login status on client mount
    setIsLoggedIn(auth.isLoggedIn());
    // Sync with landing page theme if set
    const stored = document.documentElement.style.getPropertyValue("--accent");
    if (stored) setAccent(stored);
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const cycleTheme = () => {
    const next = (themeIdx + 1) % THEMES.length;
    setThemeIdx(next);
    setAccent(THEMES[next]);
    document.documentElement.style.setProperty("--accent", THEMES[next]);
  };

  const handleLogout = () => {
    auth.clear();
    setIsLoggedIn(false);
    router.push("/");
  };

  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 700,
      padding: "0 52px", height: 54, display: "flex", alignItems: "center", justifyContent: "space-between",
      background: "#0f0f12",
      borderBottom: "3px solid #000000",
      fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
    }} className="justify-between">
      {/* Logo: Custom VeritasAI image */}
      <Link href="/dashboard" style={{ textDecoration: "none", display: "inline-flex", alignItems: "center" }}>
        <img src="/logo.png" alt="VeritasAI Logo" style={{ height: "40px", display: "block", mixBlendMode: "screen" }} />
      </Link>

      {/* Tabs */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link key={item.href} href={item.live ? item.href : "#"}
              style={{
                position: "relative",
                padding: "6px 14px",
                fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                fontSize: 12,
                fontWeight: active ? "bold" : "normal",
                textTransform: "uppercase",
                textDecoration: "none",
                transition: "color .2s",
                opacity: item.live ? 1 : 0.5,
                cursor: item.live ? "pointer" : "not-allowed",
                border: active ? "3px solid #000000" : "3px solid transparent",
                background: active ? accent : "transparent",
                color: active ? "#000000" : (item.live ? "#a0a5c0" : "#4f526b"),
                borderRadius: "20px 4px 18px 4px/4px 18px 4px 20px",
                boxShadow: active ? "2px 2px 0px 0px #000000" : "none",
              }}>
              {item.label}
              {!item.live && <span style={{ fontSize: 8, verticalAlign: "super", color: "#4f526b", marginLeft: 2 }}>soon</span>}
            </Link>
          );
        })}
      </div>

      {/* Right */}
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        {/* Theme toggle */}
        <button onClick={cycleTheme} title="Change theme"
          style={{ width: 22, height: 22, borderRadius: "50%", background: accent, border: "2px solid #000000", boxShadow: "2px 2px 0px 0px #000000", cursor: "pointer", transition: "background .5s", flexShrink: 0 }} />
        {isLoggedIn
          ? <button onClick={handleLogout}
              style={{
                fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                fontSize: 12,
                fontWeight: "bold",
                textTransform: "uppercase",
                color: "#a0a5c0",
                background: "none",
                border: "none",
                cursor: "pointer",
                transition: "color .2s"
              }}
              onMouseOver={e => (e.currentTarget.style.color = "#ffffff")}
              onMouseOut={e => (e.currentTarget.style.color = "#a0a5c0")}>
              Sign out
            </button>
          : <Link href="/auth"
              style={{
                padding: "6px 16px",
                background: accent,
                color: "#000000",
                border: "3px solid #000000",
                boxShadow: "2px 2px 0px 0px #000000",
                borderRadius: "40px 4px 30px 4px/4px 30px 4px 40px",
                fontFamily: "'Comic Neue', 'Comic Sans MS', cursive, sans-serif",
                fontSize: 11,
                fontWeight: "bold",
                textTransform: "uppercase",
                textDecoration: "none",
                transition: "all .2s"
              }}>
              Sign in
            </Link>
        }
      </div>
    </nav>
  );
}

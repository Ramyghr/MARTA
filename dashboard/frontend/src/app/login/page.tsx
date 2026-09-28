"use client";
import { useState, type FormEvent } from "react";
import Link from "next/link";
import { login } from "@/lib/auth";
import { MartaBot } from "@/components/MartaBot";
import { AuthBackground } from "@/components/AuthBackground";
import { SocialLoginButtons } from "@/components/SocialLoginButtons";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      // Full navigation — AuthGate needs a fresh /api/auth/me check, not a client
      // route change (its useCurrentUser() hook instance won't otherwise know).
      window.location.href = "/dashboard";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
      setSubmitting(false);
    }
  }

  return (
    <div className="relative h-screen w-screen flex items-center justify-center bg-[#0a0a0a] px-4 overflow-hidden">
      <AuthBackground />
      <div className="relative z-10 w-full max-w-sm">
        <div className="flex items-center justify-center mb-2">
          <MartaBot size={120} showChart />
        </div>
        <div className="flex items-center justify-center gap-2 mb-6">
          <span
            className="w-1.5 h-1.5 rounded-full inline-block animate-pulse-dot"
            style={{ background: "#00ff7f", boxShadow: "0 0 6px #00ff7f" }}
          />
          <span className="font-bold text-[#00ff7f] tracking-widest text-lg glow-green">MARTA</span>
        </div>

        <form
          onSubmit={handleSubmit}
          className="bg-[#0d0d0d]/90 backdrop-blur-sm border border-[#2a2a2a] rounded-lg p-5 flex flex-col gap-3 shadow-[0_0_50px_rgba(0,255,127,0.06)]"
        >
          <h1 className="text-[13px] font-bold text-[#e0e0e0] mb-1">Log in</h1>

          <div>
            <label className="stat-label block mb-1">Email</label>
            <input
              type="email"
              required
              autoFocus
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] outline-none focus:border-[#00ff7f44]"
            />
          </div>
          <div>
            <label className="stat-label block mb-1">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] outline-none focus:border-[#00ff7f44]"
            />
          </div>

          {error && (
            <div className="text-[11px] text-[#ff6b6b] px-2 py-1.5 border border-[#ff373733] rounded bg-[#ff373708]">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="mt-2 w-full py-2 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? "Logging in…" : "Log in"}
          </button>

          <SocialLoginButtons />

          <p className="text-[11px] text-[#555] text-center mt-1">
            No account?{" "}
            <Link href="/signup" className="text-[#4a9eff] hover:text-[#6db4ff] transition-colors">
              Sign up
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}

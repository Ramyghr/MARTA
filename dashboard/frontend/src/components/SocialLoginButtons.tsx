"use client";
import { useEffect, useState } from "react";
import { getAuthProviders, googleLoginUrl, githubLoginUrl } from "@/lib/auth";

// Renders the "or" divider + whichever OAuth buttons GET /api/auth/providers
// confirms are actually configured server-side — avoids showing buttons that
// 500, and avoids two dividers when both providers happen to be enabled.
export function SocialLoginButtons() {
  const [providers, setProviders] = useState({ google: false, github: false });

  useEffect(() => {
    getAuthProviders()
      .then(setProviders)
      .catch(() => setProviders({ google: false, github: false }));
  }, []);

  if (!providers.google && !providers.github) return null;

  return (
    <>
      <div className="flex items-center gap-2 my-1">
        <div className="flex-1 h-px bg-[#2a2a2a]" />
        <span className="text-[10px] text-[#555] uppercase tracking-wider">or</span>
        <div className="flex-1 h-px bg-[#2a2a2a]" />
      </div>

      {providers.google && (
        <a
          href={googleLoginUrl()}
          className="w-full py-2 rounded text-[11px] font-semibold flex items-center justify-center gap-2 bg-[#111] border border-[#2a2a2a] text-[#ccc] hover:border-[#3a3a3a] hover:bg-[#161616] transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 48 48" aria-hidden="true">
            <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.1 8 3l6-6C34.5 5.6 29.5 3.5 24 3.5 12.7 3.5 3.5 12.7 3.5 24S12.7 44.5 24 44.5 44.5 35.3 44.5 24c0-1.2-.1-2.4-.4-3.5z"/>
            <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.9 19 13 24 13c3.1 0 5.8 1.1 8 3l6-6C34.5 6.6 29.5 4.5 24 4.5c-7.7 0-14.3 4.4-17.7 10.2z"/>
            <path fill="#4CAF50" d="M24 44.5c5.4 0 10.3-2 14-5.5l-6.5-5.4c-2 1.5-4.6 2.4-7.5 2.4-5.3 0-9.7-3.3-11.3-8l-6.6 5C9.6 39.9 16.2 44.5 24 44.5z"/>
            <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.3-4.3 5.7l6.5 5.4C41.3 35.9 44.5 30.4 44.5 24c0-1.2-.1-2.4-.4-3.5z"/>
          </svg>
          Continue with Google
        </a>
      )}

      {providers.github && (
        <a
          href={githubLoginUrl()}
          className="w-full py-2 rounded text-[11px] font-semibold flex items-center justify-center gap-2 bg-[#111] border border-[#2a2a2a] text-[#ccc] hover:border-[#3a3a3a] hover:bg-[#161616] transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
          </svg>
          Continue with GitHub
        </a>
      )}
    </>
  );
}

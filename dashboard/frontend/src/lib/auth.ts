"use client";
import { useEffect, useState, useCallback } from "react";
import { API_URL } from "./websocket";

export interface User {
  id: number;
  email: string;
  display_name: string;
}

// Raw fetch (not fetchApi) for auth calls — we need to read the response body on
// non-2xx (e.g. "Invalid email or password") rather than fetchApi's generic
// "API error {status}", and POST bodies here always need Content-Type: application/json.
async function authFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error((body as { detail?: string } | null)?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export function login(email: string, password: string) {
  return authFetch<{ user: User }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function signup(email: string, password: string, displayName?: string) {
  return authFetch<{ user: User }>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, display_name: displayName ?? "" }),
  });
}

export function logout() {
  return authFetch<{ ok: boolean }>("/api/auth/logout", { method: "POST" }).catch(() => {});
}

export function getMe() {
  return authFetch<{ user: User }>("/api/auth/me");
}

export function updateProfile(body: { display_name?: string; current_password?: string; new_password?: string }) {
  return authFetch<{ user: User }>("/api/auth/profile", { method: "PATCH", body: JSON.stringify(body) });
}

// Lets login/signup hide the "Continue with Google/GitHub" buttons until the
// backend actually has the corresponding CLIENT_ID/SECRET configured, instead
// of showing a dead button.
export function getAuthProviders() {
  return authFetch<{ google: boolean; github: boolean }>("/api/auth/providers");
}

// Full navigations, not fetches — these need to leave the SPA and hit the provider.
export function googleLoginUrl() {
  return `${API_URL}/api/auth/google/login`;
}

export function githubLoginUrl() {
  return `${API_URL}/api/auth/github/login`;
}

// Shared by the Sidebar profile widget and the AppShell auth guard — both need to
// know "who's logged in" without each independently re-fetching /api/auth/me.
export function useCurrentUser() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    getMe()
      .then((d) => setUser(d.user))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { user, loading, refresh };
}

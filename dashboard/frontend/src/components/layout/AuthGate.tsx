"use client";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";
import { useCurrentUser } from "@/lib/auth";
import { AppShell } from "./AppShell";
import { DemoMode } from "@/components/DemoMode";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";
import { MARTAChat } from "@/components/MARTAChat";
import { MartaBot } from "@/components/MartaBot";

// /login and /signup render standalone (no sidebar/topbar, no auth check — that's
// the whole point). Everything else requires a session: redirect to /login if
// useCurrentUser() comes back empty once the initial /api/auth/me check settles.
const PUBLIC_PATHS = new Set(["/login", "/signup"]);

export function AuthGate({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const isPublic = PUBLIC_PATHS.has(pathname ?? "");
  const { user, loading } = useCurrentUser();

  useEffect(() => {
    if (!isPublic && !loading && !user) router.replace("/login");
  }, [isPublic, loading, user, router]);

  if (isPublic) return <>{children}</>;

  if (loading || !user) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center gap-4 bg-[#0a0a0a]">
        <MartaBot size={140} showChart />
        <span className="text-[#00ff7f] text-xs tracking-[0.3em] font-bold animate-pulse">MARTA</span>
      </div>
    );
  }

  return (
    <>
      <AppShell>{children}</AppShell>
      <DemoMode />
      <KeyboardShortcuts />
      <MARTAChat />
      <MartaBot size={64} showChart={false} className="fixed bottom-24 right-6 z-40" />
    </>
  );
}

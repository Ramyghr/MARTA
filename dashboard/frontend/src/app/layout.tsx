import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";
import { DemoMode } from "@/components/DemoMode";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";
import { MARTAChat } from "@/components/MARTAChat";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"),
  title: "MARTA — Multi-Agent Regime-Aware Trading",
  description: "Real-time trading dashboard with regime-aware multi-agent deliberation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#0a0a0a] text-[#e0e0e0] min-h-screen">
        <AppShell>{children}</AppShell>
        <DemoMode />
        <KeyboardShortcuts />
        <MARTAChat />
      </body>
    </html>
  );
}

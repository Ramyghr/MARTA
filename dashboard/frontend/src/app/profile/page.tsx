"use client";
import { useEffect, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Panel } from "@/components/ui/Panel";
import { useCurrentUser, updateProfile, logout } from "@/lib/auth";

export default function ProfilePage() {
  const { user, loading, refresh } = useCurrentUser();
  const [displayName, setDisplayName] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    if (user) setDisplayName(user.display_name);
  }, [user]);

  async function handleSave() {
    setError(null);
    setSavedMsg(null);
    if (newPassword && newPassword !== confirmPassword) {
      setError("New passwords don't match");
      return;
    }
    if (newPassword && newPassword.length < 8) {
      setError("New password must be at least 8 characters");
      return;
    }
    setSaving(true);
    try {
      await updateProfile({
        display_name: displayName,
        current_password: newPassword ? currentPassword : undefined,
        new_password: newPassword || undefined,
      });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setSavedMsg("Saved.");
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save changes");
    } finally {
      setSaving(false);
    }
  }

  if (loading || !user) {
    return <div className="flex flex-col gap-2 pb-4"><PageHeader title="Profile" /></div>;
  }

  return (
    <div className="flex flex-col gap-2 pb-4 max-w-lg">
      <PageHeader title="Profile" description="Account details and password" />

      <Panel title="Account">
        <div className="flex flex-col gap-3 text-[11px]">
          <div>
            <label className="stat-label block mb-1">Email</label>
            <div className="text-[#999] font-mono">{user.email}</div>
          </div>
          <div>
            <label className="stat-label block mb-1">Display name</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] outline-none focus:border-[#00ff7f44]"
            />
          </div>
        </div>
      </Panel>

      <Panel title="Change Password">
        <div className="flex flex-col gap-3 text-[11px]">
          <div>
            <label className="stat-label block mb-1">Current password</label>
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              placeholder="only required when setting a new password"
              className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] outline-none focus:border-[#00ff7f44]"
            />
          </div>
          <div>
            <label className="stat-label block mb-1">New password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] outline-none focus:border-[#00ff7f44]"
            />
          </div>
          <div>
            <label className="stat-label block mb-1">Confirm new password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] outline-none focus:border-[#00ff7f44]"
            />
          </div>
        </div>
      </Panel>

      {error && (
        <div className="text-[11px] text-[#ff6b6b] px-3 py-2 border border-[#ff373733] rounded bg-[#ff373708]">{error}</div>
      )}
      {savedMsg && (
        <div className="text-[11px] text-[#00ff7f] px-3 py-2 border border-[#00ff7f33] rounded bg-[#00ff7f08]">{savedMsg}</div>
      )}

      <div className="flex items-center gap-2 mt-1">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-3 py-1.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {saving ? "Saving…" : "Save changes"}
        </button>
        <button
          onClick={async () => { await logout(); window.location.href = "/login"; }}
          className="px-3 py-1.5 rounded text-[11px] font-semibold uppercase tracking-wider text-[#ff6b6b] border border-[#ff373730] hover:bg-[#ff3737]/10 transition-colors"
        >
          Log out
        </button>
      </div>
    </div>
  );
}

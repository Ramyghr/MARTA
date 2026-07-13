'use client'
import { useState, useRef, useEffect } from 'react'
import { MessageCircle, X, Send, Trash2, Lightbulb } from 'lucide-react'
import { useMARTAStore } from '@/lib/store'

// Generated once per page load — good enough for a single-user local dashboard.
const SESSION_ID = `marta_${Math.random().toString(36).slice(2)}`

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export function MARTAChat() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hi! I'm MARTA's AI assistant. Ask me anything about the current market regime, trading decisions, or how MARTA works.",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const selectedAsset = useMARTAStore((s) => s.selectedAsset)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/chat/suggestions/${selectedAsset}`)
      .then((r) => r.json())
      .then((d) => setSuggestions(d.suggestions || []))
      .catch(() => {})
  }, [selectedAsset])

  const sendMessage = async (text?: string) => {
    const message = text || input.trim()
    if (!message || loading) return

    setInput('')
    setShowSuggestions(false)
    setMessages((prev) => [...prev, { role: 'user', content: message, timestamp: new Date() }])
    setLoading(true)

    try {
      const resp = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: SESSION_ID, asset: selectedAsset }),
      })
      const data = await resp.json()
      setMessages((prev) => [...prev, { role: 'assistant', content: data.response, timestamp: new Date() }])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Connection error — make sure the backend is running.', timestamp: new Date() },
      ])
    } finally {
      setLoading(false)
    }
  }

  const clearChat = async () => {
    await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/chat/${SESSION_ID}`, { method: 'DELETE' })
    setMessages([{ role: 'assistant', content: 'Chat cleared. What would you like to know about MARTA?', timestamp: new Date() }])
  }

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className="fixed bottom-6 right-6 z-50 w-12 h-12 rounded-full flex items-center justify-center shadow-lg transition-all"
        style={{
          background: open ? 'var(--red)' : 'var(--green)',
          boxShadow: open ? 'var(--red-glow)' : 'var(--green-glow)',
        }}
      >
        {open ? <X size={20} color="#000" /> : <MessageCircle size={20} color="#000" />}
      </button>

      {open && (
        <div
          className="fixed bottom-20 right-6 z-50 flex flex-col rounded-lg overflow-hidden shadow-2xl"
          style={{ width: '380px', height: '520px', background: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <div
            className="flex items-center justify-between px-3 py-2"
            style={{ background: 'var(--surface-2)', borderBottom: '1px solid var(--border)' }}
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full animate-pulse-dot" style={{ background: 'var(--green)' }} />
              <span className="text-sm font-bold" style={{ color: 'var(--text)' }}>MARTA Assistant</span>
              <span className="text-xs px-1 rounded" style={{ background: 'rgba(0,255,127,0.1)', color: 'var(--green)' }}>
                {selectedAsset}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setShowSuggestions(!showSuggestions)} title="Suggested questions" style={{ color: 'var(--dim)' }}>
                <Lightbulb size={14} />
              </button>
              <button onClick={clearChat} title="Clear chat" style={{ color: 'var(--dim)' }}>
                <Trash2 size={14} />
              </button>
            </div>
          </div>

          {showSuggestions && (
            <div className="p-2 overflow-y-auto" style={{ background: 'var(--surface-2)', borderBottom: '1px solid var(--border)', maxHeight: '140px' }}>
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => { sendMessage(s); setShowSuggestions(false) }}
                  className="w-full text-left text-xs p-1 rounded mb-1 hover:opacity-80 transition-opacity"
                  style={{ background: 'var(--surface-3)', color: 'var(--dim)' }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className="rounded-lg px-3 py-2 text-xs max-w-[85%]"
                  style={{
                    background: msg.role === 'user' ? 'rgba(0,255,127,0.15)' : 'var(--surface-3)',
                    color: 'var(--text)',
                    border: msg.role === 'user' ? '1px solid rgba(0,255,127,0.3)' : '1px solid var(--border)',
                  }}
                >
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>{msg.content}</div>
                  <div className="mt-1 text-right" style={{ color: 'var(--dim)', fontSize: '10px' }}>
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="rounded-lg px-3 py-2 text-xs" style={{ background: 'var(--surface-3)', border: '1px solid var(--border)', color: 'var(--dim)' }}>
                  <span className="animate-pulse">Thinking…</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-2 flex gap-2" style={{ borderTop: '1px solid var(--border)', background: 'var(--surface-2)' }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="Ask about regime, trades, or MARTA…"
              className="flex-1 rounded px-2 py-1 text-xs outline-none"
              style={{ background: 'var(--surface-3)', border: '1px solid var(--border)', color: 'var(--text)' }}
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="rounded px-2 py-1 transition-opacity disabled:opacity-40"
              style={{ background: 'var(--green)', color: '#000' }}
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      )}
    </>
  )
}

'use client'
import { useState, useRef, useEffect, useCallback } from 'react'
import { MessageCircle, X, Send, Trash2, Lightbulb, History, Plus } from 'lucide-react'
import { useMARTAStore } from '@/lib/store'

const SESSION_STORAGE_KEY = 'marta_chat_session_id'

function newSessionId(): string {
  return `marta_${crypto.randomUUID()}`
}

function timeAgo(ts: string): string {
  const diffMs = Date.now() - new Date(ts).getTime()
  const mins = Math.floor(diffMs / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ConversationSummary {
  id: string
  title: string
  asset: string
  updated_at: string
  message_count: number
}

const GREETING: Message = {
  role: 'assistant',
  content: "Hi! I'm MARTA's AI assistant. Ask me anything about the current market regime, trading decisions, or how MARTA works.",
  timestamp: new Date(),
}

export function MARTAChat() {
  const [open, setOpen] = useState(false)
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState<Message[]>([GREETING])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const selectedAsset = useMARTAStore((s) => s.selectedAsset)
  const apiUrl = process.env.NEXT_PUBLIC_API_URL

  // Resolve (or create) the current session on first mount — reusing a stored id
  // means a page reload continues the same conversation instead of losing it.
  useEffect(() => {
    const stored = localStorage.getItem(SESSION_STORAGE_KEY)
    const id = stored || newSessionId()
    if (!stored) localStorage.setItem(SESSION_STORAGE_KEY, id)
    setSessionId(id)
  }, [])

  // Rehydrate messages for the current session whenever it changes.
  useEffect(() => {
    if (!sessionId) return
    fetch(`${apiUrl}/api/chat/conversations/${sessionId}`)
      .then((r) => r.json())
      .then((d) => {
        const stored = (d.messages || []) as { role: 'user' | 'assistant'; content: string; created_at: string }[]
        if (stored.length > 0) {
          setMessages(stored.map((m) => ({ role: m.role, content: m.content, timestamp: new Date(m.created_at) })))
        } else {
          setMessages([GREETING])
        }
      })
      .catch(() => setMessages([GREETING]))
  }, [sessionId, apiUrl])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    fetch(`${apiUrl}/api/chat/suggestions/${selectedAsset}`)
      .then((r) => r.json())
      .then((d) => setSuggestions(d.suggestions || []))
      .catch(() => {})
  }, [selectedAsset, apiUrl])

  const loadConversations = useCallback(() => {
    fetch(`${apiUrl}/api/chat/conversations`)
      .then((r) => r.json())
      .then((d) => setConversations(d.conversations || []))
      .catch(() => {})
  }, [apiUrl])

  const toggleHistory = () => {
    if (!showHistory) loadConversations()
    setShowHistory(!showHistory)
    setShowSuggestions(false)
  }

  const openConversation = (id: string) => {
    localStorage.setItem(SESSION_STORAGE_KEY, id)
    setSessionId(id)
    setShowHistory(false)
  }

  const newChat = () => {
    const id = newSessionId()
    localStorage.setItem(SESSION_STORAGE_KEY, id)
    setSessionId(id)
    setMessages([GREETING])
    setShowHistory(false)
  }

  const sendMessage = async (text?: string) => {
    const message = text || input.trim()
    if (!message || loading || !sessionId) return

    setInput('')
    setShowSuggestions(false)
    setMessages((prev) => [...prev, { role: 'user', content: message, timestamp: new Date() }])
    setLoading(true)

    try {
      const resp = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId, asset: selectedAsset }),
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
    if (sessionId) await fetch(`${apiUrl}/api/chat/${sessionId}`, { method: 'DELETE' })
    newChat()
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
              <button onClick={newChat} title="New chat" style={{ color: 'var(--dim)' }}>
                <Plus size={14} />
              </button>
              <button onClick={toggleHistory} title="Chat history" style={{ color: showHistory ? 'var(--green)' : 'var(--dim)' }}>
                <History size={14} />
              </button>
              <button onClick={() => { setShowSuggestions(!showSuggestions); setShowHistory(false) }} title="Suggested questions" style={{ color: 'var(--dim)' }}>
                <Lightbulb size={14} />
              </button>
              <button onClick={clearChat} title="Delete this conversation" style={{ color: 'var(--dim)' }}>
                <Trash2 size={14} />
              </button>
            </div>
          </div>

          {showHistory && (
            <div className="p-2 overflow-y-auto" style={{ background: 'var(--surface-2)', borderBottom: '1px solid var(--border)', maxHeight: '200px' }}>
              {conversations.length === 0 && (
                <div className="text-xs p-1" style={{ color: 'var(--dim)' }}>No past conversations yet.</div>
              )}
              {conversations.map((c) => (
                <button
                  key={c.id}
                  onClick={() => openConversation(c.id)}
                  className="w-full text-left text-xs p-2 rounded mb-1 hover:opacity-80 transition-opacity"
                  style={{
                    background: c.id === sessionId ? 'rgba(0,255,127,0.12)' : 'var(--surface-3)',
                    border: c.id === sessionId ? '1px solid rgba(0,255,127,0.3)' : '1px solid var(--border)',
                  }}
                >
                  <div style={{ color: 'var(--text)' }}>{c.title}</div>
                  <div className="flex justify-between mt-0.5" style={{ color: 'var(--dim)', fontSize: '10px' }}>
                    <span>{c.asset} · {c.message_count} messages</span>
                    <span>{timeAgo(c.updated_at)}</span>
                  </div>
                </button>
              ))}
            </div>
          )}

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

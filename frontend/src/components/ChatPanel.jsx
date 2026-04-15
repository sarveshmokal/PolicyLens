import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, CheckCircle, AlertTriangle, XCircle, Zap } from 'lucide-react'
import axios from 'axios'

function ChatPanel() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [fastMode, setFastMode] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async () => {
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const endpoint = fastMode ? '/api/query/fast' : '/api/query'
      const response = await axios.post(endpoint, {
        question: input,
        top_k: 10,
        alpha: 0.5,
        enable_debate: !fastMode,
        enable_verification: !fastMode,
      })

      const data = response.data
      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        verdict: data.verdict,
        entailment: data.entailment_score,
        faithfulness: data.faithfulness,
        citations: data.citations || [],
        provider: data.llm_provider,
        time: data.total_time_seconds,
        quality: data.quality_scores,
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        verdict: 'ERROR',
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case 'VERIFIED': return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'PARTIALLY VERIFIED': return <AlertTriangle className="w-4 h-4 text-yellow-400" />
      case 'NOT VERIFIED': return <XCircle className="w-4 h-4 text-red-400" />
      default: return null
    }
  }

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'VERIFIED': return 'text-green-400 bg-green-400/10 border-green-400/20'
      case 'PARTIALLY VERIFIED': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20'
      case 'NOT VERIFIED': return 'text-red-400 bg-red-400/10 border-red-400/20'
      default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20'
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Policy Chat</h2>
          <p className="text-xs text-gray-500">Ask questions about OECD, IMF, WHO, EU, and UNCTAD policy documents</p>
        </div>
        <button
          onClick={() => setFastMode(!fastMode)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            fastMode
              ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
              : 'bg-gray-800 text-gray-400 border border-gray-700'
          }`}
        >
          <Zap className="w-3.5 h-3.5" />
          {fastMode ? 'Fast Mode ON' : 'Fast Mode OFF'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <h3 className="text-xl font-semibold text-gray-400 mb-2">PolicyLens</h3>
              <p className="text-sm text-gray-600 mb-6">Ask a question about any of the 11 policy documents in the system.</p>
              <div className="space-y-2">
                {[
                  'What does the EU AI Act say about high-risk AI systems?',
                  'Compare IMF and OECD GDP growth projections for 2025',
                  'What are the main global health challenges according to WHO?',
                ].map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(q)}
                    className="w-full text-left px-4 py-2.5 rounded-lg bg-gray-900 border border-gray-800 text-sm text-gray-400 hover:border-gray-700 hover:text-gray-300 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-2xl rounded-xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-900 border border-gray-800'
            }`}>
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>

              {msg.role === 'assistant' && msg.verdict && msg.verdict !== 'ERROR' && (
                <div className="mt-3 pt-3 border-t border-gray-800 space-y-2">
                  <div className="flex items-center gap-4 flex-wrap">
                    <span className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium border ${getVerdictColor(msg.verdict)}`}>
                      {getVerdictIcon(msg.verdict)}
                      {msg.verdict}
                    </span>
                    {msg.faithfulness > 0 && (
                      <span className="text-xs text-gray-500">
                        Faithfulness: {(msg.faithfulness * 100).toFixed(0)}%
                      </span>
                    )}
                    {msg.entailment > 0 && (
                      <span className="text-xs text-gray-500">
                        Entailment: {msg.entailment.toFixed(3)}
                      </span>
                    )}
                    {msg.provider && (
                      <span className="text-xs text-gray-500">
                        via {msg.provider}
                      </span>
                    )}
                    {msg.time > 0 && (
                      <span className="text-xs text-gray-500">
                        {msg.time.toFixed(1)}s
                      </span>
                    )}
                  </div>

                  {msg.citations && msg.citations.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {msg.citations.map((cite, j) => (
                        <span key={j} className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded">
                          {cite.source_file}, p.{cite.page_number}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 flex items-center gap-2 text-sm text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              {fastMode ? 'Generating answer...' : 'Retrieving, synthesizing, and verifying...'}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-gray-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Ask a policy question..."
            className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
            disabled={loading}
          />
          <button
            onClick={handleSubmit}
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 text-white px-4 py-2.5 rounded-xl transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatPanel
import { useState, useEffect } from 'react'
import { Bot, RefreshCw, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import axios from 'axios'

function AgentsPanel() {
  const [agents, setAgents] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchAgents = async () => {
    setLoading(true)
    try {
      const response = await axios.get('/api/agents')
      setAgents(response.data.agents)
    } catch (error) {
      console.error('Failed to fetch agents:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchAgents() }, [])

  const groupedAgents = agents.reduce((acc, agent) => {
    if (!acc[agent.group]) acc[agent.group] = []
    acc[agent.group].push(agent)
    return acc
  }, {})

  const groupColors = {
    ingestion: 'border-emerald-500/30 bg-emerald-500/5',
    analysis: 'border-blue-500/30 bg-blue-500/5',
    verification: 'border-amber-500/30 bg-amber-500/5',
    support: 'border-purple-500/30 bg-purple-500/5',
  }

  const groupLabels = {
    ingestion: 'Ingestion Pipeline',
    analysis: 'Analysis Pipeline',
    verification: 'Verification Pipeline',
    support: 'Support Services',
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold">Agent Registry</h2>
          <p className="text-xs text-gray-500">{agents.length} agents registered</p>
        </div>
        <button
          onClick={fetchAgents}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 text-gray-400 text-xs hover:bg-gray-700 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-6 h-6 animate-spin text-gray-500" />
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedAgents).map(([group, groupAgents]) => (
            <div key={group}>
              <h3 className="text-sm font-medium text-gray-400 mb-3 uppercase tracking-wider">
                {groupLabels[group] || group}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {groupAgents.map((agent) => (
                  <div
                    key={agent.name}
                    className={`rounded-xl border p-4 ${groupColors[group] || 'border-gray-800 bg-gray-900'}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <Bot className="w-4 h-4 text-gray-400" />
                        <span className="text-sm font-medium text-white">{agent.name}</span>
                      </div>
                      {agent.enabled ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-400" />
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">{agent.description}</p>
                    <div className="flex items-center gap-3 mt-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        agent.enabled ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                      }`}>
                        {agent.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        agent.loaded ? 'bg-blue-500/10 text-blue-400' : 'bg-gray-500/10 text-gray-500'
                      }`}>
                        {agent.loaded ? 'Loaded' : 'Not loaded'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default AgentsPanel
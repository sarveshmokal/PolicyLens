import { useState, useEffect } from 'react'
import { Database, RefreshCw, Loader2, FileText, Cpu, HardDrive } from 'lucide-react'
import axios from 'axios'

function StatsPanel() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchStats = async () => {
    setLoading(true)
    try {
      const response = await axios.get('/api/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStats() }, [])

  const StatCard = ({ icon: Icon, label, value, sublabel }) => (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-4 h-4 text-gray-500" />
        <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
      {sublabel && <p className="text-xs text-gray-600 mt-1">{sublabel}</p>}
    </div>
  )

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold">System Statistics</h2>
          <p className="text-xs text-gray-500">Collection and system metrics</p>
        </div>
        <button
          onClick={fetchStats}
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
      ) : stats ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard
              icon={Database}
              label="Total Chunks"
              value={stats.collection?.total_chunks?.toLocaleString() || '0'}
              sublabel="Embedded in ChromaDB"
            />
            <StatCard
              icon={Cpu}
              label="Embedding Model"
              value={stats.collection?.embedding_model || 'N/A'}
              sublabel={`${stats.collection?.embedding_dimension || 384} dimensions`}
            />
            <StatCard
              icon={FileText}
              label="Agents Registered"
              value={stats.agents_registered || 0}
              sublabel="Across 4 pipeline groups"
            />
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Collection Details</h3>
            <div className="space-y-3">
              {stats.collection && Object.entries(stats.collection).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between py-1.5 border-b border-gray-800 last:border-0">
                  <span className="text-xs text-gray-500">{key.replace(/_/g, ' ')}</span>
                  <span className="text-sm text-white font-mono">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p className="text-gray-500 text-sm">Failed to load statistics.</p>
      )}
    </div>
  )
}

export default StatsPanel
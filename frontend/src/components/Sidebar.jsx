import { MessageSquare, Bot, BarChart3, FileText } from 'lucide-react'

const navItems = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'agents', label: 'Agents', icon: Bot },
  { id: 'stats', label: 'Stats', icon: BarChart3 },
]

function Sidebar({ activeTab, setActiveTab }) {
  return (
    <div className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <FileText className="w-6 h-6 text-blue-400" />
          <h1 className="text-lg font-semibold text-white">PolicyLens</h1>
        </div>
        <p className="text-xs text-gray-500 mt-1">Multi-Agent RAG System</p>
      </div>
      <nav className="flex-1 p-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.id
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm mb-1 transition-colors ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <Icon className="w-4 h-4" />
              {item.label}
            </button>
          )
        })}
      </nav>
      <div className="p-3 border-t border-gray-800">
        <p className="text-xs text-gray-600">v0.1.0 — Sarvesh Mokal</p>
      </div>
    </div>
  )
}

export default Sidebar
import { useState } from 'react'
import ChatPanel from './components/ChatPanel'
import Sidebar from './components/Sidebar'
import AgentsPanel from './components/AgentsPanel'
import StatsPanel from './components/StatsPanel'

function App() {
  const [activeTab, setActiveTab] = useState('chat')

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 overflow-hidden">
        {activeTab === 'chat' && <ChatPanel />}
        {activeTab === 'agents' && <AgentsPanel />}
        {activeTab === 'stats' && <StatsPanel />}
      </main>
    </div>
  )
}

export default App
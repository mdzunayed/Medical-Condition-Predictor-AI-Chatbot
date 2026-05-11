import { useState } from 'react'
import { Menu, X } from 'lucide-react'
import useChat from './hooks/useChat'
import FeatureSidebar from './components/FeatureSidebar'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'

function App() {
  const { messages, features, loading, prediction, sendMessage, resetChat } = useChat()
  const [showConfirm, setShowConfirm] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [inputValue, setInputValue] = useState('')

  const handleReset = async () => {
    setShowConfirm(true)
  }

  const confirmReset = async () => {
    setShowConfirm(false)
    await resetChat()
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden flex-col md:flex-row">
      {/* Mobile Header with Sidebar Toggle */}
      <div className="md:hidden flex items-center justify-between border-b border-gray-200 bg-white px-4 py-3 z-40">
        <h1 className="text-lg font-semibold text-gray-900">MediHelp</h1>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          title="Toggle sidebar"
        >
          {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Sidebar - Hidden on mobile, visible on md+ */}
      <div className={`
        fixed md:static inset-y-0 left-0 z-30 transition-transform duration-300 md:translate-x-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <FeatureSidebar features={features} onFillInput={setInputValue} />
      </div>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black bg-opacity-30 z-20"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main chat area - Full width on mobile */}
      <div className="flex-1 flex flex-col w-full md:w-auto">
        {/* Chat messages */}
        <ChatWindow messages={messages} prediction={prediction} loading={loading} />

        {/* Input bar */}
        <InputBar
          onSend={sendMessage}
          onReset={handleReset}
          disabled={loading}
          inputValue={inputValue}
          onInputChange={setInputValue}
        />
      </div>

      {/* Reset Confirmation Modal */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-sm mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Start New Assessment?</h3>
            <p className="text-gray-600 mb-6">
              This will clear all your current health data and start a fresh assessment.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmReset}
                className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white font-medium transition-colors"
              >
                Start New
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App

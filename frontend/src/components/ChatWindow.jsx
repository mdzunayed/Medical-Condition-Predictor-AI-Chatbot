import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import DiagnosisCard from './DiagnosisCard'
import LoadingIndicator from './LoadingIndicator'

const ChatWindow = ({ messages, prediction, loading }) => {
  const messagesEndRef = useRef(null)

  useEffect(() => {
    // Auto-scroll to bottom on new message
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, prediction])

  return (
    <div className="flex-1 overflow-y-auto bg-white px-3 sm:px-6 py-3 sm:py-6">
      <div className="max-w-2xl mx-auto w-full">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-800 mb-3">Medical Diagnosis AI</h2>
              <p className="text-gray-500 mb-2">Your personalized health assessment</p>
              <p className="text-xs text-gray-400 max-w-md mx-auto">
                Answer health questions to receive a personalized diagnosis assessment.
                All information is secure and used only for analysis.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble key={idx} role={msg.role} content={msg.content} />
        ))}

        {/* Show loading indicator when waiting for response */}
        {loading && <LoadingIndicator />}

        {/* Show diagnosis card when prediction is complete */}
        {prediction && (
          <div className="mt-6 mb-4">
            <DiagnosisCard diagnosis={prediction} />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}

export default ChatWindow

import { useState } from 'react'
import axios from 'axios'

const useChat = () => {
  const [messages, setMessages] = useState([])
  const [features, setFeatures] = useState({})
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)

  const sendMessage = async (text) => {
    if (!text.trim()) return

    // Add user message immediately
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      // Call POST /api/chat
      const res = await axios.post('/api/chat', {
        session_id: sessionId,
        message: text,
        history: messages,
      })

      // Update state from response
      setSessionId(res.data.session_id)
      setFeatures(res.data.features)
      setMessages(prev => [...prev, { role: 'assistant', content: res.data.message }])

      // Store prediction if complete
      if (res.data.is_complete && res.data.prediction) {
        setPrediction(res.data.prediction)
      }

      return res.data
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Error: Could not process your message. Please try again.'
      }])
    } finally {
      setLoading(false)
    }
  }

  const resetChat = async () => {
    if (!sessionId) {
      // Reset UI without API call if no session yet
      setMessages([])
      setFeatures({})
      setPrediction(null)
      return
    }

    try {
      await axios.post('/api/reset', { session_id: sessionId })
      setMessages([])
      setFeatures({})
      setSessionId(null)
      setPrediction(null)
    } catch (error) {
      console.error('Error resetting session:', error)
      // Still reset UI even if API call fails
      setMessages([])
      setFeatures({})
      setSessionId(null)
      setPrediction(null)
    }
  }

  return {
    messages,
    features,
    loading,
    sessionId,
    prediction,
    sendMessage,
    resetChat,
  }
}

export default useChat

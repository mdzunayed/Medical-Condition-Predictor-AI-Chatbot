import { useRef, useEffect } from 'react'
import { Send } from 'lucide-react'

const InputBar = ({ onSend, onReset, disabled, inputValue, onInputChange }) => {
  const textareaRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px'
    }
  }, [inputValue])

  // Auto-focus when loading finishes (AI response received)
  useEffect(() => {
    if (!disabled && textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [disabled])

  const handleSend = async () => {
    if (inputValue.trim() && !disabled) {
      await onSend(inputValue)
      onInputChange('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
        // Focus the textarea immediately after sending
        textareaRef.current.focus()
      }
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-gray-200 bg-white px-3 sm:px-6 py-3 sm:py-4 shadow-lg">
      <div className="max-w-2xl mx-auto">
        <div className="flex gap-2 sm:gap-3 items-end">
          {/* New button */}
          <button
            onClick={onReset}
            disabled={disabled}
            className="px-2 sm:px-4 py-2 rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50 hover:border-gray-400 text-xs sm:text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            title="Start a new assessment"
          >
            New
          </button>

          {/* Input box */}
          <div className="flex-1 border border-gray-300 rounded-lg sm:rounded-xl px-3 sm:px-4 py-2 bg-white shadow-sm hover:shadow-md focus-within:shadow-md focus-within:border-purple-400 transition-all duration-200">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Tell me about your health..."
              disabled={disabled}
              className="w-full resize-none outline-none text-xs sm:text-sm border-none p-0 bg-transparent text-gray-900 placeholder-gray-400 disabled:bg-gray-50 disabled:cursor-not-allowed"
              rows="1"
            />
          </div>

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={disabled || !inputValue.trim()}
            className="px-2 sm:px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-xs sm:text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 sm:gap-2 shadow-sm hover:shadow-md"
            title="Send message (Enter)"
          >
            <Send size={16} />
            <span className="hidden sm:inline">Send</span>
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 hidden sm:block">
          💡 Tip: Press Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}

export default InputBar

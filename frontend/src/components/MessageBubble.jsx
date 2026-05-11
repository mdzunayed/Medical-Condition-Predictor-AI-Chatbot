const MessageBubble = ({ role, content }) => {
  if (role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3 max-w-prose hover:shadow-md transition-shadow">
          <p className="text-gray-900 text-sm leading-relaxed whitespace-pre-wrap break-words">
            {content}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-[#FFD1DC] rounded-xl px-4 py-3 max-w-prose shadow-sm">
        <p className="text-gray-900 text-sm leading-relaxed whitespace-pre-wrap break-words">
          {content}
        </p>
      </div>
    </div>
  )
}

export default MessageBubble

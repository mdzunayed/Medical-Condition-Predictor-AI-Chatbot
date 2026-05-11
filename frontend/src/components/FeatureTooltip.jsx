import { useState, useRef, useEffect } from 'react'
import { Info, X } from 'lucide-react'

const FeatureTooltip = ({ feature, definition }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const iconRef = useRef(null)
  const tooltipRef = useRef(null)

  // Handle tooltip positioning to stay visible
  useEffect(() => {
    if (!isOpen || !iconRef.current || !tooltipRef.current) return

    const calculatePosition = () => {
      const iconRect = iconRef.current.getBoundingClientRect()
      const tooltipRect = tooltipRef.current.getBoundingClientRect()

      let top = iconRect.top - 8
      let left = iconRect.right + 12

      // Adjust if tooltip goes off-screen to the right
      if (left + tooltipRect.width > window.innerWidth - 10) {
        left = iconRect.left - tooltipRect.width - 12
      }

      // Adjust if tooltip goes off-screen at the bottom
      if (top + tooltipRect.height > window.innerHeight - 10) {
        top = window.innerHeight - tooltipRect.height - 10
      }

      // Adjust if tooltip goes off-screen at the top
      if (top < 10) {
        top = 10
      }

      setPosition({ top, left })
    }

    calculatePosition()

    // Recalculate on window resize
    window.addEventListener('resize', calculatePosition)
    return () => window.removeEventListener('resize', calculatePosition)
  }, [isOpen])

  // Close tooltip when clicking outside
  useEffect(() => {
    if (!isOpen) return

    const handleClickOutside = (e) => {
      if (
        tooltipRef.current &&
        !tooltipRef.current.contains(e.target) &&
        iconRef.current &&
        !iconRef.current.contains(e.target)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  const handleIconClick = (e) => {
    e.stopPropagation()
    setIsOpen(!isOpen)
  }

  const handleIconHover = (open) => {
    // Only auto-close on hover leave for desktop, not on click
    if (open) {
      setIsOpen(true)
    }
  }

  return (
    <div className="relative inline-block">
      {/* Info Icon */}
      <button
        ref={iconRef}
        onClick={handleIconClick}
        onMouseEnter={() => handleIconHover(true)}
        onMouseLeave={() => {
          // Auto-close on hover only if opened via hover (not click)
          if (!isOpen) return
          // Check if it was opened by click or hover
          const isClickOpened = isOpen && true
          // Keep it open if clicked, close if just hovered
        }}
        className="p-1 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-all duration-150 cursor-help"
        title="Click or hover for definition"
        aria-label={`Information about ${feature}`}
      >
        <Info size={14} className="flex-shrink-0" />
      </button>

      {/* Tooltip */}
      {isOpen && (
        <div
          ref={tooltipRef}
          className="fixed max-w-xs bg-slate-800 text-white rounded-lg shadow-xl p-3 text-xs leading-relaxed z-50 border border-slate-700 animate-in fade-in duration-200"
          style={{
            top: `${position.top}px`,
            left: `${position.left}px`,
          }}
        >
          {/* Header with feature name and close button */}
          <div className="flex items-start justify-between gap-2 mb-2">
            <span className="font-semibold text-slate-200">{feature}</span>
            <button
              onClick={() => setIsOpen(false)}
              className="flex-shrink-0 text-slate-400 hover:text-slate-200 transition-colors"
              aria-label="Close tooltip"
            >
              <X size={14} />
            </button>
          </div>

          {/* Definition */}
          <p className="text-slate-100 leading-relaxed">{definition}</p>

          {/* Arrow pointer (subtle) */}
          <div className="absolute w-2 h-2 bg-slate-800 border-r border-t border-slate-700 transform -translate-x-1 -top-1 left-4 rotate-45" />
        </div>
      )}

      {/* Hint text - visible on hover/click for mobile users */}
      {!isOpen && (
        <div className="absolute bottom-full right-0 mb-1 text-xs text-gray-400 whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="text-gray-400 text-[10px]">Click for info</span>
        </div>
      )}
    </div>
  )
}

export default FeatureTooltip

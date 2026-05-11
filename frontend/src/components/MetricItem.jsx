import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { CheckCircle, Circle, Info, X } from 'lucide-react'
import DietScoreModal from './DietScoreModal'

/**
 * MetricItem Component
 *
 * Handles individual health metric with isolated hover state
 * Uses React Portal to render popover outside the scrollable sidebar
 * Ensures popover floats over the chat area without being clipped
 *
 * New: Displays the actual extracted value next to the metric name
 */

// Binary features that should display "Yes" / "No"
const BINARY_FEATURES = ['Smoking', 'Alcohol', 'Family History']

// Helper function to format metric value for display
const formatMetricValue = (feature, value) => {
  if (value === null || value === undefined) return null

  // For binary features: 1 = "Yes", 0 = "No"
  if (BINARY_FEATURES.includes(feature)) {
    return value === 1 ? 'Yes' : 'No'
  }

  // For numeric features: round to appropriate decimal places
  if (typeof value === 'number') {
    // Most metrics are integers, but HbA1c might have decimals
    if (feature === 'HbA1c') {
      return value.toFixed(1)
    }
    return Math.round(value)
  }

  return String(value)
}

const MetricItem = ({ feature, displayLabel, definition, isCollected, value, onFillInput }) => {
  const [showHint, setShowHint] = useState(false)
  const [showCalculator, setShowCalculator] = useState(false)
  const [popoverPosition, setPopoverPosition] = useState({ top: 0, left: 0 })
  const iconRef = useRef(null)

  // Calculate popover position based on icon location
  useEffect(() => {
    if (!showHint || !iconRef.current) return

    const iconRect = iconRef.current.getBoundingClientRect()

    // Position popover to the right of the icon, aligned with top
    // Add some spacing (16px) to the right
    const left = iconRect.right + 16
    const top = iconRect.top - 8  // Slight vertical centering

    setPopoverPosition({ top, left })
  }, [showHint])

  // Handle mouse leaving the metric item
  const handleMouseLeave = () => {
    setShowHint(false)
  }

  // Handle mouse entering the metric item
  const handleMouseEnter = () => {
    setShowHint(true)
  }

  // Close hint when clicking X button
  const handleCloseHint = (e) => {
    e.stopPropagation()
    setShowHint(false)
  }

  return (
    <>
      {/* Metric Item Row */}
      <div
        className={`flex items-center justify-between gap-2 text-xs p-2 rounded transition-all duration-150 ${
          isCollected
            ? 'text-green-700 bg-green-50 hover:bg-green-100'
            : 'text-gray-400 hover:bg-gray-100'
        }`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* Status icon + Label */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {isCollected ? (
            <CheckCircle size={16} className="flex-shrink-0" />
          ) : (
            <Circle size={16} className="flex-shrink-0" />
          )}
          <span className="truncate">{displayLabel}</span>
        </div>

        {/* Value Badge - displays extracted value */}
        {isCollected && value !== null && value !== undefined && (
          <span className="ml-1 px-2 py-0.5 rounded-full bg-green-100 text-green-800 font-semibold flex-shrink-0 whitespace-nowrap">
            {formatMetricValue(feature, value)}
          </span>
        )}

        {/* "Don't know? Calculate" button for Diet Score on hover */}
        {showHint && onFillInput && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              setShowCalculator(true)
              setShowHint(false)
            }}
            className="px-2 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-700 hover:bg-indigo-200 transition-colors flex-shrink-0 whitespace-nowrap"
            title="Open Diet Score calculator"
          >
            Don't know? Calculate
          </button>
        )}

        {/* Info Icon Button */}
        <button
          ref={iconRef}
          onClick={() => setShowHint(!showHint)}
          className="p-1 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-200 transition-all duration-150 cursor-help flex-shrink-0"
          title="Click for definition"
          aria-label={`Information about ${displayLabel}`}
        >
          <Info size={14} />
        </button>
      </div>

      {/* Popover - Rendered via Portal (outside sidebar) with Glassmorphism */}
      {showHint &&
        createPortal(
          <div
            className="fixed z-9999 w-80 rounded-2xl shadow-lg p-4 text-xs leading-relaxed animate-in fade-in duration-200 pointer-events-auto backdrop-blur-md"
            style={{
              top: `${popoverPosition.top}px`,
              left: `${popoverPosition.left}px`,
              backgroundColor: 'rgba(220, 252, 231, 0.75)',
              borderColor: 'rgba(134, 239, 172, 0.5)',
              borderWidth: '1px',
              WebkitBackdropFilter: 'blur(12px)',
            }}
            onMouseEnter={() => setShowHint(true)}
            onMouseLeave={() => setShowHint(false)}
          >
            {/* Header with feature name and close button */}
            <div className="flex items-start justify-between gap-3 mb-3">
              <span className="font-semibold text-green-900 text-sm">{displayLabel}</span>
              <button
                onClick={handleCloseHint}
                className="flex-shrink-0 text-green-700 hover:text-green-900 transition-colors p-0.5 hover:bg-green-200/40 rounded-lg"
                aria-label="Close popover"
              >
                <X size={16} />
              </button>
            </div>

            {/* Definition text */}
            <p className="text-green-900 leading-relaxed mb-2 font-medium">{definition}</p>

            {/* Arrow pointer - points back to info icon (matches popover styling) */}
            <div
              className="absolute w-3 h-3 transform rotate-45"
              style={{
                right: '-6px',
                top: `${8}px`,
                backgroundColor: 'rgba(220, 252, 231, 0.85)',
                borderTop: '1px solid rgba(134, 239, 172, 0.4)',
                borderLeft: '1px solid rgba(134, 239, 172, 0.4)',
              }}
            />
          </div>,
          document.body
        )}

      {/* Diet Score Calculator Modal */}
      {showCalculator && (
        <DietScoreModal
          onClose={() => setShowCalculator(false)}
          onFillInput={onFillInput}
        />
      )}
    </>
  )
}

export default MetricItem

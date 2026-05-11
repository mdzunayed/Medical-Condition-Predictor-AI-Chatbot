import { useState, useEffect } from 'react'

/**
 * LoadingIndicator Component
 *
 * Displays dynamic status text with a smooth 3-dot wave animation
 * - Shows only after 1 second delay
 * - Cycles through status phrases every 1.5 seconds
 * - Three dots animate in a continuous smooth wave (iMessage style)
 */

const LOADING_PHRASES = [
  'Recognizing metrics',
  'Running health models',
  'Cross-referencing data',
  'Synthesizing response',
  'Almost there',
  'Analyzing vital signs',
  'Evaluating risk factors',
  'Processing test results',
  'Correlating symptoms',
  'Assessing health indicators',
  'Computing probability scores',
  'Comparing diagnostic patterns',
  'Validating predictions',
  'Generating personalized insights',
  'Finalizing diagnosis'
]

const LoadingIndicator = () => {
  const [show, setShow] = useState(false)
  const [phraseIndex, setPhraseIndex] = useState(0)

  // Delay showing indicator by 1 second
  useEffect(() => {
    const delayTimer = setTimeout(() => {
      setShow(true)
    }, 1000)

    return () => clearTimeout(delayTimer)
  }, [])

  // Cycle through phrases every 1.5 seconds
  useEffect(() => {
    if (!show) return

    const phraseTimer = setInterval(() => {
      setPhraseIndex(prev => (prev + 1) % LOADING_PHRASES.length)
    }, 1500)

    return () => clearInterval(phraseTimer)
  }, [show])

  if (!show) return null

  return (
    <div className="flex items-center gap-2 text-sm italic text-gray-400 py-4">
      <span>{LOADING_PHRASES[phraseIndex]}</span>

      {/* Three-dot wave animation */}
      <div className="flex items-center gap-1">
        <style>{`
          @keyframes wave {
            0%, 100% {
              transform: translateY(0px);
            }
            50% {
              transform: translateY(-8px);
            }
          }

          .dot-wave {
            display: inline-block;
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background-color: currentColor;
            animation: wave 1.4s ease-in-out infinite;
          }

          .dot-1 {
            animation-delay: 0s;
          }

          .dot-2 {
            animation-delay: 0.2s;
          }

          .dot-3 {
            animation-delay: 0.4s;
          }
        `}</style>

        <span className="dot-wave dot-1" />
        <span className="dot-wave dot-2" />
        <span className="dot-wave dot-3" />
      </div>
    </div>
  )
}

export default LoadingIndicator

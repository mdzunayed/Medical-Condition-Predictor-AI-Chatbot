import { useState } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'

/**
 * Diet Score Calculator Modal
 *
 * Questionnaire with 3 questions to calculate a diet score out of 10:
 * Q1: Fruits & Veggies (max 4 pts) → 1/2/4 options
 * Q2: Processed Foods (max 3 pts) → 0/1.5/3 options
 * Q3: Grains & Proteins (max 3 pts) → 1/2/3 options
 *
 * Auto-fills the chat input with "Diet Score is {total}" when submitted.
 */

const DietScoreModal = ({ onClose, onFillInput }) => {
  const [q1, setQ1] = useState(null) // Fruits & Veggies: 1, 2, or 4
  const [q2, setQ2] = useState(null) // Processed Foods: 0, 1.5, or 3
  const [q3, setQ3] = useState(null) // Grains & Proteins: 1, 2, or 3

  const isComplete = q1 !== null && q2 !== null && q3 !== null

  const handleCalculate = () => {
    if (!isComplete) return

    const total = q1 + q2 + q3
    const scoreText = `Diet Score is ${total}`
    onFillInput(scoreText)
    onClose()
  }

  return createPortal(
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-md mx-4 max-h-96 overflow-y-auto">
        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Diet Score Calculator</h3>
            <p className="text-xs text-gray-500 mt-1">Answer 3 quick questions (max 10 points)</p>
          </div>
          <button
            onClick={onClose}
            className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors p-1 hover:bg-gray-100 rounded-lg"
            aria-label="Close modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Question 1: Fruits & Veggies */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Q1: Fruits & Veggies servings per day? <span className="text-gray-500 font-normal">(max 4 pts)</span>
          </label>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q1"
                value={1}
                checked={q1 === 1}
                onChange={() => setQ1(1)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">0-1 servings/day → <span className="font-semibold">1 pt</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q1"
                value={2}
                checked={q1 === 2}
                onChange={() => setQ1(2)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">2-3 servings/day → <span className="font-semibold">2 pts</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q1"
                value={4}
                checked={q1 === 4}
                onChange={() => setQ1(4)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">4+ servings/day → <span className="font-semibold">4 pts</span></span>
            </label>
          </div>
        </div>

        {/* Question 2: Processed Foods */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Q2: How often do you eat processed foods? <span className="text-gray-500 font-normal">(max 3 pts)</span>
          </label>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q2"
                value={0}
                checked={q2 === 0}
                onChange={() => setQ2(0)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">4+ times a week → <span className="font-semibold">0 pts</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q2"
                value={1.5}
                checked={q2 === 1.5}
                onChange={() => setQ2(1.5)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">1-3 times a week → <span className="font-semibold">1.5 pts</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q2"
                value={3}
                checked={q2 === 3}
                onChange={() => setQ2(3)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">Rarely/Never → <span className="font-semibold">3 pts</span></span>
            </label>
          </div>
        </div>

        {/* Question 3: Grains & Proteins */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Q3: What grains & proteins do you prefer? <span className="text-gray-500 font-normal">(max 3 pts)</span>
          </label>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q3"
                value={1}
                checked={q3 === 1}
                onChange={() => setQ3(1)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">Mostly refined/red meat → <span className="font-semibold">1 pt</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q3"
                value={2}
                checked={q3 === 2}
                onChange={() => setQ3(2)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">Mix of both → <span className="font-semibold">2 pts</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-gray-50 transition-colors">
              <input
                type="radio"
                name="q3"
                value={3}
                checked={q3 === 3}
                onChange={() => setQ3(3)}
                className="accent-indigo-600"
              />
              <span className="text-sm text-gray-700">Mostly whole grains/lean proteins → <span className="font-semibold">3 pts</span></span>
            </label>
          </div>
        </div>

        {/* Score preview */}
        {isComplete && (
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 mb-4">
            <p className="text-sm font-medium text-indigo-900">
              Your Diet Score: <span className="text-lg font-bold">{q1 + q2 + q3}</span>/10
            </p>
          </div>
        )}

        {/* Buttons */}
        <div className="flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCalculate}
            disabled={!isComplete}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Calculate & Add
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}

export default DietScoreModal

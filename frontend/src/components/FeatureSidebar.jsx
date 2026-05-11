import MetricItem from './MetricItem'

const FEATURE_LABELS = {
  'Age': 'Age',
  'Glucose': 'Blood Glucose',
  'HbA1c': 'HbA1c',
  'BMI': 'BMI',
  'Cholesterol': 'Cholesterol',
  'Triglycerides': 'Triglycerides',
  'Blood Pressure': 'Blood Pressure',
  'Physical Activity': 'Physical Activity',
  'Sleep Hours': 'Sleep Hours',
  'Stress Level': 'Stress Level',
  'Diet Score': 'Diet Score',
  'Smoking': 'Smoking',
  'Alcohol': 'Alcohol',
  'Family History': 'Family History',
  'LengthOfStay': 'Length of Stay',
  'Oxygen Saturation': 'Oxygen Saturation',
}

const FEATURE_DEFINITIONS = {
  'Age': 'Your current age in years.',
  'Glucose': 'Current blood sugar level (mg/dL).',
  'HbA1c': 'Average blood sugar over the last 3 months (%).',
  'BMI': 'Body Mass Index; a measure of body fat based on height and weight.',
  'Cholesterol': 'Total amount of cholesterol in your blood (mg/dL).',
  'Triglycerides': 'A type of fat found in your blood (mg/dL).',
  'Blood Pressure': 'The force of your blood against artery walls (e.g., 120).',
  'Physical Activity': 'Total hours of exercise or active movement per week.',
  'Sleep Hours': 'Average hours of sleep you get per 24-hour period.',
  'Stress Level': '1-3 (Low): Calm, in control. 4-6 (Moderate): Busy, pressured. 7-8 (High): Anxious, overwhelmed. 9-10 (Very High): Exhausted, unable to cope.',
  'Diet Score': 'Self-rating of how healthy your meals are (1-10).',
  'Smoking': 'Whether you currently smoke or have a history of smoking.',
  'Alcohol': 'Your frequency of alcohol consumption.',
  'Family History': 'Whether close relatives have had chronic conditions like heart disease or diabetes.',
  'LengthOfStay': 'Total days spent in a hospital during your last visit.',
  'Oxygen Saturation': 'The percentage of oxygen in your blood (SpO2).',
}

const DEFAULT_FEATURES = [
  'Age', 'Glucose', 'HbA1c', 'BMI',
  'Cholesterol', 'Triglycerides', 'Blood Pressure', 'Physical Activity',
  'Sleep Hours', 'Stress Level', 'Diet Score', 'Smoking',
  'Alcohol', 'Family History', 'LengthOfStay', 'Oxygen Saturation'
]

const FeatureSidebar = ({ features, onFillInput }) => {
  const collectedCount = DEFAULT_FEATURES.filter(
    feature => features[feature] !== null && features[feature] !== undefined
  ).length

  const progressPercent = (collectedCount / 16) * 100

  return (
    <div className="w-72 bg-gray-50 border-r border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-900 mb-3">Health Metrics</h2>

        {/* Progress bar */}
        <div className="mb-2">
          <div className="flex justify-between mb-1">
            <span className="text-xs text-gray-600">{collectedCount}/16</span>
            <span className="text-xs text-gray-600">{Math.round(progressPercent)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-600 transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Features list */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <div className="space-y-2">
          {DEFAULT_FEATURES.map(feature => {
            const value = features[feature]
            const isCollected = value !== null && value !== undefined
            const displayLabel = FEATURE_LABELS[feature] || feature
            const definition = FEATURE_DEFINITIONS[feature] || 'No description available.'

            return (
              <MetricItem
                key={feature}
                feature={feature}
                displayLabel={displayLabel}
                definition={definition}
                isCollected={isCollected}
                value={value}
                onFillInput={feature === 'Diet Score' ? onFillInput : undefined}
              />
            )
          })}
        </div>

        {/* Help text at bottom */}
        <div className="mt-4 pt-3 border-t border-gray-200 text-xs text-gray-500">
          <p>💡 <strong>Tip:</strong> Hover over or click the <span className="text-gray-600">ⓘ</span> icon next to each metric to learn what it measures.</p>
        </div>
      </div>
    </div>
  )
}

export default FeatureSidebar

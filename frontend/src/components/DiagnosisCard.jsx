/**
 * DiagnosisCard Component
 *
 * Professional diagnosis result card with:
 * - Light background with professional styling
 * - Comprehensive Patient Data Report grid
 * - Responsive mobile design (1 col → 2 col → 4 col)
 * - Clean, focused layout
 */
const DiagnosisCard = ({ diagnosis }) => {
  if (!diagnosis) return null

  const {
    prediction_name,
    explanation,
    features = {},
  } = diagnosis

  // Feature display labels for the Patient Data Report
  const featureLabels = {
    Age: 'Age',
    Glucose: 'Blood Glucose',
    HbA1c: 'HbA1c',
    BMI: 'BMI',
    Cholesterol: 'Cholesterol',
    Triglycerides: 'Triglycerides',
    BloodPressure: 'Blood Pressure',
    PhysicalActivity: 'Physical Activity',
    SleepHours: 'Sleep Hours',
    StressLevel: 'Stress Level',
    DietScore: 'Diet Score',
    Smoking: 'Smoking',
    Alcohol: 'Alcohol',
    FamilyHistory: 'Family History',
    LengthOfStay: 'Length of Stay',
    OxygenSaturation: 'Oxygen Saturation',
  }

  return (
    <div className="rounded-xl sm:rounded-2xl p-4 sm:p-6 border-2 bg-green-50 border-green-200">
      {/* Header */}
      <h2 className="text-lg sm:text-xl font-bold text-green-900 mb-4">
        Health Summary
      </h2>

      {/* Diagnosis Display */}
      <div className="mb-6">
        <p className="text-xs sm:text-sm font-medium text-green-700 mb-2">Diagnosis</p>
        <h3 className="text-2xl sm:text-3xl font-bold text-green-900">
          {prediction_name}
        </h3>
      </div>

      {/* Explanation */}
      <div className="bg-white/50 rounded-lg p-3 sm:p-4 border border-green-200 mb-6">
        <p className="text-xs sm:text-sm text-green-800 leading-relaxed">
          {explanation}
        </p>
      </div>

      {/* Patient Data Report */}
      <div className="border-t border-green-200 mt-4 pt-4">
        <h3 className="text-base sm:text-lg font-bold text-green-900 mb-4">Patient Data Report</h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          {Object.entries(features).map(([key, value]) => {
            // Get the display label for this feature
            const displayLabel = featureLabels[key] || key

            return (
              <div key={key} className="space-y-1">
                <p className="text-xs font-medium text-green-600 uppercase tracking-wider break-words">
                  {displayLabel}
                </p>
                <p className="text-sm sm:text-base font-semibold text-green-900 break-words">
                  {value !== null && value !== undefined ? String(value) : '—'}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default DiagnosisCard

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { proceduresApi } from '../api/procedures';
import type { ProcedureStep } from '../types/procedure';
import { Plus, Trash2, ArrowLeft, Save, Upload } from 'lucide-react';

export default function CreateTemplate() {
  const navigate = useNavigate();
  const [isUploading, setIsUploading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [title, setTitle] = useState('');
  const [steps, setSteps] = useState<ProcedureStep[]>([]);
  const [templateId, setTemplateId] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.endsWith('.txt') && !file.name.endsWith('.pdf')) {
      setError('Please upload a .txt or .pdf file');
      return;
    }

    setIsUploading(true);
    setError('');

    try {
      const result = await proceduresApi.upload(file);

      // Set title from filename
      setTitle(file.name.replace(/\.(txt|pdf)$/, ''));

      // Generate template automatically
      setSuccess('Steps extracted! Generating template with AI validation...');

      const generated = await proceduresApi.generate({
        title: file.name.replace(/\.(txt|pdf)$/, ''),
        steps: result.steps,
        user_id: 'default_user',
      });

      setSteps(generated.template.steps);
      setTemplateId(generated.template.template_id);

      const aiStatus = generated.llama_enabled
        ? '✅ AI-validated with Llama 3'
        : '⚠️ Regex-based extraction only';

      setSuccess(`Template generated successfully! ${aiStatus}`);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to process file');
    } finally {
      setIsUploading(false);
    }
  };

  const updateStep = (index: number, field: keyof ProcedureStep, value: any) => {
    const updated = [...steps];
    updated[index] = { ...updated[index], [field]: value };
    setSteps(updated);
  };

  const updateParameter = (
    stepIndex: number,
    paramType: string,
    field: 'value' | 'unit',
    value: any
  ) => {
    const updated = [...steps];
    const step = updated[stepIndex];
    step.parameters = {
      ...step.parameters,
      [paramType]: {
        ...step.parameters[paramType as keyof typeof step.parameters],
        [field]: field === 'value' ? (parseFloat(value) || 0) : value,
      },
    };
    setSteps(updated);
  };

  const handleSave = async () => {
    if (!templateId) {
      setError('No template to save');
      return;
    }

    setIsGenerating(true);
    setError('');

    try {
      await proceduresApi.update(templateId, {
        template_id: templateId,
        title,
        user_id: 'default_user',
        created_at: new Date().toISOString(),
        total_steps: steps.length,
        estimated_duration_minutes: Math.ceil(
          steps.reduce((sum, s) => sum + (s.estimated_duration_seconds || 60), 0) / 60
        ),
        steps,
      });

      setSuccess('Changes saved successfully!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to save changes');
    } finally {
      setIsGenerating(false);
    }
  };

  const renderParameter = (
    stepIndex: number,
    paramType: string,
    paramData: any
  ) => {
    if (!paramData || paramData.value === null || paramData.value === undefined) {
      return null;
    }

    const labels: Record<string, string> = {
      volume: 'Volume',
      temperature: 'Temperature',
      duration: 'Duration',
      speed: 'Speed',
      count: 'Count',
      concentration: 'Concentration',
    };

    return (
      <div key={paramType} className="flex flex-col gap-1">
        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">
          {labels[paramType] || paramType}
        </label>
        <div className="flex gap-2">
          <input
            type="number"
            value={paramData.value}
            onChange={(e) => updateParameter(stepIndex, paramType, 'value', e.target.value)}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
            step={paramType === 'volume' || paramType === 'concentration' ? '0.1' : '1'}
          />
          <div className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-sm min-w-[70px] text-center">
            {paramData.unit}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Create Procedure Template</h1>
            <p className="text-gray-600 mt-1">Upload a file or create from scratch</p>
          </div>
        </div>

        {/* Alerts */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded-lg">
            {error}
          </div>
        )}

        {success && (
          <div className="bg-green-50 border-l-4 border-green-500 text-green-700 p-4 rounded-lg">
            {success}
          </div>
        )}

        {/* Upload Section */}
        {steps.length === 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-8">
            <label className="flex flex-col items-center justify-center cursor-pointer">
              <input
                type="file"
                accept=".txt,.pdf"
                onChange={handleFileUpload}
                disabled={isUploading}
                className="hidden"
              />
              <div className="flex flex-col items-center text-center space-y-4">
                {isUploading ? (
                  <>
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
                    <p className="text-gray-600">Processing file...</p>
                  </>
                ) : (
                  <>
                    <Upload className="w-16 h-16 text-gray-400" />
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900">
                        Upload Procedure File
                      </h3>
                      <p className="text-gray-600 mt-2">
                        Drag and drop a PDF or text file, or click to browse
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        Supported formats: .txt, .pdf
                      </p>
                    </div>
                  </>
                )}
              </div>
            </label>
          </div>
        )}

        {/* Edit Template */}
        {steps.length > 0 && (
          <>
            {/* Title */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Template Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Enter template title..."
              />
            </div>

            {/* Steps */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">
                  Procedure Steps
                  <span className="text-gray-500 text-base font-normal ml-2">
                    ({steps.length} steps)
                  </span>
                </h2>
              </div>

              {steps.map((step, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-5 space-y-4 hover:border-gray-300 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-100 text-primary-700 font-semibold text-sm">
                        {step.step_number}
                      </span>
                      <span className="text-sm font-medium text-primary-600 capitalize">
                        {step.expected_action}
                      </span>
                    </div>
                  </div>

                  <textarea
                    value={step.description || step.step_description}
                    onChange={(e) => updateStep(index, 'description', e.target.value)}
                    rows={3}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                    placeholder="Step description..."
                  />

                  {/* Parameters */}
                  {step.parameters && Object.keys(step.parameters).length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
                      {Object.entries(step.parameters).map(([paramType, paramData]) =>
                        renderParameter(index, paramType, paramData)
                      )}
                    </div>
                  )}

                  {/* Metadata */}
                  {(step.chemicals || step.equipment || step.safety_notes) && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-gray-100">
                      {step.chemicals && step.chemicals.length > 0 && (
                        <div>
                          <div className="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                            Chemicals
                          </div>
                          <div className="text-sm text-gray-700">
                            {step.chemicals.join(', ')}
                          </div>
                        </div>
                      )}

                      {step.equipment && step.equipment.length > 0 && (
                        <div>
                          <div className="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                            Equipment
                          </div>
                          <div className="text-sm text-gray-700">
                            {step.equipment.join(', ')}
                          </div>
                        </div>
                      )}

                      {step.safety_notes && step.safety_notes.length > 0 && (
                        <div>
                          <div className="text-xs font-medium text-red-600 uppercase tracking-wide mb-1">
                            Safety
                          </div>
                          <div className="text-sm text-red-600">
                            {step.safety_notes.join(', ')}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* CLI Instructions */}
            {templateId && (
              <div className="bg-primary-50 border border-primary-200 rounded-xl p-6">
                <h3 className="font-semibold text-primary-900 mb-2">✅ Template Ready</h3>
                <p className="text-primary-700 mb-3">
                  Template saved to: <code className="bg-primary-100 px-2 py-1 rounded">templates/{templateId}.json</code>
                </p>
                <p className="text-primary-700 mb-2">Use with CLI video processing:</p>
                <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                  python scripts/test_video_with_tep.py \{'\n'}
                  {'  '}--template templates/{templateId}.json \{'\n'}
                  {'  '}--video videos/your_video.mp4
                </pre>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => navigate('/')}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={isGenerating}
                className="flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 font-medium"
              >
                <Save className="w-5 h-5" />
                Save Changes
              </button>
            </div>
          </>
        )}
    </div>
  );
}

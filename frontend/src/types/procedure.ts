export interface ParameterValue {
  value: number | null;
  unit: string;
  confidence?: number;
}

export interface ProcedureStep {
  step_number: number;
  description: string;
  step_description: string;
  expected_action: string;
  parameters: {
    volume?: ParameterValue;
    temperature?: ParameterValue;
    duration?: ParameterValue;
    speed?: ParameterValue;
    count?: ParameterValue;
    concentration?: ParameterValue;
  };
  chemicals?: string[];
  equipment?: string[];
  safety_notes?: string[];
  technique_level?: string;
  estimated_duration_seconds?: number;
}

export interface ProcedureTemplate {
  template_id: string;
  title: string;
  user_id: string;
  created_at: string;
  modified_at?: string;
  total_steps: number;
  estimated_duration_minutes: number;
  steps: ProcedureStep[];
}

export interface TemplateCreateRequest {
  title: string;
  steps: string[];
  user_id?: string;
}

export interface TemplateListItem {
  id: string;
  template_id: string;
  title: string;
  created_at: string;
  step_count: number;
  user_id: string;
}

export interface UploadResponse {
  steps: string[];
  filename: string;
  step_count: number;
}

export interface GenerateTemplateResponse {
  template: ProcedureTemplate;
  template_id: string;
  message: string;
  llama_enabled: boolean;
}

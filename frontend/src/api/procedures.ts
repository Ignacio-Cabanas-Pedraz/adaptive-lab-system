import apiClient from './client';
import type {
  ProcedureTemplate,
  TemplateCreateRequest,
  TemplateListItem,
  UploadResponse,
  GenerateTemplateResponse,
} from '../types/procedure';

export const proceduresApi = {
  /**
   * Upload a PDF or text file and extract steps
   */
  upload: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<UploadResponse>('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  /**
   * Generate template from steps with AI validation
   */
  generate: async (data: TemplateCreateRequest): Promise<GenerateTemplateResponse> => {
    const response = await apiClient.post<GenerateTemplateResponse>(
      '/api/generate-template',
      data
    );
    return response.data;
  },

  /**
   * List all templates
   */
  list: async (): Promise<TemplateListItem[]> => {
    const response = await apiClient.get<{ templates: TemplateListItem[] }>('/api/templates');
    return response.data.templates;
  },

  /**
   * Get a specific template
   */
  get: async (templateId: string): Promise<ProcedureTemplate> => {
    const response = await apiClient.get<{ template: ProcedureTemplate }>(
      `/api/templates/${templateId}`
    );
    return response.data.template;
  },

  /**
   * Update a template
   */
  update: async (
    templateId: string,
    template: ProcedureTemplate
  ): Promise<ProcedureTemplate> => {
    const response = await apiClient.put<{ template: ProcedureTemplate }>(
      `/api/templates/${templateId}`,
      template
    );
    return response.data.template;
  },

  /**
   * Delete a template
   */
  delete: async (templateId: string): Promise<void> => {
    await apiClient.delete(`/api/templates/${templateId}`);
  },

  /**
   * Export template as JSON download
   */
  export: async (templateId: string, title: string): Promise<void> => {
    const response = await apiClient.get(`/api/templates/${templateId}/export`, {
      responseType: 'blob',
    });

    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${templateId}.json`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  },
};

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ParameterValue(BaseModel):
    value: Optional[float] = None
    unit: str
    confidence: Optional[float] = None

class ProcedureStep(BaseModel):
    step_number: int
    description: str
    step_description: str
    expected_action: str
    parameters: Dict[str, Any] = {}
    chemicals: List[str] = []
    equipment: List[str] = []
    safety_notes: List[str] = []
    technique_level: str = "standard"
    estimated_duration_seconds: int = 60

class ProcedureTemplate(BaseModel):
    template_id: str
    title: str
    user_id: str
    created_at: str
    modified_at: Optional[str] = None
    total_steps: int
    estimated_duration_minutes: int
    steps: List[ProcedureStep]

class TemplateCreateRequest(BaseModel):
    title: str
    steps: List[str]
    user_id: str = "default_user"

class TemplateListItem(BaseModel):
    id: str
    template_id: str
    title: str
    created_at: str
    step_count: int
    user_id: str

class UploadResponse(BaseModel):
    steps: List[str]
    filename: str
    step_count: int

class GenerateTemplateResponse(BaseModel):
    template: ProcedureTemplate
    template_id: str
    message: str
    llama_enabled: bool

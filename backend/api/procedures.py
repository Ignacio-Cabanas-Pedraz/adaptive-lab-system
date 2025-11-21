from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import uuid
import json
from datetime import datetime
from typing import List
import PyPDF2

from schemas.procedure import (
    TemplateCreateRequest,
    TemplateListItem,
    UploadResponse,
    GenerateTemplateResponse,
    ProcedureTemplate
)

# Import existing template generator
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.procedure.template_generator import ProcedureTemplateGenerator

try:
    from src.procedure.llm_validator import LlamaParameterValidator
    LLAMA_AVAILABLE = True
except:
    LLAMA_AVAILABLE = False
    print("Warning: Llama validator not available")

router = APIRouter()

# Configuration
UPLOAD_FOLDER = Path("uploads")
TEMPLATE_FOLDER = Path("templates")
ALLOWED_EXTENSIONS = {'.txt', '.pdf'}

UPLOAD_FOLDER.mkdir(exist_ok=True)
TEMPLATE_FOLDER.mkdir(exist_ok=True)

# Global instances (lazy loaded)
template_generator = None
llama_validator = None

def get_template_generator():
    """Lazy load template generator"""
    global template_generator
    if template_generator is None:
        template_generator = ProcedureTemplateGenerator()
    return template_generator

def get_llama_validator():
    """Lazy load Llama validator"""
    global llama_validator
    if llama_validator is None and LLAMA_AVAILABLE:
        try:
            from src.procedure.llm_validator import LlamaParameterValidator
            llama_validator = LlamaParameterValidator()
        except Exception as e:
            print(f"Warning: Could not load Llama validator: {e}")
            llama_validator = None
    return llama_validator

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    return text

def extract_steps_from_text(text: str) -> List[str]:
    """Extract procedure steps from text"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Filter out very short lines (likely headers/noise)
    steps = [line for line in lines if len(line) > 10]
    return steps

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload PDF or text file and extract procedure steps
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Use {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Save uploaded file
        file_path = UPLOAD_FOLDER / file.filename
        content = await file.read()

        with open(file_path, 'wb') as f:
            f.write(content)

        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        else:
            text = content.decode('utf-8')

        # Extract steps
        steps = extract_steps_from_text(text)

        # Clean up uploaded file
        file_path.unlink()

        return UploadResponse(
            steps=steps,
            filename=file.filename,
            step_count=len(steps)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-template", response_model=GenerateTemplateResponse)
async def generate_template(request: TemplateCreateRequest):
    """
    Generate template from steps with optional Llama validation
    """
    if not request.steps or len(request.steps) == 0:
        raise HTTPException(status_code=400, detail="No steps provided")

    try:
        # Generate template using existing generator
        generator = get_template_generator()
        template = generator.generate_template(
            title=request.title,
            user_id=request.user_id,
            step_descriptions=request.steps
        )

        # Run Llama validation if available
        validator = get_llama_validator()
        llama_enabled = False

        if validator is not None:
            llama_enabled = True
            validated_steps = []

            for i, step_data in enumerate(template['steps']):
                original_text = request.steps[i]

                # Validate with Llama
                validated = validator.validate_step(
                    step_number=step_data['step_number'],
                    original_text=original_text,
                    extracted_params=step_data['parameters']
                )

                # Update step with validated data
                step_data['parameters'] = validated['parameters']
                step_data['chemicals'] = validated.get('chemicals', [])
                step_data['equipment'] = validated.get('equipment', [])
                step_data['safety_notes'] = validated.get('safety_notes', [])
                step_data['technique_level'] = validated.get('technique_level', 'standard')

                validated_steps.append(step_data)

            template['steps'] = validated_steps

        # Generate unique ID
        template_id = str(uuid.uuid4())[:8]
        template['template_id'] = template_id
        template['created_at'] = datetime.now().isoformat()

        # Save template (save raw template from generator)
        template_path = TEMPLATE_FOLDER / f'{template_id}.json'
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        validation_method = 'AI-validated with Llama 3' if llama_enabled else 'Regex-based extraction only'

        # Transform template to match Pydantic schema
        # Extract duration in minutes
        duration_str = template.get('metadata', {}).get('estimated_duration', '0 minutes')
        duration_minutes = 0
        if 'hour' in duration_str:
            duration_minutes = int(float(duration_str.split()[0]) * 60)
        else:
            duration_minutes = int(duration_str.split()[0])

        # Transform steps to match ProcedureStep schema
        transformed_steps = []
        for step in template['steps']:
            transformed_step = {
                'step_number': step['step_number'],
                'description': step['description'],
                'step_description': step['description'],  # Duplicate for schema
                'expected_action': step['expected_action'],
                'parameters': step.get('parameters', {}),
                'chemicals': step.get('chemicals', []),
                'equipment': step.get('equipment', []),
                'safety_notes': step.get('safety', []),
                'technique_level': step.get('technique', 'standard'),
                'estimated_duration_seconds': 60  # Default 1 minute per step
            }
            transformed_steps.append(transformed_step)

        # Create Pydantic model with transformed data
        template_model = ProcedureTemplate(
            template_id=template_id,
            title=template['title'],
            user_id=template.get('created_by', request.user_id),
            created_at=template['created_at'],
            modified_at=template.get('modified_at'),
            total_steps=template['metadata']['step_count'],
            estimated_duration_minutes=duration_minutes,
            steps=transformed_steps
        )

        return GenerateTemplateResponse(
            template=template_model,
            template_id=template_id,
            message=f'Template generated successfully ({validation_method})',
            llama_enabled=llama_enabled
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates", response_model=dict)
async def list_templates():
    """List all saved templates"""
    try:
        templates = []

        for template_file in TEMPLATE_FOLDER.glob('*.json'):
            if template_file.name in ['template_schema.json', 'README.md']:
                continue

            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                    templates.append(TemplateListItem(
                        id=template_file.stem,
                        template_id=data.get('template_id', template_file.stem),
                        title=data.get('title', 'Untitled'),
                        created_at=data.get('created_at', 'Unknown'),
                        step_count=len(data.get('steps', [])),
                        user_id=data.get('user_id', 'Unknown')
                    ))
            except Exception as e:
                print(f"Error reading template {template_file}: {e}")
                continue

        # Sort by created_at (newest first)
        templates.sort(key=lambda x: x.created_at, reverse=True)

        return {"templates": templates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{template_id}", response_model=dict)
async def get_template(template_id: str):
    """Get a specific template by ID"""
    template_path = TEMPLATE_FOLDER / f'{template_id}.json'

    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    try:
        with open(template_path, 'r') as f:
            template = json.load(f)

        return {"template": template}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/templates/{template_id}", response_model=dict)
async def update_template(template_id: str, template: ProcedureTemplate):
    """Update a template"""
    template_path = TEMPLATE_FOLDER / f'{template_id}.json'

    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    try:
        # Update modified timestamp
        template_dict = template.dict()
        template_dict['modified_at'] = datetime.now().isoformat()

        # Save updated template
        with open(template_path, 'w') as f:
            json.dump(template_dict, f, indent=2)

        return {
            "template": template_dict,
            "message": "Template updated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a template"""
    template_path = TEMPLATE_FOLDER / f'{template_id}.json'

    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    try:
        template_path.unlink()
        return {"message": "Template deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{template_id}/export")
async def export_template(template_id: str):
    """Download template as JSON file"""
    template_path = TEMPLATE_FOLDER / f'{template_id}.json'

    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    return FileResponse(
        path=template_path,
        media_type='application/json',
        filename=f'{template_id}.json'
    )

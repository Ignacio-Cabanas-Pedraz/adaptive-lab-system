"""
Flask REST API for Procedure Template Management
Provides endpoints for PDF/text upload, template generation, and CRUD operations
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2

# Import existing template generator and Llama validator
from src.procedure.template_generator import ProcedureTemplateGenerator
from src.procedure.llm_validator import LlamaParameterValidator

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
TEMPLATE_FOLDER = 'templates'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATE_FOLDER'] = TEMPLATE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
    if llama_validator is None:
        try:
            llama_validator = LlamaParameterValidator()
        except Exception as e:
            print(f"Warning: Could not load Llama validator: {e}")
            print("Continuing without AI validation. Install and authenticate with HuggingFace to enable.")
            llama_validator = None
    return llama_validator


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    return text


def extract_steps_from_text(text):
    """Extract procedure steps from text"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Filter out very short lines (likely headers/noise)
    steps = [line for line in lines if len(line) > 10]

    return steps


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Flask API is running'})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload PDF or text file and extract procedure steps
    Returns: { steps: string[], filename: string }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use .txt or .pdf'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

        # Extract steps
        steps = extract_steps_from_text(text)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'steps': steps,
            'filename': filename,
            'step_count': len(steps)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-template', methods=['POST'])
def generate_template():
    """
    Generate template from steps with Llama validation
    Request body: {
        title: string,
        steps: string[],
        user_id?: string
    }
    Returns: Complete template JSON
    """
    try:
        data = request.json

        if not data or 'steps' not in data or 'title' not in data:
            return jsonify({'error': 'Missing required fields: title, steps'}), 400

        title = data['title']
        steps = data['steps']
        user_id = data.get('user_id', 'default_user')

        if not steps or len(steps) == 0:
            return jsonify({'error': 'No steps provided'}), 400

        # Generate template using existing generator
        generator = get_template_generator()
        template = generator.generate_template(
            title=title,
            user_id=user_id,
            step_descriptions=steps
        )

        # Run Llama validation on all steps (if available)
        validator = get_llama_validator()

        if validator is not None:
            validated_steps = []
            for i, step_data in enumerate(template['steps']):
                original_text = steps[i]

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
        else:
            # No Llama validation - use regex-only results
            print("Warning: Generating template without AI validation (Llama not available)")

        # Generate unique ID and save
        template_id = str(uuid.uuid4())[:8]
        template['template_id'] = template_id
        template['created_at'] = datetime.now().isoformat()

        # Save to templates folder
        template_path = os.path.join(app.config['TEMPLATE_FOLDER'], f'{template_id}.json')
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        validation_method = 'AI-validated with Llama 3' if validator is not None else 'Regex-based extraction only'

        return jsonify({
            'template': template,
            'template_id': template_id,
            'message': f'Template generated successfully ({validation_method})',
            'llama_enabled': validator is not None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates', methods=['GET'])
def list_templates():
    """
    List all saved templates
    Returns: [{ id, title, created_at, step_count }]
    """
    try:
        templates = []
        template_folder = Path(app.config['TEMPLATE_FOLDER'])

        for template_file in template_folder.glob('*.json'):
            if template_file.name == 'template_schema.json' or template_file.name == 'README.md':
                continue

            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                    templates.append({
                        'id': template_file.stem,
                        'template_id': data.get('template_id', template_file.stem),
                        'title': data.get('title', 'Untitled'),
                        'created_at': data.get('created_at', 'Unknown'),
                        'step_count': len(data.get('steps', [])),
                        'user_id': data.get('user_id', 'Unknown')
                    })
            except Exception as e:
                print(f"Error reading template {template_file}: {e}")
                continue

        # Sort by created_at (newest first)
        templates.sort(key=lambda x: x['created_at'], reverse=True)

        return jsonify({'templates': templates})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/<template_id>', methods=['GET'])
def get_template(template_id):
    """Get a specific template by ID"""
    try:
        template_path = os.path.join(app.config['TEMPLATE_FOLDER'], f'{template_id}.json')

        if not os.path.exists(template_path):
            return jsonify({'error': 'Template not found'}), 404

        with open(template_path, 'r') as f:
            template = json.load(f)

        return jsonify({'template': template})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/<template_id>', methods=['PUT'])
def update_template(template_id):
    """
    Update a template
    Request body: Complete template JSON
    """
    try:
        template_path = os.path.join(app.config['TEMPLATE_FOLDER'], f'{template_id}.json')

        if not os.path.exists(template_path):
            return jsonify({'error': 'Template not found'}), 404

        updated_template = request.json

        if not updated_template:
            return jsonify({'error': 'No template data provided'}), 400

        # Update modified timestamp
        updated_template['modified_at'] = datetime.now().isoformat()

        # Save updated template
        with open(template_path, 'w') as f:
            json.dump(updated_template, f, indent=2)

        return jsonify({
            'template': updated_template,
            'message': 'Template updated successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/<template_id>', methods=['DELETE'])
def delete_template(template_id):
    """Delete a template"""
    try:
        template_path = os.path.join(app.config['TEMPLATE_FOLDER'], f'{template_id}.json')

        if not os.path.exists(template_path):
            return jsonify({'error': 'Template not found'}), 404

        os.remove(template_path)

        return jsonify({'message': 'Template deleted successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/<template_id>/export', methods=['GET'])
def export_template(template_id):
    """Download template as JSON file"""
    try:
        template_path = os.path.join(app.config['TEMPLATE_FOLDER'], f'{template_id}.json')

        if not os.path.exists(template_path):
            return jsonify({'error': 'Template not found'}), 404

        return send_file(
            template_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'{template_id}.json'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask API server...")
    print("Template folder:", app.config['TEMPLATE_FOLDER'])
    print("Upload folder:", app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5050)

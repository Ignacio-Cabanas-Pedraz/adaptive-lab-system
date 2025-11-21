# Frontend Setup Guide

Complete guide to set up and run the React + Flask procedure template management system.

## Overview

This system provides a web interface for:
- Uploading PDF/text procedure files
- Extracting and editing procedure steps
- AI-powered parameter validation with Llama
- Template management (create, edit, delete, export)
- Saving templates for CLI video processing

## Architecture

```
┌─────────────────────┐
│  React Frontend     │  Port 3000
│  (Vite + React 18)  │
└──────────┬──────────┘
           │ HTTP/REST
           ↓
┌─────────────────────┐
│  Flask REST API     │  Port 5000
│  (Python Backend)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────────────────────┐
│  Template Generator + Llama         │
│  • PDF parsing (PyPDF2)             │
│  • Step extraction (regex)          │
│  • AI validation (Llama 3 8B)       │
│  • Template storage (/templates)    │
└─────────────────────────────────────┘
```

## Prerequisites

### Backend Requirements
- Python 3.10+
- CUDA-capable GPU (for Llama 3 8B)
- 16GB+ RAM recommended
- All Python dependencies from `requirements.txt`

### Frontend Requirements
- Node.js 18+
- npm or yarn

## Installation

### 1. Install Python Dependencies

From the project root:

```bash
# Install all dependencies including Flask
pip install -r requirements.txt

# If you get import errors for Flask packages:
pip install flask flask-cors PyPDF2
```

### 2. Install Node.js Dependencies

```bash
cd frontend
npm install
```

This will install:
- React 18
- Vite (build tool)
- Axios (HTTP client)

## Running the System

### Step 1: Start Flask Backend

From the project root directory:

```bash
python app.py
```

Expected output:
```
Starting Flask API server...
Template folder: /path/to/templates
Upload folder: /path/to/uploads
 * Running on http://0.0.0.0:5000
```

**Important Notes:**
- First run may take 2-3 minutes as Llama loads
- Keep this terminal open while using the frontend
- API will be accessible at `http://localhost:5000`

### Step 2: Start React Frontend

Open a **new terminal** and run:

```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

### Step 3: Open in Browser

Navigate to: **http://localhost:3000**

You should see the "Laboratory Procedure Template Manager" interface.

## Usage Workflow

### Creating Your First Template

1. **Upload File**
   - Click the upload area or drag & drop a PDF/text file
   - Supported formats: `.txt`, `.pdf`
   - Max file size: 16MB

2. **AI Processing** (automatic)
   - Steps are extracted from the file
   - Llama 3 8B validates and enhances each step:
     - Extracts parameters (volume, temperature, duration, etc.)
     - Identifies chemicals and equipment
     - Generates safety notes
     - Classifies action types
   - This takes 30-120 seconds depending on step count

3. **Review & Edit**
   - Edit step descriptions in the text areas
   - Modify parameter values (volume, temperature, etc.)
   - Review AI-extracted chemicals, equipment, safety notes

4. **Save**
   - Template is automatically saved to `templates/{id}.json`
   - ID is shown in the success message
   - You can continue editing and save changes

### Managing Templates

1. Click **"Manage Templates"** tab
2. View all saved templates with metadata
3. Actions available:
   - **Click card**: Load template for editing
   - **Export**: Download JSON file
   - **Delete**: Remove template (with confirmation)

### Using Templates for Video Processing

After creating a template, use it with the CLI:

```bash
python scripts/test_video_with_tep.py \
  --template templates/abc12345.json \
  --video videos/your_lab_video.mp4
```

The TEP system will compare detected actions in the video against your template.

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload PDF/text file |
| POST | `/api/generate-template` | Generate template with Llama |
| GET | `/api/templates` | List all templates |
| GET | `/api/templates/:id` | Get specific template |
| PUT | `/api/templates/:id` | Update template |
| DELETE | `/api/templates/:id` | Delete template |
| GET | `/api/templates/:id/export` | Download JSON |

### Example API Calls

**Upload File:**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@procedure.pdf"
```

**List Templates:**
```bash
curl http://localhost:5000/api/templates
```

**Generate Template:**
```bash
curl -X POST http://localhost:5000/api/generate-template \
  -H "Content-Type: application/json" \
  -d '{
    "title": "DNA Extraction",
    "steps": ["Add 200µL buffer", "Incubate at 37°C for 30 min"],
    "user_id": "researcher_1"
  }'
```

## Troubleshooting

### Flask won't start

**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
pip install flask flask-cors PyPDF2
```

**Error**: `Address already in use` (port 5000)

**Solution**: Kill process on port 5000 or change port in `app.py`:
```bash
# Find process
lsof -ti:5000

# Kill it
kill -9 $(lsof -ti:5000)

# Or change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### React won't start

**Error**: `command not found: npm`

**Solution**: Install Node.js from https://nodejs.org/

**Error**: `Error: Cannot find module 'react'`

**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS errors in browser

**Symptom**: Red errors in browser console mentioning CORS

**Solution**: Ensure Flask-CORS is installed:
```bash
pip install flask-cors
```

Verify `CORS(app)` is in `app.py` (line 17)

### Llama takes forever to load

**Symptom**: Template generation stuck at "Generating template with AI validation..."

**Causes**:
1. First load always takes 1-2 minutes (model loading)
2. Running on CPU instead of GPU (very slow)

**Solution**:
- Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, Llama will run on CPU (expect 5-10 min per template)
- Consider disabling Llama validation for faster testing

### PDF upload fails

**Error**: `Error reading PDF`

**Causes**:
- Encrypted/password-protected PDF
- Scanned image PDF (no text layer)
- Corrupted file

**Solutions**:
- Remove password protection
- Use OCR tool to extract text first
- Convert to .txt format manually

### Templates not appearing in "Manage" tab

**Check**:
```bash
ls -la templates/
```

Templates should have `.json` extension and valid JSON format.

**Solution**: Delete corrupted JSON files:
```bash
cd templates
python -m json.tool template_name.json  # Validate
```

## Development Tips

### Hot Reload

Both Flask and React support hot reload:
- **React**: Changes to `.jsx` files reload instantly
- **Flask**: Changes to `app.py` restart the server (set `debug=True`)

### Debugging Flask

Add debug prints in `app.py`:
```python
print(f"DEBUG: Received file: {filename}")
```

View output in Flask terminal.

### Debugging React

Use browser DevTools:
- **Console**: View errors and `console.log()` output
- **Network**: Inspect API requests/responses
- **React DevTools**: Install browser extension for component inspection

### Testing API Independently

Test Flask without React:
```bash
# Health check
curl http://localhost:5000/api/health

# Upload file
curl -X POST http://localhost:5000/api/upload \
  -F "file=@test.txt"
```

## File Locations

| Item | Location |
|------|----------|
| Flask API | `app.py` |
| React Source | `frontend/src/` |
| Templates | `templates/*.json` |
| Uploaded Files | `uploads/` (temporary) |
| Frontend Build | `frontend/dist/` |

## Production Deployment

### Building React for Production

```bash
cd frontend
npm run build
```

Outputs to `frontend/dist/`

### Serve with Flask

Update `app.py` to serve static files:

```python
from flask import send_from_directory

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend/dist', path)
```

Then run:
```bash
python app.py
```

Access at: http://localhost:5000

## Next Steps

1. **Create your first template** using a sample procedure file
2. **Test video processing** with the generated template
3. **Iterate on parameters** to improve accuracy
4. **Build a template library** for common procedures

## Support

For issues or questions:
- Check Flask terminal for backend errors
- Check browser console for frontend errors
- Review API responses in Network tab
- Consult main README.md for system architecture

---

**Built with React + Flask + Llama 3 8B**

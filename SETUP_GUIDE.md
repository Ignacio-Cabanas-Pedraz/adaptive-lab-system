# Lab Procedure Template Manager - Setup Guide

Complete guide to set up and run the TypeScript + FastAPI procedure template management system.

## Architecture Overview

This system uses the proven **LabVision architecture**:

```
┌─────────────────────┐         ┌──────────────────────┐
│  React Frontend     │         │  FastAPI Backend     │
│  (TypeScript)       │  HTTP   │  (Python)            │
│  Port 3000          │◄───────►│  Port 8000           │
│                     │  REST   │                      │
│  - TailwindCSS      │         │  - Pydantic schemas  │
│  - Lucide icons     │         │  - Template gen      │
│  - React Router     │         │  - PDF parsing       │
└─────────────────────┘         └──────────────────────┘
```

## Prerequisites

### Backend
- Python 3.10+
- pip package manager
- Optional: CUDA GPU for Llama validation

### Frontend
- Node.js 18+
- npm package manager

## Installation

### 1. Install Backend Dependencies

```bash
# Install FastAPI dependencies
pip install -r backend/requirements.txt

# Install core system dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Running the System

### Option A: Automated Start (Recommended)

**Terminal 1 - Backend:**
```bash
./start-backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start-frontend.sh
```

### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

Expected output:
```
🚀 Starting FastAPI server on http://localhost:8000
📚 API docs available at http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  press h to show help
```

### Access the Application

1. **Frontend**: http://localhost:3000
2. **API Docs**: http://localhost:8000/docs
3. **API Health**: http://localhost:8000/health

## Features

### 1. Create Templates
- Upload PDF or text files
- Automatic step extraction
- AI validation with Llama 3 (optional)
- Edit steps, parameters, and metadata
- Professional UI with TailwindCSS

### 2. Manage Templates
- List all templates
- Export as JSON
- Delete templates
- View template details

### 3. Use with CLI
After creating a template, use it for video processing:

```bash
python scripts/test_video_with_tep.py \
  --template templates/{template_id}.json \
  --video videos/your_video.mp4
```

## Project Structure

```
adaptive-lab-system/
├── frontend/                       # TypeScript + React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── TemplateList.tsx   # Main template list page
│   │   │   └── CreateTemplate.tsx # Template creation page
│   │   ├── api/
│   │   │   ├── client.ts          # Axios client
│   │   │   └── procedures.ts      # API functions
│   │   ├── types/
│   │   │   └── procedure.ts       # TypeScript types
│   │   ├── utils/
│   │   │   └── cn.ts              # Utility functions
│   │   ├── App.tsx                # Main app with routing
│   │   ├── main.tsx               # Entry point
│   │   └── index.css              # TailwindCSS
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── backend/                        # FastAPI backend
│   ├── api/
│   │   └── procedures.py          # REST API endpoints
│   ├── schemas/
│   │   └── procedure.py           # Pydantic models
│   ├── main.py                    # FastAPI app
│   └── requirements.txt
│
├── src/                            # Core TEP system
│   ├── procedure/                 # Template generator
│   ├── tep/                       # Temporal Event Parser
│   └── integration/               # Video processing
│
├── templates/                     # Generated templates
├── uploads/                       # Temporary uploads
│
├── start-backend.sh               # Quick start script
├── start-frontend.sh              # Quick start script
└── SETUP_GUIDE.md                 # This file
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload PDF/text file |
| POST | `/api/generate-template` | Generate template |
| GET | `/api/templates` | List all templates |
| GET | `/api/templates/:id` | Get specific template |
| PUT | `/api/templates/:id` | Update template |
| DELETE | `/api/templates/:id` | Delete template |
| GET | `/api/templates/:id/export` | Download JSON |

Full API documentation: http://localhost:8000/docs

## Technology Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **TailwindCSS** - Utility-first styling
- **Vite** - Build tool
- **React Router** - Routing
- **Axios** - HTTP client
- **Lucide React** - Icons

### Backend
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server
- **PyPDF2** - PDF text extraction

### Core System
- **Python 3.10+** - Core language
- **Llama 3 8B** - AI validation (optional)
- **PyTorch** - Deep learning
- **Transformers** - LLM support

## Troubleshooting

### Backend won't start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install -r backend/requirements.txt
```

**Error**: `Address already in use` (port 8000)

**Solution**:
```bash
# Find process on port 8000
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)
```

### Frontend won't start

**Error**: `command not found: npm`

**Solution**: Install Node.js from https://nodejs.org/

**Error**: `Cannot find module 'react'`

**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS errors in browser

**Symptom**: Red errors in browser console mentioning CORS

**Solution**: Ensure both backend and frontend are running:
- Backend on port 8000
- Frontend on port 3000

### Llama validation not working

**Symptom**: Templates created with "⚠️ Regex-based extraction only"

**Cause**: Llama 3 not available or not authenticated

**Solution**: See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for authentication steps

**Quick fix**: System works without Llama, just with lower accuracy

### TypeScript errors

**Error**: Type errors in IDE

**Solution**:
```bash
cd frontend
npm run build  # Check for type errors
```

## Development

### Hot Reload

Both servers support hot reload:
- **Frontend**: Changes to `.tsx` files reload instantly
- **Backend**: Changes to `.py` files restart server automatically

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
# Output in frontend/dist/
```

**Backend**: FastAPI doesn't need building, but disable reload:
```python
# In backend/main.py
uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
```

### Testing API

Use the interactive docs at http://localhost:8000/docs to test endpoints.

Or use curl:
```bash
# Health check
curl http://localhost:8000/health

# List templates
curl http://localhost:8000/api/templates
```

## Next Steps

1. **Create your first template** by uploading a procedure file
2. **Test video processing** with the generated template
3. **Iterate on parameters** to improve accuracy
4. **Build a template library** for common procedures

## Comparison with Previous Setup

| Aspect | Old (Flask + JS) | New (FastAPI + TS) |
|--------|-----------------|-------------------|
| Frontend | JavaScript | TypeScript ✅ |
| Styling | Custom CSS | TailwindCSS ✅ |
| Type Safety | None | Full TypeScript ✅ |
| Icons | Emojis | Lucide React ✅ |
| Backend | Flask | FastAPI ✅ |
| API Docs | Manual | Auto-generated ✅ |
| Validation | Manual | Pydantic ✅ |
| Maintainability | Medium | High ✅ |

## Support

For issues or questions:
- Check this guide first
- Review API docs at http://localhost:8000/docs
- Check main [README.md](README.md) for system architecture
- Review [LabVision](../LabVision) for reference implementation

---

**Built with TypeScript + TailwindCSS + FastAPI**
Based on proven LabVision architecture

# Procedure Template Manager - Frontend

React-based web interface for creating and managing laboratory procedure templates.

## Features

- 📄 Upload PDF or text files with procedure descriptions
- 🤖 Automatic step extraction and AI validation with Llama
- ✏️ Intuitive editing of step descriptions and parameters
- 💾 Save, load, edit, and delete templates
- 📥 Export templates as JSON files
- 🎯 Templates ready for CLI video processing

## Prerequisites

- Node.js 18+ and npm
- Flask API running on port 5000 (see backend setup)

## Installation

```bash
# From the frontend directory
cd frontend

# Install dependencies
npm install
```

## Running the Application

### 1. Start the Flask Backend

In the project root directory:

```bash
# Install Python dependencies if not already done
pip install -r requirements.txt

# Start Flask API
python app.py
```

The API will run on `http://localhost:5050`

### 2. Start the React Frontend

In a new terminal, from the frontend directory:

```bash
npm run dev
```

The application will open at `http://localhost:3000`

## Usage

### Creating a New Template

1. Click the **"Create/Edit Template"** tab
2. Upload a PDF or text file containing procedure steps
3. The system will:
   - Extract steps from the file
   - Run Llama AI validation to extract parameters, chemicals, equipment, and safety notes
   - Generate a structured template
4. Edit any steps, descriptions, or parameters as needed
5. Click **"Save Changes"** to update the template
6. The template is automatically saved to `templates/{id}.json`

### Managing Templates

1. Click the **"Manage Templates"** tab
2. View all saved templates
3. Click a template card to load and edit it
4. Use **"Export"** to download the JSON file
5. Use **"Delete"** to remove a template

### Using Templates with Video Processing

After creating a template, use it with the CLI:

```bash
python scripts/test_video_with_tep.py \
  --template templates/{template_id}.json \
  --video videos/your_video.mp4
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── FileUpload.jsx      # File upload component
│   │   ├── StepEditor.jsx      # Step editing interface
│   │   └── TemplateList.jsx    # Template management
│   ├── App.jsx                 # Main application
│   ├── App.css                 # Styles
│   └── main.jsx                # Entry point
├── index.html                  # HTML template
├── vite.config.js             # Vite configuration
└── package.json               # Dependencies
```

## API Endpoints

The frontend communicates with these Flask endpoints:

- `POST /api/upload` - Upload and extract steps from PDF/text
- `POST /api/generate-template` - Generate template with Llama validation
- `GET /api/templates` - List all templates
- `GET /api/templates/:id` - Get specific template
- `PUT /api/templates/:id` - Update template
- `DELETE /api/templates/:id` - Delete template
- `GET /api/templates/:id/export` - Download template JSON

## Development

### Building for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Troubleshooting

### CORS Errors

Make sure Flask-CORS is installed and the Flask API is running:

```bash
pip install flask-cors
python app.py
```

### Port Already in Use

If port 3000 is in use, Vite will automatically try the next available port.

### Upload Fails

Check that:
- Flask API is running on port 5000
- The `uploads/` folder exists
- File is .txt or .pdf format
- File size is under 16MB

### Llama Validation Slow

First-time template generation may take 1-2 minutes as Llama loads. Subsequent generations will be faster.

## Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **Flask** - Python REST API
- **PyPDF2** - PDF text extraction
- **Llama 3 8B** - AI parameter validation

## License

MIT

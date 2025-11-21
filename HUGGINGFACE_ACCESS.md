# HuggingFace Llama 3 Access Setup

You're already logged into HuggingFace as **thatbiologicalprogrammer**! ✅

## Step 1: Request Access to Llama 3

1. **Visit the Llama 3 Model Page:**
   ```
   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   ```

2. **Click "Request Access"** at the top of the page

3. **Fill out the form:**
   - Provide your contact information
   - Agree to Meta's license terms
   - Submit the request

4. **Wait for approval:**
   - Usually takes a few hours to 1-2 days
   - You'll receive an email when approved

## Step 2: Verify Your HuggingFace Token

Your HuggingFace CLI is already authenticated. To check your token:

```bash
# Check current authentication
huggingface-cli whoami

# View your token (if needed)
cat ~/.huggingface/token
```

## Step 3: Test Access

Once approved, test that you can access the model:

```bash
# Test from Python
python << EOF
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    print("✅ Access granted! You can use Llama 3")
except Exception as e:
    print(f"❌ Access denied: {e}")
EOF
```

## Step 4: Update Environment Variables (RunPod)

When deploying to RunPod, you'll need to set your token as an environment variable:

```bash
# Your HuggingFace token
export HF_TOKEN="your_token_here"

# Or add to .env file
echo "HF_TOKEN=your_token_here" >> .env
```

## Troubleshooting

### Error: "Cannot access gated repo"
- You haven't been granted access yet
- Request access at the link above and wait for approval

### Error: "401 Unauthorized"
- Your token may have expired
- Re-login: `huggingface-cli login`

### Error: "403 Forbidden"
- You may have accepted the old license
- Visit the model page and re-accept the latest license

## Alternative: Use Without Llama (Regex Only)

The system works perfectly fine without Llama! It will use:
- ✅ Regex-based parameter extraction (volumes, temps, durations, etc.)
- ✅ Action classification
- ✅ Safety notes and equipment detection
- ❌ AI-enhanced validation (only with Llama)

To disable Llama and use regex-only:

```python
from src.procedure.template_generator import ProcedureTemplateGenerator

# Create generator without LLM
generator = ProcedureTemplateGenerator(use_llm_validation=False)
```

The backend will automatically fall back to regex-only mode if Llama access fails.

## Current Status

- ✅ Logged in to HuggingFace as **thatbiologicalprogrammer**
- ⏳ Need to request Llama 3 access
- ⏳ Need to wait for Meta's approval

Once you have access, the system will automatically use Llama 3 for enhanced template generation!

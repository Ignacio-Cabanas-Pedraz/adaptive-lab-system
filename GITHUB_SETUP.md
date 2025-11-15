# GitHub Setup Guide

## Quick Steps to Push to GitHub

### 1. Create a new repository on GitHub

1. Go to https://github.com/new
2. Repository name: `adaptive-lab-system`
3. Description: "Multi-mode object detection for lab equipment using YOLO, SAM 2, and CLIP"
4. Set to **Public**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **"Create repository"**

### 2. Push your local repository

After creating the GitHub repository, run these commands:

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/adaptive-lab-system.git

# Verify the remote was added
git remote -v

# Push to GitHub
git push -u origin main
```

### 3. Verify on GitHub

Visit your repository at:
```
https://github.com/YOUR_USERNAME/adaptive-lab-system
```

You should see all files including the README displayed on the main page.

## Using on RunPod

Once pushed to GitHub, you can clone it on any RunPod instance:

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/adaptive-lab-system.git
cd adaptive-lab-system

# Run setup
bash setup_runpod.sh
```

## Updating the Repository

When you make changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with a message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Common Git Commands

```bash
# View commit history
git log --oneline

# View changes
git diff

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Pull latest changes
git pull
```

## Repository Structure

Your repository includes:
- ✅ Complete adaptive lab system implementation
- ✅ Video processing pipeline
- ✅ RunPod deployment script
- ✅ Comprehensive documentation
- ✅ Requirements file
- ✅ .gitignore (excludes large model files)

## Important Notes

1. **Model checkpoints are NOT included** in the repository (they're in .gitignore)
   - They will be downloaded automatically by `setup_runpod.sh`
   - SAM 2 checkpoint: ~150MB
   - YOLO models: downloaded on first use

2. **Videos are NOT included** in the repository
   - Users upload their own videos to the `videos/` directory
   - The directory structure is preserved with `.gitkeep` files

3. **Output files are NOT included** in the repository
   - Generated locally during processing
   - Each run creates a timestamped subdirectory

## Making Your Repository Public

Your repository is ready to share! You can:
- Share the GitHub URL with collaborators
- Add it to your portfolio
- Include it in documentation
- Use it across multiple RunPod instances

## License

Consider adding a LICENSE file if you want to specify how others can use your code:

```bash
# MIT License (permissive)
curl https://opensource.org/licenses/MIT -o LICENSE

# Or edit manually to add your name and year
```

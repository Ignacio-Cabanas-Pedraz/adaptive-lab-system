# How to Download Output Files from RunPod

## Quick Start - HTTP Server Method

### Step 1: On Your RunPod Terminal

```bash
# Navigate to your project
cd /adaptive-lab-system

# Start the download server
bash start_download_server.sh
```

You should see:
```
========================================
Starting Download Server
========================================

Available result folders:
run_20251115_121816/

Starting HTTP server on port 8000...

========================================
Access your files via RunPod's web interface:

1. Go to your RunPod Pod page
2. Click 'Connect' button
3. Look for 'HTTP Service [Port 8000]'
4. Click the generated URL
5. Browse and download your files
========================================

Server is running... Press Ctrl+C to stop
```

### Step 2: Access the Download URL

**In your RunPod web interface:**

1. Go to your **Pods** page
2. Find your running pod
3. Click the **"Connect"** button
4. Look for one of these:
   - **"HTTP Service [Port 8000]"** - Click the link
   - **"Connect to HTTP Service on Port 8000"** - Click it
   - **TCP Port Mappings** section - Find port 8000

You'll get a URL like:
```
https://abc123def456-8000.proxy.runpod.net
```

### Step 3: Browse and Download

Your browser will open showing a file listing:

```
output/
  run_20251115_121816/
    ├── annotated_video.mp4
    ├── results.json
    ├── summary.json
    └── masks/
```

**Click on any file to download it!**

### Step 4: Stop the Server

When done downloading, go back to your RunPod terminal and press:
```
Ctrl + C
```

## What Files to Download

### Essential Files

**results.json**
- Complete detection data for every frame
- Object classes, confidences, bounding boxes
- Processing times
- Use this for analysis, visualization, or further processing

**summary.json**
- Overall statistics
- Unique objects detected
- Average processing time
- Mode distribution

### Optional Files

**annotated_video.mp4** (if created)
- Video with bounding boxes and labels
- Can be large (100MB+)
- Only created if you didn't use `--no-video-output`

**masks/** folder (if saved)
- Individual PNG files for each detected object mask
- Only created if you used `--save-masks`
- Can be many files (hundreds/thousands)

## Alternative: Download Specific Files Only

If you only want specific files, you can use `curl` or `wget`:

### From Your RunPod Terminal

```bash
# Compress results for faster download
cd /adaptive-lab-system/output
tar -czf results_package.tar.gz run_*/

# Start server
python3 -m http.server 8000
```

### From Your Local Machine

After getting the RunPod URL (e.g., `https://abc123-8000.proxy.runpod.net`):

```bash
# Download compressed package
wget https://abc123-8000.proxy.runpod.net/results_package.tar.gz

# Extract
tar -xzf results_package.tar.gz
```

## Tips for Large Files

### Compress Before Downloading

Video files can be large. Compress them first:

```bash
cd /adaptive-lab-system/output/run_TIMESTAMP

# Compress video (reduces size by 20-50%)
ffmpeg -i annotated_video.mp4 -vcodec libx264 -crf 28 annotated_video_compressed.mp4

# Or create a zip of everything
zip -r results.zip .
```

### Download in Chunks

For very large files, use a download manager or:

```bash
# On your local machine
# Download with resume capability
wget -c https://abc123-8000.proxy.runpod.net/run_TIMESTAMP/annotated_video.mp4
```

### JSON Files Only (Fastest)

If you only need the data (not video):

```bash
# On RunPod
cd /adaptive-lab-system/output/run_TIMESTAMP

# Start server in this directory
python3 -m http.server 8000

# Download only results.json and summary.json from browser
```

## Troubleshooting

### "Connection Refused" or Can't Access URL

1. **Check server is running** on RunPod terminal
2. **Verify port 8000** is exposed in RunPod pod settings
3. Try a different port:
   ```bash
   bash start_download_server.sh 8080
   ```

### "No HTTP Service" in RunPod Interface

1. **Expose the port** in RunPod:
   - Stop your pod
   - Edit pod settings
   - Add port 8000 to "Expose HTTP Ports"
   - Restart pod

2. Or use **SSH tunnel**:
   ```bash
   # On your local machine
   ssh -L 8000:localhost:8000 -p PORT root@ssh.runpod.io

   # Then access http://localhost:8000 in your browser
   ```

### Server Stops When Terminal Closes

Run in background:

```bash
# On RunPod
cd /adaptive-lab-system/output
nohup python3 -m http.server 8000 > server.log 2>&1 &

# Check it's running
ps aux | grep http.server

# Stop it later
pkill -f "http.server 8000"
```

### Download is Very Slow

1. **Compress files first**:
   ```bash
   tar -czf results.tar.gz run_TIMESTAMP/
   ```

2. **Use cloud storage** for large files (see main README)

3. **Download JSON only**, skip video

## File Organization After Download

After downloading, organize like this:

```
local_project/
├── videos/
│   └── your-video.mp4 (original)
├── results/
│   ├── run_20251115_121816/
│   │   ├── results.json
│   │   ├── summary.json
│   │   ├── annotated_video.mp4
│   │   └── masks/
│   └── run_20251115_130000/
│       └── ...
```

## Quick Reference

```bash
# Start download server
cd /adaptive-lab-system
bash start_download_server.sh

# Start on different port
bash start_download_server.sh 8080

# Compress before downloading
cd output
tar -czf results.tar.gz run_*/

# List all result folders
ls -lh /adaptive-lab-system/output/

# Check file sizes
du -sh /adaptive-lab-system/output/run_*

# Stop server
Ctrl + C
```

## Next Steps After Download

Once you have your files locally:

1. **View results.json**:
   ```bash
   cat results.json | jq '.[0]'  # View first frame
   ```

2. **View summary**:
   ```bash
   cat summary.json | jq '.'
   ```

3. **Play annotated video**:
   ```bash
   vlc annotated_video.mp4
   # or any video player
   ```

4. **Analyze in Python**:
   ```python
   import json

   with open('results.json') as f:
       results = json.load(f)

   # Count objects by class
   all_objects = []
   for frame in results:
       for obj in frame['objects']:
           all_objects.append(obj['class'])

   from collections import Counter
   print(Counter(all_objects))
   ```

Happy downloading! 📥

# Why GPU Installation Takes Time

## Why `tensorflow[and-cuda]` Installation is Slow

The `pip install tensorflow[and-cuda]` command is taking a long time because:

### 1. **Large Package Size**
- TensorFlow with CUDA support: **~500MB - 2GB+**
- CUDA libraries: **~200-500MB**
- cuDNN: **~100-300MB**
- Additional dependencies: **~100-200MB**
- **Total download: ~1-3GB**

### 2. **Multiple Components**
The installation includes:
- TensorFlow core
- CUDA Toolkit libraries
- cuDNN (Deep Neural Network library)
- cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE
- NCCL (for multi-GPU)
- TensorRT (optional)

### 3. **Download Time**
- Depends on internet speed
- At 10 Mbps: ~15-30 minutes
- At 100 Mbps: ~2-5 minutes
- At 1 Gbps: ~30 seconds - 2 minutes

### 4. **Installation/Extraction**
- Extracting compressed packages
- Installing to site-packages
- Setting up library paths
- **Can take 5-15 minutes even after download**

## Expected Timeline

| Step | Time (Fast Internet) | Time (Slow Internet) |
|------|---------------------|---------------------|
| Download TensorFlow | 2-5 min | 15-30 min |
| Download CUDA libs | 1-3 min | 10-20 min |
| Extract & Install | 3-5 min | 5-10 min |
| **Total** | **6-13 min** | **30-60 min** |

## How to Monitor Progress

### Option 1: Check Process Status
```bash
ps aux | grep "pip install" | grep -v grep
```

### Option 2: Check Network Activity
```bash
# Monitor network usage
iftop -i eth0  # or your network interface
# or
nethogs  # shows per-process network usage
```

### Option 3: Check Disk Activity
```bash
# Monitor disk I/O
iotop
# or
iostat -x 1
```

### Option 4: Check Installation Directory
```bash
# Watch TensorFlow directory grow
watch -n 2 "du -sh /home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/tensorflow*"
```

## Current Status Check

The installation is running if you see:
- Process: `pip install --upgrade tensorflow[and-cuda]`
- CPU usage: 5-20% (downloading/extracting)
- Network activity: High (downloading)
- Disk activity: High (writing files)

## What's Happening Now

Based on the process check:
- ✅ Installation is **actively running**
- ✅ Pip process is **downloading/installing**
- ⏳ **Still in progress** (this is normal)

## Options

### Option 1: Wait for Completion (Recommended)
- Let it finish (usually 10-30 minutes)
- You'll get fully working GPU support
- One-time setup

### Option 2: Cancel and Use Alternative
If it's taking too long, you can:
```bash
# Cancel current installation (Ctrl+C or kill process)
# Then try:
pip install tensorflow[and-cuda] --no-cache-dir
# Or install specific version:
pip install tensorflow==2.15.0 tensorflow-gpu==2.15.0
```

### Option 3: Use Pre-installed CUDA (If Available)
If you have CUDA already installed:
```bash
# Just install TensorFlow (without CUDA bundle)
pip install tensorflow==2.20.0
# Then set LD_LIBRARY_PATH to your CUDA installation
```

## After Installation

Once complete, verify GPU:
```bash
python scripts/gpu/check_gpu.py
```

You should see:
```
✓ GPU is configured and ready to use!
```

## Performance After Setup

- **GPU simulation**: 10-100x faster than CPU
- **Worth the wait**: Significant speedup for large simulations
- **One-time cost**: Only need to install once

## Troubleshooting Slow Installation

If it's been > 1 hour:
1. Check internet connection
2. Check disk space: `df -h`
3. Check if process is stuck: `strace -p <PID>`
4. Consider using a faster mirror or VPN


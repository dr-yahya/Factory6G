# Quick GPU Fix

## Problem
TensorFlow cannot find GPU libraries, showing:
```
Cannot dlopen some GPU libraries
Skipping registering GPU devices...
```

## Quick Solution

Run this command to install TensorFlow with CUDA support:

```bash
cd /home/ysabe/personal/Factory6G
source .venv/bin/activate
pip install --upgrade tensorflow[and-cuda]
```

Or use the automated script:

```bash
python scripts/fix_gpu.py
```

## Verify GPU

After installation, verify GPU is working:

```bash
python scripts/check_gpu.py
```

You should see:
```
✓ GPU is configured and ready to use!
```

## Run Simulation

Once GPU is working, run the simulation:

```bash
python scripts/run_6g_simulation.py
```

The simulation will now use GPU for much faster execution!

## Alternative: Manual CUDA Installation

If the above doesn't work, see `docs/GPU_SETUP.md` for detailed manual installation instructions.


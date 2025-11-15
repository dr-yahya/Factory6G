# GPU Setup & Troubleshooting

This directory contains documentation for GPU setup, configuration, and troubleshooting.

## Files

1. **[01_setup.md](01_setup.md)** - GPU setup guide
   - Installing TensorFlow with CUDA support
   - Verifying GPU installation
   - Basic configuration

2. **[02_wsl_troubleshooting.md](02_wsl_troubleshooting.md)** - WSL-specific GPU troubleshooting
   - Common WSL GPU issues
   - Solutions and workarounds
   - WSL-specific configuration

3. **[03_installation_time.md](03_installation_time.md)** - GPU installation time estimates
   - Expected installation duration
   - Progress monitoring
   - Troubleshooting slow installations

## Quick Links

- **First time setup?** Start with `01_setup.md`
- **Using WSL?** Check `02_wsl_troubleshooting.md`
- **Installation taking too long?** See `03_installation_time.md`

## Related Scripts

GPU-related scripts are located in `scripts/gpu/`:
- `check_gpu.py` - Check GPU availability
- `fix_gpu.py` - Fix GPU setup issues
- `fix_wsl_gpu.py` - Fix WSL GPU issues
- `monitor_gpu_install.py` - Monitor installation progress


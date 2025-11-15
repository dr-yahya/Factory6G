#!/usr/bin/env python3
"""
Monitor GPU installation progress.

This script tracks the progress of tensorflow[and-cuda] installation
by monitoring process status, network activity, and disk usage.
"""

import os
import sys
import time
import subprocess
import psutil
from pathlib import Path
from datetime import datetime, timedelta

def get_process_info(pid):
    """Get information about a process."""
    try:
        proc = psutil.Process(pid)
        return {
            'cpu_percent': proc.cpu_percent(interval=0.1),
            'memory_mb': proc.memory_info().rss / (1024 * 1024),
            'status': proc.status(),
            'create_time': proc.create_time(),
            'elapsed': time.time() - proc.create_time()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def get_directory_size(path):
    """Get total size of directory in MB."""
    try:
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0

def get_network_connections(pid):
    """Get network connections for a process."""
    try:
        proc = psutil.Process(pid)
        connections = proc.connections()
        return len(connections)
    except:
        return 0

def find_pip_process():
    """Find the pip install process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'pip' in ' '.join(cmdline).lower():
                if 'tensorflow' in ' '.join(cmdline).lower() or 'cuda' in ' '.join(cmdline).lower():
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def main():
    print("=" * 80)
    print("GPU Installation Progress Monitor")
    print("=" * 80)
    print()
    
    # Find pip process
    pid = find_pip_process()
    if not pid:
        print("⚠ No pip installation process found.")
        print("  The installation may have completed or not started yet.")
        return
    
    print(f"✓ Found installation process: PID {pid}")
    print()
    
    # Get paths
    project_root = Path(__file__).parent.parent
    venv_path = project_root / ".venv" / "lib" / "python3.10" / "site-packages"
    tensorflow_path = venv_path / "tensorflow"
    nvidia_path = venv_path / "nvidia"
    
    # Initial measurements
    print("Taking initial measurements...")
    time.sleep(2)
    
    initial_tf_size = get_directory_size(tensorflow_path) if tensorflow_path.exists() else 0
    initial_nvidia_size = get_directory_size(nvidia_path) if nvidia_path.exists() else 0
    
    print(f"Initial TensorFlow size: {initial_tf_size:.1f} MB")
    print(f"Initial NVIDIA libraries size: {initial_nvidia_size:.1f} MB")
    print()
    print("=" * 80)
    print("Monitoring installation progress...")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    print()
    
    start_time = time.time()
    last_tf_size = initial_tf_size
    last_nvidia_size = initial_nvidia_size
    last_update = time.time()
    
    try:
        while True:
            # Check if process still exists
            proc_info = get_process_info(pid)
            if not proc_info:
                print()
                print("=" * 80)
                print("✓ Installation process completed!")
                print("=" * 80)
                break
            
            # Get current sizes
            current_tf_size = get_directory_size(tensorflow_path) if tensorflow_path.exists() else 0
            current_nvidia_size = get_directory_size(nvidia_path) if nvidia_path.exists() else 0
            
            # Calculate changes
            tf_delta = current_tf_size - last_tf_size
            nvidia_delta = current_nvidia_size - last_nvidia_size
            total_delta = tf_delta + nvidia_delta
            
            # Calculate rate (MB/s)
            time_delta = time.time() - last_update
            if time_delta > 0:
                rate = total_delta / time_delta
            else:
                rate = 0
            
            # Get network connections
            network_conns = get_network_connections(pid)
            
            # Clear screen and print status
            os.system('clear' if os.name != 'nt' else 'cls')
            print("=" * 80)
            print("GPU Installation Progress Monitor")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Process status
            print("📊 Process Status:")
            print(f"  PID: {pid}")
            print(f"  Status: {proc_info['status']}")
            print(f"  CPU: {proc_info['cpu_percent']:.1f}%")
            print(f"  Memory: {proc_info['memory_mb']:.1f} MB")
            print(f"  Elapsed Time: {format_time(proc_info['elapsed'])}")
            print(f"  Network Connections: {network_conns}")
            print()
            
            # Installation progress
            print("📦 Installation Progress:")
            print(f"  TensorFlow: {current_tf_size:.1f} MB (Δ {tf_delta:+.1f} MB)")
            if nvidia_path.exists():
                print(f"  NVIDIA Libraries: {current_nvidia_size:.1f} MB (Δ {nvidia_delta:+.1f} MB)")
            else:
                print(f"  NVIDIA Libraries: Not installed yet")
            
            total_size = current_tf_size + current_nvidia_size
            print(f"  Total: {total_size:.1f} MB")
            print()
            
            # Activity indicators
            if total_delta > 0:
                print(f"⚡ Activity: Downloading/Installing ({rate:.2f} MB/s)")
            elif proc_info['cpu_percent'] > 1:
                print(f"⚡ Activity: Processing (CPU: {proc_info['cpu_percent']:.1f}%)")
            elif network_conns > 0:
                print(f"⚡ Activity: Network active ({network_conns} connections)")
            else:
                print("⏸️  Activity: Waiting/Idle")
            print()
            
            # Estimated time (rough estimate)
            if total_size > 100:  # Only estimate if we have some data
                # Rough estimate: total package is ~2000-3000 MB
                estimated_total = 2500  # MB
                if rate > 0:
                    remaining_mb = max(0, estimated_total - total_size)
                    eta_seconds = remaining_mb / rate if rate > 0 else 0
                    if eta_seconds > 0 and eta_seconds < 3600:  # Less than 1 hour
                        print(f"⏱️  Estimated Time Remaining: {format_time(eta_seconds)}")
                    else:
                        print(f"⏱️  Estimated Time Remaining: Calculating...")
                else:
                    print(f"⏱️  Estimated Time Remaining: Waiting for activity...")
            else:
                print(f"⏱️  Estimated Time Remaining: Initializing...")
            print()
            
            print("=" * 80)
            print("Press Ctrl+C to stop monitoring (installation will continue)")
            print("=" * 80)
            
            # Update last values
            last_tf_size = current_tf_size
            last_nvidia_size = current_nvidia_size
            last_update = time.time()
            
            # Wait before next update
            time.sleep(3)
            
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("Monitoring stopped.")
        print("Installation will continue in the background.")
        print("=" * 80)
        print()
        print("To check if installation completed, run:")
        print("  python scripts/check_gpu.py")
        print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


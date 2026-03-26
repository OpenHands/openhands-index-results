#!/usr/bin/env python3
"""
Monitor inference progress by tracking line counts in output files.

This script monitors the progress of inference runs by counting lines in
output.jsonl and critic attempt files, then appending a timestamped record
to inference_progress.txt.

Usage:
    # One-shot: count lines once and append to inference_progress.txt
    python monitor_infer_progress.py /path/to/output/dir
    
    # Continuous monitoring: run every 10 minutes
    python monitor_infer_progress.py /path/to/output/dir --monitor
    
    # Custom interval (in seconds)
    python monitor_infer_progress.py /path/to/output/dir --monitor --interval 600
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path


def count_lines(file_path: Path) -> int:
    """Count lines in a file. Returns 0 if file doesn't exist or can't be read."""
    try:
        if not file_path.exists():
            return 0
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except (IOError, OSError):
        return 0


def collect_progress(output_dir: Path) -> dict:
    """Collect line counts from output files.
    
    Args:
        output_dir: Directory containing output.jsonl and critic attempt files
        
    Returns:
        Dictionary with timestamp and line counts
    """
    files = [
        'output.jsonl',
        'output.critic_attempt_1.jsonl',
        'output.critic_attempt_2.jsonl',
        'output.critic_attempt_3.jsonl'
    ]
    
    counts = {}
    for filename in files:
        file_path = output_dir / filename
        counts[filename] = count_lines(file_path)
    
    return {
        'timestamp': datetime.now(timezone.utc),
        'counts': counts
    }


def write_progress(output_dir: Path, progress: dict) -> None:
    """Append progress record to inference_progress.txt.
    
    Args:
        output_dir: Directory where inference_progress.txt will be written
        progress: Dictionary with timestamp and line counts
    """
    progress_file = output_dir / 'inference_progress.txt'
    timestamp_str = progress['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
    counts = progress['counts']
    
    line = (
        f"{timestamp_str}, "
        f"{counts['output.jsonl']}, "
        f"{counts['output.critic_attempt_1.jsonl']}, "
        f"{counts['output.critic_attempt_2.jsonl']}, "
        f"{counts['output.critic_attempt_3.jsonl']}\n"
    )
    
    with open(progress_file, 'a') as f:
        f.write(line)


def monitor_once(output_dir: Path) -> None:
    """Run monitoring once: collect progress and write to file."""
    progress = collect_progress(output_dir)
    write_progress(output_dir, progress)


def monitor_continuous(output_dir: Path, interval: int = 600) -> None:
    """Run monitoring continuously at specified interval.
    
    Args:
        output_dir: Directory containing output files
        interval: Time in seconds between monitoring checks (default: 600 = 10 minutes)
    """
    print(f"Starting continuous monitoring of {output_dir}")
    print(f"Checking every {interval} seconds (Ctrl+C to stop)")
    
    try:
        while True:
            monitor_once(output_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monitor inference progress by tracking output file line counts'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Directory containing output.jsonl and critic attempt files'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Run continuously, checking every interval seconds'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=600,
        help='Monitoring interval in seconds (default: 600 = 10 minutes)'
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Directory not found: {args.output_dir}")
        return 1
    
    if not args.output_dir.is_dir():
        print(f"Error: Not a directory: {args.output_dir}")
        return 1
    
    if args.monitor:
        monitor_continuous(args.output_dir, args.interval)
    else:
        monitor_once(args.output_dir)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

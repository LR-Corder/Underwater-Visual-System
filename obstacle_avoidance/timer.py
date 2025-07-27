# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import time

# Global task timing dictionary
task_times = {}

def start(name: str) -> None:
    """Start timing for a task."""
    task_times[name] = time.time()

def end(name: str) -> None:
    """End timing for a task and print elapsed time."""
    if name not in task_times:
        print(f"Task {name} has not been started!")
        return
    start_time = task_times.pop(name)
    elapsed_time = time.time() - start_time
    print(f"{name} elapsed time: {elapsed_time:.6f} seconds")
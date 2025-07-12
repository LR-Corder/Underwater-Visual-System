import time
global task_times
task_times = {}
def start(name):
    """开始计时"""
    task_times[name] = time.time()
def end(name):
    """结束计时并打印"""
    if name not in task_times:
        print(f"任务 {name} 未开始计时！")
        return
    start_time = task_times.pop(name)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"{name} 消耗时间：{elapsed_time:.6f}秒")




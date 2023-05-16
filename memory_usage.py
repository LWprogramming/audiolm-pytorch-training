import psutil
import gpustat
import time

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    gpu_stats = gpustat.GPUStatCollection.new_query()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp} - CPU memory used: {mem_info.rss / (1024 * 1024)} MB")
    for gpu in gpu_stats:
        print(f"{timestamp} - GPU {gpu.index} memory used: {gpu.memory_used} MB, available: {gpu.memory_total - gpu.memory_used} MB")

# Call log_memory_usage() whenever needed
log_memory_usage()
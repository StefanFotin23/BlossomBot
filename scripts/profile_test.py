import subprocess
import time
import os
import sys

# Dynamically resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_LOG_PATH = os.path.join(SCRIPT_DIR, "stress_test_raw.log")
REPORT_PATH = os.path.join(ROOT_DIR, "profiling.txt")

def get_pod_metrics():
    try:
        res = subprocess.run(["kubectl", "top", "pods"], capture_output=True, text=True, check=True, timeout=5.0)
        lines = res.stdout.strip().split('\n')
        metrics = {}
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                pod_name = parts[0]
                cpu_str = parts[1]
                mem_str = parts[2]
                
                # Parse CPU
                if cpu_str.endswith('m'):
                    cpu = float(cpu_str[:-1])
                else:
                    try:
                        cpu = float(cpu_str) * 1000.0
                    except ValueError:
                        cpu = 0.0
                        
                # Parse Memory
                if mem_str.endswith('Mi'):
                    mem = float(mem_str[:-2])
                elif mem_str.endswith('Gi'):
                    mem = float(mem_str[:-2]) * 1024.0
                elif mem_str.endswith('Ki'):
                    mem = float(mem_str[:-2]) / 1024.0
                else:
                    try:
                        mem = float(mem_str)
                    except ValueError:
                        mem = 0.0
                        
                # Group by base service name
                service_key = None
                if "chat-app" in pod_name:
                    service_key = "chat-app"
                elif "qdrant" in pod_name:
                    service_key = "qdrant"
                elif "rabbitmq" in pod_name:
                    service_key = "rabbitmq"
                elif "redis" in pod_name:
                    service_key = "redis"
                    
                if service_key:
                    if service_key not in metrics:
                        metrics[service_key] = {"cpu": 0.0, "mem": 0.0}
                    metrics[service_key]["cpu"] += cpu
                    metrics[service_key]["mem"] += mem
        return metrics
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return {}

def main():
    print("🚀 Starting Profiling Runner...")
    
    # Start the stress-test process redirecting to file to prevent buffer lockup
    print("🔥 Spawning 'make stress-test' in background (redirected to log file)...")
    with open(RAW_LOG_PATH, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            ["make", "stress-test"],
            stdout=log_file,
            stderr=log_file,
            text=True,
            cwd=ROOT_DIR
        )
        
        samples = []
        start_time = time.time()
        
        # Poll metrics while the process is running
        while proc.poll() is None:
            metrics = get_pod_metrics()
            if metrics:
                samples.append((time.time() - start_time, metrics))
            time.sleep(1.0)
            
    # Read the output log
    with open(RAW_LOG_PATH, "r", encoding="utf-8") as log_file:
        stdout = log_file.read()
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"🏁 Stress test finished in {duration:.2f} seconds.")
    
    # Process stats
    services = ["chat-app", "qdrant", "rabbitmq", "redis"]
    stats = {s: {"cpu_samples": [], "mem_samples": []} for s in services}
    
    for t, sample in samples:
        for s in services:
            if s in sample:
                stats[s]["cpu_samples"].append(sample[s]["cpu"])
                stats[s]["mem_samples"].append(sample[s]["mem"])
                
    report_lines = []
    report_lines.append("=========================================================")
    report_lines.append("BLOSSOMBOT CONCURRENT STRESS TEST PROFILING REPORT")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Test Duration: {duration:.2f} seconds")
    report_lines.append(f"Total Metric Samples Collected: {len(samples)}")
    report_lines.append("=========================================================\n")
    
    report_lines.append("1. RESOURCE UTILIZATION STATISTICS (AGGREGATED BY SERVICE)")
    report_lines.append("-" * 50)
    for s in services:
        cpu_s = stats[s]["cpu_samples"]
        mem_s = stats[s]["mem_samples"]
        
        if cpu_s and mem_s:
            avg_cpu = sum(cpu_s) / len(cpu_s)
            peak_cpu = max(cpu_s)
            avg_mem = sum(mem_s) / len(mem_s)
            peak_mem = max(mem_s)
            
            report_lines.append(f"Service: {s}")
            report_lines.append(f"  * CPU (milli-cores): Average = {avg_cpu:.1f}m, Peak = {peak_cpu:.1f}m")
            report_lines.append(f"  * Memory (MiB):      Average = {avg_mem:.1f}Mi, Peak = {peak_mem:.1f}Mi")
        else:
            report_lines.append(f"Service: {s} - No metrics collected.")
        report_lines.append("")
        
    # Get RabbitMQ queue count
    print("🔍 Fetching final RabbitMQ queue count...")
    try:
        q_res = subprocess.run(
            ["kubectl", "exec", "deployment/rabbitmq", "--", "rabbitmqadmin", "list", "queues", "name", "messages"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0
        )
        report_lines.append("2. RABBITMQ QUEUE STATUS")
        report_lines.append("-" * 50)
        report_lines.append(q_res.stdout.strip())
        report_lines.append("")
    except Exception as e:
        report_lines.append("2. RABBITMQ QUEUE STATUS")
        report_lines.append("-" * 50)
        report_lines.append(f"Error fetching RabbitMQ queue count: {e}\n")
        
    # Add stress_test.py output highlight
    report_lines.append("3. STRESS TEST OUTPUT SUMMARY")
    report_lines.append("-" * 50)
    
    summary_lines = []
    for line in stdout.split('\n'):
        if "finished" in line.lower() or "completed" in line.lower() or "error" in line.lower() or "failed" in line.lower():
            summary_lines.append(line)
    if summary_lines:
        report_lines.extend(summary_lines)
    else:
        # Just write last 15 lines of stdout
        report_lines.extend(stdout.split('\n')[-15:])
        
    report_text = "\n".join(report_lines)
    
    # Save to profiling.txt in project root
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"🎉 Profiling report successfully written to: {REPORT_PATH}")

if __name__ == "__main__":
    main()

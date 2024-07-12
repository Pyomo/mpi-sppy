import os
import subprocess
from multiprocessing import Pool

def run_python_script(rho=0, numScen = 3, smooth_type=0, p=0.0, prox_linear = 0, log_dir="logs"):
    itermax = 100
    beta = 0.1
    timeout_seconds = 1000
    
    os.makedirs(log_dir, exist_ok=True)
    
    if smooth_type == 0:
        log_file = os.path.join(log_dir, f"log_sizes_{numScen}_rho_setter_{rho}_linearized_{prox_linear}.txt")
    elif smooth_type == 1:
        log_file = os.path.join(log_dir, f"log_sizes_{numScen}_rho_setter_{rho}_pvalue_{p}_linearized_{prox_linear}.txt")
    elif smooth_type == 2:
        log_file = os.path.join(log_dir, f"log_sizes_{numScen}_rho_setter_{rho}_pratio_{p}_linearized_{prox_linear}.txt")
    python_command = f"python sizes_ph.py {numScen} {itermax} {rho} {smooth_type} {p} {beta} {prox_linear}"
    with open(log_file, 'w') as f:
        print(f"Running {python_command}")
        try:
            result = subprocess.run(python_command, shell=True, check=True, stdout=f, stderr=subprocess.PIPE, timeout=timeout_seconds)
            print(f"Command executed successfully for {python_command}")
        except subprocess.TimeoutExpired:
            f.write(f"Timeout {timeout_seconds}s for {python_command}")
            print(f"\n Timeout {timeout_seconds}s for {python_command}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {python_command}: {e.stderr.decode('utf-8')}")

if __name__ == "__main__":
    numScen = 3
    rhos = [0.001,0.01,0.1]
    prox_linears = [0,1]
    log_directory = f"logs_sizes_{numScen}"  
    num_processes = 8
    smooth_paras = [(0, 0.0),(1,1e-7),(1,1e-6),(1,1e-5),(1,1e-4),(1,1e-3),
                    (2,0.001),(2,0.005),(2,0.01),(2,0.05),(2,0.1),(2,0.5)]
    # smooth_paras = [(0, 0.0),(1,0.1),(1,0.5),(1,1.0),(2,0.1),(2,0.05),(2,0.01)]

    with Pool(processes=num_processes) as pool:
        pool.starmap(run_python_script, [(rho, numScen, smooth_type, p, prox_linear, log_directory) 
                                         for rho in rhos 
                                         for prox_linear in prox_linears 
                                         for (smooth_type,p) in smooth_paras])
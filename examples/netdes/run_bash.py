import os
import subprocess
from multiprocessing import Pool
import numpy as np

def run_python_script(rho, smooth_type=0, p=0.0, instname = "network-10-20-L-01", log_dir="logs"):
    itermax = 500
    beta = 0.1
    
    os.makedirs(log_dir, exist_ok=True)
    
    if smooth_type == 0:
        log_file = os.path.join(log_dir, f"log_{instname}_rho_{rho}.txt")
    elif smooth_type == 1:
        log_file = os.path.join(log_dir, f"log_{instname}_rho_{rho}_pvalue_{p}.txt")
    elif smooth_type == 2:
        log_file = os.path.join(log_dir, f"log_{instname}_rho_{rho}_pratio_{p}.txt")
    python_command = f"python netdes_ph.py {instname} {itermax} {rho} {smooth_type} {p} {beta}"
    with open(log_file, 'w') as f:
        try:
            print(f"Running {python_command}")
            result = subprocess.run(python_command, shell=True, check=True, stdout=f, stderr=subprocess.PIPE)
            print(f"Command executed successfully for {python_command}. Log saved to {log_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {python_command}: {e.stderr.decode('utf-8')}")

if __name__ == "__main__":
    rhos = np.round(np.arange(1000,31000,1000),1)
    instname = "network-10-20-L-01"
    log_directory = f"logs_{instname}"  
    num_processes = 8
    smooth_paras = [(0, 0.0),(1,1.0),(1,5.0),(1,10.0),(1,50.0),(1,100.0),(1,500.0),(1,1000.0),
                    (2,0.0005),(2,0.001),(2,0.005),(2,0.01),(2,0.05),(2,0.1),(2,0.5)]

    with Pool(processes=num_processes) as pool:
        pool.starmap(run_python_script, [(rho, smooth_type, p, instname, log_directory) for rho in rhos for (smooth_type,p) in smooth_paras])
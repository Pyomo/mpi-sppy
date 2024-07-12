import os
import subprocess
from multiprocessing import Pool
import numpy as np


def run_python_script(rho, smooth_type=0, p=0.0, instname = "sslp_15_45_5", log_dir="logs"):
    bundles_per_rank = 0
    itermax = 500
    beta = 0.2
    timeout_seconds = 600
    
    os.makedirs(log_dir, exist_ok=True)
    
    if smooth_type == 0:
        log_file = os.path.join(log_dir, f"log_{instname}_rho_{rho}.txt")
    elif smooth_type == 1:
        log_file = os.path.join(log_dir, f"log_{instname}_rho_{rho}_pvalue_{p}.txt")
    elif smooth_type == 2:
        log_file = os.path.join(log_dir, f"log_{instname}_rho_{rho}_pratio_{p}.txt")
    python_command = f"python sslp_ph.py {instname} {bundles_per_rank} {itermax} {rho} {smooth_type} {p} {beta}"
    with open(log_file, 'w') as f:
        print(f"Running {python_command}")
        try:
            result = subprocess.run(python_command, shell=True, check=True, stdout=f, stderr=subprocess.PIPE, timeout=timeout_seconds)
            print(f"Command executed successfully for {python_command}. Log saved to {log_file}")
        except subprocess.TimeoutExpired:
            f.write(f"Timeout {timeout_seconds}s for {python_command}")
            print(f"\nTimeout {timeout_seconds}s for {python_command}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {python_command}: {e.stderr.decode('utf-8')}")


if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__)) + os.sep + "data"
    instnames = [name for name in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory, name))]
    rhos = np.round(np.hstack((np.arange(0.1,1,0.1),np.arange(1,16))),1)
    # instname = "sslp_15_45_5"
    # log_directory = f"logs_{instname}"
    log_directories = [f"logs_{instname}" for instname in instnames]
    ins_dir_pairs = list(zip(instnames,log_directories))
    num_processes = 8
    smooth_paras = [(0, 0.0),(1,0.001),(1,0.005),(1,0.01),(1,0.05),(1,0.1),(1,0.5),(1,1.0),
                    (2,0.1),(2,0.05),(2,0.01),(2,0.005),(2,0.001),(2,0.0005)]

    with Pool(processes=num_processes) as pool:
        pool.starmap(run_python_script, [(rho, smooth_type, p, instname, log_directory) 
                                         for rho in rhos 
                                         for (smooth_type,p) in smooth_paras
                                         for (instname, log_directory) in ins_dir_pairs])
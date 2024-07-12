import os
import subprocess
from multiprocessing import Pool
import numpy as np

def run_python_script(rho, smooth_type=0, p=0.0, use_int=1, log_dir="logs"):
    crops_multiplier = 1
    scen_count = 3
    itermax = 500
    beta = 0.2
    solver_name = 'xpress'
    
    os.makedirs(log_dir, exist_ok=True)
    
    if smooth_type == 0:
        log_file = os.path.join(log_dir, f"log_rho_{rho}.txt")
        python_command = f"python farmer_ph_multi.py {crops_multiplier} {scen_count} {use_int} {rho} {itermax} {solver_name}"
    elif smooth_type == 1:
        log_file = os.path.join(log_dir, f"log_rho_{rho}_pvalue_{p}.txt")
        python_command = f"python farmer_ph_multi.py {crops_multiplier} {scen_count} {use_int} {rho} {itermax} {smooth_type} {p} {beta} {solver_name}"
    elif smooth_type == 2:
        log_file = os.path.join(log_dir, f"log_rho_{rho}_pratio_{p}.txt")
        python_command = f"python farmer_ph_multi.py {crops_multiplier} {scen_count} {use_int} {rho} {itermax} {smooth_type} {p} {beta} {solver_name}"
    with open(log_file, 'w') as f:
        try:
            print(f"Running {python_command}")
            result = subprocess.run(python_command, shell=True, check=True, stdout=f, stderr=subprocess.PIPE)
            print(f"Command executed successfully for {python_command}. Log saved to {log_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {python_command}: {e.stderr.decode('utf-8')}")

if __name__ == "__main__":
    rhos = np.round(np.hstack((np.arange(0.1,1,0.1),np.arange(1,11))),1)
    log_int_paras = [(0,"logs"),(1,"logs_int")]
    num_processes = 8
    smooth_paras = [(0, 0.0),(1,0.001),(1,0.005),(1,0.01),(1,0.05),(1,0.1),
                    (2,0.1),(2,0.05),(2,0.01),(2,0.005),(2,0.001),(2,0.0005)]

    with Pool(processes=num_processes) as pool:
        pool.starmap(run_python_script, [(rho, smooth_type, p, use_int, log_directory) 
                                         for rho in rhos 
                                         for (smooth_type,p) in smooth_paras 
                                         for (use_int, log_directory) in log_int_paras])
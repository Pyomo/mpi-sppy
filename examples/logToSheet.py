import os
import re
import pandas as pd


def extract_obj(line):
    pattern = r"Final objective from XhatClosest: (-?[\d.]+|inf)"
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    else:
        pattern = r"Timeout"
        match = re.search(pattern, line)
        if match:
            return "Timeout"
        return None
    

def extract_iternumber(line):
    pattern = r"PH Iteration (\d+)"
    match = re.search(pattern, line)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_info_from_filename(filename):
    rho_match = re.search(r"rho_([\deE.+-]+\d)", filename)
    if rho_match:
        rho = float(rho_match.group(1))
    else:
        rho = None
    rho_match = re.search(r"rho_setter_([\deE.+-]+\d)", filename)
    if rho_match:
        rho = float(rho_match.group(1))
    
    pvalue_match = re.search(r"pvalue_([\deE.+-]+\d)", filename)
    pratio_match = re.search(r"pratio_([\deE.+-]+\d)", filename)
    if pvalue_match:
        pvalue = "pvalue " + pvalue_match.group(1)
    elif pratio_match:
        pvalue = "pratio " + pratio_match.group(1)
    else:
        pvalue = "No smoothing"
    
    lin_match = re.search(r"linearized_(\d+)", filename)
    if lin_match:
        prox_lin = int(lin_match.group(1))
    else:
        prox_lin = 0

    return rho, pvalue, prox_lin


def process_log_files(log_files):
    results_obj = {}
    results_iternum = {}
    for log_file in log_files:
        rho, pvalue, prox_lin = extract_info_from_filename(log_file)
        if prox_lin not in results_obj:
            results_obj[prox_lin] = {}
            results_obj[prox_lin][rho] = {}
        elif rho not in results_obj[prox_lin]:
            results_obj[prox_lin][rho] = {}
        if prox_lin not in results_iternum:
            results_iternum[prox_lin] = {}
            results_iternum[prox_lin][rho] = {}
        elif rho not in results_iternum[prox_lin]:
            results_iternum[prox_lin][rho] = {}
        if rho is not None:
            with open(log_file, 'r') as f:
                for line in f:
                    obj = extract_obj(line)
                    if obj is not None:
                        results_obj[prox_lin][rho][pvalue] = obj
                    iternum = extract_iternumber(line)
                    if iternum is not None:
                        results_iternum[prox_lin][rho][pvalue] = iternum
            if pvalue not in results_iternum[prox_lin][rho]:
                results_iternum[prox_lin][rho][pvalue] = 0
    return results_obj, results_iternum


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    # logs_dir = os.path.join(script_dir, "farmer", "logs")  
    # output_excel = os.path.join(script_dir, "farmer_real.xlsx") 

    # logs_dir = os.path.join(script_dir, "farmer", "logs_int")  
    # output_excel = os.path.join(script_dir, "farmer_int.xlsx") 

    # logs_dir = os.path.join(script_dir, "sslp", "logs_sslp_15_45_5")  
    # output_excel = os.path.join(script_dir, "sslp_15_45_5.xlsx")

    # logs_dir = os.path.join(script_dir, "netdes", "logs_network-10-20-L-01")  
    # output_excel = os.path.join(script_dir, "network-10-20-L-01.xlsx")

    logs_dir = os.path.join(script_dir, "sizes", "logs_sizes_3")  
    output_excel = os.path.join(script_dir, "logs_sizes_3.xlsx")

    # logs_dir = os.path.join(script_dir, "sizes", "logs_sizes_10")  
    # output_excel = os.path.join(script_dir, "logs_sizes_10.xlsx")
    
    os.makedirs(logs_dir, exist_ok=True)
    
    log_files = [os.path.join(logs_dir, file) for file in os.listdir(logs_dir) if file.endswith('.txt')]
    
    results_obj, results_iternum = process_log_files(log_files)

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        if 0 in results_obj:
            df_obj = pd.DataFrame(results_obj[0]).fillna('')
            df_obj = df_obj.transpose()
            df_obj.sort_index(inplace=True,axis=0)
            df_obj.sort_index(inplace=True,axis=1)
            df_obj.index.name = 'rho values'
            df_obj.to_excel(writer, sheet_name="prox_linear 0")
            if 0 in results_iternum:
                df_iternum = pd.DataFrame(results_iternum[0]).fillna('')
                df_iternum = df_iternum.transpose()
                df_iternum.sort_index(inplace=True,axis=0)
                df_iternum.sort_index(inplace=True,axis=1)
                df_iternum.index.name = 'rho values'
                df_iternum.to_excel(writer, sheet_name="prox_linear 0", startcol=df_obj.shape[1]+4)
        
        if 1 in results_obj:
            df_obj = pd.DataFrame(results_obj[1]).fillna('')
            df_obj = df_obj.transpose()
            df_obj.sort_index(inplace=True,axis=0)
            df_obj.sort_index(inplace=True,axis=1)
            df_obj.index.name = 'rho values'
            df_obj.to_excel(writer, sheet_name="prox_linear 1")
            if 1 in results_iternum:
                df_iternum = pd.DataFrame(results_iternum[1]).fillna('')
                df_iternum = df_iternum.transpose()
                df_iternum.sort_index(inplace=True,axis=0)
                df_iternum.sort_index(inplace=True,axis=1)
                df_iternum.index.name = 'rho values'
                df_iternum.to_excel(writer, sheet_name="prox_linear 1", startcol=df_obj.shape[1]+4)
    
    
    print(f"Results saved to {output_excel}")

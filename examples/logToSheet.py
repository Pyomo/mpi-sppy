import os
import re
import pandas as pd


def extract_obj(line):
    pattern = r"Final objective from XhatClosest: (-?[\d.]+)"
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    else:
        return None
    

def extract_iternumber(line):
    pattern = r"PH Iteration (\d+)"
    match = re.search(pattern, line)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_rho_pvalue(filename):
    rho_match = re.search(r"log_rho_(\d+)", filename)
    if rho_match:
        rho = int(rho_match.group(1))
    else:
        rho = None
    

    pvalue_match = re.search(r"pvalue_(\d+\.\d+)", filename)
    pratio_match = re.search(r"pratio_(\d+\.\d+)", filename)
    if pvalue_match:
        pvalue = "pvalue " + pvalue_match.group(1)
    elif pratio_match:
        pvalue = "pratio " + pratio_match.group(1)
    else:
        pvalue = "No smoothing"
    
    return rho, pvalue


def process_log_files(log_files):
    results_obj = {}
    results_iternum = {}
    for log_file in log_files:
        rho, pvalue = extract_rho_pvalue(log_file)
        if rho is not None and pvalue is not None:
            with open(log_file, 'r') as f:
                for line in f:
                    obj = extract_obj(line)
                    if obj is not None:
                        if rho not in results_obj:
                            results_obj[rho] = {}
                        results_obj[rho][pvalue] = obj
                    iternum = extract_iternumber(line)
                    if iternum is not None:
                        if rho not in results_iternum:
                            results_iternum[rho] = {}
                        results_iternum[rho][pvalue] = iternum
    return results_obj, results_iternum


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    # logs_dir = os.path.join(script_dir, "farmer", "logs")  
    # output_excel = os.path.join(script_dir, "farmer_real.xlsx") 

    # logs_dir = os.path.join(script_dir, "farmer", "logs_int")  
    # output_excel = os.path.join(script_dir, "farmer_int.xlsx") 

    logs_dir = os.path.join(script_dir, "sslp", "logs")  
    output_excel = os.path.join(script_dir, "sslp.xlsx")
    
    os.makedirs(logs_dir, exist_ok=True)
    
    log_files = [os.path.join(logs_dir, file) for file in os.listdir(logs_dir) if file.endswith('.txt')]
    
    results = process_log_files(log_files)
    
    df_obj = pd.DataFrame(results[0]).fillna('')
    df_obj = df_obj.transpose()
    df_obj.sort_index(inplace=True,axis=0)
    df_obj.sort_index(inplace=True,axis=1)
    df_obj.index.name = 'rho values'

    df_iternum = pd.DataFrame(results[1]).fillna('')
    df_iternum = df_iternum.transpose()
    df_iternum.sort_index(inplace=True,axis=0)
    df_iternum.sort_index(inplace=True,axis=1)
    df_iternum.index.name = 'rho values'

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df_obj.to_excel(writer)
        df_iternum.to_excel(writer, startcol=df_obj.shape[1]+4)
    
    print(f"Extracted numbers saved to {output_excel}")

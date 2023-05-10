# coding=gbk
# Time Created: 2023/5/9 11:19
# Author  : Lucid
# FileName: run_all_tasks.py.py
# Software: PyCharm
import os
import subprocess

def main():
    project_folders = ['prj_high_freq', 'prj_low_freq', 'prj_risk_parity', 'prj_T_basis']
    python_executable = "D:\\anaconda\\envs\\FICC_research\\python.exe"

    for folder in project_folders:
        script_path = os.path.join(folder, 'run_script.py')
        subprocess.run([python_executable, script_path])

if __name__ == "__main__":
    main()


import subprocess
from CS15_2_virtual.memotion import run_memotion
from CS15_2_virtual.mvsa import run_mvsa  
from CS15_2_virtual.mvsa_single import run_mvsa_single  
from CS15_2_virtual.mvsa_multi_adam import run_mvsa_adam
from CS15_2_virtual.mvsa_single_adam import run_mvsa_single_adam

def run_memotion_script():
    print("Running memotion_baseline_overallSentiment.py...")
    # subprocess.run(["python", "CS15_2_virtual/memotion.py"])
    run_memotion()

def run_mvsa_multi_script():
    print("Running MVSA_baseline_overallSentiment.py...")
    # subprocess.run(["python", "CS15_2_virtual/mvsa.py"])
    run_mvsa()
    run_mvsa_adam()

def run_mvsa_single_script():
    run_mvsa_single()
    run_mvsa_single_adam()

if __name__ == "__main__":
    run_memotion_script()
    run_mvsa_multi_script()
    run_mvsa_single_script()

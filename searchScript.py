from typing import List, Dict, Tuple
import subprocess

import numpy

import argparse
parser = argparse.ArgumentParser()

interpreter = "~/miniconda3/envs/lsjenv/bin/python"
run_file = "main.py"

seed = 12
dataset = 'cs'
cudaId = 0
noise = "pair"
ptb_rate = 0.4
tau = 0.1
n_neg = 50 

alpha_list = list([0.001, 0.03, 0.1, 0.3])
K_list = list([   25, 50, 5])
th_list = list([0.95, 0.9, 0.8,0.7])
p_threshold_list = list([0.4,0.5,0.6,0.7])



parser.add_argument('--out', default='./results/'+dataset+'_'+str(ptb_rate)+ '_' + str(noise)+'_out.txt', type=str)
args = parser.parse_args()


print(f"Interpreter: {interpreter}")
print(f"File: {run_file}")
                
for alpha in alpha_list:
    for K in K_list:
            for th in th_list:
                for p_threshold in p_threshold_list:
                        acc_list = list()
                        for split in range(0, 5):
                                sh_repr: str = f"python {run_file}  --split {split} --seed {seed} --dataset {dataset} --tau {tau}  --alpha {alpha} --K {K} --n_neg {n_neg} --th {th} --ptb_rate {ptb_rate} --p_threshold {p_threshold} --cudaId {cudaId} --noise {noise}"
                                sh: str = f"{interpreter} {run_file}  --split {split} --seed {seed}  --dataset {dataset} --tau {tau}  --alpha {alpha} --K {K} --n_neg {n_neg} --th {th} --ptb_rate {ptb_rate} --p_threshold {p_threshold} --cudaId {cudaId} --noise {noise}"

                                sh_output = subprocess.getoutput(sh)
                                sh_output = [elem for elem in sh_output.splitlines() if elem is not None and len(elem) > 0]
                                cnt_acc: float = float(sh_output[-1])
                                acc_list.append(cnt_acc)

                                msg = f"Test acc = {cnt_acc} , split = {split}, seed = {seed}, --tau {tau} --alpha {alpha} --K {K} --n_neg {n_neg} --th {th} --ptb_rate {ptb_rate} --p_threshold {p_threshold}"
                                print(msg)
                                with open(args.out, 'a') as f:
                                    f.write("\n" + msg)


                        accuracy = numpy.array(acc_list)

                        msg = f"mean = {accuracy.mean()}, std = {accuracy.std()}  --alpha {alpha} --K {K} --th {th}  --p_threshold {p_threshold} "
                        print(msg)
                        with open(args.out, 'a') as f:
                            f.write("\n" + msg)
                            f.write("\n")


from typing import List, Dict, Tuple
import subprocess

import numpy

import argparse
parser = argparse.ArgumentParser()

interpreter = "~/miniconda3/envs/lsjenv/bin/python"
run_file = "main.py"
seed = 12
dataset = 'citeseer'
cudaId = 0
tau = 0.1
n_neg = 50 

lableRate = 2
co_lambda = 0.1

noise = "pair"
ptb_rate = 0.2
K = 20 
num_warmup = 10 
alpha = 0.03
th = 0.8
p_threshold = 0.4



# noise = "uniform"
# ptb_rate = 0.2
# K = 20 
# num_warmup = 10 
# alpha = 0.03
# th = 0.7
# p_threshold = 0.6



# pair 0.3
# noise = "pair"
# ptb_rate = 0.3
# K = 50 
# alpha = 0.03
# th = 0.7
# p_threshold = 0.4



# un 0.3
# noise = "uniform"
# ptb_rate = 0.3
# num_warmup = 20 
# K = 50 # now
# alpha = 0.03
# th = 0.7
# p_threshold = 0.6

# un 0.4
# noise = "uniform"
# ptb_rate = 0.4
# num_warmup = 10 
# K = 50 # now
# alpha = 0.03
# th = 0.95
# p_threshold = 0.6

# pair 0.4
# noise = "pair"
# ptb_rate = 0.4
# num_warmup = 10 
# K = 50 # now
# alpha = 0.03
# th = 0.8
# p_threshold = 0.4


parser.add_argument('--out', default='./results/ablation/'+dataset+'_'+str(ptb_rate)+ '_' + str(noise)+'_out.txt', type=str)
args = parser.parse_args()

acc_list = list()

print(f"Interpreter: {interpreter}")
print(f"File: {run_file}")

for split in range(0, 10):
    print("split",split)
    sh_repr: str = f"python {run_file}  --split {split} --seed {seed} --dataset {dataset} --tau {tau}   --alpha {alpha} --K {K} --n_neg {n_neg} --th {th} --ptb_rate {ptb_rate} --p_threshold {p_threshold} --cudaId {cudaId} --noise {noise}"
    sh: str = f"{interpreter} {run_file}  --split {split} --lableRate {lableRate} --co_lambda {co_lambda}   --seed {seed} --num_warmup {num_warmup} --dataset {dataset}   --tau {tau}  --alpha {alpha} --K {K} --n_neg {n_neg} --th {th} --ptb_rate {ptb_rate} --p_threshold {p_threshold} --cudaId {cudaId} --noise {noise}"

    sh_output = subprocess.getoutput(sh)
    sh_output = [elem for elem in sh_output.splitlines() if elem is not None and len(elem) > 0]
    noise_list = sh_output[-1]
    cnt_acc: float = float(sh_output[-2])
    acc_list.append(cnt_acc)
    print("======noise_list=====",noise_list)
    msg = f"Test acc = {cnt_acc} , split = {split}, seed = {seed},--co_lambda {co_lambda}, --lableRate {lableRate}, --num_warmup {num_warmup},--dataset {dataset}, --tau {tau}, --alpha {alpha} --K {K} --n_neg {n_neg} --th {th} --ptb_rate {ptb_rate} --p_threshold {p_threshold} "
    print(msg)
    with open(args.out, 'a') as f:
        f.write("\n" + msg)


accuracy = numpy.array(acc_list)

msg = f"mean = {accuracy.mean()}, std = {accuracy.std()}  --alpha {alpha} --K {K} --th {th} --p_threshold {p_threshold}"
print(msg)
with open(args.out, 'a') as f:
    f.write("\n" + msg)
    f.write("\n")


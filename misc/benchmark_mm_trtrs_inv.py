"""Benchmarking matrix multiplication, back substitution and inverse
"""
import torch
import time
import random
import statistics

n = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = []#check whether NaN shows up
times_mm, times_trtrs, times_inv = [], [], []
with torch.no_grad():
    while min([len(times_mm), len(times_trtrs), len(times_inv)])<3:       
        A = torch.randn(n, n, device=device) + 10*torch.eye(n, device=device)
        b = torch.randn(n, n, device=device)        
        j = random.randint(0, 2)#call mm, trtrs, inverse in random order
        if j==0:            
            t0 = time.time()
            x = A.mm(b)
            results.append(x[0,0])
            times_mm.append(time.time() - t0)
        elif j==1:                 
            t0 = time.time()
            x = torch.trtrs(b, A)[0]#just take triangular part of A
            results.append(x[0,0])
            times_trtrs.append(time.time() - t0)
        else:
            t0 = time.time()
            x = torch.inverse(A)
            results.append(x[0,0])
            times_inv.append(time.time() - t0)
    
print('Median Time in ms:')        
print('Multiplication {}; BackSubstitution {}; Inversion {}'.format(
        1000*statistics.median(times_mm), 
        1000*statistics.median(times_trtrs), 
        1000*statistics.median(times_inv)))
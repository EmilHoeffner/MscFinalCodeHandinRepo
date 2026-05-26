
import os 

import matplotlib.pyplot as plt
import numpy as np
import math

def BM25(k1, b, q_ts, f_ts, l_d, l_avg, N, N_ts):
    relevance = 0

    for (q_t, f_t, N_t) in zip(q_ts, f_ts, N_ts):

        t1 = q_t 
        t2 = (f_t * (k1 + 1.0)) / (k1 * ((1.0-b) + b * (l_d / l_avg)) + f_t)
        t3 = math.log(N / N_t)

        rel = t1 + t2 + t3

        relevance += rel 

    return relevance

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    q_ts = [1]

    l_avg = 100
    l_d = 50

    b = 0.75

    k1_vals = [1, 2, 5, 10, 20, 50]
    colors = ["red", "green", "blue", "yellow", "black", "orange"]


    for (k1,color) in zip(k1_vals, colors):
        f_ts = list(np.arange(start = 1, stop = 21, step = 1))
        N = len(f_ts)
        N_t = N
        relevance_vals = [BM25(k1 = k1, b = b, q_ts = q_ts, f_ts = [f_t], l_d = l_d, l_avg = l_avg, N = 1, N_ts = [N_t]) for f_t in f_ts]

        desc = f"k1 = {k1}"
        plt.title(f"Illustration of effect of k1")
        plt.xlabel("f_t,c")
        plt.ylabel("Context Relevance")
        plt.plot(f_ts, relevance_vals, color = color, label = desc)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
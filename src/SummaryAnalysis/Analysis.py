
import sys 
from SummaryAnalyser import SummaryAnalyser

import time

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    analyser = SummaryAnalyser(data_set_iden)
    time_start = time.time()
    analyser.summary_model_analysis()
    analyser.bootstrap_analysis()
    time_end = time.time()

    time_total = (time_end - time_start) / 60.0

    print(f"Time was {time_total} minutes")


if __name__ == "__main__":
    main()
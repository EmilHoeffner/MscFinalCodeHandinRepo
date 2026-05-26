import sys 
from SummaryAnalyser import SummaryAnalyser

import time

def main():

    analyser = SummaryAnalyser("WikiMultiHop_Test")
    time_start = time.time()
    analyser.pruned_summary_analysis()
    time_end = time.time()

    time_total = (time_end - time_start) / 60.0

    print(f"Time was {time_total} minutes")


if __name__ == "__main__":
    main()
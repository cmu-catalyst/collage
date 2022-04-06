#!/bin/bash

python3 plots/plot_e2e_perf.py -hw rtx2070 -bs 1
python3 plots/plot_e2e_perf.py -hw v100 -bs 1
python3 plots/plot_e2e_perf.py -hw xeon -bs 1
#python3 plots/plot_e2e_perf.py -hw diff_batch_v100 -bs 1
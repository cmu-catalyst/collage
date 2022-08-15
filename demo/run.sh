CUDA_VISIBLE_DEVICES=1 python3 artifact_eval.py
python3 plots/plot_e2e_perf.py -hw rtx2070 -bs 1

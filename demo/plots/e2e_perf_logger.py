import os
import csv
import os.path

this_code_path = os.path.dirname(os.path.abspath(__file__))
EXP_RESULT_PATH = this_code_path
E2E_PERF_LOG_PATH = f"{EXP_RESULT_PATH}/e2e_perf.csv"
E2E_PERF_COLS = ["HW", "BatchSize", "Network", "Method", "Mean Perf", "Std Perf"]

class E2EPerfLogger:
    def __init__(self):
        pass

    def gen_dic_key(self, hw, batch_size, network, method):
        return (hw, batch_size, network, method)

    def read_dict_from_csv(self):
        key_val_dic = {}
        if os.path.isfile(E2E_PERF_LOG_PATH):
            with open(E2E_PERF_LOG_PATH, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    hw, batch_size, net, method, mean, std = row
                    key_val_dic[self.gen_dic_key(hw, batch_size, net, method)] = (mean, std)

        return key_val_dic

    def log_all_perf(self, memo_dic):
        perf_str_row = []
        for key, val in memo_dic.items():
            hw, batch_size, net, method = key
            mean, std = float(val[0]), float(val[1])
            perf_str_row.append(f"{hw},{batch_size},{net},{method},{mean:.4f},{std:.4f}\n")

        perf_str_row.sort()

        with open(E2E_PERF_LOG_PATH, 'w') as e2e_log:
            for row in perf_str_row:
                e2e_log.write(row)

    def log_perf(self, hw, batch_size, network, method, mean_perf, std_perf):
        memo_dic = self.read_dict_from_csv()
        key = self.gen_dic_key(hw, batch_size, network, method)

        # Update performance if it already exists
        memo_dic[key] = (mean_perf, std_perf)
        self.log_all_perf(memo_dic)

DP_TUNING_TIME_LOG_PATH = f"{EXP_RESULT_PATH}/dp_tuning_time.csv"
DP_TUNING_TIME_COLS = ["HW", "Network", "Method", "Mean Perf", "Std Perf"]

class DPTuningTimeLogger:
    def __init__(self):
        pass

    def gen_dic_key(self, hw, network, method):
        return (hw, network, method)

    def read_dict_from_csv(self):
        key_val_dic = {}
        if os.path.isfile(DP_TUNING_TIME_LOG_PATH):
            with open(DP_TUNING_TIME_LOG_PATH, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    hw, net, method, mean, std = row
                    key_val_dic[self.gen_dic_key(hw, net, method)] = (mean, std)

        return key_val_dic

    def log_all_perf(self, memo_dic):
        perf_str_row = []
        for key, val in memo_dic.items():
            hw, net, method = key
            mean, std = float(val[0]), float(val[1])
            perf_str_row.append(f"{hw},{net},{method},{mean:.4f},{std:.4f}\n")

        perf_str_row.sort()

        with open(DP_TUNING_TIME_LOG_PATH, 'w') as e2e_log:
            for row in perf_str_row:
                e2e_log.write(row)

    def log_perf(self, hw, network, method, mean_perf, std_perf):
        memo_dic = self.read_dict_from_csv()
        key = self.gen_dic_key(hw, network, method)

        # Update performance if it already exists
        memo_dic[key] = (mean_perf, std_perf)
        self.log_all_perf(memo_dic)
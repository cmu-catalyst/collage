import matplotlib.pyplot as plt
import pandas as pd
from e2e_perf_logger import *
from scipy import stats
import argparse
import numpy as np

# Avoid display of figure
import matplotlib
matplotlib.use('Agg')

NET_NAME_TO_OFFICIAL = {'bert_full': 'BERT', 'nasneta':"NasNet-A", 'resnet50': 'ResNet50',
                      'resnext50_32x4d':"ResNeXt50", 'resnet50_3d':"3D-ResNet50",
                      'mobilenet_v2':"Mobilenet V2", 'nasrnn':"NasRNN", 'dcgan':'DCGAN'}

FINAL_NETWORKS = ["bert_full", "dcgan", "nasneta", 'resnet50_3d', "resnext50_32x4d"]
NVIDIA_GPUS = ['rtx2070', 'rtx3070', 'jetson', 'v100']
INTEL_CPUS = ['xeon']

METHOD_TO_COLOR = {
    'Collage': "C0", # blue, my favorite
    "TF":"C1", # orange
    "TF-XLA":"C5", # brown
    "PyTorch":"C3", # is red
    "TensorRT":"C2", # is close to green
    "TVM":"C4", # is purple (instead of blue)
    # C6 is pink
}

def set_plt_font_size():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.style.use('seaborn-paper')
    plt.rc('axes', axisbelow=True)
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes', linewidth=2)
    plt.rc('lines', linewidth=3)

def setup_df_with_baselines_and_method(df, methods):
    # Make sure column order is following before we normalize perf
    df = df[methods]

    df = df.rename(columns={'AutoTVM-libs': 'TVM', 'DP':'Op-level'})

    df = df.loc[FINAL_NETWORKS]
    df = df.rename(index=NET_NAME_TO_OFFICIAL)
    # df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])#, '3D-ResNet50', 'DCGAN', 'NasNet-A', 'ResNeXt50'])

    # print(df)
    return df

def setup_df_with_baselines_and_method_diff_batch(df, methods):
    # Make sure column order is following before we normalize perf
    df = df[methods]

    df = df.rename(columns={'AutoTVM-libs': 'TVM', 'DP':'Op-level'})

    df = df.loc[[1,4,8,16]]
    BATCH_SIZE_TO_TEXT = {
        1: "BatchSize=1",
        4: "BatchSize=4",
        8: "BatchSize=8",
        16: "BatchSize=16",
    }
    df = df.rename(index=BATCH_SIZE_TO_TEXT)
    # df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])#, '3D-ResNet50', 'DCGAN', 'NasNet-A', 'ResNeXt50'])

    # print(df)
    return df

def setup_df_for_normalized_perf_plot_diff_batch(df):
    # Normalize the performance
    for method in df:
        # print(method)
        df[method] = df['Two-level'] / df[method]

    # print(df.iloc[0:5, :])

    # Add Geomean
    # df.loc['GeoMean'] = stats.gmean(df.iloc[0:5, :], axis=0)

    # Correct Geomean of TF-XLA to deal with missing values
    xla_perf = [df.loc['BatchSize=1', 'TF-XLA'], df.loc['BatchSize=4', 'TF-XLA'],
                df.loc['BatchSize=8', 'TF-XLA'], df.loc['BatchSize=16', 'TF-XLA']]
    # df.at['GeoMean', 'TF-XLA'] = stats.gmean(xla_perf, axis=0)
    # print(df)

    # Without DP
    df = df.rename(columns={'Two-level': 'Collage'})
    df = df.drop(columns=['Op-level'])

    return df

def setup_df_for_normalized_perf_plot(df):
    # Normalize the performance
    for method in df:
        # print(method)
        df[method] = df['Two-level'] / df[method]

    # print(df.iloc[0:5, :])

    # Add Geomean
    df.loc['GeoMean'] = stats.gmean(df.iloc[0:5, :], axis=0)

    # Correct Geomean of TF-XLA to deal with missing values
    xla_perf = [df.loc['BERT', 'TF-XLA'], df.loc['ResNeXt50', 'TF-XLA'], df.loc['NasNet-A', 'TF-XLA']]
    df.at['GeoMean', 'TF-XLA'] = stats.gmean(xla_perf, axis=0)
    # print(df)

    # Without DP
    df = df.rename(columns={'Two-level': 'Collage'})
    df = df.drop(columns=['Op-level'])

    return df

# Shift the bars if there is a space between two bars because of NaN value in the middle.
def draw_plot_without_nan_values(df, is_diff_batch=False):
    if is_diff_batch:
        fig = plt.figure(figsize=(16, 5))
    else:
        fig = plt.figure(figsize=(21, 3.5))

    ax = plt.gca()

    # Have a dataframe (row: method, col: network)
    df = df.T
    # Sort data frame by geomean
    if is_diff_batch:
        df = df.sort_values(by='BatchSize=1')
    else:
        df = df.sort_values(by='GeoMean')
    #print(df)

    shifted = df.notnull().cumsum()
    # width of each bar
    width = 1 / len(df.columns) * 0.8
    if is_diff_batch:
        width*=0.7


    n_methods = df.shape[0]
    n_networks = df.shape[1]
    pos_of_method_in_each_net = np.arange(n_networks) # df.shape[1] is # of networks
    colors = [METHOD_TO_COLOR[method] for method in df.index]

    # Offset
    n_method_per_networks = shifted.loc['Collage'].to_numpy()
    net_offset = (n_methods - n_method_per_networks) * width/2.0

    for i, method in enumerate(df.index):
        offsets = shifted.loc[method]
        values = df.loc[method]
        # print(pos_of_method_in_each_net + width*offsets)
        ax.bar(pos_of_method_in_each_net + width*offsets-width*(n_methods+1)/2.0 + net_offset, values,
               color=colors[i], width=width, label=method)
    ax.set_xticks(pos_of_method_in_each_net)
    ax.set_xticklabels(df.columns)
    ax.legend()

def draw_e2e_perf_plot_normalized(df, args, is_diff_batch=False):
    draw_plot_without_nan_values(df, is_diff_batch)

    # df = df.fillna()
    # df.plot.bar(figsize=(24, 5), width=0.7)

    # Save figures
    plt.xlabel("")
    # plt.ylabel('Normalized Performance')
    plt.ylabel('Normalized Throughput')
    # plt.ylabel('Inference Time (ms)')

    plt.grid(axis='y', zorder=-2.0)
    plt.xticks(rotation=0)
    plt.yticks(np.arange(0,1.01,0.2))

    box_y_pos = 1.26 if not is_diff_batch else 1.2
    plt.legend(ncol=args.n_method, loc='upper center', bbox_to_anchor=(0.48, box_y_pos), handletextpad=0.3, borderpad=0.3, labelspacing=0.15)
    plt.savefig(f"{EXP_RESULT_PATH}/e2e_perf_norm_{args.hw}_{args.batch_size}.png", bbox_inches='tight')

def update_rtx2070_number(df):
    indices = ['bert_full', 'dcgan', 'nasneta', 'resnet50_3d', 'resnext50_32x4d']
    with open(f'{EXP_RESULT_PATH}/temp/e2e_perf_two_level.log') as f:
        for idx, line in enumerate(f.readlines()):
            mean_perf = float(line.split(",")[0])
            # print(df['Two-level'][idx], mean_perf)
            df.at[indices[idx], 'Two-level'] = mean_perf
    return df

if __name__ == "__main__":
    set_plt_font_size()
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    if args.hw!='diff_batch_v100':
        plt.rc('axes', labelsize=18)

    #print(args)
    if args.hw in NVIDIA_GPUS or args.hw in INTEL_CPUS:
        df = pd.read_csv(E2E_PERF_LOG_PATH, header=None)
        df.columns = E2E_PERF_COLS
        df = df[(df['HW'] == args.hw) & (df['BatchSize'] == args.batch_size)]
        df = df.drop(columns=['HW', 'BatchSize', 'Std Perf'])
        df = df.set_index('Network')
        df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')

        if args.hw in NVIDIA_GPUS:
            # methods = ['cuDNN', 'AutoTVM', 'TensorRT', 'TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']
            methods = ['TensorRT', 'TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']
        elif args.hw in INTEL_CPUS:
            # methods = ['AutoTVM', 'DNNL', 'TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']
            methods = ['TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']
        else:
            raise Exception(f"{args.hw} is unexpected hw, we need to set default backends for this hw.")

        args.n_method = len(methods)

        # Update numbers for RTX2070 with newly measured number
        update_rtx2070_number(df)

        df = setup_df_with_baselines_and_method(df, methods)
        df = setup_df_for_normalized_perf_plot(df)
        draw_e2e_perf_plot_normalized(df, args)

        final_path = f"{EXP_RESULT_PATH}/e2e_perf_norm_{args.hw}_{args.batch_size}.png"
        print(f"The plot (e2e perf) is saved to \'{final_path}\'")

    elif args.hw == 'diff_batch_v100':
        df = pd.read_csv(E2E_PERF_LOG_PATH, header=None)
        df.columns = E2E_PERF_COLS
        df = df[(df['HW'] == "v100") & (df['Network'] == "resnext50_32x4d")]
        df = df.drop(columns=['HW', 'Network', 'Std Perf'])
        df = df.set_index('BatchSize')
        df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')

        # methods = ['cuDNN', 'AutoTVM', 'TensorRT', 'TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']
        methods = ['TensorRT', 'TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']
        args.n_method = len(methods)

        df = setup_df_with_baselines_and_method_diff_batch(df, methods)
        df = setup_df_for_normalized_perf_plot_diff_batch(df)
        draw_e2e_perf_plot_normalized(df, args, is_diff_batch=True)
        # draw_e2e_perf_plot_normalized(df, args)
    else:
        raise Exception(f"{args.hw} is unexpected hw, we need to set default backends for this hw.")

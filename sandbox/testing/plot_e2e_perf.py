import matplotlib.pyplot as plt
import pandas as pd
from e2e_perf_logger import *
from tvm.relay.transform.utility.plot_utils import set_plt_font_size

NET_NAME_TO_OFFICIAL = {'bert': 'BERT', 'nasneta':"NasNet-A", 'resnet50': 'ResNet50',
                      'resnext50_32x4d':"ResNeXt50", 'resnet50_3d':"3D-ResNet50",
                      'mobilenet_v2':"Mobilenet V2", 'nasrnn':"NasRNN", 'dcgan':'DCGAN'}

FINAL_NETWORKS = ["resnext50_32x4d", "bert", "dcgan", "nasneta", 'resnet50_3d']

def setup_df_for_perf_plot(df):
    # Make sure column order is following before we normalize perf
    df = df[['PyTorch', 'cuDNN', 'AutoTVM', 'TensorRT', 'DP', 'Two-level']]
    df = df.drop(columns=['DP'])
    df = df.rename(columns={'Two-level': 'Collage'})
    df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])

    # Normalize the performance
    for method in df:
        print(method)
        df[method] = df['Collage'] / df[method]

    return df

def draw_e2e_perf_plot(df):
    df.plot.bar(figsize=(24, 5), width=0.7)

    # x_label_invisible = False
    #
    # if x_label_invisible:
    #     ax1 = plt.axes()
    #     x_axis = ax1.axes.get_xaxis()
    #     x_axis.set_visible(False)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Normalized Performance')
    # plt.ylabel('Inference Time (ms)')

    plt.xticks(rotation=0)
    plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.savefig(f"{EXP_RESULT_PATH}/plots/e2e_perf.png", bbox_inches='tight')

def setup_df_for_perf_inf_engine_vs_op_lib_plot(df):
    # [Temporary] Drop two-level and DP perf
    print(df)
    df = df.drop(columns=['Two-level', 'DP'])

    # [Figure 1] Drop indices other than Mobilenet V2 (AutoTVM best), ResNext(AutoTVM best), and ResNet-50 (TensorRT best)
    df = df.drop(['BERT', 'NasNet-A', '3D-ResNet50', 'NasRNN'])

    df = df[['cuDNN', 'AutoTVM', 'TensorRT']]
    df = df.reindex(index = ['ResNet50','ResNeXT50','Mobilenet V2'])
    for method in df:
        print(method)
        df[method] = df['TensorRT'] / df[method]

    return df

def draw_e2e_perf_inf_engine_vs_op_lib_plot(df):
    df.plot.bar(figsize=(12, 5), width=0.7)

    # x_label_invisible = False
    #
    # if x_label_invisible:
    #     ax1 = plt.axes()
    #     x_axis = ax1.axes.get_xaxis()
    #     x_axis.set_visible(False)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Normalized Performance')
    # plt.ylabel('Inference Time (ms)')

    plt.xticks(rotation=0)
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.savefig(f"{EXP_RESULT_PATH}/plots/e2e_perf_inf_eng_vs_op_lib.png", bbox_inches='tight')


if __name__ == "__main__":
    set_plt_font_size()

    df = pd.read_csv(E2E_PERF_LOG_PATH, header=None)
    df.columns = E2E_PERF_COLS
    df = df.drop(columns=['HW', 'Std Perf'])
    df = df.set_index('Network')
    df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')
    df = df.rename(index=NET_NAME_TO_OFFICIAL)

    # df = setup_df_for_perf_inf_engine_vs_op_lib_plot(df)
    # print(df)
    # draw_e2e_perf_inf_engine_vs_op_lib_plot(df)

    df = setup_df_for_perf_plot(df)
    print(df)
    draw_e2e_perf_plot(df)



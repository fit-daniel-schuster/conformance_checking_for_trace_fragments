import pandas as pd
import os
import math
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

ALGORITHM_BASELINE_DIJKSTRA = 'BASELINE_DIJKSTRA'
ALGORITHM_BASELINE_A_STAR = 'BASELINE_A_STAR'
ALGORITHM_BASELINE_DIJKSTRA_NOT_NAIVE = 'BASELINE_DIJKSTRA_NOT_NAIVE'
ALGORITHM_BASELINE_A_STAR_NOT_NAIVE = 'BASELINE_A_STAR_NOT_NAIVE'
ALGORITHM_TP_DIJKSTRA_ADVANCED = 'TP_DIJKSTRA_NOT_NAIVE'
ALGORITHM_TP_A_STAR_ADVANCED = 'TP_A_STAR_NOT_NAIVE'

ALGORITHMS_DIJKSTRA = {ALGORITHM_TP_DIJKSTRA_ADVANCED, ALGORITHM_BASELINE_DIJKSTRA_NOT_NAIVE,
                       ALGORITHM_BASELINE_DIJKSTRA}
ALGORITHMS_A_STAR = {ALGORITHM_TP_A_STAR_ADVANCED, ALGORITHM_BASELINE_A_STAR_NOT_NAIVE, ALGORITHM_BASELINE_A_STAR}

BASELINE_LABEL = 'Baseline'
BASELINE_NOT_NAIVE_LABEL = 'Baseline + subsequent filtering'
ADVANCED_LABEL = 'Advanced'

DIJKSTRA_RENAMINGS = {
    ALGORITHM_BASELINE_DIJKSTRA: BASELINE_LABEL,
    ALGORITHM_BASELINE_DIJKSTRA_NOT_NAIVE: BASELINE_NOT_NAIVE_LABEL,
    ALGORITHM_TP_DIJKSTRA_ADVANCED: ADVANCED_LABEL,
}

A_STAR_RENAMINGS = {
    ALGORITHM_BASELINE_A_STAR: BASELINE_LABEL,
    ALGORITHM_BASELINE_A_STAR_NOT_NAIVE: BASELINE_NOT_NAIVE_LABEL,
    ALGORITHM_TP_A_STAR_ADVANCED: ADVANCED_LABEL,
}

plot_configs = [('DIJKSTRA', ALGORITHMS_DIJKSTRA, DIJKSTRA_RENAMINGS, 3),
                ('A_STAR', ALGORITHMS_A_STAR, A_STAR_RENAMINGS, 3)]

sns_palette = sns.color_palette()
palette = {BASELINE_LABEL: sns_palette[0], BASELINE_NOT_NAIVE_LABEL: sns_palette[1], ADVANCED_LABEL: sns_palette[2]}

sns.set_theme(style="whitegrid", font_scale=0.92)


def remove_longest_n_percent_of_infixes(df, n):
    long_df = df[['Infix', 'Infix Length']].groupby(by=['Infix']).max()
    n_infixes_to_remove = math.ceil(len(long_df) / 100) * n
    infixes_to_remove = set()

    for i in range(n_infixes_to_remove):
        max_idx = long_df['Infix Length'].idxmax()
        infixes_to_remove.add(max_idx)
        long_df = long_df.drop(index=max_idx)

    results_df = df[~df['Infix'].isin(infixes_to_remove)]

    return results_df


def __create_plots(filename: str, directory: str, infix_type: str, results_df: pd.DataFrame):
    for name, algorithms, renaming, n_cols in plot_configs:
        Path(os.path.join(directory, 'plots', filename, infix_type, name)).mkdir(parents=True, exist_ok=True)

        for with_outliers in [True, False]:
            d = results_df.copy()
            d = results_df[results_df['Algorithm'].isin(algorithms)]

            d['Infix Length Bin'] = pd.cut(d['Infix Length'], bins=6)
            d['Infix Length Bin Formatted'] = d['Infix Length Bin'].apply(
                lambda d: str(math.floor(d.left) + 1) + '-' + str(math.floor(d.right)))
            for curr_name, renamed_name in renaming.items():
                d.loc[d['Algorithm'] == curr_name, 'Algorithm'] = renamed_name

            attributes_with_title = [('Consumed Time', 'Consumed Time (in seconds)', True),
                                     ('Preprocessing Duration', 'Preprocessing Duration (in seconds)', True),
                                     ('Alignment Duration', 'Alignment Duration (in seconds)', True),
                                     ('Visited States', 'Visited States', True),
                                     ('Queued States', 'Queued States', True),
                                     ('Added Tau Transitions', 'Relevant Markings', True),
                                     ]

            for attribute, y_label, should_save in attributes_with_title:
                fig, ax = plt.subplots()
                fig.set_size_inches(6.4, 3.8)
                d_cpy = d.copy()
                if attribute == 'Preprocessing Duration' or attribute == 'Added Tau Transitions':
                    d_cpy = d_cpy[d_cpy['Algorithm'] != BASELINE_LABEL]
                    baseline_mean = d[d['Algorithm'] == BASELINE_LABEL][attribute].median()
                    x_values = d['Infix Length Bin Formatted'].unique()
                    y_values = [baseline_mean for i in range(len(x_values))]
                    sns.lineplot(y=y_values, x=x_values, ax=ax, label=BASELINE_LABEL, palette=palette)
                sns.boxplot(data=d_cpy, ax=ax, x='Infix Length Bin Formatted', y=attribute, hue='Algorithm',
                            palette=palette, showfliers=with_outliers,
                            hue_order=[BASELINE_LABEL, BASELINE_NOT_NAIVE_LABEL, ADVANCED_LABEL])
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [labels.index(BASELINE_LABEL), labels.index(BASELINE_NOT_NAIVE_LABEL),
                         labels.index(ADVANCED_LABEL)]
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                           bbox_to_anchor=(0, 1.1), loc=2, borderaxespad=0., ncol=3)
                if infix_type == 'infix':
                    plt.xlabel('Infix Length')
                else:
                    plt.xlabel('Postfix Length')

                plt.ylabel(y_label)
                if should_save:
                    outlier_extension = ''
                    if not with_outliers:
                        outlier_extension = ' without outliers'
                    plt.savefig(os.path.join(directory, 'plots', filename, infix_type, name,
                                             attribute + outlier_extension + '.pdf'),
                                bbox_inches='tight')
                plt.close()


def create_plots(filename: str, directory: str, infix_type: str):
    Path(os.path.join(directory, 'plots', filename)).mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(os.path.join(directory, filename + f'_{infix_type}_results.csv'))
    results_df = remove_longest_n_percent_of_infixes(results_df, 2)
    __create_plots(filename, directory, infix_type, results_df)

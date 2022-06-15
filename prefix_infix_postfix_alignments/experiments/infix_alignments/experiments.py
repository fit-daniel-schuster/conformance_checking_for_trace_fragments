import os
import random
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import pickle

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import Trace, Event, EventLog
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.statistics.variants.log.get import get_variants
from pm4py.util.variants_util import variant_to_trace
from pm4py.objects.process_tree.importer import importer as ptml_importer
from pm4py.objects.process_tree.exporter import exporter as ptml_exporter

from prefix_infix_postfix_alignments.alignments.infix_alignments import algorithm as infix_alignments
from prefix_infix_postfix_alignments.alignments.postfix_alignments import algorithm as suffix_alignments
from prefix_infix_postfix_alignments.alignments.infix_alignments.algorithm import VARIANT_BASELINE_APPROACH, \
    VARIANT_TREE_BASED_PREPROCESSING
from prefix_infix_postfix_alignments.alignments.postfix_alignments.algorithm import \
    VARIANT_BASELINE_APPROACH as SUFFIX_BASELINE_APPROACH, \
    VARIANT_TREE_BASED_PREPROCESSING as SUFFIX_TREE_BASED_APPROACH
from prefix_infix_postfix_alignments.experiments.infix_alignments import create_plots

COLUMNS = ['Infix', 'Infix Length', 'Algorithm', 'Consumed Time', 'Visited States', 'Queued States', 'Cost',
           'Preprocessing Duration', 'Alignment Duration', 'Timeout', 'Added Tau Transitions', 'Alignment', 'LP Solved']
ALGORITHM_BASELINE_DIJKSTRA = 'BASELINE_DIJKSTRA'
ALGORITHM_BASELINE_A_STAR = 'BASELINE_A_STAR'
ALGORITHM_BASELINE_DIJKSTRA_NOT_NAIVE = 'BASELINE_DIJKSTRA_NOT_NAIVE'
ALGORITHM_BASELINE_A_STAR_NOT_NAIVE = 'BASELINE_A_STAR_NOT_NAIVE'
ALGORITHM_INFIX_ALIGNMENTS_DIJKSTRA_NOT_NAIVE = 'TP_DIJKSTRA_NOT_NAIVE'
ALGORITHM_INFIX_ALIGNMENTS_A_STAR_NOT_NAIVE = 'TP_A_STAR_NOT_NAIVE'
PROCESS_TREE_EXTENSION = '.ptml'
MODE_FILE = 'file'
MODE_DIRECTORY = 'dir'
INFIX_TYPE_INFIX = 'infix'
INFIX_TYPE_POSTFIX = 'postfix'

NOISE_THRESHOLD = float(os.getenv('NOISE_THRESHOLD', '0.9'))
TIMEOUT = int(os.getenv('TIMEOUT', '10'))
RANDOM_SAMPLE_SIZE = int(os.getenv('RANDOM_SAMPLE_SIZE', '10000'))
DATA_PATH = os.getenv('DATA_PATH', '../event_logs/')
DATA_FILENAME = os.getenv('DATA_FILENAME', 'RoadTrafficFineManagement')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
MODE = os.getenv('MODE', MODE_FILE)
INFIX_TYPE = os.getenv('INFIX_TYPE', INFIX_TYPE_INFIX)

processed_infixes = 0
number_infixes = 0

algorithm_variants_infix = {
    ALGORITHM_BASELINE_DIJKSTRA: lambda trace, tree: infix_alignments.calculate_optimal_infix_alignment(trace, tree,
                                                                                                        use_dijkstra=True,
                                                                                                        naive=True,
                                                                                                        timeout=TIMEOUT,
                                                                                                        variant=VARIANT_BASELINE_APPROACH),
    ALGORITHM_BASELINE_A_STAR: lambda trace, tree: infix_alignments.calculate_optimal_infix_alignment(trace, tree,
                                                                                                      use_dijkstra=False,
                                                                                                      naive=True,
                                                                                                      timeout=TIMEOUT,
                                                                                                      variant=VARIANT_BASELINE_APPROACH),
    ALGORITHM_BASELINE_DIJKSTRA_NOT_NAIVE: lambda trace, tree: infix_alignments.calculate_optimal_infix_alignment(trace,
                                                                                                                  tree,
                                                                                                                  use_dijkstra=True,
                                                                                                                  naive=False,
                                                                                                                  timeout=TIMEOUT,
                                                                                                                  variant=VARIANT_BASELINE_APPROACH),
    ALGORITHM_BASELINE_A_STAR_NOT_NAIVE: lambda trace, tree: infix_alignments.calculate_optimal_infix_alignment(trace,
                                                                                                                tree,
                                                                                                                use_dijkstra=False,
                                                                                                                timeout=TIMEOUT,
                                                                                                                naive=False,
                                                                                                                variant=VARIANT_BASELINE_APPROACH),
    ALGORITHM_INFIX_ALIGNMENTS_DIJKSTRA_NOT_NAIVE: lambda trace,
                                                          tree: infix_alignments.calculate_optimal_infix_alignment(
        trace, tree, naive=False, use_dijkstra=True, timeout=TIMEOUT, variant=VARIANT_TREE_BASED_PREPROCESSING),
    ALGORITHM_INFIX_ALIGNMENTS_A_STAR_NOT_NAIVE: lambda trace, tree: infix_alignments.calculate_optimal_infix_alignment(
        trace, tree, naive=False, use_dijkstra=False, timeout=TIMEOUT, variant=VARIANT_TREE_BASED_PREPROCESSING),
}

algorithm_variants_postfix = {
    ALGORITHM_BASELINE_DIJKSTRA_NOT_NAIVE: lambda trace, tree: suffix_alignments.calculate_optimal_suffix_alignment(
        trace, tree,
        use_dijkstra=True,
        naive=False,
        timeout=TIMEOUT,
        variant=SUFFIX_BASELINE_APPROACH),
    ALGORITHM_BASELINE_A_STAR_NOT_NAIVE: lambda trace, tree: suffix_alignments.calculate_optimal_suffix_alignment(trace,
                                                                                                                  tree,
                                                                                                                  naive=False,
                                                                                                                  use_dijkstra=False,
                                                                                                                  timeout=TIMEOUT,
                                                                                                                  variant=SUFFIX_BASELINE_APPROACH),
    ALGORITHM_BASELINE_DIJKSTRA: lambda trace, tree: suffix_alignments.calculate_optimal_suffix_alignment(trace, tree,
                                                                                                          use_dijkstra=True,
                                                                                                          naive=True,
                                                                                                          timeout=TIMEOUT,
                                                                                                          variant=SUFFIX_BASELINE_APPROACH),
    ALGORITHM_BASELINE_A_STAR: lambda trace, tree: suffix_alignments.calculate_optimal_suffix_alignment(trace, tree,
                                                                                                        use_dijkstra=False,
                                                                                                        naive=True,
                                                                                                        timeout=TIMEOUT,
                                                                                                        variant=SUFFIX_BASELINE_APPROACH),
    ALGORITHM_INFIX_ALIGNMENTS_DIJKSTRA_NOT_NAIVE: lambda trace,
                                                          tree: suffix_alignments.calculate_optimal_suffix_alignment(
        trace, tree, naive=False, use_dijkstra=True, timeout=TIMEOUT, variant=SUFFIX_TREE_BASED_APPROACH),
    ALGORITHM_INFIX_ALIGNMENTS_A_STAR_NOT_NAIVE: lambda trace,
                                                        tree: suffix_alignments.calculate_optimal_suffix_alignment(
        trace, tree, naive=False, use_dijkstra=False, timeout=TIMEOUT, variant=SUFFIX_TREE_BASED_APPROACH),
}


def run_infix_alignment_experiments():
    print('Mode:', MODE)
    print('Infix Type:', INFIX_TYPE)
    if MODE == MODE_FILE:
        filenames = [os.path.join(DATA_PATH, DATA_FILENAME)]
    else:
        filenames = glob.glob(os.path.join(DATA_PATH, '*.xes'))
        filenames = [os.path.splitext(f)[0] for f in filenames]

    print('Used event logs:', filenames)
    for filename in filenames:
        run_infix_alignment_experiments_for_file(filename)


def run_infix_alignment_experiments_for_file(log_filename_prefix):
    __print_parameters(log_filename_prefix)
    log = xes_importer.apply(log_filename_prefix + ".xes")
    tree = __get_model(log_filename_prefix, log)

    variants = get_variants(log)

    infixes = get_infixes(variants, log_filename_prefix)

    global processed_infixes, number_infixes
    processed_infixes = 0
    number_infixes = len(infixes)

    pool = Pool()
    processes = []
    results = []

    for infix in infixes:
        p = pool.apply_async(__run_experiments_for_infix, args=(infix, tree,),
                             callback=print_progress_on_console)
        processes.append(p)

    pool.close()
    pool.join()
    for p in processes:
        results = results + p.get()

    df = pd.DataFrame(results, columns=COLUMNS)
    filename = os.path.basename(log_filename_prefix) + f'_{INFIX_TYPE}_results.csv'
    df.to_csv(os.path.join(os.path.dirname(log_filename_prefix), filename))

    create_plots.create_plots(os.path.basename(log_filename_prefix), os.path.dirname(log_filename_prefix), INFIX_TYPE)


def get_infixes(variants, log_filename_prefix):
    infixes_filename = log_filename_prefix + '_infixes_' + str(RANDOM_SAMPLE_SIZE) + '.pkl'
    if os.path.exists(infixes_filename):
        print('Using stored infixes from file', infixes_filename)
        file = open(infixes_filename, 'rb')
        infixes = pickle.load(file)
        file.close()
        return infixes

    infixes = __generate_all_infixes(variants)
    infixes = __random_sample_infixes(infixes)
    file = open(infixes_filename, 'wb')
    pickle.dump(infixes, file)
    file.close()
    return infixes


def __print_parameters(log_filename_prefix: str):
    print('Using existing model:', os.path.exists(log_filename_prefix + PROCESS_TREE_EXTENSION))
    print('Filename:', log_filename_prefix)
    print('Noise threshold:', NOISE_THRESHOLD)
    print('Timeout:', TIMEOUT)
    print('Random sample size:', RANDOM_SAMPLE_SIZE)


def __get_model(log_filename_prefix: str, log: EventLog) -> ProcessTree:
    if os.path.exists(log_filename_prefix + PROCESS_TREE_EXTENSION):
        print('Using tree with filename ', log_filename_prefix + PROCESS_TREE_EXTENSION)
        tree = ptml_importer.apply(log_filename_prefix + PROCESS_TREE_EXTENSION)
    else:
        print('Generating tree from log using inductive miner')
        tree = inductive_miner.apply_tree(log, variant=inductive_miner.Variants.IM_CLEAN,
                                          parameters={
                                              inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: NOISE_THRESHOLD})
        ptml_exporter.apply(tree, log_filename_prefix + PROCESS_TREE_EXTENSION)

    if DEBUG:
        from pm4py.visualization.process_tree import visualizer as pt_visualizer
        gviz = pt_visualizer.apply(tree,
                                   parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
        pt_visualizer.view(gviz)

    return tree


def __run_experiments_for_infix(infix, tree: ProcessTree):
    try:
        result = []
        infix = list(infix)
        infix_trace = __generate_trace(infix)
        infix_length = len(infix)

        algorithm_variants = algorithm_variants_infix if INFIX_TYPE == INFIX_TYPE_INFIX else algorithm_variants_postfix

        for algorithm, fn in algorithm_variants.items():
            alignment = fn(infix_trace, tree)
            result.append(__build_measurement_data_for_alignment(infix, infix_length, alignment, algorithm))

        return result
    except Exception as e:
        print(e, 'Infix:', infix)


def __build_measurement_data_for_alignment(infix, infix_length, alignment, algorithm):
    if __is_timeout(alignment):
        return [infix, infix_length, algorithm, None, None, None, None, None, None, True, None, None, None]

    if algorithm == ALGORITHM_BASELINE_A_STAR or algorithm == ALGORITHM_BASELINE_DIJKSTRA:
        # for the baseline approach, we do not have to compute the auxiliary Petri net every time
        consumed_time = alignment['alignment_duration']
    else:
        consumed_time = alignment['alignment_duration'] + alignment['preprocessing_duration']

    return [infix, infix_length, algorithm, consumed_time,
            alignment['visited_states'], alignment['queued_states'],
            alignment['cost'], alignment['preprocessing_duration'], alignment['alignment_duration'],
            False, alignment['added_tau_transitions'], alignment['alignment'], alignment['lp_solved']]


def print_progress_on_console(res):
    global processed_infixes, number_infixes
    processed_infixes += 1
    print(processed_infixes, "/", number_infixes)


def __generate_all_infixes(variants):
    infixes = set()

    for variant in tqdm(variants):
        variant_trace = variant_to_trace(variant)
        trace_array = [e['concept:name'] for e in variant_trace]
        infixes = infixes.union(__generate_trace_infixes(trace_array, len(trace_array)))

    return infixes


def __random_sample_infixes(infixes):
    if len(infixes) < RANDOM_SAMPLE_SIZE:
        return infixes

    return random.sample(list(infixes), RANDOM_SAMPLE_SIZE)


def __generate_trace_infixes(trace_array: List[str], max_length: int):
    if max_length == 1:
        return set((e,) for e in trace_array)

    infixes = set()
    for i in range(len(trace_array) - max_length + 1):
        infixes.add(tuple(trace_array[i:max_length + i]))

    return infixes.union(__generate_trace_infixes(trace_array, max_length - 1))


def __generate_trace(trace_unformatted):
    trace = Trace()
    for event_unformatted in trace_unformatted:
        event = Event()
        event["concept:name"] = event_unformatted
        trace.append(event)

    return trace


def __is_timeout(alignment) -> bool:
    return 'timeout' in alignment and alignment['timeout']


if __name__ == '__main__':
    run_infix_alignment_experiments()

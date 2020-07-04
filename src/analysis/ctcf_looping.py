import sys
sys.path.append("./src/utilities")

from data_utilities import interim_data_path, external_data_path, figures_path, get_gaps
from plot_utilities import initialize_plotting_parameters, get_default_figsize
from bed_utilities import coords
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from pybedtools.bedtool import BedTool
import matplotlib.pyplot as plt

figures_path = figures_path / "analysis"


# ---------- UTILITY FUNCTIONS ---------- #

random_seed = 42
N_CORES = 10

LEFT_STOP_SYMBOL = '>'
BIDIRECTIONAL_STOP_SYMBOL = 'o'
RIGHT_STOP_SYMBOL = '<'

def logistic(x, L=1, k=1, mu=0):
    return L / (1 + np.exp(-k*(x - mu)))


def min_max_normalization(data):
    return (data - data.min()) / (data.max() - data.min())


def prob_perfect_looping(data):
    return np.ones(data.shape[0])

def assign_probs(ctcf_scores, probs):
    return ctcf_scores.iloc[:, :5].assign(stopping_prob=probs)


def get_loops(ctcfs_with_probs, random_seed=42):
    np.random.seed(random_seed)
    ctcfs_with_probs = ctcfs_with_probs.sort_values("ctcf_id")
    ids = ctcfs_with_probs.ctcf_id.values
    probs = ctcfs_with_probs.stopping_prob.values
    orients = ctcfs_with_probs.orientation.values
    
    loops = []
    middle_poss = np.arange(ids.shape[0] - 1)
    for mp in middle_poss:
        #left 
        lp = mp
        while ( np.random.uniform() > (0 if orients[lp] not in [LEFT_STOP_SYMBOL, BIDIRECTIONAL_STOP_SYMBOL] else 1)*probs[lp] ) and (lp >= 0):
            lp = lp - 1
        #right
        rp = mp + 1
        while ( np.random.uniform() > (0 if orients[rp] not in [RIGHT_STOP_SYMBOL, BIDIRECTIONAL_STOP_SYMBOL] else 1)*probs[rp] ) and (rp < middle_poss.shape[0]):
            rp = rp + 1

        if (lp >= 0) and (rp < middle_poss.shape[0]):
            loops.append({
                'left_id': ids[lp],
                'right_id': ids[rp],
                'count': 1
            })
    loops = pd.DataFrame.from_dict(loops)
    return loops

def merge_simulations(loops):
    result = loops.groupby(['left_id', 'right_id'])['count'].sum().reset_index()
    return result.sort_values(['left_id', 'right_id'])

def simulate_one_epoch(ctcfs_with_probs, random_seed=42):
    loops = []
    for chrom in sorted(ctcfs_with_probs.chr.unique()):
        chrom_data = ctcfs_with_probs[ctcfs_with_probs.chr == chrom]
        before_centromere = chrom_data[chrom_data.end < gaps[(gaps.chr == chrom) & (gaps.gap_type == 'centromere')].iloc[0].start]
        loops_before = get_loops(before_centromere, random_seed)
        loops.append(loops_before)
        after_centromere = chrom_data[chrom_data.start > gaps[(gaps.chr == chrom) & (gaps.gap_type == 'centromere')].iloc[0].end]
        loops_after = get_loops(after_centromere, random_seed)
        loops.append(loops_after)
    loops = merge_simulations(pd.concat(loops, axis=0, ignore_index=True))
    return loops

def simulate(ctcfs_with_probs, 
             epochs=10):
    with Pool(N_CORES) as pool:
        result = pool.starmap(simulate_one_epoch, [(ctcfs_with_probs, random_seed + i) for i in range(epochs)])
        result = merge_simulations(pd.concat(result, axis=0, ignore_index=True))
    return result

def metrics(simulation, hiccups_loops, min_threshold=0, max_threshold=1, n_points=None, exclude=None):
    hiccups_loops_set = set(hiccups_loops.iloc[:, :2].to_records(index=False).tolist())
    if exclude is not None:
        hiccups_loops_set = hiccups_loops_set.difference(exclude)
    thresholds = []
    precisions = []
    recalls = []
    n_points = max_threshold - min_threshold + 1 if n_points is None else n_points
    for thresh in np.linspace(min_threshold, max_threshold, n_points):
        ts = simulation[simulation['count'] > thresh]
        ts_loops = set(ts.iloc[:, :2].to_records(index=False).tolist())
        if exclude is not None:
            ts_loops = ts_loops.difference(exclude)
        if len(ts_loops) > 0:
            # true positives
            tp = len(ts_loops.intersection(hiccups_loops_set))
            # false positives
            fp = len(ts_loops.difference(hiccups_loops_set))
            # false negatives
            fn = len(hiccups_loops_set.difference(ts_loops))

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0
            recall = 0

        thresholds.append(thresh)
        precisions.append(precision)
        recalls.append(recall)
    return thresholds, recalls, precisions

def order_for_ROC(recalls, precisions):
    if isinstance(recalls, list):
        recalls = np.array(recalls)
    if isinstance(precisions, list):
        precisions = np.array(precisions)
    idx = np.argsort(recalls)
    return recalls[idx], precisions[idx]

# --------------------------------------- #

# --------- PLOTTING FUNCTIONS ---------- #

def plot_simulation_stats(simulation_metrics,
                          name_to_full_name,
                          name_to_color,
                          name_to_style):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(1,2, figsize=(figsize[0]*2, figsize[1]), tight_layout=True)
    for name, stats in simulation_metrics.items():
        axes[0].plot(stats[0], stats[1], label=name_to_full_name[name], color=name_to_color[name], style=name_to_style[name])
        axes[1].plot(*order_for_ROC(stats[1], stats[2]), label=name_to_full_name[name], color=name_to_color[name], style=name_to_style[name])

    yticks = axes[0].get_yticks()
    axes[0].set_yticklabels(["{:.0f}%".format(yi*100) for yi in yticks])
    axes[0].legend(title='Simulation')
    axes[0].set_xlabel("Minimum threshold\n(Loop occurences in the simulation)")
    axes[0].set_ylabel("RECALL (% HiCCUP loops\nrecovered by simulation)")
    axes[0].grid()

    yticks = axes[1].get_yticks()
    axes[1].set_yticklabels(["{:.0f}%".format(yi*100) for yi in yticks])
    xticks = axes[1].get_xticks()
    axes[1].set_xticklabels(["{:.0f}%".format(yi*100) for yi in xticks])
    axes[1].grid()
    axes[1].legend(title='Simulation')
    axes[1].set_xlabel("RECALL (% HiCCUP loops\nrecovered by simulation)")
    axes[1].set_ylabel("PRECISION (% simulated\nloops confirmed by HiCCUP)")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.savefig(figures_path / "simulation_stats.pdf", bbox_inches='tight', transparent=True)

# --------------------------------------- #


initialize_plotting_parameters()
print("Loading CTCF binding sites")
ctcf_scores = pd.read_csv(interim_data_path / "ctcf_scores.tsv", sep="\t")
gaps = get_gaps()


ctcf_scores['MotifScore_rank'] = ctcf_scores.MotifScore.rank()
ctcf_scores['ChipSeqScore_rank'] = ctcf_scores.ChipSeqScore.rank()


print("Assigning probabilities")
experiments = {
    'chipseqscore': assign_probs(ctcf_scores, min_max_normalization(ctcf_scores.ChipSeqScore.values)),
    'motifscore': assign_probs(ctcf_scores, min_max_normalization(ctcf_scores.MotifScore.values)),
    'rankscore': assign_probs(ctcf_scores, min_max_normalization(ctcf_scores.rank_score_aggregate.values)),
    'perfect': assign_probs(ctcf_scores, prob_perfect_looping(ctcf_scores)),

    'motifscore_rank': assign_probs(ctcf_scores, min_max_normalization(ctcf_scores.MotifScore_rank.values)),
    'chipseqscore_rank': assign_probs(ctcf_scores, min_max_normalization(ctcf_scores.ChipSeqScore_rank.values)),
}

name_to_full_name = {
    'chipseqscore': 'ChipSeq score',
    'motifscore': 'Motif score',
    'rankscore': 'Rank score',
    'perfect': 'Full-stop',

    'motifscore_rank': 'Motif Rank score',
    'chipseqscore_rank': 'ChipSeq Rank score'
}

name_to_color = {
    'chipseqscore': 'green',
    'motifscore': 'orange',
    'rankscore': 'red',
    'perfect': 'blue',

    'motifscore_rank': 'orange',
    'chipseqscore_rank': 'green'
}

name_to_style = {
    'chipseqscore': '-',
    'motifscore': '-',
    'rankscore': '-',
    'perfect': '-',

    'motifscore_rank': '--',
    'chipseqscore_rank': '--'
}

print("Experiments")
n_epochs = 1000
file_template = "looping_simulation_{}_{}_epochs.tsv"
simulations = {}
for name, data in experiments.items():
    print("\t" + name)
    file_path = interim_data_path / file_template.format(name, n_epochs)
    if os.path.isfile(file_path):
        simulation = pd.read_csv(file_path, sep="\t")
    else:
        simulation = simulate(data, epochs=n_epochs)
        simulation.to_csv(interim_data_path / file_template.format(name, n_epochs), sep="\t", index=False)
    simulations[name] = simulation
    
print("Comparing to HiCCUP loops")
hiccups = pd.read_csv(external_data_path / "GSE63525_GM12878_primary+replicate_HiCCUPS_looplist.txt", sep="\t")
hiccups['chr1'] = 'chr' + hiccups.chr1.astype(str)
hiccups['chr2'] = 'chr' + hiccups.chr2.astype(str)

hiccups['loop_id'] = hiccups.index
source_anchors = hiccups[['chr1', 'x1', 'x2']].copy()
source_anchors = source_anchors.drop_duplicates().sort_values(['chr1', 'x1', 'x2']).reset_index(drop=True)
print("\tSource anchors: {}".format(source_anchors.shape[0]))
target_anchors = hiccups[['chr2', 'y1', 'y2']].copy()
target_anchors = target_anchors.drop_duplicates().sort_values(['chr2', 'y1', 'y2']).reset_index(drop=True)
print("\tTarget anchors: {}".format(target_anchors.shape[0]))

source_names = source_anchors.columns.tolist() + ctcf_scores.columns.tolist()
source_anchors_to_ctcfs = BedTool.from_dataframe(source_anchors)\
                                    .intersect(BedTool.from_dataframe(ctcf_scores), wa=True, wb=True)\
                                    .to_dataframe(names=source_names)
source_anchors_to_ctcfs = source_anchors_to_ctcfs[['chr1', 'x1', 'x2', 'ctcf_id', 'orientation']]
source_anchors_to_ctcfs.columns = ['chr1', 'x1', 'x2', 'left_id', 'left_orientation']

target_names = target_anchors.columns.tolist() + ctcf_scores.columns.tolist()
target_anchors_to_ctcfs = BedTool.from_dataframe(target_anchors)\
                                    .intersect(BedTool.from_dataframe(ctcf_scores), wa=True, wb=True)\
                                    .to_dataframe(names=target_names)
target_anchors_to_ctcfs = target_anchors_to_ctcfs[['chr2', 'y1', 'y2', 'ctcf_id', 'orientation']]
target_anchors_to_ctcfs.columns = ['chr2', 'y1', 'y2', 'right_id', 'right_orientation']

hiccups_loops = hiccups\
                    .merge(source_anchors_to_ctcfs)\
                    .merge(target_anchors_to_ctcfs)
# remove non-convergent loops
hiccups_loops = hiccups_loops[(hiccups_loops.left_orientation == '>') & \
                              (hiccups_loops.right_orientation == "<")].reset_index(drop=True)
hiccups_loops = hiccups_loops.groupby(['left_id', 'right_id']).agg({
    'chr1': 'size',
    'loop_id': lambda x: list(x)
}).rename(columns={'chr1': 'count'}).reset_index()
print("HiCCUPS predicted {} loops".format(hiccups_loops.shape[0]))


max_threshold = max([x['count'].max() for x in simulations.values()])
simulation_metrics = {}
for name, simulation in simulations.items():
    print("\t" + name)
    ths, rec, prec = metrics(simulation, hiccups_loops, max_threshold=max_threshold, exclude=None)
    simulation_metrics[name] = (ths, rec, prec)
    
plot_simulation_stats(simulation_metrics, name_to_full_name, name_to_color, name_to_style)

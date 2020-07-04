import sys
sys.path.append("./src/utilities")

from data_utilities import processed_data_path, external_data_path, figures_path, interim_data_path, get_gaps, ctcf_context_categories
from plot_utilities import initialize_plotting_parameters, get_default_figsize, ctcf_colors
from bed_utilities import coords
import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from statannot import add_stat_annotation
from statsmodels.regression.linear_model import OLS
from pybedtools.bedtool import BedTool
from collections import defaultdict
import itertools
import os
from scipy.stats import binom_test, pearsonr, chisquare, ks_2samp, mannwhitneyu
from scipy.stats.distributions import kstwobign
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests
from datetime import datetime

figures_path = figures_path / "analysis"
os.makedirs(figures_path, exist_ok=True)

# ---------- UTILITY FUNCTIONS ---------- #

def load_ctcf_sites(ctcf_sites_path):
    
    __orientation_to_symbol = {
        'Forward': '>',
        'Reverse': '<',
        'NoMotifMatch': 'o'
    }
    
    df = pd.read_excel(ctcf_sites_path, sheet_name="GM12878_CTCF_orientation", header=None,
                       names=['chr', 'start', 'end', 'orientation'], usecols=range(4))
    # remove NoMotifMatch CTCF sites
#     df = df[df.orientation != 'NoMotifMatch']
    # convert orientation to symbol
    df['orientation'] = df.orientation.map(lambda x: __orientation_to_symbol[x])
    # sort by region
    df = df.sort_values(coords)
    return df

def read_narrow_peak(path):
    return pd.read_csv(path, 
                       names=['chr', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak'],
                       sep="\t")

def map_signal_on_sites(ctcf_sites, sample, name, slop=0):
    ctcf_bed = BedTool.from_dataframe(ctcf_sites)
    if slop > 0:
        ctcf_bed = ctcf_bed.slop(b=slop, genome='hg19')
    ctcf_on_sample = ctcf_bed.map(BedTool.from_dataframe(sample).sort(), c=4, o='max')\
                             .to_dataframe(names=ctcf_sites.columns.tolist() + [name])
    ctcf_on_sample[name] = ctcf_on_sample[name].map(lambda y: float(y) if y != '.' else 0)
    return ctcf_on_sample

def quantile_normalization(m):
    sort_idx = np.argsort(m, axis=0)
    ranks = np.argsort(sort_idx, axis=0)
    sorted_cols = np.sort(m, axis=0)
    col_ranks = sorted_cols.mean(1)
    qnorm = col_ranks[ranks]
    return qnorm

def cluster_ctcf_sites(ctcfs, distance):
    return BedTool.from_dataframe(ctcfs).cluster(d=distance).to_dataframe(names=ctcfs.columns.tolist() + ['cluster'])

def to_triplets(x):
    res = []
    if len(x) > 0:
        res.append(["T"])
        if len(x) > 1:
            for i in range(1, len(x) - 1):
                res.append(x[i - 1: i + 2])
            res.append(["T"])
    return res

def find_kmers(orientations, k, alphabet=['>', '<']):
    orientations = np.array(list(orientations))
    patterns = defaultdict.fromkeys(
        list(map(lambda x: "".join(x), itertools.product(alphabet, repeat=k))), 0)

    for i in range(orientations.shape[0] - k + 1):
        current_window = "".join(orientations[i: i + k])
        patterns[current_window] = patterns[current_window] + 1
    return dict(patterns)

def find_patterns(orientations, from_k=1, to_k=3):
    all_mers = dict()
    for i in range(from_k, to_k + 1):
        i_mers = find_kmers(orientations, i)
        all_mers.update(i_mers)
    return all_mers


def get_pattern_count(ctcf_clusters, from_k, to_k, npartitions=10):
    ctcf_clusters = dd.from_pandas(ctcf_clusters, npartitions=npartitions)
    ctcf_clusters_pattern_counts = ctcf_clusters.groupby("cluster")\
                                                .apply(lambda x: x.sort_values(coords)['orientation']\
                                                       .sum())\
                                                .map(lambda x: find_patterns(x, from_k, to_k), 
                                                     meta=('cluster', object))\
                                                .compute()
    ctcf_clusters_pattern_counts = pd.DataFrame(index=ctcf_clusters_pattern_counts.index, data=ctcf_clusters_pattern_counts.tolist())
    patterns = sorted(ctcf_clusters_pattern_counts.columns, key=len)
    ctcf_clusters_pattern_counts = ctcf_clusters_pattern_counts[patterns]
    return ctcf_clusters_pattern_counts

def whole_genome_CTCF_patterns_by_window(all_clusters, from_k, to_k, ws):
    pattern_counts_by_distance = []
    for distance in ws:
        print(" "*100, end='\r')
        print("\t{}".format(distance), end='\r')
        ctcf_clusters = all_clusters[coords + ['orientation', '{}'.format(distance)]]\
                                        .rename(columns = {'{}'.format(distance) : 'cluster'})
        ctcf_clusters_pattern_counts = get_pattern_count(ctcf_clusters, from_k, to_k).sum(0).to_frame(name=distance)
        pattern_counts_by_distance.append(ctcf_clusters_pattern_counts)
    pattern_counts_by_distance = pd.concat(pattern_counts_by_distance, axis=1).T
    return pattern_counts_by_distance

def strideby(v, k):
    res = np.zeros((v.shape[0] - k + 1, k), dtype=int)
    for i in range(v.shape[0] - k + 1):
        res[i, :] = v[i: i + k]
    return res

def get_all_patterns(ctcf_clusters, from_k, to_k, with_cluster=False):
    all_patterns = []

    for cn, cluster in ctcf_clusters.assign(ctcf_id=lambda x: x.index).groupby("cluster"):
        print(" "*100, end='\r')
        print("\t{}".format(cn), end='\r')
        for psize in range(from_k, to_k + 1):
            idxs = strideby(np.arange(cluster.shape[0], dtype=int), psize)
            patterns = cluster.orientation.values[idxs].sum(1)
            starts = cluster.start.values[idxs].min(1)
            ends = cluster.end.values[idxs].max(1)

            start_ctcf_id = cluster.ctcf_id.values[idxs].min(1)
            end_ctcf_id = cluster.ctcf_id.values[idxs].max(1)
            d = {
                'chr': cluster.chr.iloc[0],
                'start': starts,
                'end': ends,
                'pattern': patterns,
                'n_ctcf_sites': psize,
                'start_ctcf_id': start_ctcf_id,
                'end_ctcf_id': end_ctcf_id,
                'size': ends - starts
            }
            if with_cluster:
            	d['cluster'] = cn 
            df = pd.DataFrame(d)
            all_patterns.append(df)
    all_patterns = pd.concat(all_patterns, axis=0)
    all_patterns = all_patterns.sort_values(coords).reset_index(drop=True)
    return all_patterns

def get_enrichment_of_patterns_by_window(pattern_counts_by_distance):
    pattern_enrichment_by_distance = pattern_counts_by_distance.astype(float).copy()
    for pattern in pattern_counts_by_distance.columns:
        psize = len(pattern)
        p_n_combinations = 2**psize
        p_expected_prob = 1/p_n_combinations
        same_size_patterns = [p for p in pattern_counts_by_distance.columns if len(p) == psize]
        tot_n_samples_same_size = pattern_counts_by_distance[same_size_patterns].sum(1)
        p_expected = tot_n_samples_same_size * p_expected_prob + 1
        pattern_enrichment_by_distance[pattern] = pattern_counts_by_distance[pattern]/p_expected
    return pattern_enrichment_by_distance

def window_string_converter(x):
    if x < 1000:
        return str(x) + ' b'
    if (x >= 1000) and (x < 1000000):
        return str(x//1000)+' kb'
    if (x >= 1000000):
        return str(x//1000000)+' Mb'
    
def get_all_clusters(ctcfs, window_swap):
    all_clusters = ctcfs.copy()
    for w in window_swap:
        print(" "*100, end='\r')
        print("\t{}".format(w), end='\r')
        wclusters = cluster_ctcf_sites(ctcfs, w)
        wclusters.rename(columns={"cluster": '{}'.format(w)}, inplace=True)
        all_clusters = all_clusters.merge(wclusters, on=ctcfs.columns.tolist())
    return all_clusters


categories = ctcf_context_categories

# # --------------------------------------- #

# # --------- PLOTTING FUNCTIONS ---------- #

def plot_conservation_CTCF_sites(n_overlaps):
    figsize = get_default_figsize()
    fig = plt.figure(figsize=(figsize[0]*2.5, figsize[1]*0.5))
    plt.hist(n_overlaps, bins=np.arange(max(n_overlaps) + 2), rwidth=0.9)
    plt.grid(axis="y")
    plt.xlabel("# of samples in which the CTCF site is conserved")
    plt.ylabel("# CTCF sites")
    plt.xticks(np.arange(max(n_overlaps) + 1) + 0.5, labels=np.arange(max(n_overlaps) + 1))
    fig.savefig(figures_path / "CTCF_sites_number_overlaps_with_chipseqs.pdf", bbox_inches='tight', transparent=True)
    

def plot_rank_score_distribution(ctcfs_ms_cs):
    quantiles = np.quantile(ctcfs_ms_cs.rank_score_aggregate, q=[0.25, 0.50, 0.75])
    fig = plt.figure()
    plt.hist(ctcfs_ms_cs['rank_score_aggregate'], bins=30, rwidth=0.9)
    [plt.axvline(q, color='black') for q in quantiles]
    plt.grid(axis='y')
    plt.ticklabel_format(useMathText=True)
    plt.xlabel("Rank aggregated score")
    plt.ylabel('Frequency')
    fig.savefig(figures_path / "CTCF_sites_rank_aggregated_score_distribution.pdf", bbox_inches='tight', transparent=True)
    
def plot_chipseq_score_distribution(ctcfs_ms_cs):
    fig = plt.figure()
    plt.hist(ctcfs_ms_cs['ChipSeqScore'], bins=30, rwidth=0.9)
    plt.grid(axis='y')
    plt.ticklabel_format(useMathText=True)
    plt.xlabel("ChipSeq score")
    plt.ylabel('Frequency')
    fig.savefig(figures_path / "CTCF_sites_chipseq_score_distribution.pdf", bbox_inches='tight', transparent=True)
    
def plot_motif_score_distribution(ctcfs_ms_cs):
    fig = plt.figure()
    plt.hist(ctcfs_ms_cs['MotifScore'], bins=30, rwidth=0.9)
    plt.grid(axis='y')
    plt.ticklabel_format(useMathText=True)
    plt.xlabel("Motif score")
    plt.ylabel('Frequency')
    fig.savefig(figures_path / "CTCF_sites_motif_score_distribution.pdf", bbox_inches='tight', transparent=True)
    
def plot_n_clusters_vs_window(all_clusters, all_clusters_random, window_swap):
    fig, axes = plt.subplots(1, 1, sharex=True, tight_layout=True)
    axes.plot(window_swap, all_clusters_random.iloc[:, 4:].nunique(), linewidth=2, label='random')
    axes.plot(window_swap, all_clusters.iloc[:, 4:].nunique(), linewidth=2, label='real')
    axes.grid(axis='both')
    axes.axvline(25000, color='red', label='25kb')
    axes.set_xlabel("CTCF clustering window (bp)")
    axes.set_ylabel("CTCF clusters")
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    axes.legend(loc='lower left')
    axes.set_xscale("log")
    axes.set_xlim(0.5, 2e7)
    fig.savefig(figures_path / "CTCF_cluster_size_vs_window_with_random.pdf", bbox_inches='tight', transparent=True)

    
def plot_ctcf_patterns_vs_size(all_patterns):
    figsize = get_default_figsize()
    fig = plt.figure(figsize=(figsize[0]*2, figsize[1]*1.5))
    ax = sns.boxplot(data=all_patterns, 
    	x='n_ctcf_sites', y='size',
                hue_order=['Same', 
                           'Convergent', 
                           'Divergent', 
                           "Convergent-Divergent"],
                hue='pattern_class',
                showfliers=False, 
                order=[2, 3, 4],
                palette=ctcf_colors, medianprops={'color':'red'})
    # add_stat_annotation(ax, data=all_patterns, x='n_ctcf_sites', y='size',
    #                     order=[2, 3, 4],
    #                     hue_order=['Same', 
    #                        'Convergent', 
    #                        'Divergent', 
    #                        "Convergent-Divergent"],
    #                     hue='pattern_class', 
    #                     box_pairs=[
    #                         ( (2, "Same"), (2, "Convergent") ),
    #                         ( (2, "Same"), (2, "Divergent") ),
    #                         ( (2, "Divergent"), (2, "Convergent") ),
    #                         ( (3, "Same"), (3, "Convergent") ),
    #                         ( (3, "Same"), (3, "Divergent") ),
    #                         ( (3, "Same"), (3, "Convergent-Divergent") ),
    #                         ( (3, "Divergent"), (3, "Convergent") ),
    #                         # ( (3, "Convergent"), (3, "Convergent-Divergent") ),
    #                         # ( (3, "Divergent"), (3, "Convergent-Divergent") ),
    #                         # ( (4, "Same"), (4, "Convergent") ),
    #                         # ( (4, "Same"), (4, "Divergent") ),
    #                         # ( (4, "Same"), (4, "Convergent-Divergent") ),
    #                         # ( (4, "Divergent"), (4, "Convergent") ),
    #                         # ( (4, "Convergent"), (4, "Convergent-Divergent") ),
    #                         # ( (4, "Divergent"), (4, "Convergent-Divergent") ),
    #                     ],
    #                     test='t-test_ind', 
    #                     text_format='full', loc='outside', 
    #                     verbose=2)
    plt.legend(title="Pattern class")
    plt.xlabel("N. CTCF sites composing the pattern")
    plt.ylabel("Pattern size (bp)")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    fig.savefig(figures_path / "CTCF_patterns_vs_size.pdf", bbox_inches='tight', transparent=True)

    
patternclass_to_pattern = {
    'Same': ['>>', '<<', '>>>', '<<<', '>>>>', '<<<<'],
    'Convergent': ['><', '>><', '><<', '>><<', '>>><', '><<<'],
    'Divergent': ['<>', '<>>', '<<>', '<<>>', '<<<>', '<>>>'],
    'Convergent-Divergent': ['><>', '<><', '>><>', '><>>', '<><<', '<<><', '><><', '><<>', '<>><', '<><>']
}

patter_to_patternclass = { k:v for v,l in patternclass_to_pattern.items() for k in l}

def plot_patterns_by_window_and_class(pattern_enrichment_by_distance, 
                                      path=figures_path / "CTCF_patterns_vs_size.pdf", 
                                      vmin=-0.06, vmax=0.06):
    figsize=get_default_figsize()
    height_ratios=[
            len(patternclass_to_pattern['Same']),
            len(patternclass_to_pattern['Convergent']),
            len(patternclass_to_pattern['Divergent']),
            len(patternclass_to_pattern['Convergent-Divergent'])
        ]
    fig, axes = plt.subplots(len(patternclass_to_pattern.keys()), 
                           1, sharex=True, 
                           figsize=(figsize[0]*1.6, 
                                    figsize[1]*len(patternclass_to_pattern.keys())*0.5),
                           gridspec_kw={'height_ratios': height_ratios},
                           tight_layout=True)
    cbar_ax = fig.add_axes([1, .3, .03, .4])
    pebd_names = pattern_enrichment_by_distance\
                        .assign(index=lambda x: x.index.map(window_string_converter))\
                        .set_index('index')
    for i, p in enumerate(patternclass_to_pattern.items()):
        pclass, pnames = p
        sns.heatmap(pebd_names.T.loc[pnames]\
                    .applymap(lambda x: np.log10(x)), cbar= i == 0,
                    cmap='bwr', linewidth=0.01, vmin=vmin, vmax=vmax, ax=axes[i],
                    linecolor='black', cbar_ax= None if i != 0 else cbar_ax,
                    cbar_kws={'label': "$log_{10}(enrichment)$"})
        for tick in axes[i].get_yticklabels():
            tick.set_rotation(0)
        if i != len(patternclass_to_pattern.keys()) - 1:
            axes[i].set_xlabel("")
            axes[i].tick_params(axis='x', bottom=False)
        else:
            axes[i].set_xlabel("Clustering window")
        axes[i].text(x=-0.12, y=0.5, s=pclass, horizontalalignment='center',
                     verticalalignment='center',
                     rotation=90, transform=axes[i].transAxes)
    fig.savefig(path, bbox_inches='tight', transparent=True)

# --------------------------------------- #

initialize_plotting_parameters()
print("Loading CTCFs")
ctcfs = load_ctcf_sites(external_data_path / "Human_CTCF_orientation_distances.xlsx")
ctcfs.to_csv(processed_data_path / "ctcfs.tsv", sep="\t", index=False, header=True)
# plot_n_orientations(ctcfs)


print("Loading CTCF ChipSeqs")
ctcf_files_info = pd.read_csv(external_data_path / "CTCF_signals" / 'info.tsv', sep="\t")
ctcf_files_info['merged_name'] = ctcf_files_info.cell + "__" + \
                                 ctcf_files_info.treatment.map(lambda x: x if (isinstance(x, str)) and (x != 'no') else '') + "__" + \
                                 ctcf_files_info.Lineage.map(lambda x: x.replace(" ", "_")) + "__" + \
                                 ctcf_files_info.Tissue.map(lambda x: x.replace(" ", "_")) + "__" + \
                                 ctcf_files_info.Karyotype.map(lambda x: x.replace(" ", "_"))

all_chipseq = []
for ctcf_file in ctcf_files_info.File:
    print(" "*100, end='\r')
    print("\t{}".format(ctcf_file), end='\r')
    data = read_narrow_peak(external_data_path / "CTCF_signals" / ctcf_file)
    info = ctcf_files_info[ctcf_files_info.File == ctcf_file].iloc[0].to_dict()
    for k,v in info.items():
        data[k] = v
    all_chipseq.append(data)
all_chipseq = pd.concat(all_chipseq, axis=0)

print("Mapping signals to CTCF sites")
slop = 0
ctcf_mappings = ctcfs.copy()
for sn, sample in all_chipseq.groupby("merged_name"):
    print(" "*100, end='\r')
    print("\t{}".format(sn), end='\r')
    sample = sample.sort_values(coords)[coords + ['signalValue']]
    ctcf_on_sample = map_signal_on_sites(ctcfs, sample, sn, slop)
    ctcf_mappings[sn] = ctcf_on_sample[sn]

ctcf_mappings.to_csv(interim_data_path / "ctcf_mappings.tsv", sep='\t', index=False)
sample_names = ctcf_mappings.columns[4:].tolist()
n_overlaps = np.count_nonzero(ctcf_mappings[sample_names].values, axis=1)
plot_conservation_CTCF_sites(n_overlaps)


ctcf_mappings = pd.read_csv(interim_data_path / "ctcf_mappings.tsv", sep='\t')


print("Quantile normalization")
ctcf_mappings_qnorm = pd.concat([ctcf_mappings.iloc[:, :4], 
                                 pd.DataFrame(index=ctcf_mappings.index, 
                                              columns=sample_names, 
                                              data=quantile_normalization(ctcf_mappings[sample_names].values))], 
                                axis=1)

print("Adding motif strenght")
ctcf_motifs_strengths = pd.read_csv(processed_data_path / "GM12878_Aiden_peaks_with_motifscore.txt", sep="\t")
ctcfs_with_motif_strength = ctcfs.assign(ctcf_id=lambda x: np.arange(ctcfs.shape[0], dtype=int))\
                                  .merge(ctcf_motifs_strengths.groupby("PositionID")['MotifScore'].max().to_frame(), 
                                         left_on="ctcf_id", right_index=True,
                                         how='left').fillna(0).sort_values("ctcf_id")

ctcfs_ms_cs = ctcfs_with_motif_strength.copy()
ctcfs_ms_cs['ChipSeqScore'] = ctcf_mappings_qnorm[sample_names].sum(1).values

print("Calculating rank score")
ctcfs_ms_cs['rank_score_aggregate'] = ctcfs_ms_cs.ChipSeqScore.rank() * ctcfs_ms_cs.MotifScore.rank()
ctcfs_ms_cs.to_csv(interim_data_path / "ctcf_scores.tsv", sep='\t', index=False, header=True)
plot_rank_score_distribution(ctcfs_ms_cs)
plot_chipseq_score_distribution(ctcfs_ms_cs)
plot_motif_score_distribution(ctcfs_ms_cs)

print("Distance between CTCF sites")
ctcfs = ctcfs[ctcfs.orientation != 'o'].reset_index(drop=True)
distances_between_ctcfs = ctcfs.shift(-1).start - ctcfs.end
distances_between_ctcfs = distances_between_ctcfs[(distances_between_ctcfs > 0) & (distances_between_ctcfs < 1e7)]

print("Distance between Shuffled CTCF sites")
gaps = get_gaps()
shuffled_ctcf_sites = BedTool.from_dataframe(ctcfs)\
                             .shuffle(genome='hg19',
                                      chrom=True, 
                                      noOverlapping=True,
                                      excl=BedTool.from_dataframe(gaps).sort().fn)\
                             .sort().to_dataframe(names=ctcfs.columns.tolist())
distances_between_shuffled_ctcfs = shuffled_ctcf_sites.shift(-1).start - shuffled_ctcf_sites.end
distances_between_shuffled_ctcfs = distances_between_shuffled_ctcfs[(distances_between_shuffled_ctcfs > 0) & (distances_between_shuffled_ctcfs < 1e7)]


fig = plt.figure()
sns.distplot(distances_between_shuffled_ctcfs.map(np.log10), label='random', hist=False)
ax = sns.distplot(distances_between_ctcfs.map(np.log10), label='real', hist=False)
pvalue = mannwhitneyu(distances_between_ctcfs, distances_between_shuffled_ctcfs, alternative='two-sided').pvalue
plt.text(0.2,0.7, "p-value = {}".format(pvalue), transform=ax.transAxes)
xticks, _ = plt.xticks()
plt.xticks(xticks, labels=["$10^{{{}}}$".format(int(x)) for x in xticks])
plt.xlabel("Distance between adjacent CTCF sites (bp)")
plt.ylabel("Density")
fig.savefig(figures_path / "distplot_ctcf_distances_real_random_pvalue.pdf", bbox_inches='tight', transparent=True)



print("Clustering CTCF sites")
window_swap = [64,
			128,
			256,
			512,
			1024,
			2048,
			4096,
			8192,
			16384,
			25000,
			32768,
			65536,
			131072,
			262144,
			524288,
			1048576,
			2097152,
			4194304,
			8388608,
			16777216,
			33554432,
			67108864,
			134217728,
			268435456
			]


all_clusters = get_all_clusters(ctcfs, window_swap)
all_clusters_random = get_all_clusters(shuffled_ctcf_sites, window_swap)
    
plot_n_clusters_vs_window(all_clusters, all_clusters_random, window_swap)

print("Assigning CTCF categories")
gaps = get_gaps()

ctcf_clusters = all_clusters[coords + ['orientation', str(window_swap[-1])]].rename(columns={str(window_swap[-1]): 'cluster'}).copy()

def divide_clusters_by_chromosome_arms(ctcf_clusters, gaps):
    r = []
    for c in ctcf_clusters.cluster.unique():
        ccc = ctcf_clusters[ctcf_clusters.cluster == c]
        ccc_before = ccc[ccc.end < gaps[(gaps.chr == ccc.iloc[0].chr) & (gaps.gap_type == 'centromere')].iloc[0].start].copy()
        ccc_before['cluster'] = ccc_before['cluster'].astype(str) + "_1"
        ccc_after = ccc[ccc.start > gaps[(gaps.chr == ccc.iloc[0].chr) & (gaps.gap_type == 'centromere')].iloc[0].end].copy()
        ccc_after['cluster'] = ccc_after['cluster'].astype(str) + "_2"
        r.append(ccc_before)
        r.append(ccc_after)
    r = pd.concat(r, axis=0).sort_values(coords).reset_index(drop=True)
    return r

def assign_ctcf_categories(cc):
    ctcf_clusters_categories = cc.groupby("cluster").orientation\
                                        .sum()\
                                        .map(lambda x: to_triplets(list(x)))\
                                        .map(lambda x: list(map(lambda y: categories["".join(y)], x)))
    ctcf_clusters = cc.copy()
    for c in ctcf_clusters_categories.index:
        print(" "*100, end='\r')
        print("\t{}".format(c), end='\r')
        ctcf_clusters.loc[ctcf_clusters.cluster == c, 'context'] = ctcf_clusters_categories.loc[c]
    return ctcf_clusters_categories, ctcf_clusters

from_k, to_k = 1, 4
ctcf_clusters = divide_clusters_by_chromosome_arms(ctcf_clusters, gaps)
ctcf_clusters_categories, ctcf_clusters = assign_ctcf_categories(ctcf_clusters)
cats = ctcf_clusters_categories.sum()
ctcf_clusters[coords + ['orientation', 'context']].to_csv(interim_data_path / "ctcfs_with_context.tsv", sep="\t", index=False)

print("Getting all possible patterns")
all_patterns = get_all_patterns(ctcf_clusters, from_k, to_k)
all_patterns['pattern_class'] = all_patterns.pattern.map(patter_to_patternclass)
all_patterns.to_csv(interim_data_path / "all_patterns.tsv", sep="\t", index=False)
plot_ctcf_patterns_vs_size(all_patterns)

print("Permutation test of patterns")
if os.path.isfile(interim_data_path / "permutation_pattern_counts_log2.tsv"):
    print("\tLoading from cache")
    pattern_counts = pd.read_csv(interim_data_path / "permutation_pattern_counts_log2.tsv", sep="\t")
else:
    n_permutations = 10000
    pattern_counts = None
    start = datetime.now()
    for p in range(n_permutations):
        print("# ---- Permutation {} ---- #".format(p))
        p_ctcfs = ctcfs.copy()
        p_ctcfs['orientation'] = p_ctcfs.orientation.sample(p_ctcfs.shape[0], replace=False).values
        p_clusters = get_all_clusters(p_ctcfs, [int(1e9)]).rename(columns={'1000000000': 'cluster'})
        p_clusters = divide_clusters_by_chromosome_arms(p_clusters, gaps)
        _, p_clusters = assign_ctcf_categories(p_clusters)
        p_all_patterns = get_all_patterns(p_clusters, from_k, to_k)
        p_all_patterns = p_all_patterns[['chr', 'start', 'end', 'pattern']]#.rename(columns={'pattern': 'pattern_{}'.format(p)})
        p_pattern_counts = p_all_patterns.groupby("pattern").size().to_frame("count_{}".format(p)).reset_index()
        if pattern_counts is None:
            pattern_counts = p_pattern_counts
        else:
            pattern_counts = pattern_counts.merge(p_pattern_counts, on='pattern')
        if p % 10 == 0:
            print("\tSaving -- Elapsed {}".format(datetime.now() - start))
    pattern_counts.to_csv(interim_data_path / "permutation_pattern_counts_log2.tsv", sep="\t", index=False)

pattern_counts['pattern_class'] = pattern_counts.pattern.map(patter_to_patternclass)
pattern_counts['n_ctcf_sites'] = pattern_counts.pattern.map(len)

size_class_count_permutation = pattern_counts.drop('pattern', axis=1).groupby(['n_ctcf_sites', 'pattern_class']).sum()
size_class_count = all_patterns.groupby(['n_ctcf_sites', 'pattern_class']).size().to_frame("obs")
pc_table = size_class_count.merge(size_class_count_permutation, on=['n_ctcf_sites', 'pattern_class'])

theoretical_pattern_counts = pd.Series(patter_to_patternclass)
theoretical_pattern_counts = theoretical_pattern_counts.to_frame("pattern_class").reset_index()
theoretical_pattern_counts = theoretical_pattern_counts.rename(columns={'index': 'pattern'})
theoretical_pattern_counts['n_ctcf_sites'] = theoretical_pattern_counts.pattern.map(len)
theoretical_pattern_counts = theoretical_pattern_counts.groupby(['n_ctcf_sites', 'pattern_class']).size()
theoretical_pattern_counts = theoretical_pattern_counts.to_frame("theoretical")
pc_table = pc_table.merge(theoretical_pattern_counts, on=['n_ctcf_sites', 'pattern_class'])

perm_stats = []
for s in range(2, 5):
    s_table = pc_table.loc[s]
    s_table = s_table/s_table.sum(0)
    ref = s_table['theoretical'].values
    for c in s_table.columns:
        c_vals = s_table[c].values
        distance = (((c_vals - ref)**2)/ref).sum()
        perm_stats.append({'n_ctcf_sites': s, 'exp': c, 'dist': distance})
perm_stats = pd.DataFrame.from_dict(perm_stats)

figsize = get_default_figsize()
fig, ax = plt.subplots(1, 3, figsize=(figsize[0]*3, figsize[1]), tight_layout=True)
for i, s in enumerate(range(2, 5)):
    s_perm_stats = perm_stats[perm_stats.n_ctcf_sites == s]
    s_obs = s_perm_stats.loc[s_perm_stats.exp == 'obs', 'dist'].values[0]
    s_perm = s_perm_stats.loc[~s_perm_stats.exp.isin(['obs', 'theoretical']), 'dist'].values

    p_value = (s_perm > s_obs).sum() / s_perm.shape[0]
    print("{}-plets empirical p-value: {}".format(s, p_value))

    
    sns.distplot(s_perm, hist=False, label='Permutations', ax=ax[i])
    ax[i].axvline(s_obs, color='red', label='Observed')
    ax[i].set_title("{}-plets".format(s))
    ax[i].set_xlabel("$\\chi^2$ statistic")
    ax[i].set_ylabel("Frequency")
    ax[i].set_xlim(-0.0001, 0.003)
    ax[i].legend().set_visible(False)
handles, labels = ax[i].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=8,
           bbox_to_anchor=(0.45,1.02))
fig.savefig(figures_path / "permutation_patterns_dist_from_reference_all.pdf".format(s), bbox_inches='tight', transparent=True)
plt.close(fig)


print("Calculating the number of patterns for each clustering window")
if os.path.isfile(str(interim_data_path / "pattern_counts_by_distance_log2.tsv")):
    pattern_counts_by_distance = pd.read_csv(interim_data_path / "pattern_counts_by_distance_log2.tsv", sep="\t", index_col=0)
else:
    pattern_counts_by_distance = whole_genome_CTCF_patterns_by_window(all_clusters, from_k, to_k, window_swap)
    pattern_counts_by_distance.to_csv(interim_data_path / "pattern_counts_by_distance_log2.tsv", 
                                      sep='\t', index=True, index_label="window", header=True)

pattern_counts_by_distance.index.name = "window"

print("Statistical validation")
expected = pd.DataFrame({'pattern': list(patter_to_patternclass.keys()), 'class': list(patter_to_patternclass.values())})
expected['pattern_length'] = expected.pattern.map(len)
expected = pd.pivot_table(expected, index='pattern_length', columns='class', values='pattern', aggfunc='count', fill_value=0)
expected = expected.div(expected.sum(1), axis=0)

pattern_counts_by_distance_table = pd.melt(pattern_counts_by_distance.reset_index(), id_vars=['window'], var_name='pattern', value_name='count')
pattern_counts_by_distance_table['class'] = pattern_counts_by_distance_table.pattern.map(patter_to_patternclass)
pattern_counts_by_distance_table['pattern_length'] = pattern_counts_by_distance_table.pattern.map(len)
pattern_counts_by_distance_table = pattern_counts_by_distance_table[pattern_counts_by_distance_table.pattern_length > 1]
observed = pd.pivot_table(pattern_counts_by_distance_table, index=['window', 'pattern_length'], columns='class', values='count', aggfunc='sum', fill_value=0)

pattern_counts_by_distance_stats = []
for w in window_swap:
	w_obs = observed.loc[w]
	w_marginals = w_obs.sum(1)
	w_exp = expected.mul(w_marginals, axis=0)
	for l in range(2, 5):
		if l == 2:
			o = w_obs.loc[l, ['Same', 'Divergent', 'Convergent']].values
			e = w_exp.loc[l, ['Same', 'Divergent', 'Convergent']].values
		else:
			o = w_obs.loc[l, ['Same', 'Divergent', 'Convergent-Divergent', 'Convergent']].values
			e = w_exp.loc[l, ['Same', 'Divergent', 'Convergent-Divergent', 'Convergent']].values
		wl_stat, wl_pvalue = chisquare(o,e)
		pattern_counts_by_distance_stats.append({'window': w, 'pattern_length': l, 'stat': wl_stat, 'pvalue': wl_pvalue})

pattern_counts_by_distance_stats = pd.DataFrame.from_dict(pattern_counts_by_distance_stats)
pattern_counts_by_distance_stats = pattern_counts_by_distance_stats.fillna(1)
_, bonfcorr_pvalues, _, _ = multipletests(pattern_counts_by_distance_stats['pvalue'], method='bonferroni')
pattern_counts_by_distance_stats['pvalue_bonf'] = bonfcorr_pvalues
pattern_counts_by_distance_stats['log10pvalue'] = pattern_counts_by_distance_stats.pvalue.map(lambda x: -np.log10(x))
pattern_counts_by_distance_stats['log10pvalue_bonf'] = pattern_counts_by_distance_stats.pvalue_bonf.map(lambda x: -np.log10(x))


figsize = get_default_figsize()
fig = plt.figure(figsize=(figsize[0]*2.5, figsize[1]*1.5))
biplets = pattern_counts_by_distance_stats[pattern_counts_by_distance_stats.pattern_length == 2]
plt.plot(np.arange(biplets.shape[0]), biplets.log10pvalue_bonf, label='2-plets')
plt.xticks(np.arange(biplets.shape[0]), labels=biplets.window)
triplets = pattern_counts_by_distance_stats[pattern_counts_by_distance_stats.pattern_length == 3]
plt.plot(np.arange(triplets.shape[0]), triplets.log10pvalue_bonf, label='3-plets')
plt.xticks(np.arange(triplets.shape[0]), labels=triplets.window)
tetraplets = pattern_counts_by_distance_stats[pattern_counts_by_distance_stats.pattern_length == 4]
plt.plot(np.arange(tetraplets.shape[0]), tetraplets.log10pvalue_bonf, label='4-plets')
plt.xticks(np.arange(tetraplets.shape[0]), labels=tetraplets.window, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14, loc='upper left')
plt.xlabel("Clustering window", fontsize=18)
plt.ylabel("$-log_{10}($p-value$)$\n(Bonferroni corrected)", fontsize=18)
fig.savefig(figures_path / "pattern_counts_by_distance_log2_pvalues_bonferroni.pdf", bbox_inches='tight', transparent=True)
plt.close(fig)

print("Calculating the enrichment of patterns VS clustering window") 
pattern_enrichment_by_distance = get_enrichment_of_patterns_by_window(pattern_counts_by_distance)
pattern_enrichment_by_distance = pattern_enrichment_by_distance[(pattern_enrichment_by_distance.index >= 3000) & (pattern_enrichment_by_distance.index<=2097152)]

plot_patterns_by_window_and_class(pattern_enrichment_by_distance, 
                                      path=figures_path / "CTCF_class_enrichment_by_window_log2.pdf")

print("Sensitivity analysis")
ctcfs_with_scores = pd.read_csv(interim_data_path / "ctcf_scores.tsv", sep='\t')
ctcfs_with_scores = ctcfs_with_scores[ctcfs_with_scores.orientation != 'o'].reset_index(drop=True)

quantiles = [0.25, 0.50, 0.75]

sensitivity_counts_names = {
    (0.25, 'greater'): 'ctcfs_greater_25q_patterns_counts_by_distance_log2',
    (0.50, 'greater'): 'ctcfs_greater_50q_patterns_counts_by_distance_log2',
    (0.75, 'greater'): 'ctcfs_greater_75q_patterns_counts_by_distance_log2',
    (0.25, 'lower'):   'ctcfs_lower_25q_patterns_counts_by_distance_log2',
    (0.50, 'lower'):   'ctcfs_lower_50q_patterns_counts_by_distance_log2',
    (0.75, 'lower'):   'ctcfs_lower_75q_patterns_counts_by_distance_log2'
}

q_rank_scores = {}
ctcfs_greater_q = {}
ctcfs_lower_q = {}
sensitivity_counts = {}
sensitivity_enrich = {}
ws = window_swap

for q in quantiles:
    q_rank_scores[q] = ctcfs_with_scores.rank_score_aggregate.quantile(q)
    print("{} quantile: ".format(q), q_rank_scores[q])

    ctcfs_greater_q[q] = ctcfs_with_scores.loc[
        ctcfs_with_scores.rank_score_aggregate > q_rank_scores[q], ctcfs.columns.tolist()]
    
    fname_greater_q = sensitivity_counts_names[(q, 'greater')]
    if os.path.isfile(str(interim_data_path / "{}.tsv".format(fname_greater_q))):
        sensitivity_counts[(q, 'greater')] = pd.read_csv(interim_data_path / "{}.tsv".format(fname_greater_q),
                                                         sep='\t', index_col=0)
    else:
        sensitivity_counts[(q, 'greater')] = whole_genome_CTCF_patterns_by_window(
                                get_all_clusters(ctcfs_greater_q[q], ws), from_k, to_k, ws)
        sensitivity_counts[(q, 'greater')].to_csv(
            interim_data_path / "{}.tsv".format(fname_greater_q), 
            sep='\t', index=True, index_label="window", header=True)
    sensitivity_counts[(q, 'greater')] = sensitivity_counts[(q, 'greater')][(sensitivity_counts[(q, 'greater')].index >= 3000) & (sensitivity_counts[(q, 'greater')].index <= 2097152)]
    sensitivity_enrich[(q, 'greater')] = get_enrichment_of_patterns_by_window(
        sensitivity_counts[(q, 'greater')])
    
    ctcfs_lower_q[q] = ctcfs_with_scores.loc[
        ctcfs_with_scores.rank_score_aggregate < q_rank_scores[q], ctcfs.columns.tolist()]
    
    fname_lower_q = sensitivity_counts_names[(q, 'lower')]
    if os.path.isfile(str(interim_data_path / "{}.tsv".format(fname_lower_q))):
        sensitivity_counts[(q, 'lower')] = pd.read_csv(interim_data_path / "{}.tsv".format(fname_lower_q),
                                                         sep='\t', index_col=0)
    else:
        sensitivity_counts[(q, 'lower')] = whole_genome_CTCF_patterns_by_window(
                                get_all_clusters(ctcfs_lower_q[q], ws), from_k, to_k, ws)
        sensitivity_counts[(q, 'lower')].to_csv(
            interim_data_path / "{}.tsv".format(fname_lower_q), 
            sep='\t', index=True, index_label="window", header=True)
    sensitivity_counts[(q, 'lower')] = sensitivity_counts[(q, 'lower')][(sensitivity_counts[(q, 'lower')].index >= 3000) & (sensitivity_counts[(q, 'lower')].index <= 2097152)]
    sensitivity_enrich[(q, 'lower')] = get_enrichment_of_patterns_by_window(
        sensitivity_counts[(q, 'lower')])
    
for k,data in sensitivity_enrich.items():
    print(k)
    plot_patterns_by_window_and_class(data[data.index >= 64], 
                                      path=figures_path / "CTCF_{}_{}_class_enrichment_by_window_log2.pdf".format(k[0], k[1]))

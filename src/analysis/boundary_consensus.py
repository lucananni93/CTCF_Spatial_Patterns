import sys
sys.path.append("./src/utilities")

from data_utilities import external_data_path, figures_path, processed_data_path, read_PCHiC, get_gaps, interim_data_path, get_region_center
from plot_utilities import initialize_plotting_parameters, get_default_figsize
from bed_utilities import coords, load_chromsizes, windowing_by_size, coverage_by_window, shift_test
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pybedtools.bedtool import BedTool
import networkx as nx
from community import best_partition


figures_path = figures_path / "analysis"


# ---------- UTILITY FUNCTIONS ---------- #

def load_tad_dataset(path):
    return pd.read_csv(path, sep="\t")\
             .assign(start=lambda x: x.start.astype(int),
                     end=lambda x: x.end.astype(int),
                     chr=lambda x: 'chr' + x.chr.astype(str))\
             .drop("percentOverlap", axis=1)\
             .sort_values(coords)

def get_boundary_positions(all_tads):
    all_starts = all_tads.copy()
    all_starts['end'] = all_starts['start']
    all_starts['side'] = 'start'
    all_ends = all_tads.copy()
    all_ends['start'] = all_ends['end']
    all_ends['side'] = 'end'
    all_boundary_positions = pd.concat([all_starts, all_ends], 
                                       axis=0, ignore_index=True)\
                               .sort_values(coords)\
                               .reset_index(drop=True)
    all_boundary_positions['boundary_uid'] = all_boundary_positions.tad_uid + "_" + all_boundary_positions.side
    return all_boundary_positions


def cluster_boundary_positions(all_boundary_positions, window):
    all_boundary_extended = BedTool.from_dataframe(all_boundary_positions)\
                                    .slop(r=int(window/2), l=int(window/2), genome='hg19')
    bound_pos_VS_bound_pos = all_boundary_extended.intersect(all_boundary_extended, 
                                                             wa=True, wb=True, loj=True)
    bound_pos_VS_bound_pos = bound_pos_VS_bound_pos.to_dataframe(
                                    names=["b1_" + x for x in all_boundary_positions.columns] + \
                                          ["b2_" + x for x in all_boundary_positions.columns])
    bound_pos_VS_bound_pos = bound_pos_VS_bound_pos[
                (bound_pos_VS_bound_pos.b1_boundary_uid != bound_pos_VS_bound_pos.b2_boundary_uid) & 
                (bound_pos_VS_bound_pos.b1_cell_type != bound_pos_VS_bound_pos.b2_cell_type)]
    bound_pos_G = nx.from_pandas_edgelist(bound_pos_VS_bound_pos[['b1_boundary_uid', 'b2_boundary_uid']],
                                      source='b1_boundary_uid', target='b2_boundary_uid',
                                      create_using=nx.Graph)
    bound_pos_G.add_nodes_from(all_boundary_positions.boundary_uid)
    bound_communities = best_partition(bound_pos_G)
    bound_communities = pd.Series(bound_communities).to_frame(name='cluster')\
                          .reset_index().rename(columns={'index': 'boundary_uid'})
    return all_boundary_positions.merge(bound_communities)

def get_consensus_regions(data):
    clusters = data.groupby('cluster')\
                   .agg({'chr': 'first',
                         'start': 'min',
                         'end': 'max',
                         'boundary_uid': lambda x: ",".join(list(x)),
                         'cell_type': lambda x: ",".join(set(x))})
    clusters['n_cell_types'] = clusters.cell_type.map(lambda x: len(x.split(",")))
    clusters['n_boundaries'] = clusters.boundary_uid.map(lambda x: len(x.split(",")))
    
    for ct in sorted(data.cell_type.unique()):
        clusters[ct] = clusters.boundary_uid.map(lambda x: x.count(ct))
    
    clusters = clusters.sort_values(coords).reset_index(drop=True)
    return clusters


def get_consensus_tads(sel_bounds, gaps):
    consensus_tads = BedTool.from_dataframe(sel_bounds[coords])\
                            .complement(genome='hg19')\
                            .subtract(BedTool.from_dataframe(gaps))\
                            .to_dataframe(names=coords)
    return consensus_tads

# --------------------------------------- #

# --------- PLOTTING FUNCTIONS ---------- #
    
def plot_consensus_boundary_properties(consensus_bounds):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(2,1,sharex=True, tight_layout=True, figsize=(figsize[0]*1.5, figsize[1]*2))
    min_n, max_n = consensus_bounds.n_cell_types.min(), consensus_bounds.n_cell_types.max()
    sns.countplot(consensus_bounds.n_cell_types, ax=axes[0], 
                  order=range(min_n, max_n + 1), palette="Reds")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Boundaries")
    axes[0].grid(axis='y')
    axes[0].ticklabel_format(axis='y', scilimits=(0,0), useMathText=True)
    sns.boxplot(data=consensus_bounds, x='n_cell_types', y="length", 
                order=range(min_n, max_n + 1), ax=axes[1], palette="Reds")
    axes[1].set_xlabel("Conservation score (s)")
    axes[1].set_ylabel("Boundary size")
    axes[1].grid(axis='y')
    axes[1].ticklabel_format(axis='y', scilimits=(0,0), useMathText=True)
    fig.savefig(figures_path / "consensus_boundaries_stats.pdf", bbox_inches='tight', transparent=True)


def plot_consensus_boundaries_intersection_with_GM12878(x):
    
    consensus_boundaries_fixed_gm12878 = x.assign(
    has_gm12878_bound=lambda x: x.n_gm12878_bounds.map(lambda y: 'Intersects GM12878 boundary' if y > 0 \
                                                       else 'Does not intersect GM12878 boundary'))
    
    figsize = get_default_figsize()
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(figsize[0]*1.5, figsize[1]*2))
    sns.countplot(data=consensus_boundaries_fixed_gm12878, 
                  x='n_cell_types', 
                  hue='has_gm12878_bound', 
                  ax=axes[0], 
                  hue_order=['Intersects GM12878 boundary', 
                             'Does not intersect GM12878 boundary'])
    axes[0].legend()
    axes[0].grid(axis='y')
    axes[0].set_ylabel("# boundaries")
    axes[0].legend()

    x = consensus_boundaries_fixed_gm12878.groupby(['n_cell_types', 'has_gm12878_bound']).size().unstack()
    x = x.div(x.sum(1), axis=0)
    x['Intersects GM12878 boundary'].plot.bar(ax=axes[1])
    plt.ylim(0, 0.8)
    yticks, _ = plt.yticks()
    plt.yticks(yticks, ["{:.0f}%".format(yi*100) for yi in yticks], rotation=0)
    plt.xticks(rotation=0)
    plt.xlabel("Boundary conservation ($s$)")
    plt.ylabel("% boundaries overlapping\nwith a GM12878 boundary")
    plt.grid(axis='y')
    fig.savefig(figures_path / "cons_bounds_vs_gm12878_bounds.pdf", bbox_inches='tight', transparent=True)
    

def plot_cons_bounds_vs_gm12878_DI(aggregations_by_bound, extended, window_size, measure, ylabel="Average Directionality\nIndex on GM12878"):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(1,len(aggregations_by_bound.keys()), 
                             sharex='col', sharey=True, 
                             tight_layout=True, figsize=(figsize[0]*7,figsize[1]))
    for level in sorted(aggregations_by_bound.keys()):
        y = aggregations_by_bound[level].mean(0)
        axes[level - 1].plot(y, linewidth=1, color='black')
        axes[level - 1].fill_between(np.arange(y.shape[0]), y, where=y < 0, facecolor='red')
        axes[level - 1].fill_between(np.arange(y.shape[0]), y, where=y >= 0, facecolor='blue')
        if level == 1:
            axes[level - 1].set_ylabel(ylabel, fontsize="xx-large")
            axes[level - 1].tick_params(axis='y', which='major')
        axes[level - 1].grid()
        axes[level - 1].set_title("$s = {}$".format(level), fontweight="bold")
        axes[level - 1].set_xticks([0, extended/window_size, extended*2/window_size])
        yticklabels = axes[level - 1].get_yticklabels()
        
        length_n = "{}".format(extended//1000)
        
        axes[level - 1].set_xticklabels(['-{}kb'.format(length_n), '0', '+{}kb'.format(length_n)])
        axes[level - 1].set_xlabel("Distance from\nboundary center", fontsize='xx-large')
    fig.savefig(figures_path / "cons_bounds_vs_gm12878_{}.pdf".format(measure), bbox_inches='tight', transparent=True)
    
def plot_cons_bounds_vs_grbs(grb_vs_windows, length_vs_windows, grbs):
    X = grb_vs_windows.reindex(index = grbs.sort_values('length', ascending=False).grb_uid.tolist()).fillna(0).values
    min_cons = 2
    X[X < min_cons] = 0
    X[X >= min_cons] = 1


    L = length_vs_windows.reindex(index = grbs.sort_values('length', ascending=False).grb_uid.tolist()).fillna(0).values
    L[L > 0] = 1
    L = L[L.sum(1) != 0]

    figsize = get_default_figsize()
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(figsize[0]*2, figsize[1]))
    axes[0].imshow(X, aspect="auto", interpolation='bilinear', cmap='hot')
    axes[0].set_yticks([])
    axes[0].set_xticks([0, int(X.shape[1]/2), X.shape[1] - 1])
    axes[0].set_xticklabels(['-2.5Mb', '0', "+2.5Mb"])
    axes[0].set_xlabel("Distance from GRB center")
    axes[0].set_title("Conserved boundaries ($s \geq {}$)".format(min_cons), fontsize='large')

    axes[1].imshow(L, aspect="auto", interpolation='bilinear', cmap='Blues')
    axes[1].set_yticks([])
    axes[1].set_xticks([0, int(L.shape[1]/2), L.shape[1] - 1])
    axes[1].set_xticklabels(['-2.5Mb', '0', "+2.5Mb"])
    axes[1].set_xlabel("Distance from GRB center")
    axes[1].set_title("GRB length", fontsize='large')
    fig.savefig(figures_path / "cons_bounds_vs_GRBs.pdf", bbox_inches='tight', transparent=True)


def plot_conservation_shift_test(regions_n_cross):
    figsize = get_default_figsize()
    fig = plt.figure(figsize = (figsize[0]*2, figsize[1]*1.2))
    sns.lineplot(data=regions_n_cross.assign(conservation = lambda x: x.conservation.astype(str) + " times"), 
             x='shift', y='count', hue='conservation',
             hue_order = [ "{} times".format(x) for x in range(1, 8) ], 
             palette='Reds', linewidth=2, legend='full')
    plt.xlabel("Shift")
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    plt.ylabel("Average n. overlapping\nPC-HiC interactions")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    fig.savefig(figures_path / "cons_tads_vs_PC-HIC.pdf", bbox_inches='tight', transparent=True)    


    
# --------------------------------------- #

initialize_plotting_parameters()
tads_path = external_data_path / "TAD_definitions"

print("Loading TADs")
all_tads = []
for f in os.listdir(tads_path):
    fpath = os.path.join(tads_path, f)
    print("\t", f)
    tad_df = load_tad_dataset(fpath)
    cell_type = f.split("/")[-1].split("_")[1]
    tad_df['tad_number'] = tad_df.index
    tad_df['cell_type'] = cell_type
    tad_df['tad_uid'] = cell_type + "_" + tad_df.index.map(lambda x: "{:04d}".format(x))
    all_tads.append(tad_df)
all_tads = pd.concat(all_tads, axis=0, ignore_index=True)
all_tads = all_tads.sort_values(coords).reset_index(drop=True)
all_tads['length'] = all_tads.end - all_tads.start
all_tads = all_tads[all_tads.cell_type != 'Neu'].reset_index(drop=True)
all_tads.to_csv(interim_data_path / "all_TADs_by_celltype.tsv", 
                sep="\t", index=False, header=True)


print("Boundary consensus")
window = 25000
all_boundary_positions = get_boundary_positions(all_tads)
w_clusters = cluster_boundary_positions(all_boundary_positions, window)
consensus_bounds = get_consensus_regions(w_clusters)
consensus_bounds.to_csv(interim_data_path / "consensus_boundaries.tsv", 
                        sep="\t", index=False, header=True)

print("TAD consensus")
gaps = get_gaps()
consensus_tads = []
for mc in range(1, 8):
    print(" "*100, end='\r')
    print("\t{}".format(mc), end='\r')
    sel_bounds = consensus_bounds[consensus_bounds.n_cell_types >= mc]
    mc_consensus_tads = get_consensus_tads(sel_bounds, gaps)
    mc_consensus_tads['conservation'] = mc
    consensus_tads.append(mc_consensus_tads)
consensus_tads = pd.concat(consensus_tads, axis=0, ignore_index=True)
consensus_tads.to_csv(interim_data_path / "consensus_tads.tsv", sep="\t", index=False, header=True)
    
consensus_bounds['length'] = consensus_bounds.apply(lambda x: x.end - x.start + 1, axis=1)

chromsizes = load_chromsizes()
chroms = consensus_bounds.chr.unique()
hg19_total_length = 0
for chrom in chroms:
    hg19_total_length += chromsizes[chrom]
consensus_bounds['covered_genome'] = consensus_bounds.length / hg19_total_length
plot_consensus_boundary_properties(consensus_bounds)

print("Conserved boundaries VS GM12878 boundaries")
gm12878_bounds = pd.read_csv(processed_data_path / "GM12878_25kb_1Mb_boundary_strength.bed", sep="\t")
gm12878_bounds.columns = coords + ['bound_strenght', 'cluster_id']
gm12878_bounds = gm12878_bounds.sort_values(coords).reset_index(drop=True)
print("GM12878 bounds:", gm12878_bounds.shape[0])

half_window = 25000
consensus_bounds_fixed = get_region_center(consensus_bounds)
consensus_bounds_fixed = BedTool.from_dataframe(consensus_bounds_fixed)\
                                    .slop(b=half_window, genome='hg19')
consensus_boundaries_fixed_gm12878 = consensus_bounds_fixed\
                                            .map(BedTool.from_dataframe(gm12878_bounds), c=1, o='count')\
                                            .to_dataframe(names=consensus_bounds.columns.tolist() + \
                                                                ['n_gm12878_bounds'])
plot_consensus_boundaries_intersection_with_GM12878(consensus_boundaries_fixed_gm12878)


print("Conserved boundaries VS GM12878 directionality index")

hic_measures = {}

# DI
ins = pd.read_csv(processed_data_path / "di_score_r25kb_w1Mb.txt", index_col=0, sep="\t")
ins['chrom'] = 'chr' + ins.chrom.astype(str)
ins.columns = coords + ['bad', 'di_ratio', 'di_index']
ins.di_ratio = ins.di_ratio.fillna(0)
ins.di_index = ins.di_index.fillna(0)
di = ins[coords + ['di_index']].copy()
hic_measures['di'] = di

# Insulation score
ins = pd.read_csv(processed_data_path / "GM12878_25kb_1Mb_insulation_score.txt", sep="\t")
ins['chrom'] = 'chr' + ins.chrom.astype(str)
ins.columns = coords + ['bad', 'di_index', 'bound_strenght']
ins.di_index = ins.di_index.fillna(0)
ins.bound_strenght = ins.bound_strenght.fillna(0)
di = ins[coords + ['di_index']].copy()
hic_measures['ins'] = di

# TAD compare
ins = pd.read_csv(processed_data_path / "GSE63525_GM12878_25kb_TADCompare_scores.txt", sep="\t")
ins['start'] = ins.start.astype(int)
ins['end'] = ins.start + 25000
ins = ins[['chr', 'start', 'end', 'score']]
ins = ins.rename(columns={'score': 'di_index'})
di = ins.copy()
hic_measures['tad_compare'] = di

hic_measures_names = {
    'di': 'Average Directionality\n Index on GM12878',
    'ins': 'Average Insultation Score\non GM12878',
    'tad_compare': 'Average Boundary Score\non GM12878'
}

for measure, di in hic_measures.items():
    print(measure)
    extended = 500*1000
    window_size = 5*1000
    consensus_bounds_fixed = get_region_center(consensus_bounds)
    consensus_bounds_fixed = BedTool.from_dataframe(consensus_bounds_fixed)\
                                    .slop(b=extended, genome='hg19')\
                                    .to_dataframe(names=consensus_bounds_fixed.columns)
    consensus_bounds_fixed = consensus_bounds_fixed[consensus_bounds_fixed.end - consensus_bounds_fixed.start == extended*2]
    consensus_bounds_fixed['boundary_uid'] = np.arange(consensus_bounds_fixed.shape[0], dtype=int)
    windows = windowing_by_size(consensus_bounds_fixed[coords + ['boundary_uid']], window_size=window_size)
    windows_with_di = coverage_by_window(windows.sort_values(coords), di.sort_values(coords), {'di_index': 'max'})
    windows_with_di = windows_with_di.merge(consensus_bounds_fixed.drop(coords, axis=1), on='boundary_uid')

    aggregations_by_bound = {}
    for nc in sorted(windows_with_di.n_cell_types.unique()):
        print(" "*100, end='\r')
        print("\t{}".format(nc), end='\r')
        lw = windows_with_di[windows_with_di.n_cell_types == nc]
        cagg = lw.pivot_table(index='boundary_uid', columns='w_num', values="di_index").sort_index(axis=1)
        cagg = cagg.sort_index(axis=1)
        aggregations_by_bound[nc] = cagg
        
    plot_cons_bounds_vs_gm12878_DI(aggregations_by_bound, extended, window_size, measure, hic_measures_names[measure])

print("Conserved boundaries VS GRBs")
grbs = pd.read_excel(external_data_path / "BorisLenhard_GRB_CNE_2017NatCom_SupTab1.xlsx", 
                     usecols=[0,1,2], names=coords)
grbs = grbs.sort_values(coords).reset_index(drop=True)
grbs['grb_uid'] = np.arange(grbs.shape[0], dtype=int)
grbs['length'] = grbs.end - grbs.start

extended = (2.5e6)
window_size = 5*1000
centered_grbs = get_region_center(grbs)
centered_grbs = BedTool.from_dataframe(centered_grbs)\
                        .slop(b=extended,genome='hg19')\
                        .to_dataframe(names=centered_grbs.columns)
centered_grbs = centered_grbs[centered_grbs.end - centered_grbs.start == extended*2]
windows = windowing_by_size(centered_grbs[coords + ['grb_uid']], window_size=window_size)

windows_with_bound = coverage_by_window(windows.sort_values(coords), 
                                        consensus_bounds.sort_values(coords), 
                                        {'n_cell_types': 'max'})
windows_with_bound = windows_with_bound.merge(centered_grbs.drop(coords, axis=1), on='grb_uid')
windows_with_grbs = coverage_by_window(windows.sort_values(coords), 
                                       grbs.rename(columns={'grb_uid': 'gi'}).sort_values(coords), 
                                       {"gi": "max"})
windows_with_grbs.loc[windows_with_grbs.gi != windows_with_grbs.grb_uid, 'gi'] = 0

grb_vs_windows = windows_with_bound.pivot_table(index='grb_uid', columns='w_num', values='n_cell_types')
length_vs_windows = windows_with_grbs.pivot_table(index='grb_uid', columns='w_num', values='gi')
plot_cons_bounds_vs_grbs(grb_vs_windows, length_vs_windows, grbs)

print("Conserved boundaries VS PC-HIC")
pchic_interactions = read_PCHiC(external_data_path / "PCHiC_peak_matrix_cutoff5.txt")
shifts = np.linspace(-400000, 400000, 81, dtype=int)

consensus_bounds_centers = get_region_center(consensus_bounds)
regions_n_cross = shift_test(consensus_bounds_centers[coords + ['n_cell_types']]\
                                    .sort_values(coords),
                             pchic_interactions[coords].sort_values(coords), shifts)
regions_n_cross.rename(columns={'n_cell_types': 'conservation'}, inplace=True)
plot_conservation_shift_test(regions_n_cross)
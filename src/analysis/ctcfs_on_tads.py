import sys
sys.path.append("./src/utilities")

from data_utilities import interim_data_path, figures_path, get_gaps, processed_data_path
from bed_utilities import coords, coverage_by_window, windowing_by_size
from plot_utilities import initialize_plotting_parameters, \
                        get_default_figsize, ctcf_colors
import pandas as pd
from pybedtools.bedtool import BedTool
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np

figures_path = figures_path / "analysis"


# ---------- UTILITY FUNCTIONS ---------- #

def aggregate_by_tad(all_TADs_by_celltype,
                     aggregations,
                     other,
                     extension = 0.1,
                     n_windows = 100):
    tot_windows = n_windows + int(n_windows*extension)*2
    tad_start_window = int(n_windows*extension)
    tad_end_window = n_windows + int(n_windows*extension)
    
    regions = all_TADs_by_celltype[coords + ['tad_uid']].copy()
    regions['tad_uid'] = regions.tad_uid.map(lambda x: x.replace("_", "-"))
    windows = BedTool().window_maker(b=BedTool.from_dataframe(regions)\
                                     .slop(l=extension, r=extension, 
                                           pct=True, genome="hg19"), 
                                     n=tot_windows, i='srcwinnum')\
                           .to_dataframe(names=coords + ['window_uid'])
    windows_idxs = windows.window_uid.str.split("_", expand=True)
    windows_idxs.columns = ['tad_uid', 'win_num']
    windows = pd.concat((windows, windows_idxs), axis=1)
    windows['win_num'] = windows['win_num'].astype(int)
    windows = windows.sort_values(coords).reset_index(drop=True)

    windows_with_ctcfs = coverage_by_window(windows, other, aggregations)
    aggregations_by_tad = {}
    for c in aggregations.keys():
        print(" "*100, end='\r')
        print(c, end="\r")
        cagg = windows_with_ctcfs.pivot_table(index='tad_uid', columns='win_num', values=c).sort_index(axis=1)
        cagg = cagg.sort_index(axis=1)
        aggregations_by_tad[c] = cagg 
    return aggregations_by_tad, tad_start_window, tad_end_window

# --------------------------------------- #

# --------- PLOTTING FUNCTIONS ---------- #

def plot_aggregations_by_tad(aggregations_by_tad, 
                             tad_start_window,
                             tad_end_window,
                             path = figures_path / "aggregations_by_tad.pdf"):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(3,1,sharex=True,figsize=(figsize[0]*1.5, figsize[1]*3))
    axes[0].plot(aggregations_by_tad['ctcf_id'].mean(0), label='both',
                 color=ctcf_colors['all'])
    axes[0].set_ylabel("Avg. CTCF sites\nper % of TAD")
    axes[0].axvline(tad_start_window, color='black', linestyle=':')
    axes[0].axvline(tad_end_window, color='black', linestyle=':')
    axes[0].grid()
    axes[0].legend(loc='upper center')

    axes[1].plot(aggregations_by_tad['forward'].mean(0), label='Forward',
                 color=ctcf_colors['forward'])
    axes[1].plot(aggregations_by_tad['reverse'].mean(0), label='Reverse',
                 color=ctcf_colors['reverse'])
    axes[1].set_ylabel("Avg. CTCF sites\nper % of TAD")
    axes[1].axvline(tad_start_window, color='black', linestyle=':')
    axes[1].axvline(tad_end_window, color='black', linestyle=':')
    axes[1].grid()
    axes[1].legend(loc='upper center')

    axes[2].plot(aggregations_by_tad['S'].mean(0), label='Same', 
                 color=ctcf_colors['S'])
    axes[2].plot(aggregations_by_tad['C'].mean(0), label='Convergent',
                 color=ctcf_colors['C'])
    axes[2].plot(aggregations_by_tad['D'].mean(0), label='Divergent', 
                 color=ctcf_colors['D'])
    axes[2].plot(aggregations_by_tad['CD'].mean(0), label='Convergent-Divergent',
                 color=ctcf_colors['CD'])
    axes[2].set_ylabel("Avg. CTCF sites\nper % of TAD")
    axes[2].grid()
    axes[2].legend(loc='upper center')
    axes[2].axvline(tad_start_window, color='black', linestyle=':')
    axes[2].axvline(tad_end_window, color='black', linestyle=':')
    axes[2].set_xticks([tad_start_window, 
                        int(tad_start_window + tad_end_window)/2 ,
                        tad_end_window])
    axes[2].set_xticklabels(['TAD start\n(0%)', 'TAD center\n(50%)', 'TAD end\n(100%)'])
    axes[2].set_xlabel("Position on TAD")
    fig.savefig(path, bbox_inches='tight', transparent=True)
    
def plot_aggregations_by_boundary(aggregations_by_bound_tot,
                                  extended, window_size):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(figsize[0]*1.5, figsize[1]*3))
    axes[0].plot(aggregations_by_bound_tot['ctcf_id'].mean(0), label='both', 
                 color=ctcf_colors['all'])
    axes[0].set_ylabel("Avg. CTCF sites per 5kb")
    axes[0].axvline(extended/window_size, color='black')
    axes[0].grid()
    axes[0].legend(loc='upper left')

    axes[1].plot(aggregations_by_bound_tot['forward'].mean(0), label='Forward', 
                 color=ctcf_colors['forward'])
    axes[1].plot(aggregations_by_bound_tot['reverse'].mean(0), label='Reverse', 
                 color=ctcf_colors['reverse'])
    axes[1].set_ylabel("Avg. CTCF sites per 5kb")
    axes[1].axvline(extended/window_size, color='black')
    axes[1].grid()
    axes[1].legend(loc='upper left')

    axes[2].plot(aggregations_by_bound_tot['S'].mean(0), label='Same', 
                 color=ctcf_colors['S'])
    axes[2].plot(aggregations_by_bound_tot['C'].mean(0), label='Convergent', 
                 color=ctcf_colors['C'])
    axes[2].plot(aggregations_by_bound_tot['D'].mean(0), label='Divergent',
                 color=ctcf_colors['D'])
    axes[2].plot(aggregations_by_bound_tot['CD'].mean(0), label='Convergent-\nDivergent', 
                 color=ctcf_colors['CD'])
    axes[2].set_ylabel("Avg. CTCF sites per 5kb")
    axes[2].axvline(extended/window_size, color='black')
    axes[2].grid()
    axes[2].legend(loc='upper left')

    axes[2].set_xticks([0, extended/window_size, extended*2/window_size])
    axes[2].set_xticklabels(['-250kb', '0', '+250kb'])
    axes[2].set_xlabel("Distance from boundary center")

    fig.savefig(figures_path / "aggregations_by_boundary.pdf", 
                bbox_inches='tight', transparent=True)
    
def plot_aggregations_by_conservation(aggregations_by_bound,
                                      windows_with_ctcf):
    figsize = get_default_figsize()
    n_figs = len(windows_with_ctcf.n_cell_types.unique())
    fig, axes = plt.subplots(3,n_figs, 
                             sharex='col', sharey='row', 
                             tight_layout=False,
                             figsize=(figsize[0]*n_figs, figsize[1]*3))
    for level in sorted(windows_with_ctcf.n_cell_types.unique()):
        axes[0, level - 1].plot(aggregations_by_bound[(level,'ctcf_id')].mean(0), 
                                label='both', color=ctcf_colors['all'])
        if level == 1:
            axes[0, level - 1].legend(loc='upper left')
            axes[0, level - 1].set_ylabel("Avg. CTCF sites per 5kb")
        else:
            axes[0, level - 1].legend().set_visible(False)
        axes[0, level - 1].grid()
        axes[0, level - 1].set_title("$s = {}$".format(level), 
                                     fontweight="bold", fontsize='xx-large')
        axes[0, level - 1].tick_params(axis='y')

        axes[1, level - 1].plot(aggregations_by_bound[(level,'forward')].mean(0), 
                                label='Forward', color=ctcf_colors['forward'])
        axes[1, level - 1].plot(aggregations_by_bound[(level,'reverse')].mean(0), 
                                label='Reverse', color=ctcf_colors['reverse'])
        axes[1, level - 1].tick_params(axis='y')
        if level == 1:
            axes[1, level - 1].legend(loc='upper left')
            axes[1, level - 1].set_ylabel("Avg. CTCF sites per 5kb")
        else:
            axes[1, level - 1].legend().set_visible(False)
        axes[1, level - 1].grid()

        axes[2, level - 1].plot(aggregations_by_bound[(level,'S')].mean(0), 
                                label='Same', color=ctcf_colors['S'])
        axes[2, level - 1].plot(aggregations_by_bound[(level,'C')].mean(0), 
                                label='Convergent', color=ctcf_colors['C'])
        axes[2, level - 1].plot(aggregations_by_bound[(level,'D')].mean(0),
                                label='Divergent', color=ctcf_colors['D'])
        axes[2, level - 1].plot(aggregations_by_bound[(level,'CD')].mean(0), 
                                label='Convergent-Divergent', color=ctcf_colors['CD'])
        if level == 1:
            axes[2, level - 1].legend(loc='upper left')
            axes[2, level - 1].set_ylabel("Avg. CTCF sites per 5kb")
        else:
            axes[2, level - 1].legend().set_visible(False)
        axes[2, level - 1].grid()
        axes[2, level - 1].tick_params(axis='y')
        axes[2, level - 1].set_xticks([0, extended/window_size, extended*2/window_size])
        axes[2, level - 1].set_xticklabels(['-250kb', '0', '+250kb'])
        axes[2, level - 1].set_xlabel("Distance from\nboundary center (kb)")
    fig.savefig(figures_path / "aggregations_by_conservation.pdf",
                bbox_inches='tight', transparent=True)
    
def plot_distance_from_nearest_CTCF_site(bounds_with_ctcf, order, path):
    fig = plt.figure()
    sns.boxplot(data=bounds_with_ctcf, palette="Reds", order=order,
                x='n_cell_types', y='distance', showfliers=False)
    plt.xlabel("Boundary conservation (s)")
    plt.ylabel("Distance from nearest CTCF site")
    plt.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
    plt.grid(axis='y')
    fig.savefig(path,
                bbox_inches='tight', transparent=True)

def plot_n_ctcf_sites_in_boundaries(b):

    def __to_bin(x):
        if x in [0,1,2]:
            return str(x)
        elif x in [3,4]:
            return "3-4"
        elif x in range(5,9):
            return "5-8"
        elif x in range(9,12):
            return "9-11"

    bounds_with_ctcf_count = b.copy()
    bounds_with_ctcf_count['cat'] = bounds_with_ctcf_count.n_ctcfs.map(__to_bin)
    bounds_with_ctcf_count = pd.concat((bounds_with_ctcf_count,
                                        pd.get_dummies(bounds_with_ctcf_count.cat)),
                                       axis=1)
    groups = bounds_with_ctcf_count.groupby('n_cell_types')[['0', '1', '2', '3-4', 
                                                             '5-8', '9-11']].sum()
    cmap = plt.cm.get_cmap('Blues', 8)
    newcolors = cmap(np.linspace(0,1,8))
    blue = np.array([1,1,1, 1])
    newcolors[0, :] = blue
    newcmp = ListedColormap(newcolors)

    figsize = get_default_figsize()
    fig, axes = plt.subplots(1, 7, figsize=(figsize[0]*7, figsize[1]*2))
    for i, g in enumerate(range(1,8)):
        xg = groups.loc[[g], ['1', '2', '3-4', '5-8', '9-11']]
        xg.loc['0', '0'] = groups.loc[g, '0']
        xg = xg.fillna(0)
        xg.loc[g, 'not_intersect'] = 0
        xg = xg.loc[['0', g], ['0', '1', '2', '3-4', '5-8', '9-11']]
        xg.plot.bar(stacked=True, cmap=newcmp, ax=axes[i],edgecolor='black')
        axes[i].legend().set_visible(False)
        axes[i].set_xticklabels([])
        axes[i].grid(axis="y")
        axes[i].tick_params(axis='y')
        axes[i].set_xlabel("")
        axes[i].set_title("$s = {}$".format(g), fontweight="bold", fontsize='xx-large')
    axes[0].set_ylabel("Boundaries", fontsize='xx-large')
    handles, labels = axes[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, 
               bbox_to_anchor=(0.43,1.05), 
               title='CTCF binding sites')
    fig.text(x=0.53, y=-0.01, horizontalalignment = 'center',
             fontsize='xx-large',
             s="Boundary conservation score ($s$)")
    fig.savefig(figures_path / "n_ctcf_sites_in_boundaries_by_conservation.pdf",
                bbox_inches='tight', transparent=True)

# --------------------------------------- #

initialize_plotting_parameters()
all_TADs_by_celltype = pd.read_csv(interim_data_path / "all_TADs_by_celltype.tsv",
                                   sep="\t")

ctcfs = pd.read_csv(interim_data_path / "ctcfs_with_context.tsv", sep="\t")
ctcfs = pd.concat((ctcfs, 
                   pd.get_dummies(ctcfs[['orientation','context']], 
                                  prefix="", prefix_sep="")), 
                  axis=1)
ctcfs = ctcfs.rename(columns={'>': 'forward', '<': 'reverse'})
ctcfs = ctcfs.merge(pd.read_csv(interim_data_path / "ctcf_scores.tsv", sep="\t"),
                    on=coords + ['orientation'])

print("Aggregate CTCF on TAD positions")
aggregations = {'ctcf_id': 'count', 
                'forward': 'sum', 
                'reverse': 'sum',
                'S': 'sum',
                'CD': 'sum', 
                'D': 'sum',
                'C': 'sum'}

extension = 0.1
n_windows = 100

print("Aggregate CTCF on Consensus TADs positions")
min_conservation = 2
conserved_tads = pd.read_csv(interim_data_path / "consensus_tads.tsv", sep="\t")
conserved_tads['tad_uid'] = conserved_tads.index.astype(str)
conserved_tads = conserved_tads[conserved_tads.conservation >= min_conservation]

aggregations_by_tad_cons, tad_start_window, tad_end_window = aggregate_by_tad(
                                        conserved_tads,
                                        aggregations,
                                        ctcfs,
                                        extension = extension,
                                        n_windows = n_windows)
plot_aggregations_by_tad(aggregations_by_tad_cons,
                         tad_start_window, 
                         tad_end_window,
                         path = figures_path / "aggregations_by_tad_consensus.pdf")

print("Aggregate CTCF at consensus boundaries")
consensus_boundaries = pd.read_csv(interim_data_path / "consensus_boundaries.tsv", 
                                   sep = "\t")
consensus_boundaries['boundary_uid'] = consensus_boundaries.index

extended = 250*1000
window_size = 5*1000

centered_boundaries = consensus_boundaries.copy()
centers = ((centered_boundaries.start + centered_boundaries.end)/2).astype(int)
centered_boundaries['start'] = centers
centered_boundaries['end'] = centers

centered_boundaries = BedTool.from_dataframe(centered_boundaries)\
                             .slop(b=extended, genome='hg19')\
                             .to_dataframe(names=centered_boundaries.columns)

centered_boundaries = centered_boundaries[centered_boundaries.end - centered_boundaries.start == extended*2]
windows = windowing_by_size(centered_boundaries[coords + ['boundary_uid']],
                            window_size=window_size)

windows_with_ctcf = coverage_by_window(windows, ctcfs, aggregations)
windows_with_ctcf = windows_with_ctcf.merge(consensus_boundaries.drop(coords, axis=1),
                                            on='boundary_uid')

aggregations_by_bound = {}
for nc in sorted(windows_with_ctcf.n_cell_types.unique()):
    print(" "*100, end='\r')
    print("\t{}".format(nc), end='\r')
    lw = windows_with_ctcf[windows_with_ctcf.n_cell_types == nc]
    for c in aggregations.keys():
        cagg = lw.pivot_table(index='boundary_uid', 
                              columns='w_num', 
                              values=c).sort_index(axis=1)
        cagg = cagg.sort_index(axis=1)
        aggregations_by_bound[(nc, c)] = cagg

plot_aggregations_by_conservation(aggregations_by_bound,
                                      windows_with_ctcf)

print("Count CTCF sites on boundaries")
consensus_boundaries_noY = consensus_boundaries[consensus_boundaries.chr != 'chrY']

centered_boundaries = consensus_boundaries_noY.copy()
centers = ((centered_boundaries.start + centered_boundaries.end)/2).astype(int)
centered_boundaries['start'] = centers
centered_boundaries['end'] = centers

bounds_with_ctcf = BedTool.from_dataframe(centered_boundaries)\
                            .closest(BedTool.from_dataframe(ctcfs), d=True, t='first')\
                            .to_dataframe(names=consensus_boundaries_noY.columns.tolist() + \
                                          ctcfs.columns.map(lambda x: "ctcf_" + x).tolist() +\
                                          ['distance'])

plot_distance_from_nearest_CTCF_site(bounds_with_ctcf, 
                                     figures_path / "bound_distance_from_nearest_CTCF_site.pdf")

shuffled_boundaries = BedTool.from_dataframe(
                                centered_boundaries[coords + ['n_cell_types']])\
                             .sort()\
                             .shuffle(excl=BedTool.from_dataframe(get_gaps()).fn,
                                      chrom=True, genome='hg19')\
                             .sort()

shuffled_boundaries_with_ctcf = shuffled_boundaries\
                            .closest(BedTool.from_dataframe(ctcfs), d=True, t='first')\
                            .to_dataframe(names=coords + ['n_cell_types'] + \
                                          ctcfs.columns.map(lambda x: "ctcf_" + x).tolist() +\
                                          ['distance'])
plot_distance_from_nearest_CTCF_site(shuffled_boundaries_with_ctcf, 
                                     figures_path / "shuffle_bound_distance_from_nearest_CTCF_site.pdf")

bounds_with_ctcf_with_random = pd.concat(
    (bounds_with_ctcf[coords + ['n_cell_types', 'distance']],
    shuffled_boundaries_with_ctcf[coords + ['n_cell_types', 'distance']]\
            .assign(n_cell_types='random')), axis=0, ignore_index=True,
).sort_values(coords).reset_index(drop=True)

plot_distance_from_nearest_CTCF_site(bounds_with_ctcf_with_random, 
                                     ['random'] + [x for x in range(1, 8)],
                                     figures_path / "bound_distance_from_nearest_CTCF_site_with_random.pdf")

max_distance = 25000
bounds_with_ctcf_count = BedTool.from_dataframe(consensus_boundaries_noY).sort()\
                                .slop(b=max_distance, genome='hg19')\
                                .map(BedTool.from_dataframe(ctcfs), c=4, o='count')\
                        .to_dataframe(names=consensus_boundaries_noY.columns.tolist() + \
                                  ['n_ctcfs'])
plot_n_ctcf_sites_in_boundaries(bounds_with_ctcf_count)

import sys
sys.path.append("./src/utilities")

from data_utilities import get_gm12878_di_index, figures_path, interim_data_path, external_data_path, get_region_center, processed_data_path
from bed_utilities import coords, coverage_by_window, windowing_by_size
from plot_utilities import initialize_plotting_parameters, get_default_figsize, ctcf_colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
from pybedtools.bedtool import BedTool

figures_path = figures_path / "analysis"

# ---------- UTILITY FUNCTIONS ---------- #

    
def get_tads(ins_with_neigh):
    tads = ins_with_neigh.loc[ins_with_neigh.point_of_interest != 'middle', coords + ['point_of_interest']].copy()
    tads_after = tads.shift(-1, fill_value=-1)
    tads['point_of_interest_after'] = tads_after.point_of_interest
    tads['chr_after'] = tads_after.chr
    tads['start_after'] = tads_after.start
    tads['end_after'] = tads_after.end

    tads_before = tads.shift(1, fill_value=-1)
    tads['point_of_interest_before'] = tads_before.point_of_interest
    tads['chr_before'] = tads_before.chr
    tads['start_before'] = tads_before.start
    tads['end_before'] = tads_before.end
    tads = tads[(tads.point_of_interest_before == 'boundary') & \
                (tads.point_of_interest == 'tad_center') & \
                (tads.point_of_interest_after == 'boundary') & \
                (tads.chr == tads.chr_after) & (tads.chr == tads.chr_before)]
    tads['center_start'] = tads.start
    tads['center_end'] = tads.end
    tads['start'] = tads.start_before
    tads['end'] = tads.end_after
    tads = tads[coords + ['center_start', 'center_end']]
    tads['length'] = tads.end - tads.start
    tads['center_shift'] = (tads.center_start - tads.start) / tads.length
    return tads

def get_triangles(ins_with_neigh):
    ins_with_neigh_points = ins_with_neigh[ins_with_neigh.point_of_interest != 'middle'].sort_values(coords).copy()
    ins_with_neigh_points['point_after'] = ins_with_neigh_points.shift(-1).point_of_interest
    ins_with_neigh_points['chr_after'] = ins_with_neigh_points.shift(-1).chr
    ins_with_neigh_points['start_after'] = ins_with_neigh_points.shift(-1).start
    ins_with_neigh_points['end_after'] = ins_with_neigh_points.shift(-1).end
    ins_with_neigh_points['di_ratio_after'] = ins_with_neigh_points.shift(-1).di_ratio
    ins_with_neigh_points[ins_with_neigh_points.chr == ins_with_neigh_points.chr_after]
    
    pos_triangles = ins_with_neigh_points[(ins_with_neigh_points.point_of_interest == 'boundary') & (ins_with_neigh_points.point_after == 'tad_center')]
    pos_triangles = pd.DataFrame({
        'chr': pos_triangles.chr,
        'start': pos_triangles.start.astype(int),
        'end': pos_triangles.end_after.astype(int),
        'triangle_type': 'boundary to center',
        'side': 'positive'
    })

    neg_triangles = ins_with_neigh_points[(ins_with_neigh_points.point_of_interest == 'tad_center') & (ins_with_neigh_points.point_after == 'boundary')]
    neg_triangles = pd.DataFrame({
        'chr': neg_triangles.chr,
        'start': neg_triangles.start.astype(int),
        'end': neg_triangles.end_after.astype(int),
        'triangle_type': 'center to boundary',
        'side': 'negative'
    })

    pos_triangles['length'] = pos_triangles.end - pos_triangles.start
    pos_triangles['triangle_uid'] = np.arange(pos_triangles.shape[0], dtype=int)

    neg_triangles['length'] = neg_triangles.end - neg_triangles.start
    neg_triangles['triangle_uid'] = np.arange(neg_triangles.shape[0], dtype=int)
    triangles = pd.concat((pos_triangles, neg_triangles), axis=0, ignore_index=True)
    triangles = triangles[triangles.start <= triangles.end]
    triangles = triangles.sort_values(coords).reset_index(drop=True)
    triangles['triangle_uid'] = triangles.index
    return triangles

def enrichment(triangles_in, ctcfs, aggregations, 
               centers='start', 
               extended=1000*1000, window_size=10*1000, 
               value_function = lambda x: x.forward if x.forward > x.reverse \
                                                         else -x.reverse):
    triangles = triangles_in.copy()
    
    centers = triangles[centers]
    centered_triangles = triangles.copy()
    centered_triangles['start'] = centers
    centered_triangles['end'] = centers

    centered_triangles = BedTool.from_dataframe(centered_triangles)\
                                .slop(b=extended, genome='hg19')\
                                .to_dataframe(names=centered_triangles.columns)
    
    centered_triangles = centered_triangles[centered_triangles.end - centered_triangles.start == extended*2]
    windows = windowing_by_size(centered_triangles[coords + ['triangle_uid']], 
                                window_size=window_size)
    
    windows_with_ctcf = coverage_by_window(windows.sort_values(coords), 
                                           ctcfs.sort_values(coords), 
                                           aggregations)
    windows_with_ctcf = windows_with_ctcf.merge(centered_triangles.drop(coords, axis=1), 
                                                on='triangle_uid')
    windows_with_ctcf['value'] = windows_with_ctcf.apply(value_function, axis=1)
    triangles_vs_ctcfs = windows_with_ctcf.pivot_table(index='triangle_uid', 
                                                       columns='w_num', values='value')
    return triangles_vs_ctcfs

def get_epi_features(conserved_tads, gr_peaks, epigenetics,
                     id_name = 'triangle_uid'):
    tads_with_gr = BedTool.from_dataframe(conserved_tads).sort()\
                          .intersect(BedTool.from_dataframe(gr_peaks).sort(), wa=True, wb=True)\
                          .to_dataframe(names=conserved_tads.columns.map(lambda x: 'TAD_' + x).tolist() +\
                                        gr_peaks.columns.map(lambda x: 'GR_' + x).tolist())
    tads_with_gr = tads_with_gr.groupby('TAD_{}'.format(id_name))['GR_peak_id'].count()\
                        .reindex(conserved_tads[id_name].values)\
                        .fillna(0).astype(int).to_frame('n_GR_peaks')
    tads_with_gr.index.name = id_name


    tads_with_epi = BedTool.from_dataframe(conserved_tads)\
                           .map(BedTool.from_dataframe(epigenetics[coords].sort_values(coords)), 
                                c=3, o='count', null=0)\
                            .to_dataframe(names=conserved_tads.columns.tolist() + ['all_epi'])
    tads_with_epi = BedTool.from_dataframe(tads_with_epi)\
                           .map(BedTool.from_dataframe(epigenetic_marks), 
                                c=[7,8,9,10], o=['sum', 'sum', 'sum', 'sum'], null=0)\
                           .to_dataframe(names=tads_with_epi.columns.tolist() + \
                                         epigenetic_marks.columns[6:].tolist())
    tads_with_epi = tads_with_epi.set_index(id_name)\
                                 .drop(coords + ['triangle_type', 'side', 'length'], axis=1)
    tads_with_epi['epi_TA_UP'] = tads_with_epi.TA_UP_h3k27ac + tads_with_epi.TA_UP_h3k4me3
    tads_with_epi['epi_TA_DOWN'] = tads_with_epi.TA_DOWN_h3k27ac + tads_with_epi.TA_DOWN_h3k4me3

    tads_with_features = conserved_tads.merge(tads_with_epi, left_on=id_name,
                                              right_index=True)
    tads_with_features = tads_with_features.merge(tads_with_gr,
                                                  left_index=True, right_index=True)
    return tads_with_features

# --------------------------------------- #

# --------- PLOTTING FUNCTIONS ---------- #


def plot_n_ctcf_category_vs_genomic_region(ctcfs_with_histones_pois):
    fig = plt.figure()
    g = sns.countplot(data=ctcfs_with_histones_pois, 
                      x='gm12878_feature', hue='context', 
                      hue_order=['S', 'C', 'D', 'CD'],
                      palette=ctcf_colors,
                      order=['boundary', 'middle', 'tad_center'])
    g.set_yscale('log')
    g.grid(axis='y')
    g.set_xlabel("Genomic region")
    g.set_ylabel("Number of CTCF sites")
    fig.savefig(figures_path / "n_ctcf_category_vs_genomic_region.pdf", bbox_inches='tight', transparent=True)

    
def plot_perc_regions_bearing_CTCF_site_cat(pois_ext_ctcf, 
                                            catname='point_of_interest', 
                                            path=figures_path / "perc_regions_bearing_CTCF_site_cat.pdf"):
    figsize = get_default_figsize()
    fig = plt.figure(figsize=(figsize[0]*1.5, figsize[1]))
    sns.barplot(data=pois_ext_ctcf.groupby(catname)[['all', 'S', 'CD', 'D', 'C']]\
                                  .mean().reset_index()\
                                  .melt(id_vars=catname), 
                x=catname, y='value', hue='variable', palette=ctcf_colors)
    plt.grid(axis='y')
    plt.ylim(0, 1)
    yticks, _ = plt.yticks()
    plt.yticks(yticks, ["{:.0f}%".format(yi*100) for yi in yticks], rotation=0)
    plt.xlabel("TAD position")
    plt.ylabel("Percentage of regions\nbearing a CTCF site")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(path, bbox_inches='tight', transparent=True)
    
def plot_triangle_enrichment(pos_triangles_vs_ctcfs, 
                             pos_triangles,
                             neg_triangles_vs_ctcfs, 
                             neg_triangles):
    figsize=get_default_figsize()
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0]*1.5, figsize[1]*2))
    X_pos = pos_triangles_vs_ctcfs.reindex(index = pos_triangles\
                                           .sort_values('length', ascending=False)\
                                  .triangle_uid.tolist()).dropna().values
    X_pos[X_pos < 0] = -1
    X_pos[X_pos > 0] = 1

    axes[0].matshow(X_pos, aspect='auto',cmap='seismic_r', interpolation='bilinear')
    axes[0].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xticklabels([])

    X_neg = neg_triangles_vs_ctcfs.reindex(index = neg_triangles\
                                    .sort_values('length', ascending=False)\
                                           .triangle_uid.tolist())\
                                    .dropna().values
    X_neg[X_neg < 0] = -1
    X_neg[X_neg > 0] = 1

    axes[1].matshow(X_neg, aspect='auto',cmap='seismic_r', interpolation='bilinear')
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, X_pos.shape[1]/2, X_pos.shape[1]])
    axes[1].set_xticklabels(['-1Mb', '0', '+1Mb'])

    plt.xlabel("Distance from (+) to (-) inversion point")
    axes[1].xaxis.set_ticks_position('bottom')
    fig.savefig(figures_path / "triangle_enrichment.pdf", 
                bbox_inches='tight', transparent=True)

def plot_triangle_silouette(pos_triangles_vs_ctcfs, 
                             pos_triangles,
                             neg_triangles_vs_ctcfs, 
                             neg_triangles,
                             path):
    figsize=get_default_figsize()
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0]*1.5, figsize[1]*2))
    X_pos = pos_triangles_vs_ctcfs.reindex(index = pos_triangles\
                                           .sort_values('length', ascending=False)\
                                  .triangle_uid.tolist()).dropna().values
    X_pos[X_pos < 0] = -1
    X_pos[X_pos > 0] = 1

    axes[0].matshow(X_pos, aspect='auto',cmap='seismic_r', interpolation='bilinear')
    axes[0].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xticklabels([])

    X_neg = neg_triangles_vs_ctcfs.reindex(index = neg_triangles\
                                    .sort_values('length', ascending=False)\
                                           .triangle_uid.tolist())\
                                    .dropna().values
    X_neg[X_neg < 0] = -1
    X_neg[X_neg > 0] = 1

    axes[1].matshow(X_neg, aspect='auto',cmap='seismic_r', interpolation='bilinear')
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, X_pos.shape[1]/2, X_pos.shape[1]])
    axes[1].set_xticklabels(['-1Mb', '0', '+1Mb'])

    plt.xlabel("Distance from (+) to (-) inversion point")
    axes[1].xaxis.set_ticks_position('bottom')
    fig.savefig(figures_path / "triangle_enrichment.pdf", 
                bbox_inches='tight', transparent=True)
    
def plot_triangle_enrichment_horizontal(pos_triangles_vs_ctcfs, 
                                         pos_triangles,
                                         neg_triangles_vs_ctcfs, 
                                         neg_triangles,
                                         path = figures_path / "triangle_enrichment.pdf",
                                         cmap = 'Greens'):
    figsize=get_default_figsize()
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]*1))
    X_pos = pos_triangles_vs_ctcfs.reindex(index = pos_triangles\
                                           .sort_values('length', ascending=False)\
                                  .triangle_uid.tolist()).dropna().values
    X_pos[X_pos < 0] = -1
    X_pos[X_pos > 0] = 1

    axes[0].matshow(X_pos, aspect='auto',cmap=cmap, interpolation='bilinear')
    axes[0].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_xticks([0, X_pos.shape[1]//4, X_pos.shape[1]//2, X_pos.shape[1]*3//4, X_pos.shape[1]])
    axes[0].set_xticklabels(['-1 Mb', '-500 Kb', '0', '+500 Kb', '+1 Mb'])
    axes[0].xaxis.set_ticks_position('bottom')
    axes[0].set_xlabel("Distance from\nnegative inversion point")
    axes[0].set_yticks([])
    axes[0].set_ylabel("GM12878 TADs ordered by\ntheir Positive DI region length")
    
    X_neg = neg_triangles_vs_ctcfs.reindex(index = neg_triangles\
                                    .sort_values('length', ascending=False)\
                                           .triangle_uid.tolist())\
                                    .dropna().values
    X_neg[X_neg < 0] = -1
    X_neg[X_neg > 0] = 1

    axes[1].matshow(X_neg, aspect='auto',cmap=cmap, interpolation='bilinear')
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, X_pos.shape[1]//4, X_pos.shape[1]//2, X_pos.shape[1]*3//4, X_pos.shape[1]])
    axes[1].set_xticklabels(['-1 Mb', '-500 Kb', '0', '+500 Kb', '+1 Mb'])

    axes[1].set_xlabel("Distance from\nnegative inversion point")
    axes[1].xaxis.set_ticks_position('bottom')
    axes[1].set_yticks([])
    axes[1].set_ylabel("GM12878 TADs ordered by\ntheir Negative DI region length", rotation=270, labelpad=27)
    axes[1].yaxis.set_label_position("right")
    
    fig.savefig(path, 
                bbox_inches='tight', transparent=True)

def plot_triangle_enrichment_halves(pos_triangles_vs_ctcfs, 
                                     pos_triangles,
                                     neg_triangles_vs_ctcfs, 
                                     neg_triangles,
                                     path = figures_path / "triangle_enrichment.pdf",
                                     cmap = 'Greens'):
    figsize=get_default_figsize()
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]*1))
    X_pos = pos_triangles_vs_ctcfs.reindex(index = pos_triangles\
                                           .sort_values('length', ascending=False)\
                                  .triangle_uid.tolist()).dropna().values
    X_pos[X_pos < 0] = -1
    X_pos[X_pos > 0] = 1

    half = X_pos.shape[1]//2
    X_pos = X_pos[:, :half - 1]

    axes[0].matshow(X_pos, aspect='auto',cmap=cmap, interpolation='bilinear')
    axes[0].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_xticks([0, X_pos.shape[1]//2, X_pos.shape[1] - 1])
    axes[0].set_xticklabels(['-1Mb', '-500Kb', '0'])
    axes[0].xaxis.set_ticks_position('bottom')
    axes[0].set_xlabel("Distance from\nnegative inversion point")
    axes[0].set_title("Positive DI regions")
    
    X_neg = neg_triangles_vs_ctcfs.reindex(index = neg_triangles\
                                    .sort_values('length', ascending=False)\
                                           .triangle_uid.tolist())\
                                    .dropna().values
    X_neg[X_neg < 0] = -1
    X_neg[X_neg > 0] = 1

    half = X_neg.shape[1]//2
    X_neg = X_neg[:, half - 2:]

    axes[1].matshow(X_neg, aspect='auto',cmap=cmap, interpolation='bilinear')
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, X_neg.shape[1]//2, X_neg.shape[1] - 1])
    axes[1].set_xticklabels(['0', '+500Kb', '+1Mb'])

    axes[1].set_xlabel("Distance from\nnegative inversion point")
    axes[1].set_title("Negative DI regions")
    axes[1].xaxis.set_ticks_position('bottom')
    fig.savefig(path, 
                bbox_inches='tight', transparent=True)

    
def plot_aggregations_by_tad_center(aggregations_by_tad_center_tot,
                                   extended, window_size):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(3,1,sharex=True,figsize=(figsize[0]*1.5, figsize[1]*3))
    axes[0].plot(aggregations_by_tad_center_tot['ctcf_id'].mean(0), label='both',
                 color=ctcf_colors['all'])
    axes[0].set_ylabel("Avg. CTCF sites per 5kb")
    axes[0].axvline(extended/window_size, color='black')
    axes[0].grid()
    axes[0].legend(loc='lower left')

    axes[1].plot(aggregations_by_tad_center_tot['forward'].mean(0), label='Forward',
                 color=ctcf_colors['forward'])
    axes[1].plot(aggregations_by_tad_center_tot['reverse'].mean(0), label='Reverse',
                 color=ctcf_colors['reverse'])
    axes[1].set_ylabel("Avg. CTCF sites per 5kb")
    axes[1].axvline(extended/window_size, color='black')
    axes[1].grid()
    axes[1].legend(loc='lower left')

    axes[2].plot(aggregations_by_tad_center_tot['S'].mean(0), label='Same', 
                 color=ctcf_colors['S'])
    axes[2].plot(aggregations_by_tad_center_tot['C'].mean(0), label='Convergent',
                 color=ctcf_colors['C'])
    axes[2].plot(aggregations_by_tad_center_tot['D'].mean(0), label='Divergent',
                 color=ctcf_colors['D'])
    axes[2].plot(aggregations_by_tad_center_tot['CD'].mean(0), label='Convergent-\nDivergent',
                 color=ctcf_colors['CD'])
    axes[2].set_ylabel("Avg. CTCF sites per 5kb")
    axes[2].axvline(extended/window_size, color='black')
    axes[2].grid()
    axes[2].legend(loc='lower left')

    axes[2].set_xticks([0, extended/window_size, extended*2/window_size])
    axes[2].set_xticklabels(['-250kb', '0', '+250kb'])
    axes[2].set_xlabel("Distance from negative inversion point")
    fig.savefig(figures_path / "aggregations_by_tad_center.pdf", 
                bbox_inches='tight', transparent=True)

    
# --------------------------------------- #


initialize_plotting_parameters()

print("Loading DI index on GM12878")
ins = get_gm12878_di_index()
ins_noY = ins[~ins.chr.isin(['chrY', 'chrMT'])]

print("Classifying DI points")
ins_with_neigh = []
for chrom in ins_noY.chr.unique():
    ins_chrom = ins_noY[ins_noY.chr == chrom].sort_values(coords)
    ins_chrom['di_ratio_before'] = ins_chrom.shift(1)['di_ratio']
    ins_chrom['di_ratio_after'] = ins_chrom.shift(-1)['di_ratio']  
    ins_with_neigh.append(ins_chrom)
ins_with_neigh = pd.concat(ins_with_neigh, axis=0, ignore_index=True)

ins_with_neigh['DI_point_type'] = 'middle'
ins_with_neigh.loc[(ins_with_neigh.di_ratio_before > 0) & (ins_with_neigh.di_ratio < 0), 'DI_point_type'] = 'negative_inversion'
ins_with_neigh.loc[(ins_with_neigh.di_ratio_before < 0) & (ins_with_neigh.di_ratio > 0), 'DI_point_type'] = 'positive_inversion'

ins_with_neigh['point_of_interest'] = 'middle'
ins_with_neigh.loc[ins_with_neigh.DI_point_type == 'negative_inversion', 'point_of_interest'] = 'tad_center'
ins_with_neigh.loc[ins_with_neigh.DI_point_type == 'positive_inversion', 'point_of_interest'] = 'boundary'


print("Calling TADs from DI index")

tads = get_tads(ins_with_neigh)
tads['left_size'] = (tads.center_start + tads.center_end)//2 - tads.start
tads['right_size'] = tads.end - (tads.center_start + tads.center_end)//2


fig = plt.figure()
t = tads[(tads.center_shift > 0) & (tads.center_shift < 1) & (tads.length >= 1e5) & (tads.length <=3e6)]
ax = sns.kdeplot(t.left_size.map(np.log10), t.right_size.map(np.log10), shade=True)
ax.collections[0].set_alpha(0)
plt.xticks([4.5, 5, 5.5, 6], labels=['30 Kb', '100 Kb', '300 Kb', '1 Mb'])
plt.yticks([4.5, 5, 5.5, 6], labels=['30 Kb', '100 Kb', '300 Kb', '1 Mb'])
plt.xlabel("Length of positive DI region")
plt.ylabel("Length of negative DI region")
# sns.regplot(t.left_size.map(np.log10), t.right_size.map(np.log10), scatter=False, lowess=False)
fig.savefig(figures_path / "positive_vs_negative_DI_regions_lengths.pdf", 
                bbox_inches='tight', transparent=True)
plt.close(fig)


# print("Loading Insultation score on GM12878")
# ins_score = pd.read_csv(processed_data_path / "GM12878_25kb_1Mb_insulation_score.txt", sep="\t")
# ins_score['chrom'] = 'chr' + ins_score.chrom.astype(str)
# ins_score.columns = coords + ['bad', 'ins_score', 'bound_strenght']
# ins_score.ins_score = ins_score.ins_score.fillna(0)
# ins_score.bound_strenght = ins_score.bound_strenght.fillna(0)
# ins_score = ins_score[coords + ['ins_score', 'bound_strenght']].copy()

# ins_score['feature'] = "NONE"
# ins_score.loc[ins_score.bound_strenght > 0, 'feature'] = 'boundary'
# ins_score_bouns = ins_score[ins_score.feature == 'boundary']

# bound_to_bound = BedTool.from_dataframe(ins_score_bouns)\
#                         .sort()\
#                         .closest(BedTool.from_dataframe(ins_score_bouns).sort(), io=True, iu=True, D='ref')\
#                         .to_dataframe(names = ins_score_bouns.columns.map(lambda x: "left_" + x).tolist() + ins_score_bouns.columns.map(lambda x: "right_" + x).tolist() + ['dist'])

# plt.figure()
# plt.plot(ins_score.ins_score.iloc[1000:1700])
# plt.plot(ins_score.bound_strenght.iloc[1000:1700])
# plt.show()

# plt.figure()
# y = ins_score.bound_strenght.values
# y = y[y != 0]
# plt.hist(y, bins=100)

# plt.figure()
# res = cumfreq(y, numbins=100)
# x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
# plt.bar(x, res.cumcount, width=res.binsize)
# plt.show()





print("Mapping CTCF sites")
ctcfs = pd.read_csv(interim_data_path / "ctcfs_with_context.tsv", sep="\t")
ctcfs = pd.concat((ctcfs, pd.get_dummies(ctcfs[['orientation','context']], prefix="", prefix_sep="")), axis=1)
ctcfs = ctcfs.rename(columns={'>': 'forward', '<': 'reverse'})
ctcfs['ctcf_id'] = ctcfs.index

# pois = ins_with_neigh[coords + ['point_of_interest']].copy()
# pois['start'] = pois.start.map(lambda x: max(x - 12500, 0))
# pois['end'] = pois.end.map(lambda x: max(x - 12500, 0))

# extended = 25*1000
# pois_ext = BedTool.from_dataframe(pois).slop(b=extended, genome='hg19')\
#                                         .to_dataframe(names=pois.columns)
# aggregations = {'ctcf_id': 'count', 
#                 'forward': 'sum', 
#                 'reverse': 'sum',
#                 'S': 'sum',
#                 'CD': 'sum', 
#                 'D': 'sum',
#                 'C': 'sum'}
# pois_ext_ctcf = coverage_by_window(pois_ext.sort_values(coords), 
#                                    ctcfs.sort_values(coords), 
#                                    aggregations)
# pois_ext_ctcf.iloc[:, 4:] = pois_ext_ctcf.iloc[:, 4:].applymap(lambda x: min(x, 1))
# pois_ext_ctcf = pois_ext_ctcf.rename(columns={'ctcf_id': 'all'})
# pois_ext_ctcf['point_of_interest'] = pois_ext_ctcf.point_of_interest.replace('middle', 'inner')
# plot_perc_regions_bearing_CTCF_site_cat(pois_ext_ctcf)

# tad_centers = tads[['chr','center_start', 'center_end', 'shift_category']].copy()
# tad_centers.rename(columns={'center_start': 'start', 'center_end': 'end'}, inplace=True)
# tad_centers['start'] = tad_centers.start.map(lambda x: max(x - 12500, 0))
# tad_centers['end'] = tad_centers.end.map(lambda x: max(x - 12500, 0))

# tad_centers_ext = BedTool.from_dataframe(tad_centers).slop(b=extended, genome='hg19')\
#                                         .to_dataframe(names=tad_centers.columns)
# tad_centers_ext_ctcf = coverage_by_window(tad_centers_ext.sort_values(coords), 
#                                           ctcfs.sort_values(coords), 
#                                           aggregations)
# tad_centers_ext_ctcf.iloc[:, 4:] = tad_centers_ext_ctcf.iloc[:, 4:].applymap(lambda x: min(x, 1))
# tad_centers_ext_ctcf = tad_centers_ext_ctcf.rename(columns={'ctcf_id': 'all'})
# plot_perc_regions_bearing_CTCF_site_cat(tad_centers_ext_ctcf, catname="shift_category",
#                                        path= figures_path / "perc_tad_centers_bearing_CTCF_site_cat.pdf")

print("Loading epigenetic peaks")
if os.path.isfile(str(interim_data_path / "wang2019_histone_peaks.tsv")):
    histone_peaks = pd.read_csv(interim_data_path / "wang2019_histone_peaks.tsv", 
                                sep='\t')
else:
    epigenetics = pd.read_excel(external_data_path / 'SupTable2_All_histone_ChIP_peaks_tag_counts_DEseq2.xlsx')
    epigenetics['histone'] = epigenetics.Histone_Mark.map(lambda x: x.split("_")[0].lower())
    histone_peaks = epigenetics[coords + ['histone']].sort_values(coords).reset_index(drop=True)
    histone_peaks.to_csv(interim_data_path / "wang2019_histone_peaks.tsv", sep="\t", index=False, header=True)

# print("Mapping histone modifications to ctcf peaks")
# ctcfs_with_histones = BedTool.from_dataframe(ctcfs)\
#                              .intersect(BedTool.from_dataframe(histone_peaks), wao=True)\
#                              .to_dataframe(names = ctcfs.columns.tolist() + \
#                                            ["histone_" + x for x in histone_peaks.columns] + \
#                                            ['overlap'])
# ctcfs_with_histones = pd.concat((ctcfs_with_histones, 
#                                  pd.get_dummies(ctcfs_with_histones.histone_histone)), axis=1)
# ctcf_ids_with_histones = ctcfs_with_histones.groupby('ctcf_id')[['h3k27ac', 'h3k4me1', 'h3k4me3']]\
#                                             .sum().reset_index()
# ctcfs_with_histones = ctcfs.merge(ctcf_ids_with_histones, on='ctcf_id')
# print("Mapping points of interestes on ctcfs")

# ctcfs_with_pois = BedTool.from_dataframe(ctcfs)\
#                          .sort()\
#                          .closest(BedTool.from_dataframe(pois).sort())\
#                          .to_dataframe(names=ctcfs.columns.tolist() + 
#                                              ["poi_"+x for x in pois.columns])\
#                          [['ctcf_id', 'poi_point_of_interest']]
# ctcfs_with_pois.rename(columns={'poi_point_of_interest': 'gm12878_feature'}, inplace=True)
# ctcfs_with_histones_pois = ctcfs_with_histones.merge(ctcfs_with_pois, on='ctcf_id')
# ctcfs_with_histones_pois.to_csv(interim_data_path / "ctcfs_with_features.tsv", sep="\t", index=False, header=True)
# plot_n_ctcf_category_vs_genomic_region(ctcfs_with_histones_pois)

print("Enrichment of CTCF sites at DI-triangles")
triangles = get_triangles(ins_with_neigh)

aggregations = {'ctcf_id': 'count', 
                'forward': 'sum', 
                'reverse': 'sum',
                'S': 'sum',
                'CD': 'sum', 
                'D': 'sum',
                'C': 'sum'}

extended = 1000*1000
window_size = 10*1000

pos_triangles = triangles[triangles.side == 'positive'].copy()
pos_triangles_vs_ctcfs = enrichment(pos_triangles, 
                                    ctcfs, centers='end', 
                                    extended=extended, 
                                    window_size=window_size, 
                                    aggregations=aggregations)


neg_triangles = triangles[triangles.side == 'negative'].copy()
neg_triangles_vs_ctcfs = enrichment(neg_triangles, 
                                    ctcfs, centers='start', 
                                    extended=extended, 
                                    window_size=window_size, 
                                    aggregations=aggregations)

plot_triangle_enrichment(pos_triangles_vs_ctcfs=pos_triangles_vs_ctcfs,
                         pos_triangles=pos_triangles,
                         neg_triangles_vs_ctcfs=neg_triangles_vs_ctcfs,
                         neg_triangles=neg_triangles)

plot_triangle_enrichment_horizontal(pos_triangles_vs_ctcfs,
                                        pos_triangles,
                                        neg_triangles_vs_ctcfs,
                                        neg_triangles,
                                        figures_path / "triangle_enrichment_CTCF_complete_square.pdf",
                                        cmap = "seismic_r")

plot_triangle_enrichment_halves(pos_triangles_vs_ctcfs,
                                        pos_triangles,
                                        neg_triangles_vs_ctcfs,
                                        neg_triangles,
                                        figures_path / "triangle_enrichment_CTCF_square.pdf",
                                        cmap = "seismic_r")


figsize = get_default_figsize()
fig, ax = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
tbyleft = tads.sort_values("left_size", ascending=False)
ax[0].barh(np.arange(tbyleft.shape[0]), -tbyleft.left_size, 0.5, label='Positive DI region')
ax[0].barh(np.arange(tbyleft.shape[0]), tbyleft.right_size, 0.5, label='Negative DI region')
ax[0].invert_yaxis()
ax[0].set_xlim(-1e6, +1e6)
ax[0].set_xticks([-1e6, -5e5, 0, 5e5, 1e6])
ax[0].set_xticklabels(['-1 Mb', '-500 Kb', '0', '+500 Kb', '1 Mb'])
ax[0].legend(bbox_to_anchor=(0.5, 1.21), loc='upper center')
ax[0].set_yticks([])
ax[0].set_ylabel("GM12878 TADs ordered by\ntheir Positive DI region length")
ax[0].set_xlabel("Distance from\nnegative inversion point")

tbyright = tads.sort_values("right_size", ascending=False)
ax[1].barh(np.arange(tbyright.shape[0]), -tbyright.left_size, 0.5, label='Positive DI region')
ax[1].barh(np.arange(tbyright.shape[0]), tbyright.right_size, 0.5, label='Negative DI region')
ax[1].invert_yaxis()
ax[1].set_xlim(-1e6, +1e6)
ax[1].set_xticks([-1e6, -5e5, 0, 5e5, 1e6])
ax[1].set_xticklabels(['-1 Mb', '-500 Kb', '0', '+500 Kb', '1 Mb'])
ax[1].legend(bbox_to_anchor=(0.5, 1.21), loc='upper center')
ax[1].set_yticks([])
ax[1].set_ylabel("GM12878 TADs ordered by\ntheir Negative DI region length", rotation=270, labelpad=27)
ax[1].yaxis.set_label_position("right")
ax[1].set_xlabel("Distance from\nnegative inversion point")

fig.savefig(figures_path / "DI_sizes.pdf", 
                bbox_inches='tight', transparent=True)
plt.close(fig)

print("Enrichment of Histone peaks at DI-triangles")

histone_to_color = {
    'h3k4me1': 'Blues',
    'h3k27ac': 'Greens',
    'h3k4me3': "Reds"
}

for histone in histone_peaks.histone.unique():
    print("\t{}".format(histone))
    hm = histone_peaks[histone_peaks.histone == histone]
    pos_triangles_vs_histone = enrichment(pos_triangles,
                                          hm, centers = 'end',
                                          extended = extended,
                                          window_size = window_size, 
                                          aggregations={'histone': 'count'},
                                          value_function=lambda x: x.histone)
    neg_triangles_vs_histone = enrichment(neg_triangles,
                                          hm, centers = 'start',
                                          extended = extended,
                                          window_size = window_size, 
                                          aggregations={'histone': 'count'},
                                          value_function=lambda x: x.histone)
    plot_triangle_enrichment_halves(pos_triangles_vs_histone,
                                        pos_triangles,
                                        neg_triangles_vs_histone,
                                        neg_triangles,
                                        figures_path / "triangle_enrichment_{}.pdf".format(histone),
                                        cmap = histone_to_color[histone])
    

print("Aggregating CTCF bindings sites at DI inversion points")
di_centers = ins_with_neigh[ins_with_neigh.point_of_interest == 'tad_center'].copy()
di_centers = di_centers.merge(tads[['chr', 'center_start', 'center_end']],
                              left_on=coords, right_on=['chr', 'center_start', 'center_end'])
di_centers['di_center_uid'] = np.arange(di_centers.shape[0], dtype=int)

extended = 250*1000
window_size = 5*1000

centered_di = di_centers.copy()
centers = centered_di.start
centered_di['start'] = centers
centered_di['end'] = centers

centered_di = BedTool.from_dataframe(centered_di).slop(b=extended, genome='hg19')\
                    .to_dataframe(names=centered_di.columns)
centered_di = centered_di[centered_di.end - centered_di.start == extended*2]
windows = windowing_by_size(centered_di[coords + ['di_center_uid']], 
                            window_size=window_size)

windows_with_ctcf = coverage_by_window(windows, ctcfs, aggregations)
windows_with_ctcf = windows_with_ctcf.merge(di_centers.drop(coords, axis=1), 
                                            on='di_center_uid')

aggregations_by_tad_center_tot = {}
for c in aggregations.keys():
    print(" "*100, end='\r')
    print("\t{}".format(c), end="\r")
    cagg = windows_with_ctcf.pivot_table(index='di_center_uid', 
                                         columns='w_num', values=c)\
                            .sort_index(axis=1)
    cagg = cagg.sort_index(axis=1)
    aggregations_by_tad_center_tot[c] = cagg

plot_aggregations_by_tad_center(aggregations_by_tad_center_tot,
                                   extended, window_size)


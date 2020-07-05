import sys
sys.path.append("./src/utilities")

from data_utilities import external_data_path, figures_path, interim_data_path, processed_data_path
from plot_utilities import initialize_plotting_parameters, get_default_figsize, ctcf_colors
from bed_utilities import coords, windowing_by_size, coverage_by_window, chroms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pybedtools.bedtool import BedTool
from statannot import add_stat_annotation


figures_path = figures_path / "analysis"


# ---------- UTILITY FUNCTIONS ---------- #

score = lambda i: ("+" if i > 0 else "") + str(i)
window_size = 10
intermediate_lines = 300
promoter_size = 2000
lines = [(promoter_size - intermediate_lines)/window_size,
         promoter_size/window_size,
         (promoter_size + intermediate_lines)/window_size]
linestyles = ['--', '-', '--']

def preprocess_MoMaTA_RNAseq(gene_expression):
    processed = gene_expression.copy()
    processed = processed.drop("chr", axis=1)
    processed = processed.rename(columns={'chr.1': 'chr'})
    processed = processed[processed.gene.notnull()].reset_index(drop=True)
    processed['strand'] = processed.strand.map(lambda x: '+' if x == 1 else "-")
    processed = processed.sort_values(coords).reset_index(drop=True)
    processed['start'] = processed.start.astype(int)
    processed['end'] = processed.end.astype(int)
    processed = processed[coords + [c for c in processed.columns if c not in coords]]
    return processed

def rpkm_to_class_log(rpkm):
    def __rpkm_to_class_log(rpkm):
        if rpkm < 1:
            return 0
        elif rpkm < 10:
            return 1
        elif rpkm < 100:
            return 2
        elif rpkm < 1000:
            return 3
        else:
            return 4
    return rpkm.map(__rpkm_to_class_log)

def rpkm_to_class_quantiles(rpkm):
    return pd.qcut(rpkm, q=[0, .25, .5, .75, 1.], labels=range(4))

rpkm_class_to_name_quantiles_OLD = {
    0: "$[0; 10^{-4}[$",
    1: "$[10^{-4}; 10^{-2}[$",
    2: "$[10^{-2}; 10[$",
    3: "$[10; 10^6[$"
}

rpkm_class_to_name_quantiles = {
    0: "$[0; 10^{-2}[$",
    1: "$[10^{-2}; 10^{0}[$",
    2: "$[10^{0}; 10[$",
    3: "$[10; 10^6[$"
}

rpkm_class_to_name_log = {
    0: "$[0; 1[$",
    1: "$[1; 10[$",
    2: "$[10; 100[$",
    3: "$[100; 1000[$",
    4: "$[1000; \infty[$"
}

def __to_bin(x):
    if x in [0,1,2]:
        return str(x)
    elif x in [3,4]:
        return "3-4"
    elif x in range(5,11):
        return "5-10"
    elif x in range(11,21):
        return "11-20"
    elif x in range(21, 34):
        return "21-33"
    
def read_gene_coordinates(path):  
    tss = pd.read_csv(path, sep='\t')
    tss = tss[tss.chrom.isin(chroms)]
    def __aggregate(df):
        if (df.chrom.nunique() != 1) or (df.strand.nunique() != 1):
            res = {
                'chrom': 'unknown',
                'txStart': -1,
                'txEnd': -1,
                'strand': 'unknown'
            }
        else:
            res = {
                'chrom': df.chrom.iloc[0],
                'txStart': df.txStart.min(),
                'txEnd': df.txEnd.max(),
                'strand': df.strand.iloc[0]
            }
        return pd.Series(res)
    tss = tss.groupby("name2")['chrom', 'txStart', 'txEnd', 'strand'].agg(__aggregate)
    tss = tss[tss.chrom != 'unknown']
    tss = tss.reset_index()
    tss = tss.rename(columns={
        'name2': 'gene_name',
        'chrom': 'chr',
        'txStart': 'start',
        'txEnd': 'end',
    })
    tss = tss[['chr', 'start', 'end', 'strand', 'gene_name']]
    tss = tss.sort_values(coords).reset_index(drop=True)
    return tss

def mirror(m):
    r = np.zeros_like(m)
    for i in range(int(m.shape[1])):
        r[:,i] = m[:,-(i + 1)]
    return r

# --------------------------------------- #

# --------- PLOTTING FUNCTIONS ---------- #

def plot_number_of_genes_per_RPKM_class(genexp):
    fig = plt.figure()
    ax = sns.countplot(genexp.rpkm_class_name, palette="copper_r")
    for p in ax.patches:
        plt.text(x=p.get_x() + 0.4, 
                 y=p.get_height()+400, 
                 s=str(p.get_height()), ha='center')
    plt.ylim(0, 45000)
    # plt.xticks(rotation=45)
    plt.xlabel("RPKM range")
    plt.ylabel("genes")
    plt.grid(axis='y')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    fig.savefig(figures_path / "n_of_genes_per_RPKM_class_{}.pdf".format(classification), 
                bbox_inches='tight', transparent=True)
    
def plot_gene_counts_vs_ctcfs(genexp_with_ctcfs, 
                              ctcf_classes = ['1', '2', '3-4', '5-10', '11-20', '21-33'],
                              ctcf_empty_class = '0',
                              cmap='Greens',
                              rename_empty_class = None,
                              path = figures_path / "gene_counts_vs_ctcfs.pdf",
                              ylims = None,
                              ylabel = 'Genes',
                              legend_title = "CTCF binding sites"):
    groups = genexp_with_ctcfs.groupby('rpkm_class_name')[[ctcf_empty_class] + ctcf_classes].sum()
    
    if rename_empty_class is not None:
        groups = groups.rename(columns={ctcf_empty_class: rename_empty_class})
        ctcf_empty_class = rename_empty_class
    
    if isinstance(cmap, dict):
        colors = ['white'] + [cmap[x] for x in ctcf_classes]
        newcmp = ListedColormap(colors)
    else: 
        cmap = plt.cm.get_cmap(cmap, 7)
        newcolors = cmap(np.linspace(0,1,7))
        blue = np.array([1,1,1, 1])
        newcolors[0, :] = blue
        newcmp = ListedColormap(newcolors)
    
    figsize = get_default_figsize()
    fig, axes = plt.subplots(1, groups.shape[0], figsize=(figsize[0]*groups.shape[0], figsize[1]*1.5))
    for i, g in enumerate(rpkm_class_to_name.values()):
        xg = groups.loc[[g], ctcf_classes]
        xg.loc[ctcf_empty_class, ctcf_empty_class] = groups.loc[g, ctcf_empty_class]
        xg = xg.fillna(0)
        xg.loc[g, 'not_intersect'] = 0
        xg = xg.loc[[ctcf_empty_class, g], [ctcf_empty_class] + ctcf_classes]
        xg.plot.bar(stacked=True, cmap=newcmp, ax=axes[i],edgecolor='black')
        axes[i].legend().set_visible(False)
        axes[i].set_xticklabels([])
        axes[i].grid(axis="y")
        axes[i].tick_params(axis='y')
        axes[i].set_xlabel("")
        axes[i].set_title(g, fontweight="bold")
        if ylims is not None:
            axes[i].set_ylim(*ylims)
            if i != 0:
                axes[i].set_yticklabels([])
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8,
               bbox_to_anchor=(0.45,1.05), 
               title=legend_title)
    fig.text(x=0.5, y=-0.01, s="RPKM range",
             horizontalalignment='center', fontsize='xx-large')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    
def plot_avg_stats_on_TSS_per_RPKM_class(aggregations_by_gene_merged, genexp,
                                         window_size, shrink=1):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(6, 4, sharey='row', 
                             figsize=(figsize[0]*5, figsize[1]*6),
                             tight_layout=True)

    def __get_vector(rpkm_class, feature, shrink=1):
        genes_rpkm_class = genexp.loc[genexp.rpkm_class == rpkm_class,'gene_uid'].tolist()
        genes_rpkm_class = set(genes_rpkm_class)\
                                .intersection(set(aggregations_by_gene_merged[feature].index))
        x = aggregations_by_gene_merged[feature].loc[genes_rpkm_class].mean(0)/shrink
        return x
    
    def __annotate(ax, ylabel, pos, legend=False, xlabel="Distance from the TSS (bp)", 
                  title=''):
        for l,ls in zip(lines, linestyles):
            ax.axvline(l, color='black', linestyle=ls)
        if pos[1] == 0:
            ax.set_ylabel(ylabel)
        if pos[0] == 5:
            ax.set_xlabel(xlabel)
        if pos[0] == 0:
            ax.set_title(title)
        xticks = [0] + lines + [2*promoter_size/window_size]
        xticklabels = ['-{}'.format(promoter_size)] + \
                      ["{}".format(score(int(l*window_size - promoter_size))) for l in lines ] + \
                      ['+{}'.format(promoter_size)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        if legend:
            ax.legend(loc='upper right')
        ax.grid()

    for rpkm_class in range(0, 4):
        axes[0, rpkm_class].plot(__get_vector(rpkm_class, 'ctcf_id', shrink), 
                                 color=ctcf_colors['all'])
        __annotate(axes[0, rpkm_class], 
                   "Avg. CTCF sites per 10 bp", (0, rpkm_class),
                   title="RPKM $\in$ {}".format(rpkm_class_to_name[rpkm_class]))

        axes[1, rpkm_class].plot(__get_vector(rpkm_class, 'forward', shrink),
                                 color=ctcf_colors['forward'], label='Forward')
        axes[1, rpkm_class].plot(__get_vector(rpkm_class, 'reverse', shrink),
                                 color=ctcf_colors['reverse'], label='Reverse')
        __annotate(axes[1, rpkm_class], 
                   "Avg. CTCF sites per 10 bp", (1, rpkm_class),
                   legend=True)

        axes[2, rpkm_class].plot(__get_vector(rpkm_class, 'S', shrink),
                                 color=ctcf_colors['S'], label='Same')
        axes[2, rpkm_class].plot(__get_vector(rpkm_class, 'C', shrink),
                                 color=ctcf_colors['C'], label='Convergent')
        axes[2, rpkm_class].plot(__get_vector(rpkm_class, 'D', shrink),
                                 color=ctcf_colors['D'], label='Divergent')
        axes[2, rpkm_class].plot(__get_vector(rpkm_class, 'CD', shrink),
                                 color=ctcf_colors['CD'], label='Convergent-\nDivergent')
        __annotate(axes[2, rpkm_class], 
                   "Avg. CTCF sites per 10 bp", (2, rpkm_class),
                   legend=True)

        axes[3, rpkm_class].plot(__get_vector(rpkm_class, 'MotifScore'),
                                 color='#3E9C2B')
        __annotate(axes[3, rpkm_class], 
                   "Avg. Motif Score per 10 bp", (3, rpkm_class),
                   legend=False)

        axes[4, rpkm_class].plot(__get_vector(rpkm_class, 'ChipSeqScore'),
                                 color='#9C2B95')
        __annotate(axes[4, rpkm_class], 
                   "Avg. ChipSeq Score per 10 bp", (4, rpkm_class),
                   legend=False)

        ratio = __get_vector(rpkm_class, 'MotifScore') / __get_vector(rpkm_class, 'ChipSeqScore')
        axes[5, rpkm_class].plot(ratio,
                                 color='black')
        __annotate(axes[5, rpkm_class], 
                   "Avg. Motif Score / ChipSeq\nScore per 10 bp", (5, rpkm_class),
                   legend=False)
    fig.savefig(figures_path / "avg_stats_on_TSS_per_rpkm_class_{}.pdf".format(classification),
                bbox_inches='tight', transparent=True)


# --------------------------------------- #


initialize_plotting_parameters()

print("Pre-processing gene expression data")
gene_expression = pd.read_excel(external_data_path / "SupTab1_20180202_MoMaTA_RNAseq_DEseq2mmDIFF.xlsx")
genexp = preprocess_MoMaTA_RNAseq(gene_expression)

single_value_genes = genexp.gene.value_counts()
single_value_genes = single_value_genes[single_value_genes == 1]
single_value_genes = single_value_genes.index.tolist()
genexp = genexp[genexp.gene.isin(single_value_genes)]

rpkm_columns = ['RPKM_6124', 'RPKM_7165', 'RPKM_bc27_D6_DMSO', 'RPKM_bc28_D6_DMSO']
genexp = genexp[coords + ['strand', 'gene'] + rpkm_columns]
genexp['mean_RPKM'] = genexp[rpkm_columns].mean(1)
genexp['std_RPKM'] = genexp[rpkm_columns].std(1)
genexp = genexp[coords + ['strand', 'gene', 'mean_RPKM', 'std_RPKM']]
refGenes = read_gene_coordinates(external_data_path / "hg19_UCSC_RefSeq_refGene.txt")
genexp = refGenes.merge(genexp.drop(coords + ['strand'], axis=1).rename(columns={'gene': 'gene_name'}), on='gene_name')
genexp['gene_uid'] = genexp.index


classification = 'quantiles'
rpkm_to_class = rpkm_to_class_quantiles
rpkm_class_to_name = rpkm_class_to_name_quantiles

genexp['rpkm_class'] = rpkm_to_class(genexp.mean_RPKM)
genexp['rpkm_class_name'] = genexp.rpkm_class.map(rpkm_class_to_name)

print("Loading CTCFs")
ctcfs = pd.read_csv(interim_data_path / "ctcfs_with_context.tsv", sep="\t")
ctcfs = pd.concat((ctcfs, pd.get_dummies(ctcfs[['orientation','context']], 
                                         prefix="", prefix_sep="")), axis=1)
ctcfs = ctcfs.rename(columns={'>': 'forward', '<': 'reverse'})
ctcfs = pd.concat((ctcfs, pd.get_dummies(ctcfs.orientation + ctcfs.context)), axis=1)
ctcfs = ctcfs.merge(pd.read_csv(interim_data_path / "ctcf_scores.tsv", sep="\t"),
                    on=coords + ['orientation'])
quantiles = np.quantile(ctcfs.rank_score_aggregate, q=[0, 0.25, 0.50, 0.75, 1])
ctcfs['rank_score_quartile'] = pd.cut(ctcfs.rank_score_aggregate, quantiles, 
                                      labels=['1st quartile', 
                                              '2nd quartile', 
                                              '3rd quartile', 
                                              '4th quartile'])
ctcfs = pd.concat((ctcfs, pd.get_dummies(ctcfs.rank_score_quartile)), axis=1)

print("Mapping CTCFs on genes")
genexp_with_ctcfs = BedTool.from_dataframe(genexp.sort_values(coords))\
                            .map(BedTool.from_dataframe(ctcfs), 
                                 c=[4,8,9,10,11,28,29,30,31],
                                 o=['count', 'sum', 'sum', 'sum', 'sum', 
                                    'sum', 'sum', 'sum', 'sum'], null=0)\
                            .to_dataframe(names=genexp.columns.tolist() + \
                                          ['n_ctcfs', 'C', 'CD', 'D', 'S',
                                           '1st quartile', '2nd quartile',
                                           '3rd quartile', '4th quartile'])
genexp_with_ctcfs.loc[:, ['C', 'CD', 'D', 'S']] = genexp_with_ctcfs\
                                        .loc[:, ['C', 'CD', 'D', 'S']]\
                                        .applymap(lambda x: int(x>0))
genexp_with_ctcfs.loc[:, ['1st quartile', '2nd quartile', '3rd quartile', '4th quartile']] = genexp_with_ctcfs.loc[:, ['1st quartile', '2nd quartile', '3rd quartile', '4th quartile']]\
                                        .applymap(lambda x: int(x>0))
genexp_with_ctcfs['has_CTCF'] = genexp_with_ctcfs.n_ctcfs > 0
genexp_with_ctcfs['has_CTCF_name'] = genexp_with_ctcfs.has_CTCF.map(
    lambda x: 'Intersects a CTCF site' if x else "Does not intersect a CTCF site")
genexp_with_ctcfs['cat'] = genexp_with_ctcfs.n_ctcfs.map(__to_bin)
genexp_with_ctcfs = pd.concat((genexp_with_ctcfs, pd.get_dummies(genexp_with_ctcfs.cat)), axis=1)

ylims = None#(0, 16000)
plot_gene_counts_vs_ctcfs(genexp_with_ctcfs,
                          ['1', '2', '3-4', '5-10', '11-20', '21-33'],
                          '0', 'Greens', None,
                          figures_path / "gene_counts_with_ctcfs_{}.pdf".format(classification), 
                          ylims)

plot_gene_counts_vs_ctcfs(genexp_with_ctcfs,
                          ['C', 'CD', 'D', 'S'],
                          '0', ctcf_colors,
                          'None',
                          figures_path / "gene_counts_with_ctcfs_class_{}.pdf".format(classification),
                          ylims,
                          ylabel='Instances of genes with at least\none CTCF of specified class')

plot_gene_counts_vs_ctcfs(genexp_with_ctcfs,
                          ['1st quartile', '2nd quartile', 
                           '3rd quartile', '4th quartile'],
                          '0', 'Purples',
                          'None',
                          figures_path / "gene_counts_with_ctcfs_rank_{}.pdf".format(classification),
                          ylims,
                          ylabel='Instances of genes with at least\none CTCF of specified quartile')

print("Mapping consensus boundaries on genes")
consensus_bounds = pd.read_csv(interim_data_path / 'consensus_boundaries.tsv', sep="\t")
consensus_bounds = pd.concat((consensus_bounds, pd.get_dummies(consensus_bounds.n_cell_types)), axis=1)

genexp_with_bounds = BedTool.from_dataframe(genexp.sort_values(coords))\
                            .map(BedTool.from_dataframe(consensus_bounds), 
                                 c=list(range(15,22)), o=['sum' for _ in range(15,22)], null=0)\
                            .to_dataframe(names=genexp.columns.tolist()+list(range(1,8)))
genexp_with_bounds['None'] = (genexp_with_bounds[list(range(1, 8))].sum(1) == 0).astype(int)

plot_gene_counts_vs_ctcfs(genexp_with_bounds, 
                          ctcf_classes=list(range(1, 8)),
                          ctcf_empty_class='None',
                          cmap='Reds',
                          legend_title="Boundary conservation",
                          path=figures_path / "gene_counts_with_cons_bounds_{}.pdf".format(classification))

print("Enrichment of CTCF sites at TSS of genes")
# genes = read_gene_coordinates(external_data_path / "hg19_UCSC_RefSeq_refGene.txt")
# genes = genes.merge(genexp[['gene', "mean_RPKM",'rpkm_class', 'rpkm_class_name', 'gene_uid']], 
#                 left_on='gene_name', right_on='gene')
genes = genexp[coords + ['strand', 'gene_name', 'mean_RPKM', 'rpkm_class', 'rpkm_class_name', 'gene_uid']].copy()
# genes = genexp[coords + ['strand', 'gene', 'rpkm_class', 'rpkm_class_name', 'gene_uid']].copy()
# genes = genes[genes.chr !='chrMT']
# genes.rename(columns={'gene': 'gene_name'}, inplace=True)
genes['TSS'] = genes.apply(lambda x: x.start if x.strand == '+' else x.end, axis=1)
tss = pd.DataFrame({
    'chr': genes.chr,
    'start': genes.TSS,
    'end': genes.TSS,
    'strand': genes.strand,
    'gene_name': genes.gene_name,
    'gene_uid': genes.gene_uid,
    'mean_RPKM': genes.mean_RPKM,
    'rpkm_class': genes.rpkm_class,
    'rpkm_class_name': genes.rpkm_class_name
})
# tss = tss.drop_duplicates(coords)


print("Analysis of promoters")
x = tss[['chr', 'start', 'end', 'gene_name', 'gene_uid', 'strand', 'rpkm_class', 'mean_RPKM']]
promoters = BedTool.from_dataframe(x)\
					.sort().slop(l=promoter_size, r=promoter_size, genome='hg19', s=True)\
					.to_dataframe(names=x.columns)
ctcfs_on_promoters = BedTool.from_dataframe(ctcfs).sort().map(BedTool.from_dataframe(promoters).sort(), c=[7,8], o=['max', 'max'], null=-1)\
							.to_dataframe(names=ctcfs.columns.tolist() + ['rpkm_class', 'mean_RPKM'])
ctcfs_on_promoters['in_promoter'] = ctcfs_on_promoters.rpkm_class.map(lambda x: "on promoter" if x > -1 else "outside promoter")
ctcfs_on_promoters['ratio'] = ctcfs_on_promoters.MotifScore / ctcfs_on_promoters.ChipSeqScore

ctcfs_on_promoters[['chr', 'start', 'end', 'orientation', 'context', 'MotifScore', 'ChipSeqScore', 'ratio',
'rank_score_aggregate', 'rank_score_quartile', 'rpkm_class', 'in_promoter']].to_csv(interim_data_path / "ctcfs_on_promoters.tsv", sep="\t", index=False, header=True)


promoters_on_ctcfs = BedTool.from_dataframe(promoters).sort().map(BedTool.from_dataframe(ctcfs).sort(), c=[1, 24, 25, 26], o=['count', 'max', 'max', 'max'], null=0)\
      .to_dataframe(names=promoters.columns.tolist()+["n_ctcfs", 'MotifScore', 'ChipSeqScore', 'rank_score_aggregate'])

promoters_on_ctcfs = BedTool.from_dataframe(promoters).sort().map(BedTool.from_dataframe(ctcfs[coords]).sort(), c=1, o='count')\
      .to_dataframe(names=promoters.columns.tolist()+["n_ctcfs"])

# promoters_on_ctcfs.groupby('rpkm_class')['n_ctcfs'].mean()
# promoters_on_ctcfs[promoters_on_ctcfs.n_ctcfs > 0]
# sns.boxplot(data=promoters_on_ctcfs, x='rpkm_class', y='n_ctcfs', showfliers=False)
# plt.show()

# plt.figure()
# for i in range(0, 4):
#   sns.distplot(promoters_on_ctcfs[promoters_on_ctcfs.rpkm_class == i].rank_score_aggregate, kde=True, hist=False, label=rpkm_class_to_name_quantiles[i])
# plt.legend()
# plt.show()

x = ctcfs_on_promoters[ctcfs_on_promoters.mean_RPKM > -1].assign(mean_RPKM=lambda x: np.log10(x.mean_RPKM + 1))
fig = plt.figure()
sns.kdeplot(x.MotifScore, x.mean_RPKM)
sns.regplot(data=x, x='MotifScore', y='mean_RPKM',
            scatter_kws={'s': 1}, scatter=False)
plt.xlabel("Motif Score")
plt.ylabel("$log_{10}$(RPKM)")
plt.title("N = {}".format(x.shape[0]))
fig.savefig(figures_path / "MotifScore_vs_log10RPKM.pdf",
                bbox_inches='tight', transparent=True)

fig = plt.figure()
sns.kdeplot(x.ChipSeqScore, x.mean_RPKM)
sns.regplot(data=x, x='ChipSeqScore', y='mean_RPKM',
            scatter_kws={'s': 1}, scatter=False)
plt.xlim(-10, 50)
plt.xlabel("ChipSeq Score")
plt.ylabel("$log_{10}$(RPKM)")
plt.title("N = {}".format(x.shape[0]))
fig.savefig(figures_path / "ChipSeq_vs_log10RPKM.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()


# fig = plt.figure()
# sns.kdeplot(x.ratio, x.mean_RPKM)
# sns.regplot(data=x, x='ratio', y='mean_RPKM',
#             scatter_kws={'s': 1}, scatter=False)
# plt.xlim(-10, 50)
# plt.xlabel("Motif Score / ChipSeq Score")
# plt.ylabel("$log_{10}$(RPKM)")
# plt.title("N = {}".format(x.shape[0]))
# fig.savefig(figures_path / "ratio_vs_log10RPKM.pdf",
#                 bbox_inches='tight', transparent=True)
# plt.show()

fig = plt.figure()
sns.kdeplot(x.rank_score_aggregate, x.mean_RPKM)
sns.regplot(data=x, x='rank_score_aggregate', y='mean_RPKM',
            scatter_kws={'s': 1}, scatter=False)
# plt.xlim(-10, 50)
plt.xlabel("Rank score aggregate")
plt.ylabel("$log_{10}$(RPKM)")
plt.title("N = {}".format(x.shape[0]))
fig.savefig(figures_path / "rankscore_vs_log10RPKM.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()

# plt.figure()
# sns.boxplot(data=ctcfs_on_promoters, x='in_promoter', y='ChipSeqScore')
# plt.show()

figsize = get_default_figsize()
fig = plt.figure(figsize=(figsize[0]*0.7, figsize[1]))
sns.countplot(data=ctcfs_on_promoters, x='in_promoter')
plt.xlabel("")
plt.ylabel("N. of CTCF binding sites")
fig.savefig(figures_path / "ctcfs_on_promoters.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()
figsize = get_default_figsize()
fig = plt.figure(figsize=(figsize[0], figsize[1]))
sns.countplot(data=ctcfs_on_promoters, x='rank_score_quartile', hue='in_promoter', order=["1st quartile", "2nd quartile", "3rd quartile", '4th quartile'])
plt.xlabel("Rank score aggregate quartile")
plt.ylabel("N. of CTCF binding sites")
plt.legend(loc="upper right", bbox_to_anchor=(1, 0.7))
fig.savefig(figures_path / "ctcf_quartiles_vs_in_promoter.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()


figsize = get_default_figsize()
fig = plt.figure(figsize=(figsize[0]*1.2, figsize[1]*1.2))
sns.boxplot(data=ctcfs_on_promoters.assign(name = lambda x: x.rpkm_class.map(lambda x: rpkm_class_to_name_quantiles.get(x, "None"))), x='name', y='rank_score_aggregate', order= ['None'] + list(rpkm_class_to_name_quantiles.values()))
plt.xlabel("RPKM quartiles")
plt.ylabel("Rank score aggregate")
fig.savefig(figures_path / "rpkm_quartiles_vs_rankscore.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()

figsize = get_default_figsize()
fig = plt.figure(figsize = (figsize[0]*1.2, figsize[1]*1.2))
sns.countplot(data=ctcfs_on_promoters.assign(name = lambda x: x.rpkm_class.map(lambda x: rpkm_class_to_name_quantiles.get(x, "None"))), x='name', 
	order= ['None'] + list(rpkm_class_to_name_quantiles.values()))
plt.xlabel("RPKM quartiles")
plt.ylabel("N. of CTCF binding sites")
fig.savefig(figures_path / "ctcfs_on_promoters_by_class.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()


figsize = get_default_figsize()
fig = plt.figure(figsize = (figsize[0]*0.7, figsize[1]))
ax = sns.boxplot(data=ctcfs_on_promoters, x='in_promoter', y='MotifScore')
test_results = add_stat_annotation(ax, data=ctcfs_on_promoters, x='in_promoter', y='MotifScore', order=['outside promoter', 'on promoter'],
                                   box_pairs=[('outside promoter', 'on promoter')],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=2)
plt.xlabel("")
plt.ylabel("Motif Score")
fig.savefig(figures_path / "motifscore_on_promoters.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()

# figsize = get_default_figsize()
# fig = plt.figure(figsize = (figsize[0], figsize[1]))
# ax = sns.boxplot(data=ctcfs_on_promoters, x='rpkm_class', y='MotifScore', showfliers=False)
# # test_results = add_stat_annotation(ax, data=ctcfs_on_promoters, x='in_promoter', y='MotifScore', order=['outside promoter', 'on promoter'],
# #                                    box_pairs=[('outside promoter', 'on promoter')],
# #                                    test='t-test_ind', text_format='full',
# #                                    loc='outside', verbose=2)
# plt.xlabel("")
# plt.ylabel("Motif Score")
# # fig.savefig(figures_path / "motifscore_on_promoters.pdf",
# #                 bbox_inches='tight', transparent=True)
# plt.show()
# plt.close(fig)


figsize = get_default_figsize()
fig = plt.figure(figsize = (figsize[0]*0.7, figsize[1]))
ax = sns.boxplot(data=ctcfs_on_promoters, x='in_promoter', y='ChipSeqScore', showfliers=False)
test_results = add_stat_annotation(ax, data=ctcfs_on_promoters, x='in_promoter', y='ChipSeqScore', order=['outside promoter', 'on promoter'],
                                   box_pairs=[('outside promoter', 'on promoter')],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=2)
plt.xlabel("")
plt.ylabel("ChipSeq Score")
fig.savefig(figures_path / "chipseqscore_on_promoters.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()

figsize = get_default_figsize()
fig = plt.figure(figsize = (figsize[0]*0.7, figsize[1]))
ax = sns.boxplot(data=ctcfs_on_promoters, x='in_promoter', y='ratio', showfliers=False)
test_results = add_stat_annotation(ax, data=ctcfs_on_promoters, x='in_promoter', y='ratio', order=['outside promoter', 'on promoter'],
                                   box_pairs=[('outside promoter', 'on promoter')],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=2)
plt.xlabel("")
plt.ylabel("Motif Score / ChipSeq Score")
fig.savefig(figures_path / "ratio_on_promoters.pdf",
                bbox_inches='tight', transparent=True)
# plt.show()


figsize = get_default_figsize()
fig = plt.figure(figsize = (figsize[0]*0.7, figsize[1]))
ax = sns.boxplot(data=ctcfs_on_promoters, x='in_promoter', y='rank_score_aggregate', showfliers=False)
test_results = add_stat_annotation(ax, data=ctcfs_on_promoters, x='in_promoter', y='rank_score_aggregate', order=['outside promoter', 'on promoter'],
                                   box_pairs=[('outside promoter', 'on promoter')],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=2)
plt.xlabel("")
plt.ylabel("Rank score aggregate")
fig.savefig(figures_path / "rankscore_on_promoters.pdf",
                bbox_inches='tight', transparent=True)

tss_extended = promoters_on_ctcfs[promoters_on_ctcfs.n_ctcfs > 0].reset_index(drop=True)
tss_extended = tss_extended[tss_extended.end - tss_extended.start == promoter_size*2]
# window_size = 100
windows = windowing_by_size(tss_extended[coords + ['gene_uid']],
                            window_size=window_size)


def plot_avg_ctcf_sites_per_rpkm_class(x, 
                                       window_size,
                                       path,
                                       cmap='Greys',
                                       ylabel = "Avg. CTCF sites per 10 bp",
                                       ylim=None):
    figsize=get_default_figsize()
    fig, axes = plt.subplots(1, 1, figsize=(figsize[0]*1.5, figsize[1]))
    colors = plt.get_cmap(cmap)
    colors = colors(np.linspace(0, 1, 4 + 1))
    for rpkm_class in range(4):
        axes.plot(x[rpkm_class], label=rpkm_class_to_name[rpkm_class], color=colors[rpkm_class + 1])
    for l,ls in zip(lines, linestyles):
        axes.axvline(l, color='black', linestyle=ls)
    axes.legend(title='RPKM class', loc='upper left')
    xticks = [0] + lines + [2*promoter_size/window_size]
    xticklabels = ['-{}'.format(promoter_size)] + \
                  ["{}".format(score(int(l*window_size - promoter_size))) for l in lines ] + \
                  ['+{}'.format(promoter_size)]
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels)
    axes.set_xlabel("Distance from the TSS (bp)")
    axes.set_ylabel(ylabel)
    if ylim is not None:
      axes.set_ylim(ylim)
    fig.savefig(path,
                bbox_inches='tight', transparent=True)

def split_in_rpkm_classes(x):
    r = {}
    for rpkm_class in range(4):
        rpkm_class_genes = sorted(tss_extended[tss_extended.rpkm_class == rpkm_class].gene_uid.tolist())
        print("{}: {} genes".format(rpkm_class_to_name[rpkm_class], len(rpkm_class_genes)))
        r[rpkm_class] = x.reindex(index=rpkm_class_genes)
    return r


#### Methylation probes

meth = pd.read_csv(processed_data_path / "GM12878_methylation.bed", sep="\t", header=None, names = coords + ['score'])
meth['down'] = meth.score.map(lambda x: 1 if x < 50 else 0)
meth['up'] = meth.score.map(lambda x: 0 if x < 50 else 1)
meth['v'] = 1

aggregations = {'v': 'sum', 'score': 'mean', 'down': 'sum', 'up': 'sum'}
windows_with_meth = coverage_by_window(windows, meth.sort_values(coords), aggregations, null=0)
windows_with_meth = windows_with_meth.merge(genexp.drop(coords, axis=1),
                                              on='gene_uid')

aggregations_by_gene_meth = {}
for c in aggregations.keys():
    print(" "*100, end='\r')
    print("\t{}".format(c), end='\r')
    cagg = windows_with_meth.pivot_table(index='gene_uid', columns='w_num', values=c).sort_index(axis=1)
    aggregations_by_gene_meth[c] = cagg 
    
aggregations_by_gene_merged_meth = {}
for c in aggregations_by_gene_meth.keys():
    print(" "*100, end='\r')
    print("\t{}".format(c), end='\r')
    pos_set = tss.loc[tss.strand == '+', 'gene_uid'].tolist()
    neg_set = tss.loc[tss.strand == '-', 'gene_uid'].tolist()
    mirrored_neg = pd.DataFrame(data=mirror(aggregations_by_gene_meth[c].loc[neg_set].values), 
                                index=neg_set)
    cagg = pd.concat((aggregations_by_gene_meth[c].loc[pos_set], mirrored_neg), axis=0)
    aggregations_by_gene_merged_meth[c] = cagg


aggregations_by_gene_merged_meth_by_class = {}
for c in aggregations_by_gene_merged_meth.keys():
    aggregations_by_gene_merged_meth_by_class[c] = split_in_rpkm_classes(aggregations_by_gene_merged_meth[c])


meth_aggregations_averages = {}
# number of methylation probes
n_meth_probes_averages = {}
n_meth_probes_sum = {}
for rpkm_class in range(4):
    n_meth_probes_averages[rpkm_class] = aggregations_by_gene_merged_meth_by_class["v"][rpkm_class].mean(0)
    n_meth_probes_sum[rpkm_class] = aggregations_by_gene_merged_meth_by_class["v"][rpkm_class].sum(0)
meth_aggregations_averages["n_meth_probes_mean"] = n_meth_probes_averages
meth_aggregations_averages['n_meth_probes_sum'] = n_meth_probes_sum

plot_avg_ctcf_sites_per_rpkm_class(meth_aggregations_averages['n_meth_probes_mean'], window_size,
                                 figures_path / "avg_meth_probes_per_rpkm_class_{}.pdf".format(classification), 
                                   'Greys', "Avg. N. methylation probes per {} bp".format(window_size))


# number of UP methylation probes
n_up_meth_probes_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_meth_by_class["up"][rpkm_class].sum(0)
    m_avg = m_sum / meth_aggregations_averages['n_meth_probes_sum'][rpkm_class]
    n_up_meth_probes_averages[rpkm_class] = m_avg
meth_aggregations_averages['n_up_meth_probes_mean'] = n_up_meth_probes_averages


plot_avg_ctcf_sites_per_rpkm_class(meth_aggregations_averages['n_up_meth_probes_mean'], window_size,
                                   figures_path / "avg_meth_up_per_rpkm_class_{}.pdf".format(classification), 
                                   'Greys', "Avg. N. UP methylation probes per {} bp".format(window_size))


# number of UP methylation probes
n_down_meth_probes_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_meth_by_class["down"][rpkm_class].sum(0)
    m_avg = m_sum / meth_aggregations_averages['n_meth_probes_sum'][rpkm_class]
    n_down_meth_probes_averages[rpkm_class] = m_avg
meth_aggregations_averages['n_down_meth_probes_mean'] = n_down_meth_probes_averages


plot_avg_ctcf_sites_per_rpkm_class(meth_aggregations_averages['n_down_meth_probes_mean'], window_size,
                                   figures_path / "avg_meth_down_per_rpkm_class_{}.pdf".format(classification), 
                                   'Greys', "Avg. N. DOWN methylation probes per {} bp".format(window_size))


c = pd.read_csv(interim_data_path / "ctcfs_on_promoters.tsv", sep="\t")
ctcfs_with_meth = BedTool.from_dataframe(c).sort()\
             .map(BedTool.from_dataframe(meth).sort(), c=4, o="max", null=-1)\
             .to_dataframe(names=c.columns.tolist() + ['meth_score'])
ctcfs_with_meth['has_meth'] = ctcfs_with_meth.meth_score.map(lambda x: "With probe" if x>-1 else "Without probe")

ctcfs_with_meth.to_csv(interim_data_path / "ctcfs_with_meth.tsv", sep="\t", index=False, header=True)



ctcfs_with_meth = pd.read_csv(interim_data_path / "ctcfs_with_meth.tsv", sep="\t")
ctcfs = ctcfs_on_promoters.merge(ctcfs_with_meth[coords + ['meth_score', 'has_meth']], on=coords)
ctcfs['meth_down'] = 0
ctcfs.loc[(ctcfs.meth_score > -1) & (ctcfs.meth_score < 50), "meth_down"] = 1
ctcfs['meth_up'] = 0
ctcfs.loc[(ctcfs.meth_score > -1) & (ctcfs.meth_score >= 50), "meth_up"] = 1
ctcfs['with_meth'] = 0
ctcfs.loc[ctcfs.has_meth == 'With probe', 'with_meth'] = 1
ctcfs['without_meth'] = 0
ctcfs.loc[ctcfs.has_meth == 'Without probe', 'without_meth'] = 1


aggregations = {'ctcf_id': 'count', 
                'forward': 'sum', 
                'reverse': 'sum',
                'S': 'sum',
                'CD': 'sum', 
                'D': 'sum',
                'C': 'sum',
                'MotifScore': 'sum',
                'ChipSeqScore': 'sum',
                'meth_up': 'sum',
                'meth_down': 'sum',
                'with_meth': 'sum',
                'without_meth': 'sum'
                }

windows_with_ctcfs = coverage_by_window(windows, ctcfs, aggregations)
# windows_with_filtered_ctcfs = coverage_by_window(windows, ctcfs[~ctcfs.has_meth], aggregations)

windows_with_ctcfs = windows_with_ctcfs.merge(genexp.drop(coords, axis=1),
                                              on='gene_uid')

windows_with_ctcfs.rpkm_class.nunique()

aggregations_by_gene = {}
for c in aggregations.keys():
    print(" "*100, end='\r')
    print("\t{}".format(c), end='\r')
    cagg = windows_with_ctcfs.pivot_table(index='gene_uid', columns='w_num', values=c).sort_index(axis=1)
    aggregations_by_gene[c] = cagg 
    print(cagg.shape)
    
aggregations_by_gene_merged = {}
for c in aggregations_by_gene.keys():
    print(" "*100, end='\r')
    print("\t{}".format(c), end='\r')
    pos_set = tss_extended.loc[tss_extended.strand == '+', 'gene_uid'].tolist()
    neg_set = tss_extended.loc[tss_extended.strand == '-', 'gene_uid'].tolist()
    print("Pos: {} - Neg: {}".format(len(pos_set), len(neg_set)))
    mirrored_neg = pd.DataFrame(data=mirror(aggregations_by_gene[c].loc[neg_set].values), 
                                index=neg_set)
    cagg = pd.concat((aggregations_by_gene[c].loc[pos_set], mirrored_neg), axis=0)
    aggregations_by_gene_merged[c] = cagg

shrink_factor = 1#(ctcfs.end - ctcfs.start).mean() / window_size

#### Figure 1: CTCF sites at TSS

aggregations_by_gene_merged_by_class = {}
for c in aggregations_by_gene_merged.keys():
    aggregations_by_gene_merged_by_class[c] = split_in_rpkm_classes(aggregations_by_gene_merged[c])


aggregations_averages = {}

##### All CTCF sites at promoters (Figure 1)


# number of ctcf binding sites
n_ctcfs_averages = {}
n_ctcfs_sum = {}
for rpkm_class in range(4):
    n_ctcfs_averages[rpkm_class] = aggregations_by_gene_merged_by_class["ctcf_id"][rpkm_class].mean(0)
    n_ctcfs_sum[rpkm_class] = aggregations_by_gene_merged_by_class["ctcf_id"][rpkm_class].sum(0)
aggregations_averages["n_ctcfs_mean"] = n_ctcfs_averages
aggregations_averages['n_ctcfs_sum'] = n_ctcfs_sum

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['n_ctcfs_mean'], window_size,
                                 figures_path / "avg_ctcf_sites_per_rpkm_class_{}.pdf".format(classification), 
                                   'Greys', "Avg. CTCF sites per {} bp".format(window_size))

# Motif score
motif_score_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_by_class["MotifScore"][rpkm_class].sum(0)
    m_avg = m_sum / aggregations_averages['n_ctcfs_sum'][rpkm_class]
    motif_score_averages[rpkm_class] = m_avg
aggregations_averages['MotifScore_mean'] = motif_score_averages


plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['MotifScore_mean'], window_size,
                                 figures_path / "avg_MotifScore_per_rpkm_class_{}.pdf".format(classification), 
                                   'Greens', "Avg. Motif Score per {} bp".format(window_size))

# ChipSeq score
chipseq_score_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_by_class["ChipSeqScore"][rpkm_class].sum(0)
    m_avg = m_sum / aggregations_averages['n_ctcfs_sum'][rpkm_class]
    chipseq_score_averages[rpkm_class] = m_avg
aggregations_averages['ChipSeqScore_mean'] = chipseq_score_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ChipSeqScore_mean'], window_size,
                                 figures_path / "avg_ChipSeq_per_rpkm_class_{}.pdf".format(classification), 
                                   'Purples', "Avg. ChipSeq Score per {} bp".format(window_size))


# Ratio between ChipSeq score and Motif score
ratios_averages = {}
for rpkm_class in range(4):
    ratios_averages[rpkm_class] = aggregations_averages['MotifScore_mean'][rpkm_class] / aggregations_averages['ChipSeqScore_mean'][rpkm_class]
aggregations_averages['ratio_mean'] = ratios_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ratio_mean'], window_size,
                                 figures_path / "avg_MotifChipSeqRatio_per_rpkm_class_{}.pdf".format(classification), 
                                   'Greys', "Avg. Motif Score / ChipSeq\nScore per {} bp".format(window_size))


##### Methylation probe vs No Methylation (Rebuttal)

# N. CTCF sites with a methylation probe
with_meth_averages = {}
with_meth_sums = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_by_class["with_meth"][rpkm_class].sum(0)
    m_avg = m_sum / aggregations_averages['n_ctcfs_sum'][rpkm_class]
    with_meth_averages[rpkm_class] = m_avg
    with_meth_sums[rpkm_class] = m_sum
aggregations_averages['with_meth_mean'] = with_meth_averages
aggregations_averages['with_meth_sum'] = with_meth_sums


plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['with_meth_mean'], window_size,
                                   figures_path / "avg_ctcf_sites_per_rpkm_class_{}_with_meth.pdf".format(classification),
                                   'Greys', "Avg. CTCF sites per {} bp".format(window_size))

# N. CTCF sites without a methylation probe
without_meth_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_by_class["without_meth"][rpkm_class].sum(0)
    m_avg = m_sum / aggregations_averages['n_ctcfs_sum'][rpkm_class]
    without_meth_averages[rpkm_class] = m_avg
aggregations_averages['without_meth_mean'] = without_meth_averages


plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['without_meth_mean'], window_size,
                                   figures_path / "avg_ctcf_sites_per_rpkm_class_{}_without_meth.pdf".format(classification),
                                   'Greys', "Avg. CTCF sites per {} bp".format(window_size))

# ChipSeq score for CTCF sites with a methylation probe
chipseq_score_with_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["ChipSeqScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["with_meth"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    chipseq_score_with_meth_averages[rpkm_class] = m_avg
aggregations_averages['ChipSeqScore_with_meth_mean'] = chipseq_score_with_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ChipSeqScore_with_meth_mean'], window_size,
                                   figures_path / "avg_ChipSeq_per_rpkm_class_{}_with_meth.pdf".format(classification),
                                   'Purples', "Avg. ChipSeq Score per {} bp".format(window_size))

# ChipSeq score for CTCF sites without a methylation probe
chipseq_score_without_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["ChipSeqScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["without_meth"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    chipseq_score_without_meth_averages[rpkm_class] = m_avg
aggregations_averages['ChipSeqScore_without_meth_mean'] = chipseq_score_without_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ChipSeqScore_without_meth_mean'], window_size,
                                   figures_path / "avg_ChipSeq_per_rpkm_class_{}_without_meth.pdf".format(classification),
                                   'Purples', "Avg. ChipSeq Score per {} bp".format(window_size))

# Motif Score for CTCF sites with a methylation probe
motif_score_with_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["MotifScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["with_meth"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    motif_score_with_meth_averages[rpkm_class] = m_avg
aggregations_averages['MotifScore_with_meth_mean'] = motif_score_with_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['MotifScore_with_meth_mean'], window_size,
                                   figures_path / "avg_MotifScore_per_rpkm_class_{}_with_meth.pdf".format(classification),
                                   'Greens', "Avg. Motif Score per {} bp".format(window_size))

# Motif Score for CTCF sites without a methylation probe
motif_score_without_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["MotifScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["without_meth"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    motif_score_without_meth_averages[rpkm_class] = m_avg
aggregations_averages['MotifScore_without_meth_mean'] = motif_score_without_meth_averages


plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['MotifScore_without_meth_mean'], window_size,
                                   figures_path / "avg_MotifScore_per_rpkm_class_{}_without_meth.pdf".format(classification),
                                   'Greens', "Avg. Motif Score per {} bp".format(window_size))


# Motif/ChipSeq score ratio for CTCF sites with a methylation probe
ratio_score_with_meth_averages = {}
for rpkm_class in range(4):
    ratio_score_with_meth_averages[rpkm_class] = aggregations_averages['MotifScore_with_meth_mean'][rpkm_class] / aggregations_averages['ChipSeqScore_with_meth_mean'][rpkm_class]
aggregations_averages['ratio_with_meth_mean'] = ratio_score_with_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ratio_with_meth_mean'], window_size,
                                   figures_path / "avg_MotifChipSeqRatio_per_rpkm_class_{}_with_meth.pdf".format(classification),
                                   'Greys', "Avg. Motif Score / ChipSeq\nScore per {} bp".format(window_size))

# Motif/ChipSeq score ratio for CTCF sites without a methylation probe
ratio_score_without_meth_averages = {}
for rpkm_class in range(4):
    ratio_score_without_meth_averages[rpkm_class] = aggregations_averages['MotifScore_without_meth_mean'][rpkm_class] / aggregations_averages['ChipSeqScore_without_meth_mean'][rpkm_class]
aggregations_averages['ratio_without_meth_mean'] = ratio_score_without_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ratio_without_meth_mean'], window_size,
                                   figures_path / "avg_MotifChipSeqRatio_per_rpkm_class_{}_without_meth.pdf".format(classification),
                                   'Greys', "Avg. Motif Score / ChipSeq\nScore per {} bp".format(window_size))


##### Methylation UP vs Methylation DOWN


# N. CTCF sites with a UP methylation probe
with_up_meth_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_by_class["meth_up"][rpkm_class].sum(0)
    m_avg = m_sum / aggregations_averages['with_meth_sum'][rpkm_class]
    with_up_meth_averages[rpkm_class] = m_avg
aggregations_averages['with_up_meth_mean'] = with_up_meth_averages


plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['with_up_meth_mean'], window_size,
                                   figures_path / "avg_ctcf_sites_per_rpkm_class_{}_meth_up.pdf".format(classification),
                                   'Greys', "Avg. CTCF sites per {} bp".format(window_size))

# N. CTCF sites without a DOWN methylation probe
with_down_meth_averages = {}
for rpkm_class in range(4):
    m_sum = aggregations_by_gene_merged_by_class["meth_down"][rpkm_class].sum(0)
    m_avg = m_sum / aggregations_averages['with_meth_sum'][rpkm_class]
    with_down_meth_averages[rpkm_class] = m_avg
aggregations_averages['with_down_meth_mean'] = with_down_meth_averages


plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['with_down_meth_mean'], window_size,
                                   figures_path / "avg_ctcf_sites_per_rpkm_class_{}_meth_down.pdf".format(classification),
                                   'Greys', "Avg. CTCF sites per {} bp".format(window_size))

# ChipSeq score for CTCF sites with a UP methylation probe
chipseq_score_with_up_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["ChipSeqScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["meth_up"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    chipseq_score_with_up_meth_averages[rpkm_class] = m_avg
aggregations_averages['ChipSeqScore_with_up_meth_mean'] = chipseq_score_with_up_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ChipSeqScore_with_up_meth_mean'], window_size,
                                   figures_path / "avg_ChipSeq_per_rpkm_class_{}_meth_up.pdf".format(classification),
                                   'Purples', "Avg. ChipSeq Score per {} bp".format(window_size))


# ChipSeq score for CTCF sites with a DOWN methylation probe
chipseq_score_with_down_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["ChipSeqScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["meth_down"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    chipseq_score_with_down_meth_averages[rpkm_class] = m_avg
aggregations_averages['ChipSeqScore_with_down_meth_mean'] = chipseq_score_with_down_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ChipSeqScore_with_down_meth_mean'], window_size,
                                   figures_path / "avg_ChipSeq_per_rpkm_class_{}_meth_down.pdf".format(classification),
                                   'Purples', "Avg. ChipSeq Score per {} bp".format(window_size))



# Motif score for CTCF sites with a UP methylation probe
motif_score_with_up_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["MotifScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["meth_up"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    motif_score_with_up_meth_averages[rpkm_class] = m_avg
aggregations_averages['MotifScore_with_up_meth_mean'] = motif_score_with_up_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['MotifScore_with_up_meth_mean'], window_size,
                                   figures_path / "avg_MotifScore_per_rpkm_class_{}_meth_up.pdf".format(classification),
                                   'Greens', "Avg. Motif Score per {} bp".format(window_size))


# Motif score for CTCF sites with a DOWN methylation probe
motif_score_with_down_meth_averages = {}
for rpkm_class in range(4):
    m = aggregations_by_gene_merged_by_class["MotifScore"][rpkm_class]
    s = aggregations_by_gene_merged_by_class["meth_down"][rpkm_class]
    ms = m*s
    m_sum = ms.sum(0)
    m_avg = m_sum / s.sum(0)
    motif_score_with_down_meth_averages[rpkm_class] = m_avg
aggregations_averages['MotifScore_with_down_meth_mean'] = motif_score_with_down_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['MotifScore_with_down_meth_mean'], window_size,
                                   figures_path / "avg_MotifScore_per_rpkm_class_{}_meth_down.pdf".format(classification),
                                   'Greens', "Avg. Motif Score per {} bp".format(window_size))


# Motif/ChipSeq score ratio for CTCF sites with a methylation probe
ratio_score_with_up_meth_averages = {}
for rpkm_class in range(4):
    ratio_score_with_up_meth_averages[rpkm_class] = aggregations_averages['MotifScore_with_up_meth_mean'][rpkm_class] / aggregations_averages['ChipSeqScore_with_up_meth_mean'][rpkm_class]
aggregations_averages['ratio_with_up_meth_mean'] = ratio_score_with_up_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ratio_with_up_meth_mean'], window_size,
                                   figures_path / "avg_MotifChipSeqRatio_per_rpkm_class_{}_meth_up.pdf".format(classification),
                                   'Greys', "Avg. Motif Score / ChipSeq\nScore per {} bp".format(window_size), ylim=(0, 1))

# Motif/ChipSeq score ratio for CTCF sites without a methylation probe
ratio_score_with_down_meth_averages = {}
for rpkm_class in range(4):
    ratio_score_with_down_meth_averages[rpkm_class] = aggregations_averages['MotifScore_with_down_meth_mean'][rpkm_class] / aggregations_averages['ChipSeqScore_with_down_meth_mean'][rpkm_class]
aggregations_averages['ratio_with_down_meth_mean'] = ratio_score_with_down_meth_averages

plot_avg_ctcf_sites_per_rpkm_class(aggregations_averages['ratio_with_down_meth_mean'], window_size,
                                   figures_path / "avg_MotifChipSeqRatio_per_rpkm_class_{}_meth_down.pdf".format(classification),
                                   'Greys', "Avg. Motif Score / ChipSeq\nScore per {} bp".format(window_size), ylim=(0, 1))


def plot_avg_stats_on_TSS_per_RPKM_class(aggregations_averages,
                                         window_size):
    figsize = get_default_figsize()
    fig, axes = plt.subplots(6, 4, sharey='row', 
                             figsize=(figsize[0]*5, figsize[1]*6),
                             tight_layout=True)
    
    def __annotate(ax, ylabel, pos, legend=False, xlabel="Distance from the TSS (bp)", 
                  title=''):
        for l,ls in zip(lines, linestyles):
            ax.axvline(l, color='black', linestyle=ls)
        if pos[1] == 0:
            ax.set_ylabel(ylabel)
        if pos[0] == 5:
            ax.set_xlabel(xlabel)
        if pos[0] == 0:
            ax.set_title(title)
        xticks = [0] + lines + [2*promoter_size/window_size]
        xticklabels = ['-{}'.format(promoter_size)] + \
                      ["{}".format(score(int(l*window_size - promoter_size))) for l in lines ] + \
                      ['+{}'.format(promoter_size)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        if legend:
            ax.legend(loc='upper right')
        ax.grid()

    for rpkm_class in range(0, 4):
        axes[0, rpkm_class].plot(aggregations_averages['n_ctcfs_mean'][rpkm_class], 
                                 color=ctcf_colors['all'])
        __annotate(axes[0, rpkm_class], 
                   "Avg. CTCF sites per 10 bp", (0, rpkm_class),
                   title="RPKM $\in$ {}".format(rpkm_class_to_name[rpkm_class]))

        axes[1, rpkm_class].plot(aggregations_averages['n_forward_mean'][rpkm_class],
                                 color=ctcf_colors['forward'], label='Forward')
        axes[1, rpkm_class].plot(aggregations_averages['n_reverse_mean'][rpkm_class],
                                 color=ctcf_colors['reverse'], label='Reverse')
        __annotate(axes[1, rpkm_class], 
                   "Avg. CTCF sites per 10 bp", (1, rpkm_class),
                   legend=True)

        axes[2, rpkm_class].plot(aggregations_averages['n_S_mean'][rpkm_class],
                                 color=ctcf_colors['S'], label='Same')
        axes[2, rpkm_class].plot(aggregations_averages['n_C_mean'][rpkm_class],
                                 color=ctcf_colors['C'], label='Convergent')
        axes[2, rpkm_class].plot(aggregations_averages['n_D_mean'][rpkm_class],
                                 color=ctcf_colors['D'], label='Divergent')
        axes[2, rpkm_class].plot(aggregations_averages['n_CD_mean'][rpkm_class],
                                 color=ctcf_colors['CD'], label='Convergent-\nDivergent')
        __annotate(axes[2, rpkm_class], 
                   "Avg. CTCF sites per 10 bp", (2, rpkm_class),
                   legend=True)

        axes[3, rpkm_class].plot(aggregations_averages["MotifScore_mean"][rpkm_class],
                                 color='#3E9C2B')
        __annotate(axes[3, rpkm_class], 
                   "Avg. Motif Score per 10 bp", (3, rpkm_class),
                   legend=False)

        axes[4, rpkm_class].plot(aggregations_averages["ChipSeqScore_mean"][rpkm_class],
                                 color='#9C2B95')
        __annotate(axes[4, rpkm_class], 
                   "Avg. ChipSeq Score per 10 bp", (4, rpkm_class),
                   legend=False)
        axes[5, rpkm_class].plot(aggregations_averages["ratio_mean"][rpkm_class],
                                 color='black')
        __annotate(axes[5, rpkm_class], 
                   "Avg. Motif Score / ChipSeq\nScore per 10 bp", (5, rpkm_class),
                   legend=False)
    fig.savefig(figures_path / "avg_stats_on_TSS_per_rpkm_class_{}.pdf".format(classification),
                bbox_inches='tight', transparent=True)


for c in ["forward", "reverse", "S", "CD", "D", "C"]:
	n_c_averages = {}
	for rpkm_class in range(4):
	    n_c_averages[rpkm_class] = aggregations_by_gene_merged_by_class[c][rpkm_class].mean(0)
	aggregations_averages["n_{}_mean".format(c)] = n_c_averages

# Supplementary figure
plot_avg_stats_on_TSS_per_RPKM_class(aggregations_averages, window_size)

plt.close("all")
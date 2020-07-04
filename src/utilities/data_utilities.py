import os
from pathlib import Path
import pandas as pd
from bed_utilities import coords
from collections import defaultdict
import numpy as np

project_path = Path(".")

data_path = project_path / "data"
raw_data_path = data_path / 'raw'
processed_data_path = data_path / 'processed'
interim_data_path = data_path / "interim"
external_data_path = data_path / 'external'

# results_path = project_path / "results"
figures_path = project_path / "figures"

os.makedirs(data_path, exist_ok=True)
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(processed_data_path, exist_ok=True)
os.makedirs(interim_data_path, exist_ok=True)
os.makedirs(external_data_path, exist_ok=True)
# os.makedirs(results_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)


def read_PCHiC(path):
    pchic_interactions = pd.read_csv(path, sep='\t')
    pchic_interactions['baitChr'] = pchic_interactions.baitChr.astype(str)
    pchic_interactions['oeChr'] = pchic_interactions.oeChr.astype(str)
    pchic_interactions = pchic_interactions[pchic_interactions.baitChr == pchic_interactions.oeChr]
    pchic_interactions.columns
    
    pchic_interactions = pd.DataFrame({
        'chr': 'chr' + pchic_interactions.baitChr.astype(str),
        'start': pchic_interactions.apply(lambda x: min(x.baitStart, x.oeStart), axis=1),
        'end': pchic_interactions.apply(lambda x: max(x.baitEnd, x.oeEnd), axis=1),
        'name': '.',
        'score': 1,
        'value': 1,
        'exp': '.',
        'color': '148,0,190',
        'baitChr': 'chr' + pchic_interactions.baitChr.astype(str),
        'baitStart': pchic_interactions.baitStart,
        'baitEnd': pchic_interactions.baitEnd,
        'baitID': pchic_interactions.baitID,
        'baitStrand': '.',
        'oeChr': 'chr' + pchic_interactions.oeChr.astype(str),
        'oeStart': pchic_interactions.oeStart,
        'oeEnd': pchic_interactions.oeEnd,
        'oeID': pchic_interactions.oeID,
        'oeStrand': '.',
    })
    pchic_interactions = pchic_interactions.sort_values(coords).reset_index(drop=True)
    return pchic_interactions


def get_gm12878_di_index():
    ins = pd.read_csv(processed_data_path / "di_score_r25kb_w1Mb.txt", index_col=0, sep="\t")
    ins['chrom'] = 'chr' + ins.chrom.astype(str)
    ins.columns = coords + ['bad', 'di_ratio', 'di_index']
    return ins


def get_gaps():
    gaps = pd.read_csv(external_data_path / "hg19_gaps.txt", sep="\t")
    gaps.columns = ['chr', 'start', 'end', 'gap_type']
    gaps = gaps.sort_values(coords)
    return gaps

def get_region_center(consensus_bounds):
    consensus_bounds_fixed = consensus_bounds.copy()
    consensus_bounds_centers = (consensus_bounds.start + consensus_bounds.end)//2
    consensus_bounds_fixed['start'] = consensus_bounds_centers
    consensus_bounds_fixed['end'] = consensus_bounds_centers
    return consensus_bounds_fixed

ctcf_context_categories = {
    '>>>': 'S',
    '<<<': 'S',
    '><<': 'C',
    '>><': 'C',
    '<>>': 'D',
    '<<>': 'D',
    '<><': 'CD',
    '><>': 'CD',
    'T': 'T'
}

ctcf_context_categories = defaultdict(lambda: np.NaN, ctcf_context_categories)

def get_context(ctcf_list):
    if isinstance(ctcf_list, list):
        ctcf_list = np.array(ctcf_list)
    res = []
    previous = None
    for i in range(len(ctcf_list)):
        if (i == 0) or (i == len(ctcf_list) - 1):
            res.append('T')
        else:
            cat = ctcf_context_categories["".join(ctcf_list[i-1:i+2])]
            res.append(cat)
    return res
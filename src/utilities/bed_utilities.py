from pybedtools.bedtool import BedTool
from pathlib import Path
import os
import pandas as pd

resources_path = Path(".") / "src" / "resources"

coords = ['chr', 'start', 'end']
chroms = ['chr{}'.format(x) for x in range(1, 23)] + ['chrX', 'chrY']

def coverage_by_window(windows, ctcfs, agg, null=0):
    cns = []
    c = []
    o = []
    for cn,v in agg.items():
        ci = ctcfs.columns.tolist().index(cn) + 1
        c.append(ci)
        cns.append(cn)
        o.append(v)
    windows_with_ctcfs = BedTool.from_dataframe(windows).map(BedTool.from_dataframe(ctcfs), c=c, o=o, null=null)\
                                .to_dataframe(names=windows.columns.tolist() + cns)
    return windows_with_ctcfs


def windowing_by_size(centered_boundaries, window_size):    
    windows = BedTool().window_maker(b=BedTool.from_dataframe(centered_boundaries), 
                                     w=window_size, i='srcwinnum')\
                       .to_dataframe(names=centered_boundaries.columns.tolist())
    idxs = windows[centered_boundaries.columns[-1]].str.split("_", expand=True)
    tad_ids = idxs.iloc[:, :-1].apply(lambda x: "_".join(x), axis=1)
    w_nums = idxs.iloc[:, -1].astype(int) - 1
    windows[centered_boundaries.columns[-1]] = tad_ids
    windows['w_num'] = w_nums
    windows = windows.sort_values(coords).reset_index(drop=True)
    return windows


def windowing_by_number(all_TADs_by_celltype, n_windows):
    windows = BedTool().window_maker(b=BedTool.from_dataframe(all_TADs_by_celltype), 
                                     n=n_windows, i='srcwinnum')\
                       .to_dataframe(names=all_TADs_by_celltype.columns.tolist())
    idxs = windows[all_TADs_by_celltype.columns[-1]].str.split("_", expand=True)
    tad_ids = idxs.iloc[:, :-1].apply(lambda x: "_".join(x), axis=1)
    w_nums = idxs.iloc[:, -1].astype(int) - 1
    windows[all_TADs_by_celltype.columns[-1]] = tad_ids
    windows['w_num'] = w_nums
    windows = windows.sort_values(coords).reset_index(drop=True)
    return windows


def load_chromsizes(assembly='hg19'):
    fpath = os.path.join(resources_path, "{}_chromsizes.txt".format(assembly))
    r = {}
    with open(fpath, "r") as f:
        line = f.readline()
        while line:
            splits = line.strip().split("\t")
            chrom, length = splits[0], int(splits[1])
            r[chrom] = length
            line = f.readline()
    return r

def shift_test(cons_TADs_pos, int_on_TADs, shifts):
    n_cross = cons_TADs_pos.reset_index(drop=True).copy()
    n_cross['id'] = n_cross.index
    
    res = []
    
    for shift in shifts:
        print(" "*100, end='\r')
        print("\t{}".format(shift), end='\r')
        shifted_bounds = n_cross.copy()
        pos = shifted_bounds.apply(lambda x: max(0, x.start + shift), axis=1)
        shifted_bounds['start'] = pos
        shifted_bounds['end'] = pos

        gh_with_bounds = BedTool.from_dataframe(shifted_bounds)\
                                .map(BedTool.from_dataframe(int_on_TADs), c=1, o='count', null=0, f=1)\
                                .to_dataframe(names=shifted_bounds.columns.tolist() + ['count'])
        gh_with_bounds['shift'] = shift
        res.append(gh_with_bounds)
    return pd.concat(res, axis=0, ignore_index=True)
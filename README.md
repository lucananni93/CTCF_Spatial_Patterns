# Spatial patterns of CTCF sites define the anatomy of TADs and their boundaries
Analysis code repository for the paper "Spatial patterns of CTCF sites define the anatomy of TADs and their boundaries", published in Genome Biology.


## Installing the dependencies
The analysis pipeline runs on Python 3+. Please install the dependencies by running:
```
pip install -r requirements.txt
```

## Downloading the data for the analysis


## Running the analysis code
To reproduce the figures of the paper, run the following scripts from the root folder of this repository. The scripts have to be run in order. All the figures will be generated and put in the `figures/analysis` folder.

### Analysis of CTCF binding sites
```
python src/analysis/ctcf_analysis.py
```

### TAD boundary consensus calls
```
python src/analysis/boundary_consensus.py
```

### CTCF looping simulations
```
python src/analysis/ctcf_looping.py
```

### The CTCF grammar of TADs and boundaries
```
python src/analysis/ctcfs_on_tads.py
python src/analysis/directionality.py
```

### CTCF binding sites at genes and TSSs
```
python src/analysis/ctcfs_on_genes.py
```
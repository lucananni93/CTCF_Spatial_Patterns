from matplotlib import rcParams

mtparams = {
    'axes.labelsize': 'large',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'legend.fontsize': 'medium',
    'axes.titlesize': 'xx-large',
    'lines.markersize': 3,
    'font.size': 9,
    'font.family': 'sans-serif',
    'text.usetex': False,
    'savefig.dpi': 300,
    'figure.figsize': [3.36, 3.36],
    'pdf.fonttype': 42
}


def initialize_plotting_parameters():
	rcParams.update(mtparams)
    
def get_default_figsize():
    return mtparams['figure.figsize']


ctcf_colors = {
	'all': 'grey',
	'forward': 'blue', 
	'reverse': 'red',
    '>': 'blue',
    '<': 'red',

	'S': 'black',
	'C': '#A67B04',
	'D': 'green',
	'CD': '#C59CDB',
    
    '>>': 'grey',
    '<<': 'grey',
    '<>': 'green',
    '><': '#A67B04',
    
    'Same': 'black',
    'Convergent': '#A67B04',
    'Divergent': 'green',
    'Convergent-Divergent': '#C59CDB'
}
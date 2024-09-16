import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'text.usetex': True})
# sns.set()
palette = sns.color_palette('bright')
palette = sns.color_palette('dark')


# set up plot configuration
plot_configs = {}
plot_configs['plot_1'] = {
    'xlabel': r'parallelized $f$ + parallelized $\partial{f}/\partial{(y,p_d,p_s)}$ count',
    # 'xlabel': ' count (with full parallelization)',
    'ylabel': r'optimality',
    'title':  r'Optimization Convergence History (with full parallelization)',
    'plot_type': 'semilogy',
    'plot_indices': ((1,), (0, 1)),
    'final_marker': True,
}
plot_configs['plot_2'] = {
    'xlabel': r'$f$ + $\partial{f}/\partial{(y,p_d,p_s)}$ count',
    # 'xlabel': 'count',
    'ylabel': r'optimality',
    'title':  r'Optimization Convergence History',
    'plot_type': 'semilogy',
    'plot_indices': ((0,), (0, 1)),
    'final_marker': True,
}
plot_configs['plot_3'] = {
    'ylabel': r'relative error',
    # 'ylabel': 'f count',
    'xlabel': r'memory (mb)',
    # 'xlabel': 'partial f count',
    'title':  r'Error Memory Tradeoff',
    'plot_type': 'scatter loglog',
    'plot_indices': ((0,), (3,)),
}
# plot_configs['plot_3'] = {
#     'ylabel': r'$f$ count',
#     # 'ylabel': 'f count',
#     'xlabel': r'$\partial{f}/\partial{(y,p_d,p_s)}$ count',
#     # 'xlabel': 'partial f count',
#     'title':  r'Function Count (with full parallelization)',
#     'plot_type': 'scatter',
#     'plot_indices': ((1,), (2,)),
# }
# plot_configs['plot_4'] = {
#     'ylabel': r'$f$ count',
#     # 'ylabel': 'f count',
#     'xlabel': r'$\partial{f}/\partial{(y,p_d,p_s)}$ count',
#     # 'xlabel': 'partial f  count',
#     'title':  r'Function Count',
#     'plot_type': 'scatter',
#     'plot_indices': ((0,), (2,)),
# }
plot_configs['plot_4'] = {
    'ylabel': r'relative error',
    # 'ylabel': 'f count',
    'xlabel': r'optimization wall time (s)',
    # 'xlabel': 'partial f  count',
    'title':  r'Error Time Tradeoff',
    'plot_type': 'scatter loglog',
    'plot_indices': ((0,), (2,)),
}
# plot_configs['plot_5'] = {
#     'xlabel': r'Finite difference step size',
#     'ylabel': r'Relative error',
#     'title':  r'Derivative Verification',
#     'plot_type': 'loglog',
#     'plot_indices': ((2,), (0,)),
# }
plot_configs['plot_6'] = {
    'xlabel': r'optimization wall time (s)',
    'ylabel': r'memory (mb)',
    'title':  r'Memory Time Tradeoff',
    'plot_type': 'scatter',
    'plot_indices': ((1,), (2,)),
    # 'bottom_y': -0.5,
    # 'left_x': -0.05,
}

N_ROWS = max([max(config_dict['plot_indices'][0]) for config_dict in plot_configs.values()])+1
N_COLS = max([max(config_dict['plot_indices'][1]) for config_dict in plot_configs.values()])+1

# approeach configuration
method_config = {}
method_config['ForwardEuler'] = {
    'linestyle': '-',
    'color_ratio': 0.4,
    'legend name': 'Explicit (low-order)',
    'marker': 's',
}
method_config['RK4'] = {
    'linestyle': ':',
    'color_ratio': 0.6,
    'legend name': 'Explicit (high-order)',
    'marker': '^',
}
method_config['ImplicitMidpoint'] = {
    'linestyle': '-.',
    'color_ratio': 1.0,
    'legend name': 'Implicit (low-order)',
    'marker': 'D',
}
method_config['GaussLegendre6'] = {
    'linestyle': '--',
    'color_ratio': 0.8,
    'legend name': 'Implicit (high-order)',
    'marker': 'o',
}
# Method configuration
approach_config = {}
approach_config['Collocation'] = {
    'color': palette[8],
    'legend name': 'Collocation',
}
approach_config['TimeMarching'] = {
    'color': palette[6],
    'legend name': 'Time-marching',
}
approach_config['Picard Iteration'] = {
    'color': palette[2],
    'legend name': 'Picard iteration',
}
approach_config['TimeMarching (Checkpointing)'] = {
    'color': 'dodgerblue',
    'legend name': 'Time-marching \n(checkpointing)',
}


# function to build color
def build_color(color, ratio):
    tuple_return = []
    for i in color:

        if i*ratio < 1.0:
            tuple_return.append(i*ratio)
        else:
            tuple_return.append(1.0)

    return tuple(tuple_return)

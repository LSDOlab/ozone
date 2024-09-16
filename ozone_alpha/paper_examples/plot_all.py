from ozone_alpha.paper_examples.utils.plot_configurations import  plot_configs, method_config, approach_config, build_color, N_ROWS, N_COLS

if __name__ == '__main__':
    import os
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from ozone_alpha.paper_examples.utils.read_snopt import pad_opt_data

    sns.set(rc={'text.usetex': True})
    # sns.set_style("white")
    sns.set_style("ticks")

    # sns.set()

    palette = sns.color_palette('deep')

    dir_list_data = os.listdir('plot_data')
    dir_list = []
    for dir_path_w_data in dir_list_data:
        dir_list.append(f'plot_data/{dir_path_w_data}')

    data_plot_dict = {}
    for dir in dir_list:
        if os.path.isdir(dir):
            if 'DATA' in dir:
                sys_name = dir
                processed_sys_name = sys_name[10:].replace('_', ' ')
                print(sys_name)
                # if processed_sys_name in ['ASCENT SYSTEM','VAN DER POL OSCILATOR']:
                #     continue
                data_plot_dict[sys_name] = {}
                data_plot_dict[sys_name]['suptitle'] = processed_sys_name
                data_files = os.listdir(dir)

                # print(f'\nREADING {processed_sys_name}')

                MAX_OPT_TIME = 0.0
                for data_file in data_files:
                    split = data_file.split('__')

                    # extract method and approach
                    approach = split[0]
                    method = split[1].split('.')[0]

                    # if method in ['ImplicitMidpoint', 'ForwardEuler']:
                    #     continue
                    if approach in ['time-marching checkpointing']:
                        if processed_sys_name in ['Nonlinear Reaction Diffusion Process']:
                            pass
                        else:
                            continue

                    # get actual data of method and approach to plot
                    with open(f'{dir}/{data_file}', 'rb') as handle:
                        data_pickle = pickle.load(handle)

                    if data_pickle['LINE_EXIT'].split()[2] != '0':
                        continue

                    opt_time = data_pickle['OPT_TIME']
                    if opt_time > MAX_OPT_TIME:
                        MAX_OPT_TIME = opt_time

                for data_file in data_files:
                    split = data_file.split('__')

                    # extract method and approach
                    approach = split[0]
                    method = split[1].split('.')[0]

                    # if method in ['ImplicitMidpoint', 'ForwardEuler']:
                    #     continue
                    # print(processed_sys_name)


                    # get actual data of method and approach to plot
                    with open(f'{dir}/{data_file}', 'rb') as handle:
                        data_pickle = pickle.load(handle)

                    exit_status = data_pickle['LINE_EXIT'].split()[2]
                    # if isinstance(data_pickle['MAJOR'], np.ndarray):
                    #     num_major = data_pickle['MAJOR'][-1]
                    # else:
                    #     num_major = data_pickle['MAJOR']
                    if isinstance(data_pickle['OPTIMALITY'], np.ndarray):
                        opt = data_pickle['OPTIMALITY'][-1]
                        opt_fail = False
                    else:
                        opt = data_pickle['OPTIMALITY']
                        opt_fail = True

                    print(f'\t{data_file}')
                    print(f'\t    EXIT STATUS:    {exit_status}')
                    # print(f'\t    NUM MAJOR ITER: {num_major}')
                    print(f'\t    OPTIMALITY:     {opt}')

                    # Do not care about results that do not converge
                    if exit_status != '0':
                        continue
                    if opt_fail:
                        continue
                    obj_val = data_pickle['OBJECTIVE']
                    mem = data_pickle['ALLOCATED_MEMORY']
                    nfc = data_pickle['HIST_NUM_F_CALLS'][-1]
                    ndfc = data_pickle['HIST_NUM_DF_CALLS'][-1]
                    print(f'\t    OBJECTIVE:      {obj_val}')
                    print(f'\t    # F CALLS:      {nfc}')
                    print(f'\t    # DF CALLS:     {ndfc}')
                    print(f'\t    MEMORY:         {mem}mb')
                    print(f'\t    Opt Time:       {data_pickle["OPT_TIME"]}')
                    print(f'\t    Num sim calls:  {len(data_pickle["HIST_NUM_F_CALLS"])}')

                    if approach in ['time-marching checkpointing']:
                        if processed_sys_name in ['Nonlinear Reaction Diffusion Process']:
                            pass
                        else:
                            continue
                    # create data for each planned plot
                    # for key in data_pickle:
                    #     print(f'data_pickle[\'{key}\']')

                    data_plot_dict[sys_name][(approach, method)] = {}

                    # PLOT 1: Optimization Progress
                    data_plot_dict[sys_name][(approach, method)]['plot_1'] = {}
                    # "fix" data, if the optimality iters and f_call iters do not match by small error, pad shorter one.
                    history_vf_calls = [sum(x) for x in zip(data_pickle['HIST_NUM_VECTORIZED_F_CALLS'], data_pickle['HIST_NUM_VECTORIZED_DF_CALLS'])]
                    opt_hist = data_pickle['OPTIMALITY'].copy().tolist()
                    history_vf_calls, opt_hist = pad_opt_data(history_vf_calls, opt_hist)

                    data_plot_dict[sys_name][(approach, method)]['plot_1']['y'] = [opt_hist]
                    data_plot_dict[sys_name][(approach, method)]['plot_1']['x'] = history_vf_calls

                    # PLOT 2: Optimization Progress
                    data_plot_dict[sys_name][(approach, method)]['plot_2'] = {}

                    # "fix" data, if the optimality iters and f_call iters do not match by small error, pad shorter one.
                    history_f_calls = [sum(x) for x in zip(data_pickle['HIST_NUM_F_CALLS'], data_pickle['HIST_NUM_DF_CALLS'])]
                    opt_hist = data_pickle['OPTIMALITY'].copy().tolist()
                    history_f_calls, opt_hist = pad_opt_data(history_f_calls, opt_hist)

                    data_plot_dict[sys_name][(approach, method)]['plot_2']['y'] = [opt_hist]
                    data_plot_dict[sys_name][(approach, method)]['plot_2']['x'] = history_f_calls

                    # PLOT 3: RHS eval vs vectorized RHS
                    data_plot_dict[sys_name][(approach, method)]['plot_3'] = {}
                    data_plot_dict[sys_name][(approach, method)]['plot_3']['y'] = data_pickle['HIST_NUM_VECTORIZED_F_CALLS'][-1]
                    data_plot_dict[sys_name][(approach, method)]['plot_3']['x'] = data_pickle['HIST_NUM_VECTORIZED_DF_CALLS'][-1]

                    # PLOT 4: dRHS eval vs vectorized dRHS
                    data_plot_dict[sys_name][(approach, method)]['plot_4'] = {}
                    data_plot_dict[sys_name][(approach, method)]['plot_4']['y'] = data_pickle['HIST_NUM_F_CALLS'][-1]
                    data_plot_dict[sys_name][(approach, method)]['plot_4']['x'] = data_pickle['HIST_NUM_DF_CALLS'][-1]
                    nfc = data_pickle['HIST_NUM_F_CALLS'][-1]
                    ndfc = data_pickle['HIST_NUM_DF_CALLS'][-1]

                    # PLOT 5: step size vs derivative error
                    data_plot_dict[sys_name][(approach, method)]['plot_5'] = {}
                    data_plot_dict[sys_name][(approach, method)]['plot_5']['x'] = []
                    data_plot_dict[sys_name][(approach, method)]['plot_5']['y'] = []

                    if 'CHECK' in data_pickle:
                        for i, step_size in enumerate(data_pickle['CHECK']):
                            data_plot_dict[sys_name][(approach, method)]['plot_5']['x'].append(step_size)

                            check_dict = data_pickle['CHECK'][step_size]
                            if i == 0:
                                data_plot_dict[sys_name][(approach, method)]['plot_5']['y'].append([])

                            avg_rel_error = 0.0
                            num_keys = 0.0
                            # num_keys = len(list(check_dict.keys()))
                            for of_wrt in check_dict.keys():

                                # if (check_dict[of_wrt]['abs_error_norm'] < 1e-5) and (check_dict[of_wrt]['relative_error_norm'] > 1e-4):
                                #     check_dict[of_wrt]['relative_error_norm'] = check_dict[of_wrt]['abs_error_norm']
                                    # continue
                                error = np.linalg.norm(check_dict[of_wrt]['value'] - check_dict[of_wrt]['fd_value'])
                                rel_error = error/np.linalg.norm(check_dict[of_wrt]['fd_value'])
                                # if (rel_error == 1e-5):
                                #     continue

                                avg_rel_error += rel_error
                                num_keys += 1
                                # if approach == 'solver-based':
                                # print(check_dict)
                            #         print(step_size, of_wrt, '\t\t',check_dict[of_wrt]['relative_error_norm'],check_dict[of_wrt]['abs_error_norm'], check_dict[of_wrt]['analytical_norm'])
                            # print(check_dict)
                            avg_rel_error = avg_rel_error/num_keys
                            data_plot_dict[sys_name][(approach, method)]['plot_5']['y'][0].append(avg_rel_error)

                    # PLOT 6:
                    opt_time = data_pickle['OPT_TIME']
                    data_plot_dict[sys_name][(approach, method)]['plot_6'] = {}
                    data_plot_dict[sys_name][(approach, method)]['plot_6']['y'] = data_pickle['ALLOCATED_MEMORY']
                    data_plot_dict[sys_name][(approach, method)]['plot_6']['x'] = opt_time/MAX_OPT_TIME

                    # mem = data_pickle['ALLOCATED_MEMORY']
                    # print(f'\t    MEMORY:         {mem}mb')
                    # print(f'\t    # F CALLS:      {nfc}')
                    # print(f'\t    # DF CALLS:     {ndfc}')

                    # data_plot_dict[sys_name][(approach, method)]['plot_6']['x'] = data_pickle['NUM_F_CALLS'] + data_pickle['NUM_DF_CALLS']


    # create subplot
    num_cols = N_COLS
    num_rows = N_ROWS


    # loop through systems
    for (sys_name, sys_data_dict) in (data_plot_dict.items()):
        # plot
        # f, ax = plt.subplots(num_rows, num_cols, figsize=(10, 5))
        # f = plt.figure(figsize=(19, 10))
        f = plt.figure(figsize=(11, 9.5))

        f.suptitle(sys_data_dict['suptitle'], fontsize = 20, y = 0.945)
        gs = f.add_gridspec(num_rows, num_cols)
        # loop through plot type
        for (plot_key, plot_config) in (plot_configs.items()):

            col_index = plot_config['plot_indices'][1]
            row_index = plot_config['plot_indices'][0]

            # get current axis
            if (len(col_index) == 1) and (len(row_index) == 1):
                a = f.add_subplot(gs[row_index[0], col_index[0]])
            elif (len(col_index) == 2) and (len(row_index) == 2):
                a = f.add_subplot(gs[row_index[0]:row_index[1]+1, col_index[0]:col_index[1]+1])
            elif len(col_index) == 2:
                a = f.add_subplot(gs[row_index[0], col_index[0]:col_index[1]+1])
            elif len(row_index) == 2:
                a = f.add_subplot(gs[row_index[0]:row_index[1]+1, col_index[0]])

            ttl = plot_config['title']
            print(f'PLOTTING {plot_key} ({ttl})')

            # loop through methods and approaches of system and plot type
            for method_approach_key, method_approach_data in sys_data_dict.items():
                if method_approach_key == 'suptitle':
                    continue
                method = method_approach_key[1]
                approach = method_approach_key[0]
                legend_name = method+' '+approach
                linewidth = 1.0

                # colors
                m_config = method_config[method]
                a_config = approach_config[approach]
                color = a_config['color']
                # color = build_color(a_config['color'], m_config['color_ratio'])
                linestyle = m_config['linestyle']

                # plot actual data
                plot_data = method_approach_data[plot_key]

                if plot_config['plot_type'] == 'scatter':
                    a.plot(plot_data['x'],
                        plot_data['y'],
                        marker=m_config['marker'],
                        c=color,
                        label=legend_name,
                        )
                elif plot_config['plot_type'] == 'scatter loglog':
                    a.loglog(plot_data['x'],
                            plot_data['y'],
                            marker=m_config['marker'],
                            c=color,
                            label=legend_name,
                            )
                else:
                    for y_data in plot_data['y']:
                        if plot_config['plot_type'] == 'plot':
                            a.plot(plot_data['x'], y_data, linestyle, color=color, linewidth=linewidth)
                        elif plot_config['plot_type'] == 'semilogy':
                            # print(method, approach, plot_data['x'], y_data)
                            # print(method, approach, len(plot_data['x']), len(y_data))
                            a.semilogy(plot_data['x'], y_data, linestyle, color=color, linewidth=linewidth)
                        elif plot_config['plot_type'] == 'loglog':
                            a.loglog(plot_data['x'], y_data, linestyle, color=color, linewidth=linewidth)

                        if 'final_marker' in plot_config:
                            if plot_config['final_marker']:
                                a.plot(plot_data['x'][-1],
                                y_data[-1],
                                marker=m_config['marker'],
                                c=color,
                                label=legend_name,
                                )                
                

                # labels
                a.set_title(plot_config['title'])
                a.set_xlabel(plot_config['xlabel'])
                a.set_ylabel(plot_config['ylabel'])
            
            if "left_x" in plot_config:
                a.set_xlim(left=plot_config['left_x'])
            if "bottom_y" in plot_config:
                a.set_ylim(bottom=plot_config['bottom_y'])
            plt.grid()

        custom_lines_approach = []
        custom_names_approach = []
        n_ma = 0
        for approach_name, a_config in approach_config.items():
            color = a_config['color']
            custom_lines_approach.append(Line2D([0], [0], color=color, lw=4))

            approach_legend_name = a_config['legend name']
            custom_names_approach.append(f'{approach_legend_name}')
            n_ma += 1

        custom_lines_method = []
        custom_names_method = []
        for method_name, m_config in method_config.items():
            custom_lines_method.append(Line2D([0], [0], lw=1.0, color='black', markersize=4.0, linestyle=m_config['linestyle'], marker=m_config['marker']))
            custom_names_method.append(m_config['legend name'])

        f.legend(custom_lines_approach, custom_names_approach, bbox_to_anchor=(.87, 0.31), ncol=1, prop={'size': 8.9})
        f.legend(custom_lines_method, custom_names_method, bbox_to_anchor=(.89, 0.19), ncol=1, prop={'size': 8.9}, handlelength=3.5)
        # f.tight_layout()
        # plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1, wspace=0.2, hspace=0.2)
        plt.subplots_adjust(wspace=0.53, hspace=0.55)

        if not os.path.isdir('figures'):
            os.mkdir('figures')
        filename = 'figures/'+sys_data_dict['suptitle']
        plt.savefig(filename, dpi=400,bbox_inches='tight')

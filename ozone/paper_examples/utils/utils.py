import modopt as mo
import csdl_alpha as csdl
import time
import ozone as ozone
from ozone.paper_examples.utils.read_snopt import read_most_recent_snopt_hist
import numpy as np

def count_f_df_calls(
        rec, 
        ode_problem, 
        snopt_options, 
        recorder_options, 
        save_dict:dict,
        measure_memory, 
        count_f_df_calls,
        num_outs = None,
        ):

    recorder_options['gpu'] = False
    # Build modified problem/simulator to count nlsolvers stuff
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        **recorder_options,
    )
    if hasattr(rec, 'nlsolvers_instances'):
        for instances in rec.nlsolvers_instances:
            instances.do_count = True

    jax_sim.compute_optimization_derivatives()
    if measure_memory:
        print('Measuring memory...')
        # Compute memory:
        def f():
            for i in range(5):
                start = time.time()
                jax_sim.compute_optimization_derivatives()
                # jax_sim.run_forward()
                print(f'end {i}:', time.time()- start)
        from memory_profiler import memory_usage
        mem_usage = memory_usage(f, interval=1e-6)
        ALLOCATED_MEMORY = max(mem_usage) - min(mem_usage)

        import resource
        before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        for i in range(5):
            start = time.time()
            jax_sim.compute_optimization_derivatives()
            # jax_sim.run_forward()
            print(f'end {i}:', time.time()- start)

        after_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ALLOCATED_MEMORY_r = 1024.0*(after_mem - before_mem)

        save_dict['ALLOCATED_MEMORY'] = ALLOCATED_MEMORY
        save_dict['ALLOCATED_MEMORY_r'] = ALLOCATED_MEMORY_r
        print('Memory:', ALLOCATED_MEMORY)
        print('Memory_r:', ALLOCATED_MEMORY_r)

    if count_f_df_calls:
        jax_sim.nl_timemarching_counts = []
        jax_sim.nl_picard_counts = []
        jax_sim.calls = 0
        rec.nlsolvers = {}

        def update_counts(sim):
            if 'ozone_stage_solver' in rec.nlsolvers:
                sim.nl_timemarching_counts.append(rec.nlsolvers['ozone_stage_solver'])
                print('stage solver:', rec.nlsolvers['ozone_stage_solver'])
            if 'ozone_picard_solver' in rec.nlsolvers:
                sim.nl_picard_counts.append(rec.nlsolvers['ozone_picard_solver'])
            sim.calls += 1
        jax_sim.add_callback(update_counts)

        prob = mo.CSDLAlphaProblem(problem_name='name',simulator=jax_sim)
        optimizer = mo.SNOPT(
            prob,
            solver_options = snopt_options,
            turn_off_outputs = True,
        )
        start = time.time()
        optimizer.solve()
        end = time.time()

        # Post-process to compute the number of calls
        num_calls = jax_sim.calls
        num_f = [0]
        num_vec_f = [0]
        num_df = [0]
        num_vec_df = [0]

        approach = ode_problem.integrator.approach
        method = ode_problem.integrator.method
        num_stages = method.num_stages
        num_steps = ode_problem.integrator.num_steps

        if num_outs is None:
            num_outs = 1 + sum(constraint.size for constraint in rec.constraints.keys())
            
        if isinstance(approach, ozone.approaches.TimeMarchingCheckpoints):
            if not method.explicit:
                assert len(jax_sim.nl_timemarching_counts) == num_calls
                for i in range(num_calls):
                    # F is called once per nlsolver iteration
                    num_f.append(jax_sim.nl_timemarching_counts[i]*num_stages)
                    num_vec_f.append(jax_sim.nl_timemarching_counts[i])

                    # DF is called once per nlsolver iteration (from newton's method)
                    num_df.append(jax_sim.nl_timemarching_counts[i]*num_stages)
                    num_vec_df.append(jax_sim.nl_timemarching_counts[i])
            else:
                for i in range(num_calls):
                    # For a forward evaluation: F is called num_stages times per timestep
                    # For a derivative evaluation: F is same as forward evaluation * num outputs
                    num_f.append(num_f[-1] + num_stages*num_steps*(1+num_outs))
                    num_vec_f.append(num_f[-1])

                    # DF is called num_stages times per timestep per output per derivative
                    num_df.append(num_df[-1]+num_stages*num_steps*(num_outs))
                    num_vec_df.append(num_df[-1])
        elif isinstance(approach, ozone.approaches.TimeMarching):
            if not method.explicit:
                assert len(jax_sim.nl_timemarching_counts) == num_calls
                for i in range(num_calls):
                    # F is called once per nlsolver iteration
                    num_f.append(jax_sim.nl_timemarching_counts[i]*num_stages)
                    num_vec_f.append(jax_sim.nl_timemarching_counts[i])

                    # DF is called once per nlsolver iteration (from newton's method)
                    num_df.append(jax_sim.nl_timemarching_counts[i]*num_stages)
                    num_vec_df.append(jax_sim.nl_timemarching_counts[i])
            else:
                for i in range(num_calls):
                    # For a forward evaluation: F is called num_stages times per timestep
                    # For a derivative evaluation: F is same as forward evaluation * num outputs
                    num_f.append(num_f[-1] + num_stages*num_steps*(1))
                    num_vec_f.append(num_f[-1])

                    # DF is called num_stages times per timestep per output per derivative
                    num_df.append(num_df[-1] + num_stages*num_steps*(num_outs))
                    num_vec_df.append(num_df[-1])
        elif isinstance(approach, ozone.approaches.PicardIteration):
            assert len(jax_sim.nl_picard_counts) == num_calls
            for i in range(num_calls):
                num_ode_states = len(ode_problem.states.states)
                # The number of individual functions calls = num_nl*num_stages*num_times*NUM_STATES(because nlbgs)
                # The number of individual derivative calls = num_stages*num_times*NUM_STATES
                num_f.append(jax_sim.nl_picard_counts[i]*num_stages*num_steps*num_ode_states)
                num_df.append(num_stages*num_steps*num_ode_states)

                # The number of vectorized function calls = num_nl*NUM_STATES
                # The number of vectorized derivative calls = 1*NUM_STATES
                num_vec_f.append(jax_sim.nl_picard_counts[i]*num_ode_states)
                num_vec_df.append(num_df[-1]+1*num_ode_states)
        elif isinstance(approach, ozone.approaches.Collocation):
            for i in range(num_calls):
                # For a forward evaluation: Function is called num_stages*num_steps
                # For a derivative evaluation: same
                num_f.append(num_f[-1] + num_stages*num_steps)
                num_df.append(num_f[-1])
                num_vec_f.append(num_vec_f[-1] + 1)
                num_vec_df.append(num_vec_f[-1])

        time.sleep(15)
        jax_sim.run()
        hist_opt, line_exit = read_most_recent_snopt_hist()

        save_dict['HIST_NUM_F_CALLS'] = num_f
        save_dict['HIST_NUM_VECTORIZED_F_CALLS'] = num_vec_f
        save_dict['HIST_NUM_DF_CALLS'] = num_df
        save_dict['HIST_NUM_VECTORIZED_DF_CALLS'] = num_vec_df
        save_dict['OPTIMALITY'] = hist_opt
        save_dict['LINE_EXIT'] = line_exit

    if hasattr(rec, 'nlsolvers_instances'):
        for instances in rec.nlsolvers_instances:
            instances.do_count = False

    return

def optimize_and_get_data(
        rec:csdl.Recorder, 
        name:str, 
        stats:dict, 
        approach, 
        method, 
        snopt_options:dict,
        recorder_options:dict = None)->dict:

    # Options on what data to save
    time_optimization = True
    measure_memory = True
    count_optimization_history = True
    check_derivatives = False

    # Initialize dictionary to save data
    save_dict = {}

    # Process recorder options
    if recorder_options is None:
        recorder_options = {}
    if 'derivative_kwargs' not in recorder_options:
        recorder_options['derivatives_kwargs'] = {'loop':False}
    else:
        recorder_options['derivatives_kwargs']['loop'] = False

    # Build Simulator
    rec.start()
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        **recorder_options
    )
    ode_problem = stats['ode_problem']
    dv_values = {dv:dv.value for dv in rec.design_variables.keys()}

    jax_sim.compute_optimization_derivatives()
    # jax_sim.run_forward()

    # print('Measuring memory...')
    # # Compute memory:
    # def f():
    #     for i in range(8):
    #         start = time.time()
    #         jax_sim.compute_optimization_derivatives()
    #         # jax_sim.run_forward()
    #         print(f'end {i}:', time.time()- start)
    # import time
    # from memory_profiler import memory_usage
    # mem_usage = memory_usage(f, interval=1e-6)
    # ALLOCATED_MEMORY = max(mem_usage) - min(mem_usage)

    # Compute finite difference:
    if check_derivatives:
        print('Validating derivatives...')
        CHECK_DICT = {}
        for dx_power in range(1, 10):
            dx = 10 ** (-dx_power)
            # if not save_for_plot:
            #     continue
            check = jax_sim.check_optimization_derivatives(step_size=dx, print_results=True)

            CHECK_DICT[dx] = {}
            for i, (key, value) in enumerate(check.items()):
                CHECK_DICT[dx][i] = value
        save_dict['CHECK'] = CHECK_DICT


    # TIME OPTIMIZATION:
    snopt_options['append2file'] = True
    snopt_options['Verbose'] = False
    if time_optimization:
        prob = mo.CSDLAlphaProblem(problem_name='name',simulator=jax_sim)
        optimizer = mo.SNOPT(
            prob,
            solver_options = snopt_options,
            turn_off_outputs = True,
        )
        start = time.time()
        optimizer.solve()
        end = time.time()
        OPT_TIME = end-start
        save_dict['OPT_TIME'] = OPT_TIME

        return_dict = jax_sim.compute_optimization_derivatives()
        OBJECTIVE = return_dict['f']
        CONSTRAINTS = return_dict['c']
        save_dict['OBJECTIVE'] = OBJECTIVE
        save_dict['CONSTRAINTS'] = CONSTRAINTS

        # dv_vector = np.zeros(jax_sim.dv_meta[])
        solved_dvs = []
        for var in jax_sim.dv_meta:
            no_save = False
            for v_name in var.names:
                if ('stage_dv' in v_name) or ('state_dv' in v_name):
                    no_save = True
            if no_save:
                continue

            solved_dvs.append((var.name, var.value))
            # var.value = dv_vector[self.dv_meta[var]['l_ind']:self.dv_meta[var]['u_ind']].reshape(var.shape)
        save_dict['DVS'] = solved_dvs

    # ========================= VERY SLOW =======================
    if name == "Ascent_System":
        num_outs = 4
    else:
        num_outs = None

    for dv in dv_values:
        dv.value = dv_values[dv]
    count_f_df_calls(
        rec, 
        ode_problem, 
        snopt_options, 
        recorder_options,
        save_dict,
        measure_memory = measure_memory,
        count_f_df_calls = count_optimization_history,
        num_outs=num_outs,
    )

    # ========================= VERY SLOW =======================

    # save_dict = {}

    # RUN:
    # save_dict['ALLOCATED_MEMORY'] = ALLOCATED_MEMORY
    # save_dict['OBJECTIVE'] = OBJECTIVE
    # save_dict['CONSTRAINTS'] = CONSTRAINTS
    save_dict['STATS'] = stats['num_times']

    # CHECK:
    # save_dict['CHECK'] = CHECK_DICT

    # OPTIMIZATION
    # save_dict['NUM_F_CALLS'] = NUM_F_CALLS
    # save_dict['NUM_VECTORIZED_F_CALLS'] = NUM_VECTORIZED_F_CALLS
    # save_dict['NUM_DF_CALLS'] = NUM_DF_CALLS
    # save_dict['NUM_VECTORIZED_DF_CALLS'] = NUM_VECTORIZED_DF_CALLS

    # save_dict['HIST_NUM_F_CALLS'] = HIST_NUM_F_CALLS
    # save_dict['HIST_NUM_VECTORIZED_F_CALLS'] = HIST_NUM_VECTORIZED_F_CALLS
    # save_dict['HIST_NUM_DF_CALLS'] = HIST_NUM_DF_CALLS
    # save_dict['HIST_NUM_VECTORIZED_DF_CALLS'] = HIST_NUM_VECTORIZED_DF_CALLS

    # save_dict['OPTIMALITY'] = OPTIMALITY
    # save_dict['FEASIBILITY'] = FEASIBILITY
    # save_dict['MERIT'] = MERIT
    # save_dict['MAJOR'] = MAJOR
    # save_dict['LINE_EXIT'] = EXIT
    # save_dict['OBJECTIVE_FINAL'] = OBJECTIVE_FINAL
    # save_dict['CONSTRAINTS_FINAL'] = CONSTRAINTS_FINAL
    # save_dict['OPT_TIME'] = OPT_TIME
    # save_dict['OPT_TIME_NEW'] = OPT_TIME_NEW

    print(f'CASE: {name} - {approach.name} - {method}')
    # print('Memory:', ALLOCATED_MEMORY)
    # print('Optimization time:', OPT_TIME)
    return save_dict

def save_pickle(results:dict, name:str, stats:dict, approach, method)->None:
    import os
    import pickle
    data_dir = f'plot_data/DATA_{name}'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Open and UPDATE the dictionary with results
    # Check if file exists first.
    if not os.path.isfile(f'{data_dir}/{approach.name}__{method}.pickle'):
        with open(f'{data_dir}/{approach.name}__{method}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Now update:
    with open(f'{data_dir}/{approach.name}__{method}.pickle', 'rb') as handle:
        old_results = pickle.load(handle)
    for key, value in results.items():
        old_results[key] = value
    with open(f'{data_dir}/{approach.name}__{method}.pickle', 'wb') as handle:
        pickle.dump(old_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

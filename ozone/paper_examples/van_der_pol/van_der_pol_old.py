import numpy as np
nthreads = 1 ### (in script)) set # of numpy threads
import os
# os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
import time

import csdl_alpha as csdl
import ozone as ozone
from ozone.paper_examples.van_der_pol.main_model import build_recorder
import numpy as np
import time

def get_model_van_der_pol(
        approach:ozone.approaches._Approach,
        method:str,
        num_times:int = 40,
        tf:float = 15,
        plot = False,
    )->tuple[csdl.Recorder, str, dict]:

    # See inside this function to see CSDL/Ozone implementation
    recorder, ode_problem, nt = build_recorder(approach, method, num_times, tf, plot)

    return recorder, "Van_Der_Pol_Oscillator", {'num_times': nt, 'ode_problem': ode_problem}

if __name__ == '__main__':
    # This script will run the van der pol oscillator optimization problem using Ozone
    # The optimization problem is to minimize the cost function J = x0(tf)^2 + x1(tf)^2
    # subject to the dynamics of the van der pol oscillator

    # The script will plot 
    # - the states, control, and cost function J compared to Dymos
    # - a comparison of the measured collocation forward and collocation adjoint evaluation times
    # - a comparison of the measured time marching forward and time marching adjoint evaluation times
    # The timing comparisons all use implicit methods of similar orders. However, the results are to be taken with a grain of salt as the implementations are not identical.
    # Differences:
    # Scipy:
    # - Scipy uses adaptive time-stepping (and no derivatives) and also only has RK45
    # Dymos:
    # - Dymos uses graph coloring for adjoints
    # Tensorflow:
    # - Only has adaptive time-stepping
    # - Assumes float32



    # User parameters:
    num_times = 40 # Can edit but will affect plots
    tf = 15 # Can edit but will affect plots
    approach_1 = ozone.approaches.TimeMarching()
    approach_2 = ozone.approaches.Collocation()
    method_1 = 'RK4'
    method_2 = 'GaussLegendre6'

    ozone_results = []
    for method, approach in zip([method_1, method_2],[approach_1, approach_2]):
        # build and get CSDL recorder containing ODE/optimization problem
        rec, name, stats = get_model_van_der_pol(
            approach = approach,
            method = method,
            num_times = num_times,
            tf = tf,
            plot = True
        )

        # JIT Compile model/derivative evaluation to JAX
        jax_sim = csdl.experimental.JaxSimulator(
            recorder=rec,
            gpu = False,
            additional_outputs=[rec._find_variables_by_name(name)[0] for name in ['full_x0', 'full_x1', 'full_J', 'full_h', 'full_u']],
            derivatives_kwargs={'loop':False}
        )
        rec.start()

        # Solve Optimization problem
        import modopt as mo
        prob = mo.CSDLAlphaProblem(problem_name='quartic',simulator=jax_sim)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter':3000, 'ftol':1e-6})

        # Time optimization
        start = time.time()
        optimizer.solve()
        end = time.time()
        opt_time = end-start
        print(f"Ozone optimization time: {opt_time}")

        forward_time = 100
        for _ in range(10):
            start = time.time()
            jax_sim.run_forward()
            end = time.time()
            forward_time_current = end-start
            if forward_time_current < forward_time:
                forward_time = forward_time_current
        print(f"Forward Evaluation time: ", forward_time)

        adjoint_time = 100
        for _ in range(10):
            start = time.time()
            derivs = jax_sim.compute_optimization_derivatives()
            end = time.time()
            adjoint_time_current = end-start
            if adjoint_time_current < adjoint_time:
                adjoint_time = adjoint_time_current
        print(f"Adjoint Evaluation time: ", adjoint_time)
        
        # Extract outputs to plot:
        jax_sim.run()
        x0 = jax_sim['full_x0'].flatten()
        x1 = jax_sim['full_x1'].flatten()
        J = jax_sim['full_J'].flatten()
        h = jax_sim['full_h'].flatten()
        u = jax_sim['full_u'].flatten()
        # print(u.tolist())
        # print(derivs['df'])

        timespan = np.zeros(x0.shape)
        timespan[1:] = np.cumsum(h)
        ozone_results.append({
            'timespan':timespan,
            'state_x0': x0,
            'state_x1': x1,
            'state_J': J,
            'control_u': u,
            'opt_time': opt_time,
            'forward_time': forward_time,
            'adjoint_time': adjoint_time
        })

    ozone_results_1 = ozone_results[0]
    ozone_results_2 = ozone_results[1]

    # Plot:
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(rc={'text.usetex': True}, font_scale=1.2)
    sns.set_style("ticks")

    palette = sns.color_palette('deep')
    sns.set_palette(palette)

    try:
        with open('dymos_vdp.pickle', 'rb') as handle:
            dymos_results = pickle.load(handle)
    except Exception as e:
        raise ValueError("Dymos results not found, please run van_der_pol_dymos.py to generate them: ", e)

    try:
        with open('ozone_col_old_vdp.pickle', 'rb') as handle:
            ozone_old_results = pickle.load(handle)
    except Exception as e:
        raise ValueError("Old Ozone results not found, please run van_der_pol_old_ozone_col.py to generate them: ", e)

    try:
        with open('ozone_tm_old_vdp.pickle', 'rb') as handle:
            ozone_old_tm_results = pickle.load(handle)
    except Exception as e:
        raise ValueError("Old Ozone results not found, please run van_der_pol_old_ozone_tm.py to generate them: ", e)

    try:
        with open('scipy_vdp.pickle', 'rb') as handle:
            scipy_results = pickle.load(handle)
    except Exception as e:
        raise ValueError("Scipy results not found, please run van_der_pol_scipy.py to generate them: ", e)

    try:
        with open('tensorflow_vdp.pickle', 'rb') as handle:
            tf_results = pickle.load(handle)
    except Exception as e:
        raise ValueError("Tensorflow results not found, please run van_der_pol_tensorflow.py to generate them: ", e)

    try: 
        with open('torchdiffeq_vdp.pickle', 'rb') as handle:
            torchdiffeq_results = pickle.load(handle)
    except Exception as e:
        raise ValueError("Torchdiffeq results not found, please run van_der_pol_pytorch.py to generate them: ", e)

    m1 = ''
    m2 = 'x'
    m3 = 'o'
    use_log = True
    upper_lim = 0.7
    lower_lim = 1e-5

    ozone_color = palette[0]
    dymos_color = palette[3]
    old_ozone_color = palette[2]
    scipy_color = palette[4]
    tf_color = palette[1]
    torchdiffeq_color = palette[5]

    # FORWARD COLLOCATION:
    f, a = plt.subplots(ncols=3,nrows=2, figsize=(15, 7))
    a_mem = a[0,1]
    a_mem.bar([r'$ozone$'], (ozone_results_2['forward_time']), color=ozone_color, log=use_log)
    a_mem.bar([r'$dymos$'], (dymos_results['forward_time']), color=dymos_color, log=use_log)
    a_mem.bar([r'$ozone$' + '\n'+ r'$(OpenMDAO)$'], (ozone_old_results['forward_time']), color=old_ozone_color, log=use_log)
    
    a_mem.set_title(r'Measured Forward Evaluation Time (Collocation)')
    # a_mem.set_ylabel(r'wall time (sec)')
    # set y lims
    a_mem.set_ylim([lower_lim, upper_lim])
    a_mem.grid()
    a_mem.set_ylabel(r'wall time (sec)')
    
    # ADJOINT COLLOCATION:
    a_mem = a[1,1]
    a_mem.bar([r'$ozone$'], (ozone_results_2['adjoint_time']), color=ozone_color, log=use_log)
    a_mem.bar([r'$dymos$'], (dymos_results['adjoint_time']), color=dymos_color, log=use_log)
    a_mem.bar([r'$ozone$' + '\n'+ r'$(OpenMDAO)$'], (ozone_old_results['adjoint_time']), color=old_ozone_color, log=use_log)

    a_mem.set_title(r'Measured Adjoint Evaluation Time (Collocation)')
    a_mem.set_ylim([lower_lim, upper_lim])
    a_mem.grid()
    a_mem.set_ylabel(r'wall time (sec)')

    # FORWARD TIME MARCHING:
    a_mem = a[0,2]
    a_mem.bar([r'$ozone$'], (ozone_results_1['forward_time']), color=ozone_color, log=use_log)
    a_mem.bar([r'$tensorflow$'], (tf_results['forward_time']), color=tf_color, log=use_log)
    a_mem.bar([r'$scipy$'], (scipy_results['forward_time']), color=scipy_color, log=use_log)
    a_mem.bar([r'$torchdiffeq$'], (torchdiffeq_results['forward_time']), color=torchdiffeq_color, log=use_log)
    a_mem.bar([r'$ozone$' + '\n'+ r'$(OpenMDAO)$'], (ozone_old_tm_results['forward_time']), color=old_ozone_color, log=use_log)
    # a_mem.bar([''], (ozone_results_1['forward_time']), color=ozone_color, log=use_log)
    # a_mem.bar([''], (tf_results['forward_time']), color=tf_color, log=use_log)
    # a_mem.bar([''], (scipy_results['forward_time']), color=scipy_color, log=use_log)
    # a_mem.bar([''], (torchdiffeq_results['adjoint_time']), color=torchdiffeq_color, log=use_log)
    # a_mem.bar([''], (ozone_old_tm_results['forward_time']), color=old_ozone_color, log=use_log)
    # a_mem.xticks([])

    a_mem.set_title(r'Measured Forward Evaluation Time (Time Marching)')
    # a_mem.set_ylabel(r'wall time (sec)')
    a_mem.set_ylim([lower_lim, upper_lim])
    a_mem.grid()
    
    # ADJOINT TIME MARCHING:
    a_mem = a[1,2]
    a_mem.bar([r'$ozone$'], (ozone_results_1['adjoint_time']), color=ozone_color, log=use_log)
    a_mem.bar([r'$tensorflow$'], (tf_results['adjoint_time']), color=tf_color, log=use_log)
    a_mem.bar([r'$scipy$'], (0.0), color=scipy_color, log=use_log)
    a_mem.bar([r'$torchdiffeq$'], (torchdiffeq_results['adjoint_time']), color=torchdiffeq_color, log=use_log)
    a_mem.bar([r'$ozone$' + '\n'+ r'$(OpenMDAO)$'], (ozone_old_tm_results['adjoint_time']), color=old_ozone_color, log=use_log)

    a_mem.set_title(r'Measured Adjoint Evaluation Time (Time Marching)')
    # plt.xticks(rotation=45, ha="right")
    # a_mem.set_ylabel(r'wall time (sec)')
    a_mem.set_ylim([lower_lim, upper_lim])
    a_mem.grid()

    # for i, text in enumerate(a.texts):
    #     print(text.get_text())
    #     # text.set_color(label_colors[i])
    # exit('pass')

    a_states = a[0,0]
    a_states.plot(dymos_results['timespan'], dymos_results['state_J'], linewidth=0.8, linestyle = '--', marker=m3, markersize=2,label=r'$J$', color=dymos_color)
    a_states.plot(dymos_results['timespan'], dymos_results['state_x0'], linewidth=0.8, linestyle = '--', marker=m3, markersize=2,label=r'$dymos$', color=dymos_color)
    a_states.plot(dymos_results['timespan'], dymos_results['state_x1'], linewidth=0.8, linestyle = '--', marker=m3, markersize=2,color=dymos_color)
    a_states.plot(ozone_results_1['timespan'], ozone_results_1['state_x0'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, label=r'$ozone (time marching)$', color=ozone_color)
    a_states.plot(ozone_results_1['timespan'], ozone_results_1['state_x1'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, color=ozone_color)
    a_states.plot(ozone_results_1['timespan'], ozone_results_1['state_J'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, label=r'$J$', color=ozone_color)
    # a_states.plot(ozone_results_2['timespan'], ozone_results_2['state_x0'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, label=r'$ozone$', color=palette[3])
    # a_states.plot(ozone_results_2['timespan'], ozone_results_2['state_x1'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, color=palette[3])
    # a_states.plot(ozone_results_2['timespan'], ozone_results_2['state_J'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, label=r'$J$', color=palette[3])
    # a_states.plot(ozone_old_results['timespan'], ozone_old_results['state_x0'], linewidth=0.8, marker=m3, markersize=2, label=r'$ozone (old)$', color=palette[2])
    # a_states.plot(ozone_old_results['timespan'], ozone_old_results['state_x1'], linewidth=0.8, marker=m3, markersize=2, color=palette[2])
    # a_states.plot(ozone_old_results['timespan'], ozone_old_results['state_J'], linewidth=0.8, marker=m3, markersize=2, label=r'$J$', color=palette[2])
    a_states.set_ylabel(r'states $x_0, x_1$ and cost $J$')
    # a_states.set_xlabel(r'time (s)')
    a_states.set_title(r'Integrated States')
    a_states.grid()

    a_u = a[1,0]
    # a_u.plot(ozone_results_1['timespan'], ozone_results_1['control_u'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, label=r'$ozone (time marching)$', color=palette[0])
    a_u.plot(dymos_results['timespan'], dymos_results['control_u'], linewidth=0.8, linestyle = '--', marker=m3, markersize=2, label=r'$dymos$', color=dymos_color)
    a_u.plot(ozone_results_2['timespan'], ozone_results_2['control_u'], linewidth=0.8, linestyle = '-', marker=m2,markersize=2, label=r'$ozone$', color=ozone_color)
    # a_u.plot(ozone_old_results['timespan'], ozone_old_results['control_u'], linewidth=0.8, marker=m3, markersize=2, label=r'$ozone$ $(old)$', color=palette[2])
    
    a_u.set_ylabel(r'control $u$')
    a_u.set_xlabel(r'time (s)')
    a_u.set_title(r'Optimized Control')
    a_u.legend()
    a_u.grid()

    plt.tight_layout()
    # plt.grid()

    plt.savefig('vdp_accuracy.png', dpi=300)
    plt.show()
        


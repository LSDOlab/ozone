import numpy as np
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
    )->tuple[csdl.Recorder, str, dict]:

    # See inside this function to see CSDL/Ozone implementation
    recorder, _, _ = build_recorder(approach, method, num_times, tf, plot=True)

    return recorder

if __name__ == '__main__':
    # This script will run the van der pol oscillator optimization problem using Ozone
    # The optimization problem is to minimize the cost function J = x0(tf)^2 + x1(tf)^2
    # subject to the dynamics of the van der pol oscillator

    # User parameters:
    num_times = 40 
    tf = 15
    approach_1 = ozone.approaches.TimeMarching()
    method_1 = ozone.methods.RK4()
    
    approach_2 = ozone.approaches.Collocation()
    method_2 = ozone.methods.GaussLegendre6()

    ozone_results = []
    for method, approach in zip([method_1, method_2],[approach_1, approach_2]):
        # build and get CSDL recorder containing ODE/optimization problem
        rec = get_model_van_der_pol(
            approach = approach,
            method = method,
            num_times = num_times,
            tf = tf,
        )

        # JIT Compile model/derivative evaluation to JAX
        jax_sim = csdl.experimental.JaxSimulator(
            recorder=rec,
            additional_outputs=[rec._find_variables_by_name(name)[0] for name in ['full_x0', 'full_x1', 'full_J', 'full_h', 'full_u']],
            derivatives_kwargs={'loop':False}
        )
        # rec.start()

        # Solve Optimization problem
        import modopt as mo
        prob = mo.CSDLAlphaProblem(problem_name='vdp',simulator=jax_sim)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter':3000, 'ftol':1e-6})

        # Time optimization
        start = time.time()
        optimizer.solve()
        end = time.time()
        opt_time = end-start
        
        # Extract outputs to plot:
        rec.execute()
        x0 = jax_sim['full_x0'].flatten()
        x1 = jax_sim['full_x1'].flatten()
        J = jax_sim['full_J'].flatten()
        h = jax_sim['full_h'].flatten()
        u = jax_sim['full_u'].flatten()

        timespan = np.zeros(x0.shape)
        timespan[1:] = np.cumsum(h)
        ozone_results.append({
            'timespan':timespan,
            'state_x0': x0,
            'state_x1': x1,
            'state_J': J,
            'control_u': u,
            'opt_time': opt_time,
        })
    ozone_results_1 = ozone_results[0]
    ozone_results_2 = ozone_results[1]

    # Plot:
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(rc={'text.usetex': True}, font_scale=1.4)
    sns.set_style("ticks")

    palette = sns.color_palette('deep')
    sns.set_palette(palette)

    m1 = 'x'
    m2 = ''
    use_log = True
    upper_lim = 0.7
    lower_lim = 3e-6

    ozone_color = 'lightcoral'
    ozone_color2 = 'cornflowerblue'

    # Plot solved values:
    f = plt.figure(figsize=(15, 9))
    gs = f.add_gridspec(ncols=2,nrows=1)

    # STATES AND CONTROL:
    a_states = f.add_subplot(gs[1])
    a_states.plot(ozone_results_1['timespan'], ozone_results_1['state_x0'], linewidth=2.0, linestyle = '-', marker=m1,markersize=6, label=r'$ozone (timemarching)$', color=ozone_color)
    a_states.plot(ozone_results_1['timespan'], ozone_results_1['state_x1'], linewidth=2.0, linestyle = '-', marker=m1,markersize=6, color=ozone_color)
    a_states.plot(ozone_results_1['timespan'], ozone_results_1['state_J'], linewidth=2.0, linestyle = '-', marker=m1,markersize=6, color=ozone_color)
    a_states.plot(ozone_results_2['timespan'], ozone_results_2['state_x0'], linewidth=2.0, linestyle = '--', marker=m2,markersize=2, label=r'$ozone (collocation)$', color=ozone_color2)
    a_states.plot(ozone_results_2['timespan'], ozone_results_2['state_x1'], linewidth=2.0, linestyle = '--', marker=m2,markersize=2, color=ozone_color2)
    a_states.plot(ozone_results_2['timespan'], ozone_results_2['state_J'], linewidth=2.0, linestyle = '--', marker=m2,markersize=2, color=ozone_color2)
    a_states.set_ylabel(r'states $x_0, x_1$ and cost $J$')
    a_states.set_xlabel(r'time (s)')
    a_states.set_title(r'Integrated States')
    a_states.grid()
    a_states.legend()

    a_u = f.add_subplot(gs[0])
    a_u.plot(ozone_results_1['timespan'], ozone_results_1['control_u'], linewidth=2.0, linestyle = '-', marker=m1,markersize=6, label=r'$ozone (timemarching)$', color=ozone_color)
    a_u.plot(ozone_results_2['timespan'], ozone_results_2['control_u'], linewidth=2.0, linestyle = '--', marker=m2,markersize=2, label=r'$ozone (collocation)$', color=ozone_color2)
    
    a_u.set_ylabel(r'control $u$')
    a_u.set_xlabel(r'time (s)')
    a_u.set_title(r'Optimized Control')
    a_u.legend()
    a_u.grid()

    plt.tight_layout()
    # plt.savefig('vdp_accuracy.png', dpi=300) #uncomment to save figure
    plt.show()
        


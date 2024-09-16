import numpy as np
nthreads = 1 ### (in script)) set # of numpy threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
import csdl_alpha as csdl
import ozone_alpha as ozone
from ozone_alpha.paper_examples.trajectory_optimization.main_model import build_recorder
import numpy as np
import time

def get_model_trajectory_optimization(
        approach:ozone.approaches._Approach,
        method:str,
        num:int=40,
    )->tuple[csdl.Recorder, str, dict]:

    # See inside this function to see CSDL/Ozone implementation
    recorder, ode_problem, nt = build_recorder(num, approach, method)

    return recorder, "Trajectory_Optimization", {'num_times': nt,'ode_problem': ode_problem}

if __name__ == '__main__':
    # User parameters:
    approach = ozone.approaches.Collocation()
    method = 'RK4'
    num = 40

    # build and get CSDL recorder containing ODE/optimization problem
    rec, name, stats = get_model_trajectory_optimization(
        approach = approach,
        method = method,
        num = num,
    )

    x_plot_variable = rec._find_variables_by_name('x_plot')[0]
    h_plot_variable = rec._find_variables_by_name('h_plot')[0]

    # JIT Compile model/derivative evaluation to JAX
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        gpu = False,
        derivatives_kwargs={'loop':False},
        additional_outputs=[x_plot_variable, h_plot_variable]
    )
    jax_sim.run()
    x_initial = jax_sim[x_plot_variable]
    h_initial = jax_sim[h_plot_variable]

    control_x_initial = rec._find_variables_by_name('control_x')[0].value
    control_z_initial = rec._find_variables_by_name('control_z')[0].value
    control_alpha_initial = rec._find_variables_by_name('control_alpha')[0].value
    # Solve Optimization problem
    import modopt as mo
    prob = mo.CSDLAlphaProblem(problem_name='quartic',simulator=jax_sim)
    # optimizer = mo.PySLSQP(prob, turn_off_outputs = True)
    # optimizer = mo.SNOPT(prob,{})
    optimizer = mo.SNOPT(
        prob,
        solver_options = {'Major optimality':2e-4, 'Major feasibility':1e-4,'Major iterations':600,'Iteration limit':100000, 'Verbose':False},
    )
    optimizer.solve()
    optimizer.print_results()

    # Plot:
    import matplotlib.pyplot as plt
    jax_sim.run()
    x = jax_sim[x_plot_variable]
    h = jax_sim[h_plot_variable]



    f = plt.figure(figsize=(11, 2.5))
    # f.suptitle('Optimized Trajectory', fontsize = 20, y = 0.945)
    a = f.add_subplot(1,1,1)
    a.plot(x_initial, h_initial, '-o' ,label = 'Initial Guess', color = 'grey', linewidth = 1.0, markersize = 2.0)
    a.plot(x, h, '-o' ,label = 'Optimized', color = 'black', linewidth = 1.0, markersize = 2.0)
    a.legend()
    a.set_xlabel('Horizontal Displacement (m)')
    a.set_ylabel('Vertical Displacement (m)')
    # plt.show()
    plt.savefig('traj_opt_comparison', dpi=400,bbox_inches='tight')


    f = plt.figure(figsize=(11, 2.5))
    a = f.add_subplot(1,1,1)
    # f.suptitle('Optimized Trajectory', fontsize = 20, y = 0.945)
    rec.execute()
    a.plot(rec._find_variables_by_name('v_plot')[0].value)
    a.plot(rec._find_variables_by_name('gamma_plot')[0].value)
    a.plot(rec._find_variables_by_name('e_plot')[0].value)
    a.legend(['v','gamma','e'])

    f = plt.figure(figsize=(11, 2.5))
    a = f.add_subplot(1,1,1)
    # f.suptitle('Optimized Trajectory', fontsize = 20, y = 0.945)
    rec.execute()
    a.plot(rec._find_variables_by_name('control_x')[0].value)
    a.plot(rec._find_variables_by_name('control_z')[0].value)
    a.plot(rec._find_variables_by_name('control_alpha')[0].value)
    # a.plot(control_x_initial)
    # a.plot(control_z_initial)
    # a.plot(control_alpha_initial)

    # a.legend(['control_x','control_z','control_alpha', 'control_x_initial', 'control_z_initial', 'control_alpha_initial'])
    a.legend(['control_x','control_z','control_alpha'])

    plt.show()
    


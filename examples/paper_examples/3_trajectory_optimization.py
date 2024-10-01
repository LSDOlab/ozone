import numpy as np
import csdl_alpha as csdl
import ozone as ozone
from ozone.paper_examples.trajectory_optimization.main_model import build_recorder
import numpy as np

def get_model_trajectory_optimization(
        approach:ozone.approaches._Approach,
        method:str,
        num:int=40,
    )->csdl.Recorder:

    # See inside this function to see CSDL/Ozone implementation
    recorder, _, _ = build_recorder(num, approach, method)

    return recorder

if __name__ == '__main__':
    # User parameters:
    approach = ozone.approaches.Collocation()
    method = ozone.methods.ImplicitMidpoint()
    num = 40
    max_iter = 200 # Maximum number of iterations for optimization

    # build and get CSDL recorder containing ODE/optimization problem
    rec = get_model_trajectory_optimization(
        approach = approach,
        method = method,
        num = num,
    )

    x_plot_variable = rec.find_variable_by_name('x_plot')
    h_plot_variable = rec.find_variable_by_name('h_plot')

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

    control_x_initial = rec.find_variable_by_name('control_x').value
    control_z_initial = rec.find_variable_by_name('control_z').value
    control_alpha_initial = rec.find_variable_by_name('control_alpha').value
    # Solve Optimization problem
    import modopt as mo
    prob = mo.CSDLAlphaProblem(problem_name='trajectory_opt',simulator=jax_sim)
    optimizer = mo.SLSQP(prob, turn_off_outputs = True, solver_options={'maxiter':max_iter})
    # optimizer = mo.SNOPT(
    #     prob,
    #     solver_options = {'Major optimality':2e-4, 'Major feasibility':1e-4,'Major iterations':600,'Iteration limit':100000, 'Verbose':False},
    # ) #Uncomment to use SNOPT
    optimizer.solve()
    optimizer.print_results()

    # Plot:
    import matplotlib.pyplot as plt
    rec.execute()
    x = jax_sim[x_plot_variable]
    h = jax_sim[h_plot_variable]

    f = plt.figure(figsize=(11, 2.5))
    f.suptitle(f'Trajectory after {max_iter} iterations vs Initial', fontsize = 20, y = 0.945)
    a = f.add_subplot(1,1,1)
    a.plot(x_initial, h_initial, '-o' ,label = 'Initial Guess', color = 'grey', linewidth = 1.0, markersize = 2.0)
    a.plot(x, h, '-o' ,label = 'Optimized', color = 'black', linewidth = 1.0, markersize = 2.0)
    a.legend()
    a.set_xlabel('Horizontal Displacement (m)')
    a.set_ylabel('Vertical Displacement (m)')
    # plt.savefig('traj_opt_comparison', dpi=400,bbox_inches='tight')

    f = plt.figure(figsize=(11, 2.5))
    a = f.add_subplot(1,1,1)
    f.suptitle(f'States after {max_iter} iterations', fontsize = 20, y = 0.945)
    a.plot(rec.find_variable_by_name('v_plot').value)
    a.plot(rec.find_variable_by_name('gamma_plot').value)
    a.plot(rec.find_variable_by_name('e_plot').value)
    a.legend(['v','gamma','e'])

    f = plt.figure(figsize=(11, 2.5))
    a = f.add_subplot(1,1,1)
    f.suptitle(f'Controls after {max_iter} iterations', fontsize = 20, y = 0.945)
    a.plot(rec.find_variable_by_name('control_x').value)
    a.plot(rec.find_variable_by_name('control_z').value)
    a.plot(rec.find_variable_by_name('control_alpha').value)
    a.legend(['control_x','control_z','control_alpha'])

    plt.show()
    


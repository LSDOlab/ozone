import numpy as np
nthreads = 1						### (in script)) set # of numpy threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
import csdl_alpha as csdl
import ozone as ozone
from ozone.paper_examples.ascent_system.main_model import build_recorder
import numpy as np
import time

def get_model_ascent_system(
        approach:ozone.approaches._Approach,
        method:str,
        num = 101,
        plot:bool = False,
    )->tuple[csdl.Recorder, str, dict]:

    # See inside this function to see CSDL/Ozone implementation
    recorder, ode_problem, nt = build_recorder(approach, method, num, plot)

    return recorder, "Ascent_System", {'num_times': nt,'ode_problem': ode_problem}

if __name__ == '__main__':
    # User parameters:
    num_times = 101
    approach = ozone.approaches.TimeMarching()
    method = 'RK6'

    # build and get CSDL recorder containing ODE/optimization problem
    rec, name, stats = get_model_ascent_system(
        approach = approach,
        method = method,
        num = num_times,
        plot=True,
    )
    additional_outputs = [rec._find_variables_by_name(name)[0] for name in ['full_rx', 'full_ry', 'full_rz', 'full_Vx', 'full_Vy', 'full_Vz', 'final_time']]

    # JIT Compile model/derivative evaluation to JAX
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        gpu = True,
        derivatives_kwargs={'loop':False},
        additional_outputs=additional_outputs,
    )
    rec.start()

    jax_sim.compute_optimization_derivatives()
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(rc={'text.usetex': True})
    sns.set_style("ticks")

    # Solve Optimization problem
    import modopt as mo
    prob = mo.CSDLAlphaProblem(problem_name='quartic',simulator=jax_sim)
    # optimizer = mo.PySLSQP(prob, turn_off_outputs = True)
    optimizer = mo.SNOPT(prob, solver_options={}, turn_off_outputs = True)
    optimizer.solve()

    # Plot:
    jax_sim.run()
    rx = jax_sim['full_rx']
    ry = jax_sim['full_ry']
    rz = jax_sim['full_rz']
    num = rx.shape[0]
    Vx = jax_sim['full_Vx']
    Vy = jax_sim['full_Vy']
    Vz = jax_sim['full_Vz']

    t  = np.linspace(0, jax_sim['final_time'], num)

    f = plt.figure(figsize=(11, 2.5))
    # f.suptitle('Optimized Trajectory', fontsize = 20, y = 0.945)
    a = f.add_subplot(1,2,1)
    r = np.zeros(num)
    for i in range(0,num):
        r[i] = np.linalg.norm([rx[i],ry[i],rz[i]],2)
    a.plot(1034.2*t, (1738100.*r - 1738100.),'-o' , color = 'grey', linewidth = 1.0, markersize = 2.0)
    a.set_xlabel('Time (s)')
    a.set_ylabel('Altitude (m)')

    a = f.add_subplot(1,2,2)
    V = np.zeros(num)
    for i in range(0,num):
        V[i] = np.linalg.norm([Vx[i],Vy[i],Vz[i]],2)
    a.plot(1034.2*t, 1680.6*V,'-o' , color = 'grey', linewidth = 1.0, markersize = 2.0)
    a.set_xlabel('Time (s)')
    a.set_ylabel('Velocity (m/s)')
    plt.savefig('ascent_opt_profile', dpi=400,bbox_inches='tight')

    # a.grid()
    plt.show()
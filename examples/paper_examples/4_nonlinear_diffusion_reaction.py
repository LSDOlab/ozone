nthreads = 1						### (in script)) set # of numpy threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
import csdl_alpha as csdl
import ozone as ozone
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, hstack, csr_matrix
from ozone.paper_examples.pde_control.pde_control import build_recorder

def get_model_pde_control(
        approach:ozone.approaches._Approach,
        method:str,
        nt:int = 5000,
        nx:int = 110,
        tf:int = 2.0,
        plot = False
    )->tuple[csdl.Recorder, str, dict]:
    m = 1000
    n_dv = nt/m+1
    if n_dv == int(n_dv):
        n_dv = int(n_dv)
    else:
        raise ValueError('n_dv = nt/m is not integer')

    ODEProblem = ozone.ODEProblem(method, approach)

    # See inside this function to see CSDL/Ozone implementation
    recorder = build_recorder(
        options_dict={'nt': nt, 'nx': nx, 'tf': tf,},
        ode_problem=ODEProblem,
        m=m,
        n_dv = n_dv,
        for_plot = plot,
    )
    return recorder

if __name__ == '__main__':
    # User parameters:
    nt = 5000
    nx = 110
    tf = 2.0
    approach = ozone.approaches.TimeMarching()
    method = ozone.methods.ForwardEuler()

    # build and get CSDL recorder containing ODE/optimization problem
    rec = get_model_pde_control(
        approach = approach,
        method = method,
        nt = nt,
        nx = nx,
        tf = tf,
        plot = True
    )

    # JIT Compile model/derivative evaluation to JAX
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_outputs=[rec._find_variables_by_name('x_full')[0]],
        derivatives_kwargs={'loop':True}
    )

    # Solve Optimization problem
    import modopt as mo
    prob = mo.CSDLAlphaProblem(problem_name='quartic',simulator=jax_sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol':1e-6, 'maxiter':200})
    optimizer.solve()

    # Print results of optimization
    optimizer.print_results()

    # Plot:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(rc={'text.usetex': True})
    sns.set_style("ticks")
    jax_sim.run()
    full_state = jax_sim['x_full']
    nt, nx = full_state.shape
    dt = tf/nt

    X = np.linspace(0, np.pi, nx)
    Y = np.zeros((nt,))
    Y[1:] = np.cumsum(np.ones((nt-1,))*dt)

    X, Y = np.meshgrid(X, Y)

    Z = full_state
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, color = 'gray',rstride=3,cstride=3)
    ax.set_xlabel('Z')
    ax.set_ylabel('t')
    ax.set_zlabel('X')
    plt.show()
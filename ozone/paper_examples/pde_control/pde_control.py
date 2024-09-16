nthreads = 1						### (in script)) set # of numpy threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
import csdl_alpha as csdl
import ozone as ozone
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, hstack, csr_matrix
import time

def ode_function(
        ozone_vars:ozone.FuncVars, dx:float, nx:int, zero_bc:int, pi_bc:int, beta_t:float, beta_u:float, gamma:float,
    ):
    beta_t_g = beta_t*(np.exp(-gamma))
    heaviside_array = np.zeros((nx, 1))
    heaviside_array[round(0.3*nx):round(0.7*nx)] = 1.0

    # inputs:
    x = ozone_vars.states['x']
    u = ozone_vars.dynamic_parameters['u']
    nn = ozone_vars.num_nodes

    # initialize time rhs
    dx_dts = []

    # Instead of vectorizing, just loop over nodes
    for node in range(nn):
        # second derivative term:
        x_vec = csdl.reshape(x[node, :], (nx, 1))

        # NEW: # No matrix
        zeros = np.zeros((1,1))
        x_vec0 = csdl.concatenate((zeros, x_vec[:-1,:]))
        x_vec1 = csdl.concatenate((x_vec[1:,:],zeros))
        dxdz2_without_bc_new = x_vec0 + (-2.0)*x_vec + x_vec1
        dxdz2_without_bc = dxdz2_without_bc_new/(dx**2)

        # dxdz2 = dxdz2_without_bc # Boundary conditions are zero so no need to incorporate BC's for now
        dx_dz2 = csdl.reshape(dxdz2_without_bc, (1, nx))

        # heat of reaction source term:
        # exp_term = -gamma*1/(1+x_vec)
        exp_term = -gamma*1*((1+x_vec)**(-1))
        reaction_src_b4_reshape = beta_t*(np.exp(1)**exp_term) + (-1 * beta_t_g)
        reaction_src = csdl.reshape(reaction_src_b4_reshape, (1, nx))

        # heat transfer term:
        t_sub_term = (-beta_u)*x_vec
        control_input = csdl.reshape(u[node], (1,))
        t_src_term = (csdl.expand(control_input, (nx, 1)))*(beta_u*heaviside_array)
        # print(t_sub_term.shape, t_src_term.shape)
        transfer_src_b4_reshape = t_src_term+t_sub_term
        transfer_src = csdl.reshape(transfer_src_b4_reshape, (1, nx))

        dx_dts.append((dx_dz2 + reaction_src + transfer_src).reshape(1, -1))
    if nn == 1:
        ozone_vars.d_states['x'] = dx_dts[0]
    else:
        ozone_vars.d_states['x'] = csdl.concatenate(dx_dts)

    # Outputs:
    dx = 1.0/nx
    xnn = x
    unn = u
    Jss = []
    for n in range(nn):
        x = csdl.reshape(xnn[n, :], (nx,))
        u = csdl.reshape(unn[n, :], (1,))

        x2 = x*1*x
        u2 = u*1*u

        x_term = (dx*100)*csdl.sum(x2)
        u_term = (dx*nx*20)*(u2)

        Jss.append((x_term + u_term).reshape(1, 1))

    if nn == 1:
        ozone_vars.field_outputs['J_space'] = Jss[0]
    else:
        ozone_vars.field_outputs['J_space'] = csdl.concatenate(Jss)


def build_recorder(
        options_dict:dict[str,float],
        ode_problem:ozone.ODEProblem,
        m:int,
        n_dv:int,
        for_plot:bool = False,
    )->csdl.Recorder:

    recorder = csdl.Recorder(inline = False)
    recorder.start()

    # Pre-processing
    tf = options_dict['tf']
    nt = options_dict['nt']
    nx = options_dict['nx']
    dx = np.pi/nx
    dt = tf/nt
    stab_condition = dt/(dx*dx)
    if stab_condition > 0.5:
        raise ValueError(f'unstable: {stab_condition}, dt = {dt}, dx = {dx}')

    # Create ODE inputs:
    h_vec = csdl.Variable(name='h_vec', value=np.ones((nt-1,))*dt)  # timespan
    timespan = csdl.sum(h_vec)
    x_0 = csdl.Variable(name='x_0', value=np.ones((nx,))*0.5)  # initial conditions
    
    # dynamic parameters
    u_guess = np.ones((n_dv, ))*-0.6
    u_dv = csdl.Variable(name='u_dv',  value=u_guess)  # design variables
    
    # Expanding and interpolating u_dv:
    u_dv_expanded_full = csdl.expand(u_dv, (n_dv, m), action = 'i->ij')
    u_dv_expanded_0 = u_dv_expanded_full[:-1, :]
    u_dv_expanded_1 = u_dv_expanded_full[1:, :]

    weights_1 = np.linspace(0, 1, m)
    weights_0 = np.linspace(1, 0, m)
    weights_1 = np.tile(weights_1, (n_dv-1,1))
    weights_0 = np.tile(weights_0, (n_dv-1,1))
    u_dv_expanded = (weights_0*u_dv_expanded_0 + weights_1*u_dv_expanded_1).flatten()
    u_dv_expanded.add_name('u')

    # Define ODE problem
    ode_problem.add_state('x', x_0, store_history=for_plot)
    ode_problem.add_dynamic_parameter('u', u_dv_expanded)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=h_vec))
    ode_problem.set_function(ode_function, dx=dx, nx=nx, zero_bc=0, pi_bc = 0, beta_t = 16.0, beta_u = 2.0, gamma = 2.0)

    # Solve ODE
    outputs = ode_problem.solve()
    
    # Set optimization results
    Js = outputs.field_outputs['J_space']*dt
    Js.set_as_objective(scaler = 0.1)
    u_dv.set_as_design_variable(lower=-0.6, upper=0.0)

    if for_plot:
        x_full = outputs.states['x']
        x_full.add_name('x_full')
    return recorder

def get_model_pde_control(
        approach:ozone.approaches._Approach,
        method:str,
        # nt:int = 1000,
        # nx:int = 20,
        nt:int = 40000,
        nx:int = 210,
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
    recorder = build_recorder(
        options_dict={'nt': nt, 'nx': nx, 'tf': tf,},
        ode_problem=ODEProblem,
        m=m,
        n_dv = n_dv,
        for_plot = plot,
    )

    stats = {
        'num_times': nt,
        'ode_problem': ODEProblem
    }
    return recorder, "PDE_Optimal_Control", stats

if __name__ == '__main__':
    # User parameters:
    nt = 40000
    nx = 210
    tf = 2.0
    approach = ozone.approaches.TimeMarching()
    method = 'ForwardEuler'

    # build and get CSDL recorder containing ODE/optimization problem
    rec, name, stats = get_model_pde_control(
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
        gpu = False,
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
    # full_state = full_state[0:2150:,:]
    nt, nx = full_state.shape
    dt = tf/nt

    X = np.linspace(0, np.pi, nx)
    Y = np.zeros((nt,))
    Y[1:] = np.cumsum(np.ones((nt-1,))*dt)

    # plt.plot(Y,np.load('saved_control.npy'))
    # plt.show()
    X, Y = np.meshgrid(X, Y)

    Z = full_state
    # print(Z.nbytes)
    # exit()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, color = 'gray',rstride=3,cstride=3)
    # surf = ax.plot_surface(X, Y, Z, linewidth=0)
    ax.set_xlabel('Z')
    ax.set_ylabel('t')
    ax.set_zlabel('X')
    plt.show()
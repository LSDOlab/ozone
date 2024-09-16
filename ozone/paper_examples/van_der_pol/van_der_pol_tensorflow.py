
if __name__ == '__main__':
    import pickle
    import time

    import numpy as np
    import tensorflow as tf

    import tensorflow_probability as tfp
    # tf.keras.backend.set_floatx('float64')

    # solve the Van der Pol equation using tensorflow's ODE solver
    # Define the time span
    solution_times = tf.linspace(0.0, 15.0, num=40)
    # solution_times = tf.cast(solution_times, tf.float64)

    u_initial_np = [
        -0.75, -0.75, 0.10858121249416318, 0.9541639795734429, 1.0, 1.0, 1.0, 0.9964565877080211, 0.6192766485536403,
        0.38775599949721695, 0.19167569702764586, 0.06450124502101293, -0.013358382898357765, -0.05165487036155912, 
        -0.06417871567038007, -0.06094428366804166, -0.050006829327462116, -0.03661728993977972, -0.02401365451518119,
        -0.013898798519293187, -0.006420499134926808, -0.00182260945540434, 0.0011079324896811188, 0.0023957457833935583,
        0.0028678363178118245, 0.002650523997455129, 0.0019395185641181765, 0.001385013158869574, 0.0010395643592066442,
        0.0004734934628664808, 0.0002730788690709177, -1.5225513917683966e-05, -0.00030755115892911573, -9.608524569066593e-05,
        -0.00010877849672274054, 4.479707061495503e-06, 1.3362150984079099e-05, -0.000381894556692044, 0.00013773632461764376,
        -0.00032799437534591046,
    ]
    u_initial = tf.constant(u_initial_np)

    # Define the Van der Pol system with control input u
    @tf.function
    def vdp_ode_fn(t, y, u_initial):
        x0, x1, J = y
        u_t = tfp.math.interp_regular_1d_grid(t, 0.0, solution_times[-1], u_initial)

        dx0 = (1.0 - (x1 ** 2.0)) * x0 - x1 + u_t
        dx1 = x0
        dJ = x0 ** 2.0 + x1 ** 2.0 + u_t ** 2.0
        return [dx0, dx1, dJ]

    # Initial conditions for x0 and x1
    initial_state = [1.0, 1.0, 0.0]  # Example initial conditions

    # Solve ODE:
    # @tf.function
    def solve_ode():
            
        # Set up the ODE solver
        solver = tfp.math.ode.DormandPrince(rtol=1e-5, atol=1e-5)
        
        # Solve the ODE
        results = solver.solve(
            ode_fn=vdp_ode_fn,
            initial_time=0.0,
            initial_state=initial_state,
            solution_times=solution_times,
            constants = {'u_initial': u_initial}
        )

        # Extract the last value of J
        last_J = results.states[-1][-1]
        # last_J = results.states

        # Compute the gradient of the last value of J with respect to u_initial
        return last_J

    # Solve the ODE and compute the gradient
    # @tf.function
    def solve_ode_and_compute_gradient():
        with tf.GradientTape() as tape:
            # Set the tape to watch the variable u_initial
            tape.watch(u_initial)
            
            # Set up the ODE solver
            solver = tfp.math.ode.DormandPrince(rtol=1e-5, atol=1e-5)
            
            # Solve the ODE
            results = solver.solve(
                ode_fn=vdp_ode_fn,
                initial_time=0.0,
                initial_state=initial_state,
                solution_times=solution_times,
                constants = {'u_initial': u_initial}
            )

            # Extract the last value of J
            last_J = results.states[-1][-1]

        # Compute the gradient of the last value of J with respect to u_initial
        grad_u = tape.gradient(last_J, u_initial)
        return last_J, grad_u

    # Run the function and print the results
    solve_ode_and_compute_gradient = tf.function(solve_ode_and_compute_gradient, autograph=False, jit_compile=True)
    solve_ode = tf.function(solve_ode, autograph=False, jit_compile=True)
    s = time.time()
    last_J, grad_u = solve_ode_and_compute_gradient()
    e = time.time()
    print('Forward time (first): ', e-s)
    
    last_J = solve_ode()
    # print(f"Last value of J: {last_J.numpy()}")
    # print(f"Gradient of last J with respect to u: {grad_u.numpy()}")
    # exit()
    # Extract the results
    # results = last_J
    # times = results.times.numpy()
    # print('Times: ', times)
    # print('States: ', results.states)

    # Print or plot results
    # states = last_J
    # import matplotlib.pyplot as plt

    # plt.plot(solution_times, states[0].numpy(), label='x0')
    # plt.plot(solution_times, states[1].numpy(), label='x1')
    # plt.plot(solution_times, states[2].numpy(), label='J')
    # plt.xlabel('Time')
    # plt.ylabel('States')
    # plt.title('Van der Pol Oscillator')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    run_time = 100
    for i in range(10):
        s = time.time()
        last_J = solve_ode()
        e = time.time()
        run_time = min(run_time, e-s)
    print('Forward time: ', run_time)

    adjoint_time = 100
    for i in range(10):
        s = time.time()
        last_J, grad_u = solve_ode_and_compute_gradient()
        e = time.time()
        adjoint_time = min(adjoint_time, e-s)
    print('Adjoint time: ', adjoint_time)
    print(grad_u)

    save_dict = {}
    save_dict['forward_time'] = run_time
    save_dict['adjoint_time'] = adjoint_time

    with open('tensorflow_vdp.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
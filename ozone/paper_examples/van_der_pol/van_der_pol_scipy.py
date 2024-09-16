
if __name__ == '__main__':
    import pickle
    import time

    # Using scipy. integrate.solve_ivp to solve the Van der Pol problem:
    # ozone_vars.d_states['x0'] = ((1.0 + (- 1.0) * (x1**2.0))*x0 + (-1*x1) + u).reshape(-1,1)
    # ozone_vars.d_states['x1'] = (x0*1.0).reshape(-1,1)
    # ozone_vars.d_states['J'] = (x0**2.0 + x1**2.0 + u**2.0).reshape(-1,1)

    import numpy as np
    from scipy.integrate import solve_ivp

    def vdp(t, y, u):
        x0, x1, J = y
        return [((1.0 + (- 1.0) * (x1**2.0))*x0 + (-1*x1) + u),
                (x0*1.0),
                (x0**2.0 + x1**2.0 + u**2.0)]
    
    run_time = 100
    for i in range(10):
        s = time.time()
        sol = solve_ivp(vdp, [0, 15], [1.0, 1.0, 0.0], args=(0.0,),method='RK45', vectorized=True, dense_output=True)
        e = time.time()
        
        run_time = min(run_time, e-s)
    print(sol.y)
    print((sol.status))
    print('Forward time: ', run_time)
    save_dict = {}
    save_dict['forward_time'] = run_time

    with open('scipy_vdp.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
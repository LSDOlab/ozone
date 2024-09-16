
if __name__ == '__main__':
    import pickle
    import time

    import torch
    import torch.nn as nn

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=10)
    from torch import Tensor
    from torchdiffeq import odeint
    from torchdiffeq import odeint_adjoint

    plot = False

    u_initial_np = [
        -0.7499999999999946, -0.7499999999999879, 0.1048557615796629, 0.9546863888821571, 0.9999999999999998,
        1.0, 1.0, 0.9947076280483206, 0.6194482389940691, 0.38744967537092556, 0.191601731318239, 
        0.06442019848729504, -0.0136413761795336, -0.0517539531423932, -0.06374117567213765, -0.06130428127083274,
        -0.049830108658241355, -0.03652865082176863, -0.024188755170452134, -0.014025768435844728, 
        -0.006391522995915824, -0.0017582623617511826, 0.0011537716705032955, 0.0026102267433633537,
        0.0027700804880032193, 0.0021919661479884015, 0.0021418729141011866, 0.0013533005997215235,
        0.0011024807828213148, 0.0003433481447394484, 1.6567875610885338e-05, -5.363965834822524e-05,
        -0.00010757395189032846, -0.00021885257367582214, -0.00030425189502262823, 0.00028646380272886947,
        -7.38930184045605e-05, -0.00013710377488088062, -4.211178581795341e-05, 0.0003520228959555908,
    ]   

    # https://github.com/pytorch/pytorch/issues/50334
    def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1]) # slopes
        b = fp[:-1] - (m * xp[:-1]) # intercepts

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]

    # Define the Van der Pol system as a PyTorch module
    class VanDerPol(torch.nn.Module):
        def __init__(self, t_span):
            super().__init__()
            self.t_span = t_span
            u_initial = nn.Parameter(torch.as_tensor(u_initial_np), requires_grad=True)
            self.u = u_initial

        def forward(self, t, y):
            x0, x1, J = y
            t = t.reshape(1)
            u_t = interp(t, self.t_span, self.u)[0]
            dx0 = (1.0 - (x1 ** 2.0)) * x0 - x1 + u_t
            dx1 = x0
            dJ = x0 ** 2.0 + x1 ** 2.0 + u_t ** 2.0
            return torch.stack([dx0, dx1, dJ])

        def integrate(self):
            initial_state = torch.tensor([1.0, 1.0, 0.0])  # Initial values for x and y
            solution = odeint(self, initial_state, self.t_span, method='rk4')
            last_J = solution[:, 2][-1]
            return last_J

        def integrate_plot(self):
            initial_state = torch.tensor([1.0, 1.0, 0.0])  # Initial values for x and y
            solution = odeint(self, initial_state, self.t_span, method='rk4')
            last_J = solution
            return last_J

        def adjoint(self):
            initial_state = torch.tensor([1.0, 1.0, 0.0])  # Initial values for x and y
            solution = odeint_adjoint(self, initial_state, self.t_span, method='rk4')
            last_J = solution[:, 2][-1]
            last_J.backward()
            return last_J, self.u.grad

    # Initial conditions and parameters
    initial_state = torch.tensor([1.0, 1.0, 0.0])  # Initial values for x and y
    t = torch.linspace(0, 15, 40)  # Time range
    vdp = torch.compile(VanDerPol(t),mode="reduce-overhead") # Doesn't do anything

    # Try various ways to compile without luck
    # vdp.forward = torch.compile(vdp.forward)
    # vdp.integrate = torch.compile(vdp.integrate)
    # vdp.adjoint = torch.compile(vdp.adjoint)
    # vdp = torch.jit.script(VanDerPol(t))
    # vdp = (VanDerPol(t))

    if plot:
        solution = vdp.integrate_plot()
        # Extract solutions
        x = solution[:, 0].detach().numpy()
        y = solution[:, 1].detach().numpy()
        J = solution[:, 2].detach().numpy()

        # Plotting the results
        import matplotlib.pyplot as plt

        plt.plot(t.numpy(), x, label='x(t)')
        plt.plot(t.numpy(), y, label='y(t)')
        plt.plot(t.numpy(), J, label='J(t)')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Van der Pol Oscillator')
        plt.legend()
        plt.grid(True)
        plt.show()

    s = time.time()
    solution = vdp.integrate()
    solution, deriv = vdp.adjoint()
    e = time.time()
    print('Forward time (first): ', e-s)
    
    # Time:
    run_time = 100
    for i in range(10):
        s = time.time()
        solution = vdp.integrate()
        e = time.time()
        run_time = min(run_time, e-s)
    print('Forward time: ', run_time)
    # print(solution)
    
    adjoint_time = 100
    for i in range(10):
        s = time.time()
        solution,grad_u =  vdp.adjoint()
        e = time.time()
        adjoint_time = min(adjoint_time, e-s)
    print('Adjoint time: ', adjoint_time)
    # print(grad_u)

    save_dict = {}
    save_dict['forward_time'] = run_time
    save_dict['adjoint_time'] = adjoint_time

    with open('torchdiffeq_vdp.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


try:
    import numpy as np
    import time
    from openmdao.api import ExplicitComponent, ExecComp,  ScipyOptimizeDriver
    from ozone.api import ODEFunction
    from ozone.api import ODEIntegrator
    from openmdao.api import IndepVarComp
    import matplotlib.pyplot as plt
    from openmdao.api import Problem
    from openmdao.utils.array_utils import evenly_distrib_idxs
    import pickle
except Exception as e:
    raise ValueError(f'Error importing pacakges. (requires openmdao version == 2.5.0, numpy version == 1.19.1, scipy version == 1.5.2, python version == 3.8)')

class VanderpolODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, nn)  # (#cpus, #inputs) -> (size array, offset array)
        self.start_idx = offsets[rank]
        self.io_size = sizes[rank]  # number of inputs and outputs managed by this distributed process
        self.end_idx = self.start_idx + self.io_size

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(nn), desc='derivative of Output')
        self.add_input('x1', val=np.ones(nn), desc='Output')
        self.add_input('J', val=np.ones(nn), desc='J')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(self.io_size))
        self.add_output('x1dot', val=np.ones(self.io_size))
        self.add_output('Jdot', val=np.ones(self.io_size))

        # self.declare_coloring(method='cs')
        # # partials
        r = np.arange(self.io_size, dtype=int)
        c = r + self.start_idx

        self.declare_partials(of='x0dot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='u',   rows=r, cols=c, val=1.0)

        self.declare_partials(of='x1dot', wrt='x0',  rows=r, cols=c, val=1.0)

        self.declare_partials(of='Jdot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='u',   rows=r, cols=c)

    def compute(self, inputs, outputs):
        # introduce slowness proportional to size of computation
        # time.sleep(self.options['delay'] * self.io_size)

        # print(self.start_idx, self.end_idx)
        # The inputs contain the entire vector, be each rank will only operate on a portion of it.
        x0 = inputs['x0'][self.start_idx:self.end_idx]
        x1 = inputs['x1'][self.start_idx:self.end_idx]
        J = inputs['J'][self.start_idx:self.end_idx]
        u = inputs['u'][self.start_idx:self.end_idx]

        outputs['x0dot'] = (1.0 - x1**2) * x0 - x1 + u
        outputs['x1dot'] = x0
        outputs['Jdot'] = x0**2 + x1**2 + u**2

    def compute_partials(self, inputs, jacobian):
        # time.sleep(self.options['delay'] * self.io_size)

        x0 = inputs['x0'][self.start_idx:self.end_idx]
        x1 = inputs['x1'][self.start_idx:self.end_idx]
        u = inputs['u'][self.start_idx:self.end_idx]

        jacobian['x0dot', 'x0'] = 1.0 - x1 * x1
        jacobian['x0dot', 'x1'] = -2.0 * x1 * x0 - 1.0

        jacobian['Jdot', 'x0'] = 2.0 * x0
        jacobian['Jdot', 'x1'] = 2.0 * x1
        jacobian['Jdot', 'u'] = 2.0 * u


class PredatorPreyODEFunction(ODEFunction):
    def initialize(self):
        self.set_system(VanderpolODE)

        # Here, we declare that we have one state variable called 'y', which has shape 1.
        # We also specify the name/path for the 'f' for 'y', which is 'dy_dt'
        # and the name/path for the input to 'f' for 'y', which is 'y'.
        # self.declare_state('y0', 'dy0_dt', targets=['y0'], shape = 1)
        # self.declare_state('y1', 'dy1_dt', targets=['y1'], shape = 1)

        self.declare_state('x0', 'x0dot', targets=['x0'], shape = 1)
        self.declare_state('x1', 'x1dot', targets=['x1'], shape = 1)
        self.declare_state('J', 'Jdot', targets=['J'], shape = 1)

        # self.declare_state('x0', 'x0dot', shape = 1)
        # self.declare_state('x1', 'x1dot', shape = 1)
        # self.declare_state('J', 'Jdot', shape = 1)
        self.declare_parameter('u', 'u', shape=1)

def optimize_vdp_old_ozone():

    num = 39
    tf = 15
    # times = np.ones((num, ))*(tf/(num))
    h = tf/(num)
    times = np.arange(0,h*num,h)

    # num = 5
    # h = 0.01
    # times = np.arange(0,h*num,h)
    ode_function = PredatorPreyODEFunction()
    # formulation = 'solver-based'
    # formulation = 'time-marching'
    formulation = 'optimizer-based'
    method_name = 'GaussLegendre6'
    initial_conditions = {'x1': 1.,'x0': 1.,'J': 0.0 }
    # initial_conditions = {'y0': 2., 'y1': 2.}
    integrator = ODEIntegrator(ode_function, formulation, method_name,
                                times=times, initial_conditions=initial_conditions)
    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('u', val = np.ones((num,1))*-0.75)
    prob.model.add_subsystem('inputs',comp)
    prob.model.add_subsystem('integrator_group',integrator)
    # prob.model.connect('inputs.y0_0', 'integrator_group.initial_condition:y0')

    # excomp = ExecComp('total=sum(state)',state = {'shape': num})
    excomp = ExecComp('total=state[-1]',state = {'shape': num})
    prob.model.add_subsystem('excomp', excomp, promotes=['*'])
    prob.model.connect('integrator_group.state:J', 'state')
    prob.model.connect('inputs.u', 'integrator_group.dynamic_parameter:u')

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.model.add_objective('total')
    prob.model.add_constraint('integrator_group.state:x1', equals=0.0 ,indices = [-1])
    prob.model.add_constraint('integrator_group.state:x0', equals=0.0 ,indices = [-1])
    prob.model.add_design_var('inputs.u', lower= -0.75, upper= 1.0)
    # prob.model.add_design_var('inputs.y0_0', lower = 0.1)

    prob.setup()

    # view_model(prob)
    # exit()
    # prob.set_val('inputs.y0_0', 2.)

    # start = time.process_time()
    # prob.run_model()
    # end = time.process_time()

    # start = time.time()
    # prob.run_model()
    # end = time.time()

    # print('run_model time: ', (end  - start), 'seconds \t\t', num,'timesteps')
    # exit()

    # start = time.time()
    # prob.compute_totals()
    # end = time.time()
    # print('compute_totals time: ', (end  - start), 'seconds \t\t', num,'timesteps')

    # prob.check_totals(compact_print=True)

    # # view_model(prob)
    # start = time.time()

    t1 = time.time()
    prob.run_driver()
    time_opt = time.time() - t1

    save_dict = {}
    save_dict['optimization_time'] = time_opt
    save_dict['state_J'] = prob['integrator_group.state:J']
    save_dict['state_x0'] = prob['integrator_group.state:x0']
    save_dict['state_x1'] = prob['integrator_group.state:x1']
    save_dict['control_u'] = prob['integrator_group.dynamic_parameter:u']
    save_dict['timespan'] = times

    with open('ozone_old_vdp.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(time_opt)

if __name__ == '__main__':
    # Requires:
    # openmdao version == 2.5.0
    # numpy version == 1.19.1
    # scipy version == 1.5.2
    # python version == 3.8
    try:
        optimize_vdp_old_ozone()
    except Exception as e:
        raise ValueError(f'Error running old ozone. (requires openmdao version == 2.5.0, numpy version == 1.19.1, scipy version == 1.5.2, python version == 3.8) {e}')
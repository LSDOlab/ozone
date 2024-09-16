
import numpy as np
import pytest
import csdl_alpha as csdl
import ozone_alpha as ozone

"""This example uses a dummy ODE problem to demonstrate the performance tradeoff between TimeMarching and TimeMarchingCheckpoints"""


def build_recorder(
        nt:int,
        nx:int,
        numerical_method,
        approach,
    )->csdl.Recorder:
    def f(ozone_vars:ozone.FuncVars):
        ozone_vars.d_states['y'] = -ozone_vars.states['y']**0.5

    rec = csdl.Recorder(inline = False, debug = False)
    rec.start()

    h_stepsize = 0.01

    # Initial condition for state
    y_0 = csdl.Variable(name = 'y_0', value = 0.5, shape = (nx,))
    h_vec = np.ones(nt-1)*h_stepsize
    h_vec = csdl.Variable(name = 'h', value = h_vec)

    ode_problem = ozone.ODEProblem(numerical_method, approach)
    ode_problem.add_state('y', initial_condition=y_0, store_final=True)
    ode_problem.set_timespan(ozone.timespans.StepVector(start = 0.0, step_vector=h_vec))
    ode_problem.set_function(f)
    integrated_outputs = ode_problem.solve()

    y_out = csdl.sum(integrated_outputs.final_states['y'])
    y_out.set_as_objective()
    y_0.set_as_design_variable()
    return rec

if __name__ == '__main__':
    approaches = {
        'TimeMarching': {'approach':ozone.approaches.TimeMarching()},
        'TimeMarching with checkpoints': {'approach':ozone.approaches.TimeMarchingCheckpoints(num_checkpoints=500)},
    }
    for name, approach in approaches.items():
        print(f"Running approach: {name}")
        approach_ozone = approach['approach']
        rec = build_recorder(
            nt=250_001,
            nx=5_000,
            numerical_method='ForwardEuler',
            approach=approach_ozone
        )

        jax_sim = csdl.experimental.JaxSimulator(
            recorder=rec,
            gpu = False,
        )

    
        # import time
        # min_time = int(1e10)
        # for _ in range(10):
        #     start = time.time()
        #     # sim.compute_total_derivatives()
        #     jax_sim.run_forward()
        #     end = time.time()
        #     if end-start < min_time:
        #         min_time = end-start
        # print(min_time)
        # exit()

        def f():
            for i in range(2):
                start = time.time()
                jax_sim.compute_optimization_derivatives()
                end = time.time()
                print(f"\tend run {i}:", end- start, "s")

        import time
        from memory_profiler import memory_usage
        
        start = time.time()
        mem_usage = memory_usage(f, interval=1e-7)
        end = time.time()
        run_time = end - start
        
        mem_used = max(mem_usage) - min(mem_usage)
        print(f"\tmemory usage: {mem_used} MB")

        approach['memory_usage'] = mem_used
        approach['time'] = run_time

        # Clear the recorders to reset memory
        from csdl_alpha.utils.hard_reload import hard_reload
        hard_reload()

    print('\n\n\n')
    print('==============================Summary==============================:')
    for name, approach in approaches.items():
        print(f"{name}:")
        print(f"\tMemory usage:           {approach['memory_usage']} MB")
        print(f"\tCompile + 2x run time:  {approach['time']}s")
    ratio = approaches['TimeMarching with checkpoints']['time']/approaches['TimeMarching']['time']
    mem_ratio = approaches['TimeMarching with checkpoints']['memory_usage']/approaches['TimeMarching']['memory_usage']
    print(f"(checkpoints time)/(timemarching time) =      {ratio}")
    print(f"(checkpoints memory)/(timemarching memory) =  {mem_ratio}")
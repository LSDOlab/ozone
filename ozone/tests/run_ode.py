import time
nthreads = 1						### (in script)) set # of numpy threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)

def run_ode(settings_dict, build_and_run_ode):
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    import ozone as ozone
    from ozone import ODEProblem

    numerical_method = settings_dict['num_method']
    approach_test = settings_dict['approach']
    system_type = settings_dict['system']
    fwd_solver = settings_dict['fwd_solver']
    jvp_solver = settings_dict['jvp_solver']
    nt = settings_dict['numtimes']

    import csdl_alpha as csdl
    rec = csdl.Recorder(inline = False, debug=False)
    rec.start()
    inputs, outputs = build_and_run_ode(numerical_method, approach_test, nt)

    if isinstance(approach_test, ozone.approaches.Collocation):
        from modopt import CSDLAlphaProblem
        from modopt import PySLSQP
        
        # Dummy objective
        x = csdl.Variable(name = 'x_dummy', value = 2.0)
        x1 = (x/2.0)**2.0
        x1.set_as_objective()
        x.set_as_design_variable()
    
    use_jax = True
    if use_jax:
        jax_sim = csdl.experimental.JaxSimulator(
            recorder=rec,
            additional_outputs=[output for output in outputs.values()],
            additional_inputs=[input for input in inputs.values()],
            derivatives_kwargs={'loop': False},
            gpu = False,
        )
    else:
        jax_sim = csdl.experimental.PySimulator(
            recorder=rec,
        )

    if isinstance(approach_test, ozone.approaches.Collocation):
        jax_sim.run_forward()
        jax_sim.compute_optimization_derivatives()
        prob = CSDLAlphaProblem(problem_name='dparam',simulator=jax_sim)
        options = dict(
            acc=1e-9,
        )

        optimizer = PySLSQP(prob, solver_options = options, turn_off_outputs = True)
        optimizer.solve()

    rec.stop()

    jax_sim.run()

    if use_jax:
        # checks = jax_sim.check_totals(raise_on_error=True)
        checks = jax_sim.compute_totals()
    else:
        # checks = jax_sim.check_totals(ofs = list(outputs.values()), wrts = list(inputs.values()), raise_on_error=True)
        checks = jax_sim.compute_totals(ofs = list(outputs.values()), wrts = list(inputs.values()))    
    derivative_checks = {}
    for output_name, output in outputs.items():
        for input_name, input in inputs.items():
            derivative_checks[(output_name, input_name)] = checks[output, input]

    return_dict = {
        'output': {name:jax_sim[var] for name, var in outputs.items()},
        'derivative_checks': derivative_checks}

    return return_dict
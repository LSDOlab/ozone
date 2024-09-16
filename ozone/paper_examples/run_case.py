nthreads = 1						### (in script)) set # of numpy threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)

import ozone
from ozone.paper_examples.utils.utils import optimize_and_get_data, save_pickle

if __name__ == '__main__':
    # Process user inputs
    import sys
    # - Check Specific example case
    given_example = sys.argv[1]
    if given_example not in ['pde_control', 'trajectory_optimization', 'ascent_system', 'vdp_oscillator']:
        raise ValueError('example must be one of ["pde_control"]')
    
    # - Check Approach
    given_approach = sys.argv[2]
    if given_approach not in ['TimeMarching', 'TimeMarchingCheckpoints', 'PicardIteration', 'Collocation']:
        raise ValueError('approach must be one of ["TimeMarching", "TimeMarchingCheckpoints", "PicardIteration", "Collocation"]')

    # - Check Method
    given_method = sys.argv[3]
    if given_method not in ['ImplicitMidpoint', 'RK4', 'ForwardEuler', 'GaussLegendre6', 'RK6']:
        raise ValueError('method must be one of ["ImplicitMidpoint", "RK4", "ForwardEuler", "GaussLegendre6"]')
    
    # Get approach
    approaches = {
        'TimeMarching': ozone.approaches.TimeMarching,
        'TimeMarchingCheckpoints': ozone.approaches.TimeMarchingCheckpoints,
        'PicardIteration': ozone.approaches.PicardIteration,
        'Collocation': ozone.approaches.Collocation
    }

    # Get example
    from ozone.paper_examples.pde_control.pde_control import get_model_pde_control
    from ozone.paper_examples.trajectory_optimization.trajectory_optimization import get_model_trajectory_optimization
    from ozone.paper_examples.ascent_system.ascent_system import get_model_ascent_system
    from ozone.paper_examples.van_der_pol.van_der_pol import get_model_van_der_pol

    examples = {
        'pde_control': (
            get_model_pde_control, 
            597, 
            {}, 
            {'gpu':False}),
        'trajectory_optimization': (
            get_model_trajectory_optimization,
            None,
            {'Major optimality':2e-4, 'Major feasibility':1e-4,'Major iterations':600,'Iteration limit':130000},
            {'gpu':False}),
        'ascent_system': (
            get_model_ascent_system,
            None, 
            {},
            {'gpu':True}),
        'vdp_oscillator': (
            get_model_van_der_pol,
            None, 
            {},
            {'gpu':False}),
    }

    example_info = examples[given_example]
    example = example_info[0]
    nc = example_info[1]
    approach = approaches[given_approach]
    method = given_method
    snopt_options = example_info[2]
    rec_options = example_info[3]

    # Run analysis:
    if approach == ozone.approaches.TimeMarchingCheckpoints:
        approach_i = approach(num_checkpoints=nc)
    else:
        approach_i = approach()
    model, name, stats = example(
        approach = approach_i,
        method = method,
    )

    # Run optimizations
    results = optimize_and_get_data(
        model,
        name,
        stats,
        approach,
        method,
        snopt_options,
        rec_options,
    )

    # Save results
    save_pickle(
        results,
        name,
        stats,
        approach,
        method
    )
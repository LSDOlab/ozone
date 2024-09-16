#!/bin/bash
# PDE
# python run_case.py pde_control TimeMarchingCheckpoints ImplicitMidpoint
# python run_case.py pde_control TimeMarching ImplicitMidpoint
# python run_case.py pde_control TimeMarching ForwardEuler
# python run_case.py pde_control TimeMarchingCheckpoints ForwardEuler
# python run_case.py pde_control TimeMarching RK4
# python run_case.py pde_control TimeMarchingCheckpoints RK4
# python run_case.py pde_control TimeMarchingCheckpoints GaussLegendre6 # Too long
# # python run_case.py pde_control TimeMarching GaussLegendre6 # Out of memory
# python run_case.py pde_control TimeMarchingCheckpoints RK6 # Base case

#  Ascent System
python run_case.py ascent_system TimeMarching ImplicitMidpoint
python run_case.py ascent_system TimeMarching ForwardEuler
python run_case.py ascent_system TimeMarching RK4
python run_case.py ascent_system TimeMarching GaussLegendre6
# python run_case.py ascent_system PicardIteration ImplicitMidpoint
# python run_case.py ascent_system PicardIteration ForwardEuler
# python run_case.py ascent_system PicardIteration RK4
# python run_case.py ascent_system PicardIteration GaussLegendre6
# python run_case.py ascent_system Collocation ImplicitMidpoint
# python run_case.py ascent_system Collocation ForwardEuler
# python run_case.py ascent_system Collocation RK4
# python run_case.py ascent_system Collocation GaussLegendre6
# python run_case.py ascent_system TimeMarching RK6 # Base case

# Van der Pol
# python run_case.py vdp_oscillator Collocation ImplicitMidpoint
# python run_case.py vdp_oscillator Collocation ForwardEuler
# python run_case.py vdp_oscillator Collocation RK4
# python run_case.py vdp_oscillator Collocation GaussLegendre6
# python run_case.py vdp_oscillator TimeMarching ImplicitMidpoint
# python run_case.py vdp_oscillator TimeMarching ForwardEuler
# python run_case.py vdp_oscillator TimeMarching RK4
# python run_case.py vdp_oscillator TimeMarching GaussLegendre6
# # python run_case.py vdp_oscillator PicardIteration ImplicitMidpoint
# # python run_case.py vdp_oscillator PicardIteration ForwardEuler
# # python run_case.py vdp_oscillator PicardIteration RK4
# # python run_case.py vdp_oscillator PicardIteration GaussLegendre6
# python run_case.py vdp_oscillator TimeMarching RK6 # Base case

# Trajectory Optimization
# python run_case.py trajectory_optimization Collocation ImplicitMidpoint
# python run_case.py trajectory_optimization Collocation ForwardEuler
# python run_case.py trajectory_optimization Collocation RK4
# python run_case.py trajectory_optimization Collocation GaussLegendre6
# python run_case.py trajectory_optimization TimeMarching ImplicitMidpoint
# # python run_case.py trajectory_optimization TimeMarching ForwardEuler # does not converge
# python run_case.py trajectory_optimization TimeMarching RK4
# # python run_case.py trajectory_optimization TimeMarching GaussLegendre6 # does not converge
# # python run_case.py trajectory_optimization PicardIteration ImplicitMidpoint # does not converge
# # python run_case.py trajectory_optimization PicardIteration ForwardEuler # does not converge
# # python run_case.py trajectory_optimization PicardIteration RK4 # does not converge
# python run_case.py trajectory_optimization PicardIteration GaussLegendre6 # does not converge
# python run_case.py trajectory_optimization Collocation RK6 # Base case

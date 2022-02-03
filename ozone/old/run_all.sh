#!/bin/bash
python benchmark/run_benchmark.py
python group/run_group.py
python general/run_general.py
python dynamic/run_dynamic.py
python simplelinearsystem_native/run_simplelinearsystem_native.py
python simplelinearsystem/run_simplelinearsystem.py
python predator_prey_native/run_predator_prey_native.py
python predator_prey/run_predator_prey.py
python dynamic_native/run_dynamic_native.py


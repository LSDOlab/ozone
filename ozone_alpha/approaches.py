from typing import Optional

class _Approach(object):
    def __init__(self) -> None:
        pass
    integrator_class = None
    name:str = 'None'

class TimeMarching(_Approach):
    from ozone_alpha.integrators.time_marching import TimeMarching
    integrator_class = TimeMarching
    name = 'TimeMarching'

class TimeMarchingCheckpoints(_Approach):
    from ozone_alpha.integrators.time_marching_uniform import TimeMarchingUniform
    integrator_class = TimeMarchingUniform
    name = 'TimeMarching (Checkpointing)'
    def __init__(self, num_checkpoints:Optional[int] = None) -> None:
        super().__init__()
        self.num_checkpoints:int = num_checkpoints

class PicardIteration(_Approach):
    from ozone_alpha.integrators.picard_iteration import PicardIteration
    integrator_class = PicardIteration
    name = 'Picard Iteration'

class Collocation(_Approach):
    from ozone_alpha.integrators.collocation import Collocation
    integrator_class = Collocation
    name = 'Collocation'
class _Method(object):
    pass

class ForwardEuler(_Method):
    name = 'ForwardEuler'

class BackwardEuler(_Method):
    name = 'BackwardEuler'

class ExplicitMidpoint(_Method):
    name = 'ExplicitMidpoint'

class ImplicitMidpoint(_Method):
    name = 'ImplicitMidpoint'

class KuttaThirdOrder(_Method):
    name = 'KuttaThirdOrder'

class RK4(_Method):
    name = 'RK4'

class RK6(_Method):
    name = 'RK6'

class RalstonsMethod(_Method):
    name = 'RalstonsMethod'

class HeunsMethod(_Method):
    name = 'HeunsMethod'

class GaussLegendre2(_Method):
    name = 'GaussLegendre2'

class GaussLegendre4(_Method):
    name = 'GaussLegendre4'

class GaussLegendre6(_Method):
    name = 'GaussLegendre6'

class Lobatto2(_Method):
    name = 'Lobatto2'

class Lobatto4(_Method):
    name = 'Lobatto4'

class RadauI3(_Method):
    name = 'RadauI3'

class RadauI5(_Method):
    name = 'RadauI5'

class RadauII3(_Method):
    name = 'RadauII3'

class RadauII5(_Method):
    name = 'RadauII5'

class Trapezoidal(_Method):
    name = 'Trapezoidal'

class AB1(_Method):
    name = 'AB1'

class AM1(_Method):
    name = 'AM1'

class BDF1(_Method):
    name = 'BDF1'

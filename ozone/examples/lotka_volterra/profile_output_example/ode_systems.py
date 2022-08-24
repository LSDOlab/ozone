import numpy as np
from ozone.api import NativeSystem
import csdl


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        # y = self.create_input('y', shape=n)
        # x = self.create_input('x', shape=n)

        # # Paramters are now inputs
        # a = self.create_input('a', shape=(n))
        # b = self.create_input('b', shape=(n))
        # g = self.create_input('g', shape=(n))
        # d = self.create_input('d')

        y = self.declare_variable('y', shape=n)
        x = self.declare_variable('x', shape=n)

        # Paramters are now inputs
        a = self.declare_variable('a', shape=(n))
        b = self.declare_variable('b', shape=(n))
        g = self.declare_variable('g', shape=(n))
        d = self.declare_variable('d')

        # Predator Prey ODE:
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - csdl.expand(d, n)*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


class ProfileModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)

        # Paramters are now inputs
        a = self.create_input('a', shape=(n))
        b = self.create_input('b', shape=(n))
        g = self.create_input('g', shape=(n))
        d = self.create_input('d')

        # Register profile outputs as a function of solved states and parameters
        self.register_output('profile_output1', y*1.0 + a)
        self.register_output('profile_output2', y+x+a+b+g+csdl.expand(d, shape=(n)))

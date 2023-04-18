from ozone.classes.integrators.vectorized_integrator import VectorBased
from ozone.classes.integrators.vectorized.MainGroup import VectorBasedGroup
import scipy.sparse.linalg as spln
import scipy.sparse as sp

class SolverBased(VectorBased):

    def post_setup_init(self):
        # Perform setup routines needed only for solver-based.

        super().post_setup_init()

        for key in self.state_dict:
            V_kron = sp.kron(sp.csc_matrix(self.GLM_V), sp.eye(
                self.state_dict[key]['num'], format='csc'), format='csr')

            ImV_full = sp.eye((self.num_steps+1)*self.state_dict[key]['num'], format='csc') - sp.kron(
                sp.eye(self.num_steps+1, k=-1, format='csc'), V_kron, format='csc')
            ImV_full_inv = spln.inv(ImV_full)
            self.state_dict[key]['ImV_inv'] = ImV_full_inv
            self.state_dict[key]['UImV_inv'] = self.state_dict[key]['U_full']*ImV_full_inv

        self.var_order_name = {}
        comp_list = ['InputProcessComp', 'ODEComp', 'StageComp', 'StateComp', 'FieldComp', 'ProfileComp']
        for comp_name in comp_list:
            # We have a dictionary for each component that we feed through
            var_dict = self.order_variables(comp_name)
            self.var_order_name[comp_name] = var_dict
            
        return self.num_stage_time, self.num_steps+1

    def get_solver_model(self):

        # return SolverBased group
        component = VectorBasedGroup(solution_type = 'solver-based')
        component.add_ODEProb(self)

        return component

import csdl
from ozone.classes.integrators.vectorized.StateComp import StateComp
from ozone.classes.integrators.vectorized.ProfileComp import ProfileComp
from ozone.classes.integrators.vectorized.InputProcessingComp import InputProcessingComp
from ozone.classes.integrators.vectorized.FieldComp import FieldComp
# from ozone.classes.integrators.vectorized.ResidualModel import ResidualModel
from ozone.classes.integrators.vectorized.ODEComp import ODEComp
from ozone.classes.integrators.vectorized.StageComp import StageComp
from ozone.classes.integrators.vectorized.plot_state_comp import PlotStateComp
import numpy as np


class VectorBasedGroup(csdl.Model):
    """
    This OpenMDAO groups all the solver-based classes into a group. This is what is returned to the user when they call for a component.
    """

    def initialize(self):
        self.parameters.declare('solution_type', default='none', types=str)

    def define(self):
        solution_approach = self.parameters['solution_type']

        if solution_approach == 'solver-based':
            pass
        elif solution_approach == 'collocation':
            pass
        else:
            raise KeyError(f'solution approach {solution_approach} does not exist')

        misc = {'num_steps': self.integrator.num_steps,
                'num_stages': self.integrator.num_stages}

        misc_state = {'num_steps': self.integrator.num_steps,
                      'num_stages': self.integrator.num_stages,
                      'visualization': self.integrator.visualization,
                      'ongoingplot': self.integrator.ongoingplot}
        # var = self.declare_variable('y_0')
        # var = self.declare_variable('x_0')

        # ----------------------Inputs Processing Component---------------------- #
        # Processes inputs by:
        # - creating timestep vector
        # - shaping dynamic parameters
        # - shaping inputs
        IPComp = InputProcessingComp(
            parameter_dict=self.integrator.parameter_dict,
            IC_dict=self.integrator.IC_dict,
            times=self.integrator.times,
            state_dict=self.integrator.state_dict,
            stage_dict=self.integrator.stage_dict,
            misc=misc,
            define_dict=self.integrator.var_order_name['InputProcessComp'],
            glm_C=self.integrator.GLM_C)

        # list of input variables ordered beforehand
        input_list_IPC = []
        for key in self.integrator.var_order_name['InputProcessComp']['inputs']:
            dic_temp = self.integrator.var_order_name['InputProcessComp']['inputs'][key]
            # print('InputProcessComp: ', dic_temp['name'])
            var = self.declare_variable(**dic_temp)
            input_list_IPC.append(var)

        output_tup_IPC = csdl.custom(*input_list_IPC, op=IPComp)

        output_tup_IPC_info = []
        for i, var in enumerate(output_tup_IPC):
            output_tup_IPC_info.append((var.name, var.shape))
            # print()
            self.register_output(f'TEMP_{var.name}', var*1)

        # -^-^-^-^-^-^-^-^-^-^-^-^-^-^ Inputs Processing Component -^-^-^-^-^-^-^-^-^-^-^-^-^-^ #

        if solution_approach == 'solver-based':
            # ----====----====----====----====----RESIDUAL----====----====----====----====---- #
            # - This nonlinear equation is solved by creating a csdl model that computes the residuals.
            # - The residual to solve for is R = Y1 - Y2 = Y1 - f(Y1)
            #    - Y1 is the stage values, f(Y1) is the approximated stage using the ODE.
            #    - We want to find Y1 s.t. f(Y1) = Y1
            # - f = dY1/dt are exposed variables used for computing the state

            # -OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD
            # I guess I need to create a model for the implicit operation
            # OLD:
            # residual_model = ResidualModel()
            # residual_model.integrator = self.integrator
            # residual_model.misc = misc
            # residual_model.output_tup_IPC_info = output_tup_IPC_info

            # =====UNCOMMENT TO VISUALIZE RESIDUAL MODEL=====:
            # import python_csdl_backend
            # res_sim = python_csdl_backend.Simulator(residual_model)
            # # res_sim.visualize_implementation(recursive=True)
            # res_sim.prob.run_model()
            # res_sim.prob.check_partials(compact_print=True)
            # exit()
            # =====UNCOMMENT TO VISUALIZE RESIDUAL MODEL=====:
            # -OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD-OLD

            # Creating residual model
            rm = csdl.Model()

            # Creating ODE model which takes in parameters, states and computes f = dy/dt
            # ==== ODE COMP: ====
            output_tup_IPC_r = []
            for tuple_info in output_tup_IPC_info:
                var = rm.declare_variable(tuple_info[0], shape=tuple_info[1])
                output_tup_IPC_r.append(var)

            IPC_out_key = list(self.integrator.var_order_name['InputProcessComp']['outputs'].keys())
            # print(IPC_out_key,output_tup_IPC_info )
            # exit()
            ODEC_out_key = list(self.integrator.var_order_name['ODEComp']['outputs'].keys())

            # For now, csdl models and NS models are treated the same. Hence the commented 'if'
            # if self.integrator.OStype == 'NS':
            if True:
                odecomp = ODEComp(
                    parameter_dict=self.integrator.parameter_dict,
                    IC_dict=self.integrator.IC_dict,
                    times=self.integrator.times,
                    state_dict=self.integrator.state_dict,
                    stage_dict=self.integrator.stage_dict,
                    f2s_dict=self.integrator.f2s_dict,
                    ODE_system=self.integrator.ode_system,
                    misc=misc,
                    define_dict=self.integrator.var_order_name['ODEComp'],
                    stage_f_dict=self.integrator.stage_f_dict,
                    recorder=self.integrator.recorder)
                odecomp.set_odesys(self.integrator.ode_system)

                input_list_OC = []

                for key in self.integrator.var_order_name['ODEComp']['inputs']:
                    dic_temp = self.integrator.var_order_name['ODEComp']['inputs'][key]

                    if key in IPC_out_key:
                        var = output_tup_IPC_r[IPC_out_key.index(key)]
                    else:
                        var = rm.declare_variable(**dic_temp)

                    input_list_OC.append(var)

                # The csdl operation
                output_tup_OC = csdl.custom(*input_list_OC, op=odecomp)
                if type(output_tup_OC) != type((1, 2)):
                    output_tup_OC = (output_tup_OC,)

            # Try to insert given csdl ODE function directly into the model. Not working as of now.
            # Partials were correct but total derivatives were wrong...
            elif self.integrator.OStype == 'OM':

                for var in output_tup_IPC_r:
                    rm.register_output(var.name+'_test', var*1.0)

                input_list_OC = []
                input_list_preOC = []

                for key in self.integrator.var_order_name['ODEComp']['inputs']:
                    dic_temp = self.integrator.var_order_name['ODEComp']['inputs'][key].copy()

                    if key in IPC_out_key:
                        # var = output_tup_IPC_r[IPC_out_key.index(key)]

                        var_t = output_tup_IPC_r[IPC_out_key.index(key)]
                        var = var_t*1.0
                        rm.connect(var.name, key)
                    else:
                        # var = rm.declare_variable(**dic_temp)

                        var_t = rm.declare_variable(**dic_temp)
                        state_name = self.integrator.stage_dict[key]['state_name']
                        sd = self.integrator.state_dict[state_name]

                        var = csdl.reshape(var_t, sd['nn_shape'])
                        # print(var.shape)
                        rm.connect(var.name, state_name)

                    # print('ODE IN TO RESIDUAL', var.name, key)

                    input_list_preOC.append(var_t)
                    input_list_OC.append(var)

                rm.add(self.integrator.ode_system.system(num_nodes=self.integrator.numnodes), 'ode_system')

                for i in input_list_OC:
                    rm.register_output(i.name + '_p', i*1.0)

                # import python_csdl_backend
                # simtest = python_csdl_backend.Simulator(self.integrator.ode_system.system(num_nodes=self.integrator.numnodes))
                # simtest.visualize_implementation()

                output_tup_OC = []
                for f_name in self.integrator.var_order_name['ODEComp']['outputs']:
                    f_dict = self.integrator.var_order_name['ODEComp']['outputs'][f_name].copy()
                    stage_f_name = self.integrator.stage_f_dict[f_name]['res_name']
                    state_name = self.integrator.f2s_dict[f_name]
                    f_dict['shape'] = self.integrator.state_dict[state_name]['nn_shape']
                    f_dict['name'] = f_name
                    var_temp = rm.declare_variable(**f_dict)
                    flat_shape = (self.integrator.state_dict[state_name]['num']*self.integrator.num_steps*self.integrator.num_stages,)

                    # rm.register_output(stage_f_name, csdl.reshape(var_temp, flat_shape))
                    # var_temp_add = rm.declare_variable(stage_f_name, shape=flat_shape)

                    var_temp_add = csdl.reshape(var_temp, flat_shape)
                    # stage_f_name = var_temp_add.name
                    self.integrator.stage_f_dict[f_name]['res_name'] = var_temp_add.name
                    # print(self.integrator.var_order_name['StageComp'])
                    self.integrator.var_order_name['StageComp']['inputs'][f_name]['name'] = var_temp_add.name

                    for dicts in self.integrator.var_order_name['StageComp']['partials']:
                        if dicts['wrt'] == stage_f_name:
                            dicts['wrt'] = var_temp_add.name

                    output_tup_OC.append(var_temp_add)
                output_tup_OC = tuple(output_tup_OC)

                # for i, f_name in enumerate(self.integrator.var_order_name['ODEComp']['outputs']):
                #     rm.connect(f_name, output_tup_OC[i].name)

            # ==== ODE COMP: ====

            # Create stage model that takes in F and computes Y2. Note the residual is Y1 - Y2.
            stage_comp = StageComp(
                parameter_dict=self.integrator.parameter_dict,
                IC_dict=self.integrator.IC_dict,
                times=self.integrator.times,
                state_dict=self.integrator.state_dict,
                stage_dict=self.integrator.stage_dict,
                f2s_dict=self.integrator.f2s_dict,
                ODE_system=self.integrator.ode_system,
                misc=misc,
                define_dict=self.integrator.var_order_name['StageComp'],
                stage_f_dict=self.integrator.stage_f_dict)

            input_list_SgC = []
            for key in self.integrator.var_order_name['StageComp']['inputs']:
                dic_temp = self.integrator.var_order_name['StageComp']['inputs'][key]

                if key in IPC_out_key:
                    var = output_tup_IPC_r[IPC_out_key.index(key)]
                elif key in ODEC_out_key:
                    var = output_tup_OC[ODEC_out_key.index(key)]

                input_list_SgC.append(var)

            output_tup_SgC = csdl.custom(*input_list_SgC, op=stage_comp)

            # for stage_out_var in output_tup_SgC:
            # rm.register_output(f'TEMP_{output_tup_SgC.name}', output_tup_SgC*1.0)
            if type(output_tup_SgC) != type((1, 2)):
                output_tup_SgC = (output_tup_SgC,)

            ODEC_in_key = list(self.integrator.var_order_name['ODEComp']['inputs'].keys())
            SgC_out_key = list(self.integrator.var_order_name['StageComp']['outputs'].keys())

            # For now, csdl models and NS models are treated the same. Hence the commented if
            # if self.integrator.OStype == 'NS':
            if True:
                # Register the exposed variable outputs
                for i, f_name in enumerate(ODEC_out_key):
                    state_f_name = self.integrator.stage_f_dict[f_name]['state_name']
                    var_temp = output_tup_OC[i]
                    # print(var_temp.shape)
                    rm.register_output(state_f_name, var_temp*1.0)

                # Register the residual outputs
                for stage_name in self.integrator.stage_dict:
                    sgd = self.integrator.stage_dict[stage_name]
                    # print(stage_name, input_list_OC[ODEC_in_key.index(stage_name)].shape, input_list_OC[ODEC_in_key.index(stage_name)].name)
                    # print(stage_name, output_tup_SgC[SgC_out_key.index(stage_name)].shape, output_tup_SgC[SgC_out_key.index(stage_name)].name)
                    temp = input_list_OC[ODEC_in_key.index(stage_name)] - output_tup_SgC[SgC_out_key.index(stage_name)]
                    rm.register_output('res_'+sgd['state_name'], temp)

            # Not currently working.
            elif self.integrator.OStype == 'OM':

                for stage_name in self.integrator.stage_dict:
                    sgd = self.integrator.stage_dict[stage_name]
                    # print(stage_name, input_list_preOC[ODEC_in_key.index(stage_name)].shape, input_list_preOC[ODEC_in_key.index(stage_name)].name)
                    # print(stage_name, output_tup_SgC[SgC_out_key.index(stage_name)].shape, output_tup_SgC[SgC_out_key.index(stage_name)].name)
                    temp = input_list_preOC[ODEC_in_key.index(stage_name)] - output_tup_SgC[SgC_out_key.index(stage_name)]
                    rm.register_output('res_'+sgd['state_name'], temp)

                for i, f_name in enumerate(ODEC_out_key):
                    state_f_name = self.integrator.stage_f_dict[f_name]['state_name']
                    rm.register_output(state_f_name, output_tup_OC[i]*1.00)

            # == == =UNCOMMENT TO VISUALIZE RESIDUAL MODEL == == =:
            # import python_csdl_backend
            # # from csdl import GraphRepresentation
            # # GraphRepresentation(rm).visualize_graph()
            # res_sim = python_csdl_backend.Simulator(rm, display_scripts=True, analytics=True)
            # # res_sim.visualize_implementation(recursive=True)
            # res_sim.run()
            # res_sim.check_partials(compact_print=True)
            # exit()
            # of_list = []
            # wrt_list = []
            # for i, f_name in enumerate(ODEC_out_key):
            #     state_f_name = self.integrator.stage_f_dict[f_name]['state_name']
            #     of_list.append(state_f_name)
            # for stage_name in self.integrator.stage_dict:
            #     sgd = self.integrator.stage_dict[stage_name]
            #     of_list.append('res_'+sgd['state_name'])

            # if self.integrator.OStype == 'OM':
            #     for var in input_list_preOC:
            #         wrt_list.append(var.name)
            # else:
            #     for var in input_list_OC:
            #         wrt_list.append(var.name)
            # for var in output_tup_IPC_r:
            #     wrt_list.append(var.name)
            # res_sim.prob.check_totals(of=of_list, wrt=wrt_list, compact_print=True)
            # exit()
            # =====UNCOMMENT TO VISUALIZE RESIDUAL MODEL=====:

            # Add in the residual model
            solve_implicit = self.create_implicit_operation(rm)

            # Declare residual, state
            for stage in self.integrator.stage_dict:
                state_stage = self.integrator.stage_dict[stage]['state_name']
                solve_implicit.declare_state(stage, residual='res_'+state_stage, val=2.0)

            # Fixed point nonlinear solver
            solve_implicit.nonlinear_solver = csdl.NonlinearBlockGS(maxiter=40, iprint=1)
            solve_implicit.linear_solver = csdl.ScipyKrylov(iprint=1)

            # solve_implicit.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
            # solve_implicit.linear_solver = csdl.DirectSolver(atol=1)

            # Get state outputs as declared variables
            expose_name = []
            for f_name in ODEC_out_key:
                expose_name.append(self.integrator.stage_f_dict[f_name]['state_name'])

            # Finally adding the csdl model
            stage_tuple = solve_implicit(*output_tup_IPC, expose=expose_name)

            # ---------------------- State Component ----------------------:
            # This component computes the state given F

            state_comp = StateComp(
                parameter_dict=self.integrator.parameter_dict,
                IC_dict=self.integrator.IC_dict,
                times=self.integrator.times,
                state_dict=self.integrator.state_dict,
                stage_dict=self.integrator.stage_dict,
                f2s_dict=self.integrator.f2s_dict,
                ODE_system=self.integrator.ode_system,
                output_state_list=self.integrator.output_state_list,
                misc=misc_state,
                define_dict=self.integrator.var_order_name['StateComp'],
                stage_f_dict=self.integrator.stage_f_dict)

            # list of input variables ordered beforehand
            input_list_StC = []
            IPC_out_key = list(self.integrator.var_order_name['InputProcessComp']['outputs'].keys())
            ODEC_out_key = list(self.integrator.var_order_name['ODEComp']['outputs'].keys())
            for key in self.integrator.var_order_name['StateComp']['inputs']:
                dic_temp = self.integrator.var_order_name['StateComp']['inputs'][key]
                # print('StateComp: ', dic_temp['name'])
                if key in ODEC_out_key:
                    var = stage_tuple[len(self.integrator.f2s_dict) + ODEC_out_key.index(key)]
                elif key in IPC_out_key:
                    var = output_tup_IPC[IPC_out_key.index(key)]
                input_list_StC.append(var)

            # Add the csdl operation
            output_tup_StC = csdl.custom(*input_list_StC, op=state_comp)
            if type(output_tup_StC) != type((1, 2)):
                output_tup_StC = (output_tup_StC,)

            StC_out_key = list(self.integrator.var_order_name['StateComp']['outputs'].keys())

            # -^-^-^-^-^-^-^-^-^-^-^-^-^-^ State Component -^-^-^-^-^-^-^-^-^-^-^-^-^-^:

        elif solution_approach == 'collocation':

            # Goal:
            # Set constraints and design variables according to:
            #   HF = Hbar*Fbar
            #   C1 = Abar*HF - Ybar + Ubar*ybar
            #   C2 = Bbar*HF + (Vbar - I)*ybar
            #
            # Subject to:
            #   constraint C1 = 0
            #   constraint C2 = 0
            # with respect to
            #   ybar,  Ybar
            #
            # Note that ybar nominally includes the initial conditions but we do not want it as a DV.

            # Do the following:
            # 1) Create both (design) variables for Ybar and ybar_1 for each state
            # 2) ybar = [y_0, ybar_1] for each state
            # 3) Evaluate Fbar = F(Ybar) for all states
            # 4) Then write both collocation constraint equations C1 and C2 for each state
            # 5) Set constraints

            #  1) and 2) for all states:
            store_vectorized_stage_var = []
            store_vectorized_state_var = []
            store_dv_state_var = []
            store_finalized_state = []
            visualization_define_dict = {}
            for i, key in enumerate(self.integrator.stage_dict):

                stage_dict_current = self.integrator.stage_dict[key]
                state_name = stage_dict_current['state_name']  # i.e. y
                stage_name = key  # i.e. stage__y --> variable name to feed into F
                state_dv_name = stage_dict_current['state_dv_name']
                state_dict_current = self.integrator.state_dict[state_name]

                # print('\nKEY:', key)
                # for key2, value2 in stage_dict_current.items():
                #     print('\t', key2)
                #     print('\t\t', value2)

                # 1) Create both design variables:
                store_vectorized_stage_var.append(self.create_input(
                    stage_name,
                    shape=state_dict_current['nn_shape'],
                    val=state_dict_current['nn_guess']))
                self.add_design_variable(stage_name, scaler=state_dict_current['scaler'])  # YBar design variable to feed into F

                state_dv_var = self.create_input(
                    state_dv_name,
                    shape=state_dict_current['state_dv_info']['shape'],
                    val=state_dict_current['state_dv_info']['guess'])
                store_dv_state_var.append(state_dv_var)
                self.add_design_variable(state_dv_name, scaler=state_dict_current['scaler'])  # ybar design variable without  initial conditions

                # 2) Add initial conditions to ybar.
                for csdl_input_var in output_tup_IPC:
                    if csdl_input_var.name == self.integrator.IC_dict[state_dict_current['IC_name']]['meta_name']:
                        ic_vector = csdl_input_var
                state_meta_name = state_dict_current['meta_name']+'_minus_initial_conditions'
                ybar_full_vector_wo_ic = self.create_output(
                    state_meta_name,
                    shape=state_dict_current['flat_output_shape'],
                    val=np.zeros(state_dict_current['flat_output_shape'])
                )
                ybar_full_vector_wo_ic[state_dict_current['num']:state_dict_current['flat_output_size']] = state_dv_var
                ybar_full_vector = self.register_output(state_dict_current['meta_name']+'_flat', ybar_full_vector_wo_ic + ic_vector)
                visualization_define_dict[state_dict_current['meta_name']] = {
                    'name': state_dict_current['meta_name']+'_flat',
                    'shape': state_dict_current['flat_output_shape'],
                }
                store_vectorized_state_var.append(ybar_full_vector)

                y_state_full = self.register_output(state_dict_current['meta_name'], csdl.reshape(ybar_full_vector, state_dict_current['output_shape']))
                store_finalized_state.append(y_state_full)

            # 3) evaluating F for Ybar design
            # ==== ODE COMP: ====
            output_tup_IPC_r = []
            for tuple_info in output_tup_IPC_info:
                var = self.declare_variable(tuple_info[0], shape=tuple_info[1])
                output_tup_IPC_r.append(var)
            IPC_out_key = list(self.integrator.var_order_name['InputProcessComp']['outputs'].keys())
            ODEC_out_key = list(self.integrator.var_order_name['ODEComp']['outputs'].keys())

            odecomp = ODEComp(
                parameter_dict=self.integrator.parameter_dict,
                IC_dict=self.integrator.IC_dict,
                times=self.integrator.times,
                state_dict=self.integrator.state_dict,
                stage_dict=self.integrator.stage_dict,
                f2s_dict=self.integrator.f2s_dict,
                ODE_system=self.integrator.ode_system,
                misc=misc,
                define_dict=self.integrator.var_order_name['ODEComp'],
                stage_f_dict=self.integrator.stage_f_dict,
                recorder=self.integrator.recorder)
            odecomp.set_odesys(self.integrator.ode_system)

            input_list_OC = []

            for key in self.integrator.var_order_name['ODEComp']['inputs']:
                dic_temp = self.integrator.var_order_name['ODEComp']['inputs'][key]

                # print('\nODE DIC KEY:', key)
                # for key2, value2 in dic_temp.items():
                #     print('\t', key2)
                #     print('\t\t', value2)

                if key in IPC_out_key:
                    var = output_tup_IPC_r[IPC_out_key.index(key)]
                else:
                    var = self.declare_variable(**dic_temp)

                input_list_OC.append(var)

            # The csdl operation
            output_tup_OC = csdl.custom(*input_list_OC, op=odecomp)
            if type(output_tup_OC) != type((1, 2)):
                output_tup_OC = (output_tup_OC,)
            for temp in output_tup_OC:
                self.register_output(temp.name, temp)

            # 4) Write both collocation constraint equations C1 and C2 for each state
            store_constraints_1 = []
            store_constraints_2 = []
            for i, key in enumerate(self.integrator.stage_dict):

                stage_dict_current = self.integrator.stage_dict[key]
                state_name = stage_dict_current['state_name']  # i.e. y
                stage_name = key  # i.e. stage__y --> variable name to feed into F
                stage_f_name = stage_dict_current['stage_f_name']
                state_dv_name = stage_dict_current['state_dv_name']
                state_dict_current = self.integrator.state_dict[state_name]

                stage_vec_shape = (state_dict_current['nn_size'],)
                Fbar = csdl.reshape(self.declare_variable(stage_f_name, shape=stage_vec_shape), stage_vec_shape)
                Hbar = self.declare_variable(stage_dict_current['h_name'], shape=stage_vec_shape)
                Y_dv = csdl.reshape(store_vectorized_stage_var[i], stage_vec_shape)
                y_dv = store_vectorized_state_var[i]
                A = state_dict_current['A_full']
                B = state_dict_current['B_full']
                U = state_dict_current['U_full']
                V = state_dict_current['V_full']

                HF = Hbar*Fbar
                constraint_1_name = f'constraint_1_{state_name}'
                constraint_2_name = f'constraint_2_{state_name}'

                c1 = csdl.matvec(A, HF) + csdl.matvec(U, y_dv) - Y_dv
                c2 = csdl.matvec(B, HF) + csdl.matvec(V, y_dv) - y_dv
                test1 = self.register_output(constraint_1_name, c1)
                test2 = self.register_output(constraint_2_name, c2[state_dict_current['num']:])

                self.add_constraint(constraint_1_name, equals=0.0, scaler=state_dict_current['scaler'])
                self.add_constraint(constraint_2_name, equals=0.0, scaler=state_dict_current['scaler'])

            if self.integrator.visualization == 'end':
                plot_var_list = []
                plot_var_list.extend(store_vectorized_state_var)
                times_vec = self.declare_variable(self.integrator.times['name'], shape=(self.integrator.num_steps,))
                plot_var_list.append(times_vec)

                visualization_define_dict['times'] = {
                    'name': self.integrator.times['name'],
                    'shape': (self.integrator.num_steps,),
                }

                plotstatecomp = PlotStateComp(
                    misc=misc_state,
                    define_dict=visualization_define_dict,
                )
                output_plot_temp = csdl.custom(*plot_var_list, op=plotstatecomp)
                self.register_output('OZONE_PLOT_DUMMY', output_plot_temp)

            # process list of csdl variables and their states to feed into profile/field/regular outputs
            StC_out_key = list(self.integrator.var_order_name['StateComp']['outputs'].keys())
            output_tup_StC = []
            for state_name in StC_out_key:
                state_out_name = self.integrator.var_order_name['StateComp']['outputs'][state_name]['name']
                for finalized_state_var in store_finalized_state:
                    if finalized_state_var.name == state_out_name:
                        output_tup_StC.append(finalized_state_var)
            output_tup_StC = tuple(output_tup_StC)
            if len(output_tup_StC) != len(StC_out_key):
                raise ValueError('size mismatch')
            # declare equality constraints for residuals
            # for stage in self.integrator.stage_dict:
            #     state_stage = self.integrator.stage_dict[stage]['state_name']

            #     if isinstance(self.integrator.state_dict[state_stage]['nn_shape'], tuple):
            #         nn_shape = self.integrator.state_dict[state_stage]['nn_shape']
            #     else:
            #         nn_shape = (self.integrator.state_dict[state_stage]['nn_shape'],)
            #     flat_shape = np.prod(nn_shape)

            #     res_var = rm.declare_variable('res_'+state_stage, shape=flat_shape)
            #     # OLD
            #     # constraint = rm.register_output('constraint_'+state_stage, csdl.pnorm(res_var))

            #     # NEW
            #     constraint = rm.register_output('constraint_'+state_stage, 1.0*res_var)
            #     rm.add_constraint(name='constraint_'+state_stage, equals=0.0)

            # # add model that computes residual model
            # self.add(rm)

            # # declare stage variables for state computation
            # stage_tuple = []
            # for stage in self.integrator.stage_dict:
            #     state_name = self.integrator.stage_dict[stage]['state_name']
            #     stage_tuple.append(self.create_input(stage,
            #                                          shape=self.integrator.state_dict[state_name]['nn_shape'],
            #                                          val=self.integrator.state_dict[state_name]['nn_guess']))
            #     self.add_design_variable(stage)

            # # Get state outputs as declared variables
            # for f_name in ODEC_out_key:
            #     state_name = self.integrator.stage_f_dict[f_name]['state_name']  # not sure what this variable is honestly.
            #     state_key = self.integrator.stage_f_dict[f_name]['state_key']  # not sure what this variable is honestly.

            #     if isinstance(self.integrator.state_dict[state_key]['nn_shape'], tuple):
            #         nn_shape = self.integrator.state_dict[state_key]['nn_shape']
            #     else:
            #         nn_shape = (self.integrator.state_dict[state_key]['nn_shape'],)
            #     flat_shape = np.prod(nn_shape)
            #     stage_tuple.append(self.declare_variable(state_name,  shape=flat_shape))
            #     # print(state_name, self.integrator.state_dict[state_key]['nn_shape'])
            # # exit()
            # # x = self.create_input('collocation_temp_in')
            # # self.register_output('collocation_temp_out', x*1.0)
            # # self.add_objective('collocation_temp_out')
            # # exit()

        # ^=^=^=^=^=^=^=^=^=^=^=^=^=^RESIDUAL ^=^=^=^=^=^=^=^=^=^=^=^=^=^ #

        # ----------------------ProfileOutputs Component----------------------:
        # If declared, compute the profile outputs
        if self.integrator.profile_outputs_bool == True:

            profile_comp = ProfileComp(
                parameter_dict=self.integrator.parameter_dict,
                IC_dict=self.integrator.IC_dict,
                times=self.integrator.times,
                state_dict=self.integrator.state_dict,
                profile_output_dict=self.integrator.profile_output_dict,
                f2s_dict=self.integrator.f2s_dict,
                profile_outputs_system=self.integrator.profile_outputs_system,
                misc=misc,
                define_dict=self.integrator.var_order_name['ProfileComp'])
            input_list_PC = []

            for key in self.integrator.var_order_name['ProfileComp']['inputs']:
                dic_temp = self.integrator.var_order_name['ProfileComp']['inputs'][key]

                if key in StC_out_key:
                    var = output_tup_StC[StC_out_key.index(key)]
                elif key in self.integrator.parameter_dict:
                    var = self.declare_variable(**dic_temp)
                else:
                    raise KeyError(f'cannot find variable {key} for profile output comp')

                input_list_PC.append(var)

            # add the operation
            output_tup_PC = csdl.custom(*input_list_PC, op=profile_comp)

            if type(output_tup_PC) != type((1, 2)):
                output_tup_PC = (output_tup_PC,)

            for var in output_tup_PC:
                self.register_output(var.name, var)
        # -^-^-^-^-^-^-^-^-^-^-^-^-^-^ Profile Component -^-^-^-^-^-^-^-^-^-^-^-^-^-^:

        # ---------------------- Field Outputs Comp ----------------------:
        # If declared, compute the field outputs
        if bool(self.integrator.field_output_dict):

            field_comp = FieldComp(
                parameter_dict=self.integrator.parameter_dict,
                IC_dict=self.integrator.IC_dict,
                times=self.integrator.times,
                state_dict=self.integrator.state_dict,
                field_output_dict=self.integrator.field_output_dict,
                f2s_dict=self.integrator.f2s_dict,
                misc=misc,
                define_dict=self.integrator.var_order_name['FieldComp'])
            input_list_FC = []

            for key in self.integrator.var_order_name['FieldComp']['inputs']:
                dic_temp = self.integrator.var_order_name['FieldComp']['inputs'][key]
                # print('FieldComp: ', dic_temp['name'])

                if key in StC_out_key:
                    var = output_tup_StC[StC_out_key.index(key)]
                else:
                    var = self.declare_variable(**dic_temp)
                input_list_FC.append(var)

            output_tup_FC = csdl.custom(*input_list_FC, op=field_comp)
            if type(output_tup_FC) != type((1, 2)):
                output_tup_FC = (output_tup_FC,)

            for var in output_tup_FC:
                self.register_output(var.name, var)
        # -^-^-^-^-^-^-^-^-^-^-^-^-^-^ Field Component -^-^-^-^-^-^-^-^-^-^-^-^-^-^:

        # ---------------------- State Outputs ----------------------:
        # If declared, register states as outputs

        for state in self.integrator.state_dict:
            sd = self.integrator.state_dict[state]
            if sd['output_bool']:
                self.register_output(sd['output_name'], csdl.reshape(output_tup_StC[StC_out_key.index(state)], sd['output_shape']))
        # ---------------------- State Outputs ----------------------:

    def add_ODEProb(self, ODEint):
        self.integrator = ODEint

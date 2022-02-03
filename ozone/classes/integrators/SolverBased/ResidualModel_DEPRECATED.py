import csdl
from ozone.classes.integrators.SolverBased.ODEComp import ODEComp
from ozone.classes.integrators.SolverBased.StageComp import StageComp


class ResidualModel(csdl.Model):
    """
    This OpenMDAO groups all the solver-based classes into a group. This is what is returned to the user when they call for a component.
    """

    def define(self):

        misc = self.misc

        output_tup_IPC = []
        for tuple_info in self.output_tup_IPC_info:
            var = self.declare_variable(tuple_info[0], shape=tuple_info[1])
            output_tup_IPC.append(var)

        # ---------------------- ODE Component: F = f(x,y,t) ---------------------- #
        odecomp = ODEComp(
            parameter_dict=self.integrator.parameter_dict,
            IC_dict=self.integrator.IC_dict,
            times=self.integrator.times,
            state_dict=self.integrator.state_dict,
            stage_dict=self.integrator.stage_dict,
            f2s_dict=self.integrator.f2s_dict,
            ODE_system=self.integrator.ode_system,
            misc=misc,
            define_dict=self.integrator.var_order_name['ODEComp'])
        odecomp.set_odesys(self.integrator.ode_system)

        # list of input variables ordered beforehand
        input_list_OC = []
        IPC_out_key = list(self.integrator.var_order_name['InputProcessComp']['outputs'].keys())
        ODEC_out_key = list(self.integrator.var_order_name['ODEComp']['outputs'].keys())

        print(IPC_out_key)
        print(ODEC_out_key)
        for key in self.integrator.var_order_name['ODEComp']['inputs']:
            dic_temp = self.integrator.var_order_name['ODEComp']['inputs'][key]

            if key in IPC_out_key:
                var = output_tup_IPC[IPC_out_key.index(key)]
            else:
                var = self.declare_variable(**dic_temp)

            input_list_OC.append(var)

        output_tup_OC = csdl.custom(*input_list_OC, op=odecomp)

        if type(output_tup_OC) != type((1, 2)):
            output_tup_OC = (output_tup_OC,)
            # for i, key in enumerate(output_tup_OC):
            #     self.register_output(str(i)+'_1', key)
            # self.add(odecomp, 'ode_comp',  promotes=['*'])
            # ---------------------- ODE Component: F = f(x,y,t) ---------------------- #

            # ---------------------- Stage Component: Y_k+1 = f(F) ---------------------- #

        stage_comp = StageComp(
            parameter_dict=self.integrator.parameter_dict,
            IC_dict=self.integrator.IC_dict,
            times=self.integrator.times,
            state_dict=self.integrator.state_dict,
            stage_dict=self.integrator.stage_dict,
            f2s_dict=self.integrator.f2s_dict,
            ODE_system=self.integrator.ode_system,
            misc=misc,
            define_dict=self.integrator.var_order_name['StageComp'])

        input_list_SgC = []
        for key in self.integrator.var_order_name['StageComp']['inputs']:
            dic_temp = self.integrator.var_order_name['StageComp']['inputs'][key]

            if key in IPC_out_key:
                var = output_tup_IPC[IPC_out_key.index(key)]
            elif key in ODEC_out_key:
                # print('aksjdnfkandf', output_tup_OC)
                var = output_tup_OC[ODEC_out_key.index(key)]

            input_list_SgC.append(var)

        output_tup_SgC = csdl.custom(*input_list_SgC, op=stage_comp)

        if type(output_tup_SgC) != type((1, 2)):
            output_tup_SgC = (output_tup_SgC,)

        # for i, stage_var in enumerate(output_tup_SgC):
        #     self.register_output(str(i)+'_1', stage_var)
        ODEC_in_key = list(self.integrator.var_order_name['ODEComp']['inputs'].keys())
        SgC_out_key = list(self.integrator.var_order_name['StageComp']['outputs'].keys())

        for stage_name in self.integrator.stage_dict:
            sgd = self.integrator.stage_dict[stage_name]
            # stage2_name = sgd['stage_comp_out_name']
            print(stage_name, input_list_OC[ODEC_in_key.index(stage_name)].shape, input_list_OC[ODEC_in_key.index(stage_name)].name)
            print(stage_name, output_tup_SgC[SgC_out_key.index(stage_name)].shape, output_tup_SgC[SgC_out_key.index(stage_name)].name)

            temp = input_list_OC[ODEC_in_key.index(stage_name)] - output_tup_SgC[SgC_out_key.index(stage_name)]
            # temp2 = csdl.pnorm(temp, pnorm_type=2)
            self.register_output('res_'+sgd['state_name'], temp)

        for i, f_name in enumerate(ODEC_out_key):
            self.register_output(f_name, output_tup_OC[i])

        # ---------------------- Stage Component: Y_k+1 = f(F) ---------------------- #

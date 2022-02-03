import ozone.api as o2
import numpy as np


class AttitudeNS(o2.NativeSystem):
    def setup(self):
        n = self.num_nodes

        self.add_input('K', shape=(3,))

        self.input_list = ['Omega', 't0', 't1',
                           't2', 'omega0', 'omega1', 'omega2', 'C00', 'C01', 'C02', 'C10', 'C11', 'C12', 'C20', 'C21', 'C22']
        self.output_list = ['do0_dt', 'do1_dt', 'do2_dt', 'dC00_dt', 'dC00_dt',
                            'dC01_dt', 'dC02_dt', 'dC10_dt', 'dC11_dt', 'dC12_dt', 'dC20_dt', 'dC21_dt', 'dC22_dt']

        for input in self.input_list:
            self.add_input(input, shape=n)

        for output in self.output_list:
            self.add_output(output, shape=n)

        # for output in self.output_list:
        #     for input in self.input_list:
        #         self.declare_partial_properties(
        #             output, input, complex_step_directional=True)

        for output in self.output_list:
            self.declare_partial_properties(
                output, 'K', empty=True)

        # self.add_input('Omega', shape=n)
        # self.add_input('t0', shape=n)
        # self.add_input('t1', shape=n)
        # self.add_input('t2', shape=n)

        # self.add_input('omega0', shape=n)
        # self.add_input('omega1', shape=n)
        # self.add_input('omega2', shape=n)

        # self.add_input('C00', shape=n)
        # self.add_input('C01', shape=n)
        # self.add_input('C02', shape=n)
        # self.add_input('C10', shape=n)
        # self.add_input('C11', shape=n)
        # self.add_input('C12', shape=n)
        # self.add_input('C20', shape=n)
        # self.add_input('C21', shape=n)
        # self.add_input('C22', shape=n)

    def compute(self, inputs, outputs):
        n = self.num_nodes

        for output in self.output_list:
            outputs[output] = np.zeros(n)

        # outputs['do0_dt'] = np.zeros(n)
        # outputs['do1_dt'] = np.zeros(n)
        # outputs['do2_dt'] = np.zeros(n)
        # outputs['dC00_dt'] = np.zeros(n)
        # outputs['dC01_dt'] = np.zeros(n)
        # outputs['dC02_dt'] = np.zeros(n)
        # outputs['dC10_dt'] = np.zeros(n)
        # outputs['dC11_dt'] = np.zeros(n)
        # outputs['dC12_dt'] = np.zeros(n)
        # outputs['dC20_dt'] = np.zeros(n)
        # outputs['dC21_dt'] = np.zeros(n)
        # outputs['dC22_dt'] = np.zeros(n)

        for i in range(n):
            K = inputs['K']
            Omega = inputs['Omega'][i]
            t0 = inputs['t0'][i]
            t1 = inputs['t1'][i]
            t2 = inputs['t2'][i]

            omega0 = inputs['omega0'][i]
            omega1 = inputs['omega1'][i]
            omega2 = inputs['omega2'][i]

            C00 = inputs['C00'][i]
            C01 = inputs['C01'][i]
            C02 = inputs['C02'][i]
            C10 = inputs['C10'][i]
            C11 = inputs['C11'][i]
            C12 = inputs['C12'][i]
            C20 = inputs['C20'][i]
            C21 = inputs['C21'][i]
            C22 = inputs['C22'][i]

            # print(i, K[0] *
            #       (omega1 * omega2 - 3 * Omega**2 * C01 * C02 + t0))
            # print(outputs['do0_dt'])
            outputs['do0_dt'][i] = K[0] * \
                (omega1 * omega2 - 3 * Omega**2 * C01 * C02 + t0)
            outputs['do1_dt'][i] = K[1] * \
                (omega2 * omega0 - 3 * Omega**2 * C02 * C00 + t1)
            outputs['do2_dt'][i] = K[2] * \
                (omega0 * omega1 - 3 * Omega**2 * C00 * C01 + t2)
            # fixed in orbit
            outputs['dC00_dt'][i] = C01 * omega2 - C02 * omega1 + \
                Omega * (C02 * C21 - C01 * C22)
            outputs['dC01_dt'][i] = C02 * omega0 - C00 * omega2 + \
                Omega * (C00 * C22 - C02 * C20)
            outputs['dC02_dt'][i] = C00 * omega1 - C01 * omega0 + \
                Omega * (C01 * C20 - C00 * C21)
            outputs['dC10_dt'][i] = C11 * omega2 - C12 * omega1 + \
                Omega * (C12 * C21 - C11 * C22)
            outputs['dC11_dt'][i] = C12 * omega0 - C10 * omega2 + \
                Omega * (C10 * C22 - C12 * C20)
            outputs['dC12_dt'][i] = C10 * omega1 - C11 * omega0 + \
                Omega * (C11 * C20 - C10 * C21)
            outputs['dC20_dt'][i] = C21 * omega2 - C22 * omega1
            outputs['dC21_dt'][i] = C22 * omega0 - C20 * omega2
            outputs['dC22_dt'][i] = C20 * omega1 - C21 * omega0

    def compute_partials(self, inputs, partials):

        n = self.num_nodes

        for output in self.output_list:
            for input in self.input_list:
                # print(output, input, partials)
                partials[output][input] = np.zeros((n, n))

        for i in range(n):

            K = inputs['K']
            Omega = inputs['Omega'][i]
            t0 = inputs['t0'][i]
            t1 = inputs['t1'][i]
            t2 = inputs['t2'][i]

            omega0 = inputs['omega0'][i]
            omega1 = inputs['omega1'][i]
            omega2 = inputs['omega2'][i]

            C00 = inputs['C00'][i]
            C01 = inputs['C01'][i]
            C02 = inputs['C02'][i]
            C10 = inputs['C10'][i]
            C11 = inputs['C11'][i]
            C12 = inputs['C12'][i]
            C20 = inputs['C20'][i]
            C21 = inputs['C21'][i]
            C22 = inputs['C22'][i]

            partials['do0_dt']['omega1'][i, i] = K[0] * omega2
            partials['do0_dt']['omega2'][i, i] = K[0] * omega1
            partials['do0_dt']['Omega'][i, i] = K[0] * -6 * Omega * C01 * C02
            partials['do0_dt']['C01'][i, i] = K[0] * -3 * Omega**2 * C02
            partials['do0_dt']['C02'][i, i] = K[0] * -3 * Omega**2 * C01
            partials['do0_dt']['t0'][i, i] = K[0]

            partials['do1_dt']['omega2'][i, i] = K[1] * omega0
            partials['do1_dt']['omega0'][i, i] = K[1] * omega2
            partials['do1_dt']['Omega'][i, i] = K[1] * -6 * Omega * C02 * C00
            partials['do1_dt']['C02'][i, i] = K[1] * -3 * Omega**2 * C00
            partials['do1_dt']['C00'][i, i] = K[1] * -3 * Omega**2 * C02
            partials['do1_dt']['t1'][i, i] = K[1]

            partials['do2_dt']['omega0'][i, i] = K[2] * omega1
            partials['do2_dt']['omega1'][i, i] = K[2] * omega0
            partials['do2_dt']['Omega'][i, i] = K[2] * -6 * Omega * C00 * C01
            partials['do2_dt']['C00'][i, i] = K[2] * -3 * Omega**2 * C01
            partials['do2_dt']['C01'][i, i] = K[2] * -3 * Omega**2 * C00
            partials['do2_dt']['t2'][i, i] = K[2]

            partials['dC00_dt']['C01'][i, i] = omega2 - Omega * C22
            partials['dC00_dt']['omega2'][i, i] = C01
            partials['dC00_dt']['C02'][i, i] = -omega1 + Omega * C21
            partials['dC00_dt']['omega1'][i, i] = -C02
            partials['dC00_dt']['Omega'][i, i] = (C02 * C21 - C01 * C22)
            partials['dC00_dt']['C21'][i, i] = Omega * C02
            partials['dC00_dt']['C22'][i, i] = -C01*Omega

            partials['dC01_dt']['C02'][i, i] = omega0 - Omega * C20
            partials['dC01_dt']['omega0'][i, i] = C02
            partials['dC01_dt']['C00'][i, i] = -omega2 + Omega * C22
            partials['dC01_dt']['omega2'][i, i] = -C00
            partials['dC01_dt']['Omega'][i, i] = (C00 * C22 - C02 * C20)
            partials['dC01_dt']['C22'][i, i] = Omega * C00
            partials['dC01_dt']['C20'][i, i] = -C02*Omega

            partials['dC02_dt']['C00'][i, i] = omega1 - Omega * C21
            partials['dC02_dt']['omega1'][i, i] = C00
            partials['dC02_dt']['C01'][i, i] = -omega0 + Omega * C20
            partials['dC02_dt']['omega0'][i, i] = -C01
            partials['dC02_dt']['Omega'][i, i] = (C01 * C20 - C00 * C21)
            partials['dC02_dt']['C20'][i, i] = Omega * C01
            partials['dC02_dt']['C21'][i, i] = -C00*Omega

            partials['dC10_dt']['C11'][i, i] = omega2 - Omega * C22
            partials['dC10_dt']['omega2'][i, i] = C11
            partials['dC10_dt']['C12'][i, i] = -omega1 + Omega * C21
            partials['dC10_dt']['omega1'][i, i] = -C12
            partials['dC10_dt']['Omega'][i, i] = (C12 * C21 - C11 * C22)
            partials['dC10_dt']['C21'][i, i] = Omega * C12
            partials['dC10_dt']['C22'][i, i] = -C11*Omega

            partials['dC11_dt']['C12'][i, i] = omega0 - Omega * C20
            partials['dC11_dt']['omega0'][i, i] = C12
            partials['dC11_dt']['C10'][i, i] = -omega2 + Omega * C22
            partials['dC11_dt']['omega2'][i, i] = -C10
            partials['dC11_dt']['Omega'][i, i] = (C10 * C22 - C12 * C20)
            partials['dC11_dt']['C22'][i, i] = Omega * C10
            partials['dC11_dt']['C20'][i, i] = -C12*Omega

            partials['dC12_dt']['C10'][i, i] = omega1 - Omega * C21
            partials['dC12_dt']['omega1'][i, i] = C10
            partials['dC12_dt']['C11'][i, i] = -omega0 + Omega * C20
            partials['dC12_dt']['omega0'][i, i] = -C11
            partials['dC12_dt']['Omega'][i, i] = (C11 * C20 - C10 * C21)
            partials['dC12_dt']['C20'][i, i] = Omega * C11
            partials['dC12_dt']['C21'][i, i] = -C10*Omega

            partials['dC22_dt']['C21'][i, i] = -omega0
            partials['dC22_dt']['C20'][i, i] = omega1
            partials['dC22_dt']['omega1'][i, i] = C20
            partials['dC22_dt']['omega0'][i, i] = -C21

            partials['dC21_dt']['C22'][i, i] = omega0
            partials['dC21_dt']['C20'][i, i] = -omega2
            partials['dC21_dt']['omega0'][i, i] = C22
            partials['dC21_dt']['omega2'][i, i] = -C20

            partials['dC20_dt']['C21'][i, i] = omega2
            partials['dC20_dt']['C22'][i, i] = -omega1
            partials['dC20_dt']['omega2'][i, i] = C21
            partials['dC20_dt']['omega1'][i, i] = -C22

        # for x in partials:
        #     for y in partials[x]:
        #         print(np.linalg.norm(partials[x][y]))


# 0.0
# 6.0
# 1.0
# 0.0
# 0.0
# 0.0
# 1.0
# 1.0
# 0.0
# 3.0
# 3.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 10.8
# 0.0
# 1.8
# 0.0
# 1.8
# 0.0
# 1.8
# 5.4
# 0.0
# 5.4
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 8.727272727272727
# 0.0
# 0.0
# 1.4545454545454546
# 1.4545454545454546
# 1.4545454545454546
# 0.0
# 4.363636363636363
# 4.363636363636363
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 2.0
# 2.0
# 0

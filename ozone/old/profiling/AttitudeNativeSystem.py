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

        for output in self.output_list:
            for input in self.input_list:
                self.declare_partial_properties(
                    output, input, rows=np.arange(n), cols=np.arange(n))

        for output in self.output_list:
            self.declare_partial_properties(
                output, 'K', empty=True)

    def compute(self, inputs, outputs):
        n = self.num_nodes

        for output in self.output_list:
            outputs[output] = np.zeros(n)

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
                partials[output][input] = np.zeros(n)

        K = inputs['K']
        Omega = inputs['Omega'][:, 0]
        t0 = inputs['t0'][:, 0]
        t1 = inputs['t1'][:, 0]
        t2 = inputs['t2'][:, 0]

        omega0 = inputs['omega0']
        omega1 = inputs['omega1']
        omega2 = inputs['omega2']

        C00 = inputs['C00']
        C01 = inputs['C01']
        C02 = inputs['C02']
        C10 = inputs['C10']
        C11 = inputs['C11']
        C12 = inputs['C12']
        C20 = inputs['C20']
        C21 = inputs['C21']
        C22 = inputs['C22']

        partials['do0_dt']['omega1'] = K[0] * omega2
        partials['do0_dt']['omega2'] = K[0] * omega1
        partials['do0_dt']['Omega'] = K[0] * -6 * Omega * C01 * C02
        partials['do0_dt']['C01'] = K[0] * -3 * Omega**2 * C02
        partials['do0_dt']['C02'] = K[0] * -3 * Omega**2 * C01
        partials['do0_dt']['t0'] = K[0]*np.ones(n)

        partials['do1_dt']['omega2'] = K[1] * omega0
        partials['do1_dt']['omega0'] = K[1] * omega2
        partials['do1_dt']['Omega'] = K[1] * -6 * Omega * C02 * C00
        partials['do1_dt']['C02'] = K[1] * -3 * Omega**2 * C00
        partials['do1_dt']['C00'] = K[1] * -3 * Omega**2 * C02
        partials['do1_dt']['t1'] = K[1]*np.ones(n)

        partials['do2_dt']['omega0'] = K[2] * omega1
        partials['do2_dt']['omega1'] = K[2] * omega0
        partials['do2_dt']['Omega'] = K[2] * -6 * Omega * C00 * C01
        partials['do2_dt']['C00'] = K[2] * -3 * Omega**2 * C01
        partials['do2_dt']['C01'] = K[2] * -3 * Omega**2 * C00
        partials['do2_dt']['t2'] = K[2]*np.ones(n)

        partials['dC00_dt']['C01'] = omega2 - Omega * C22
        partials['dC00_dt']['omega2'] = C01
        partials['dC00_dt']['C02'] = -omega1 + Omega * C21
        partials['dC00_dt']['omega1'] = -C02
        partials['dC00_dt']['Omega'] = (C02 * C21 - C01 * C22)
        partials['dC00_dt']['C21'] = Omega * C02
        partials['dC00_dt']['C22'] = -C01*Omega

        partials['dC01_dt']['C02'] = omega0 - Omega * C20
        partials['dC01_dt']['omega0'] = C02
        partials['dC01_dt']['C00'] = -omega2 + Omega * C22
        partials['dC01_dt']['omega2'] = -C00
        partials['dC01_dt']['Omega'] = (C00 * C22 - C02 * C20)
        partials['dC01_dt']['C22'] = Omega * C00
        partials['dC01_dt']['C20'] = -C02*Omega

        partials['dC02_dt']['C00'] = omega1 - Omega * C21
        partials['dC02_dt']['omega1'] = C00
        partials['dC02_dt']['C01'] = -omega0 + Omega * C20
        partials['dC02_dt']['omega0'] = -C01
        partials['dC02_dt']['Omega'] = (C01 * C20 - C00 * C21)
        partials['dC02_dt']['C20'] = Omega * C01
        partials['dC02_dt']['C21'] = -C00*Omega

        partials['dC10_dt']['C11'] = omega2 - Omega * C22
        partials['dC10_dt']['omega2'] = C11
        partials['dC10_dt']['C12'] = -omega1 + Omega * C21
        partials['dC10_dt']['omega1'] = -C12
        partials['dC10_dt']['Omega'] = (C12 * C21 - C11 * C22)
        partials['dC10_dt']['C21'] = Omega * C12
        partials['dC10_dt']['C22'] = -C11*Omega

        partials['dC11_dt']['C12'] = omega0 - Omega * C20
        partials['dC11_dt']['omega0'] = C12
        partials['dC11_dt']['C10'] = -omega2 + Omega * C22
        partials['dC11_dt']['omega2'] = -C10
        partials['dC11_dt']['Omega'] = (C10 * C22 - C12 * C20)
        partials['dC11_dt']['C22'] = Omega * C10
        partials['dC11_dt']['C20'] = -C12*Omega

        partials['dC12_dt']['C10'] = omega1 - Omega * C21
        partials['dC12_dt']['omega1'] = C10
        partials['dC12_dt']['C11'] = -omega0 + Omega * C20
        partials['dC12_dt']['omega0'] = -C11
        partials['dC12_dt']['Omega'] = (C11 * C20 - C10 * C21)
        partials['dC12_dt']['C20'] = Omega * C11
        partials['dC12_dt']['C21'] = -C10*Omega

        partials['dC22_dt']['C21'] = -omega0
        partials['dC22_dt']['C20'] = omega1
        partials['dC22_dt']['omega1'] = C20
        partials['dC22_dt']['omega0'] = -C21

        partials['dC21_dt']['C22'] = omega0
        partials['dC21_dt']['C20'] = -omega2
        partials['dC21_dt']['omega0'] = C22
        partials['dC21_dt']['omega2'] = -C20

        partials['dC20_dt']['C21'] = omega2
        partials['dC20_dt']['C22'] = -omega1
        partials['dC20_dt']['omega2'] = C21
        partials['dC20_dt']['omega1'] = -C22

        # for x in partials:
        #     for y in partials[x]:
        #         print(np.linalg.norm(partials[x][y]))
        #         if partials[x][y].shape != (4,):
        # print(partials[x][y].shape, x, y)

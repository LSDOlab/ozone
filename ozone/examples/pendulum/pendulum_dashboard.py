import lsdo_dash.api as ld
import numpy as np

class SampleDash(ld.BaseDash):

    def set_config(self):
        self.fps = 30

    def setup(self):
        # this method tell simulator which variables to save

        # Pull data from simulator
        self.set_clientID('simulator')
        self.save_variable('timestep_vector', history=True)
        self.save_variable('torque')
        self.save_variable('solved_theta')
        self.save_variable('solved_thetadot')
        self.save_variable('constraint_final_theta', history=True)
        self.save_variable('constraint_final_thetadot', history=True)
        self.save_variable('final_time', history=True)

        self.add_frame('main',
                       height_in=6,
                       width_in=12,
                       ncols=2,
                       nrows=2,
                       wspace=0.5,
                       hspace=0.5)

    def plot(self,
             frames,
             data_dict_current,
             data_dict_history,
             limits_dict,
             video=False):

        ddcs = data_dict_current['simulator']
        ddhs = data_dict_history['simulator']
        iter_list = ddhs['global_ind']

        frame = frames['main']
        frame.clear_all_axes()
        ax_states = frame[0, 0]
        time_point_vector = np.cumsum(ddcs['timestep_vector'])
        time_point_vector = np.insert(time_point_vector, 0, 0.0)
        ax_states.plot(time_point_vector, ddcs['solved_theta'])
        ax_states.plot(time_point_vector, ddcs['solved_thetadot'])
        ax_states.set_xlabel('time (s)')
        ax_states.set_ylabel('states')
        ax_states.set_title('Integrated ODE states')
        ax_states.legend(['theta (rad)', 'theta dot (rad/s)'])
        ax_states.grid()
        ax_states.set_xlim([0, 2.0])

        ax_states = frame[0, 1]
        # print(ddhs['constraint_final_theta'].shape)
        # print(ddhs['constraint_final_thetadot'].shape)
        # print(len(iter_list))
        ax_states.plot(time_point_vector, ddcs['torque'])
        ax_states.set_xlabel('time (s)')
        ax_states.set_ylabel('control input (rad/s^2)')
        ax_states.set_title('Design Variables (Control Input)')
        ax_states.legend(['theta (rad)'])
        ax_states.grid()
        ax_states.set_xlim([0, 2.0])
        ax_states.set_ylim([-105, 105.0])

        ax_states = frame[1, 0]
        # print(ddhs['constraint_final_theta'].shape)
        # print(ddhs['constraint_final_thetadot'].shape)
        # print(len(iter_list))
        ax_states.plot(iter_list, ddhs['constraint_final_theta'].flatten())
        ax_states.plot(iter_list, ddhs['constraint_final_thetadot'].flatten())
        ax_states.set_xlabel('optimization iterations')
        ax_states.set_ylabel('final states')
        ax_states.legend(['final theta (rad)', 'final theta dot (rad/s)'])
        ax_states.set_title('Final-Value Constraints')
        ax_states.grid()
        ax_states.set_xlim(left=0)

        ax_states = frame[1, 1]
        # print(ddhs['constraint_final_theta'].shape)
        # print(ddhs['constraint_final_thetadot'].shape)
        # print(len(iter_list))
        ax_states.plot(iter_list, ddhs['final_time'].flatten())
        ax_states.set_xlabel('optimization iterations')
        ax_states.set_ylabel('final time (s)')
        ax_states.set_title('Objective')
        ax_states.grid()
        ax_states.set_ylim([0.0, 5.0])
        ax_states.set_xlim(left=0)

        frame.write()


if __name__ == '__main__':

    dash = SampleDash()
    # dash.visualize_most_recent(show=True)
    # dash.run_GUI()
    # dash.visualize_auto_refresh()

    dash.visualize_auto_refresh()

    # dash.visualize_all()
    # dash.make_mov()

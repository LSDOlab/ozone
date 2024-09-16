from smt.surrogate_models import RBF, RMTB
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Century'
plt.rcParams.update({'font.size': 16})

xt = np.deg2rad(np.linspace(-90, 90, 46))
yt_cl = np.array([-0.033,-0.065,-0.097,-0.129,-0.161,-0.193,-0.225,-0.257,-0.289,-0.321,-0.353,-0.385,-0.417,-0.449,
                    -0.481,-0.513,-0.545,-0.577,-0.609,-0.634,-0.642,-0.4854,0.0914,0.6101,0.9747,1.2206,1.25,1.19,1.13,
                    1.07,1.01,0.95,0.89,0.83,0.77,0.71,0.65,0.59,0.53,0.47,0.41,0.35,0.29,0.23,0.17,0.11])

yt_cd = np.array([1.345,1.325,1.305,1.285,1.265,1.245,1.225,1.205,1.185,1.16,1.115,0.99,0.803131468,0.657798541,0.511489095,
                    0.35,0.20285,0.13657,0.08653,0.05091,0.03576,0.02778,0.01702,0.01561,0.01837,0.03392,0.07,0.088,0.14493,
                    0.20944,0.32,0.479306071,0.62122118,0.767530626,0.95,1.09,1.15,1.18,1.2,1.22,1.24,1.26,1.28,1.3,1.32,1.34])


sm_cl = RBF(d0=0.15,print_global=False,print_solver=False,)
sm_cl.set_training_values(xt, yt_cl)
sm_cl.train()

sm_cd = RBF(d0=0.35,print_global=False,print_solver=False,)
sm_cd.set_training_values(xt, yt_cd)
sm_cd.train()






if __name__ == '__main__':
    num = 1000
    x = np.deg2rad(np.linspace(-90, 90, num))

    ycl = sm_cl.predict_values(x)
    ycd = sm_cd.predict_values(x)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xt, yt_cl,'o',color='k')
    ax1.plot(x, ycl,color='k')
    ax1.set_ylabel('$C_L$')
    ax1.set_xlabel('α (rad)')
    ax1.legend(['Training Points','RBF Surrogate'])
    ax2.plot(xt, yt_cd,'o',color='k')
    ax2.plot(x, ycd,color='k')
    ax2.set_ylabel('$C_D$')
    ax2.set_xlabel('α (rad)')
    ax2.legend(['Training Points','RBF Surrogate'])
    plt.show()
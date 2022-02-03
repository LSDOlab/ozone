import matplotlib.pyplot as plt
import numpy as np

# This script plots the results of the checkpoint example

# Load in results
memory_vector_total = np.load('memory.npy')
duration_vector_total = np.load('duration_vector.npy')
derivative_norm_vector_total = np.load(
    'derivative_norm_vector.npy')
value_vector_total = np.load('value_vector.npy')
nc_vector = np.load('nc_vector.npy')

num_trials = len(memory_vector_total)
fig, ax = plt.subplots(2, 2)
plt.suptitle(
    'Memory comparison for size 500 state with 300 timesteps')
# Average Vector
memory_vector_avg = np.zeros(memory_vector_total[0].shape)
duration_vector_avg = np.zeros(duration_vector_total[0].shape)
derivative_norm_vector_avg = np.zeros(derivative_norm_vector_total[0].shape)
value_vector_avg = np.zeros(value_vector_total[0].shape)

for i in range(num_trials):
    memory_vector = memory_vector_total[i]
    duration_vector = duration_vector_total[i]
    derivative_norm_vector = derivative_norm_vector_total[i]
    value_vector = value_vector_total[i]
    # Creating plots/figures:
    ax[0, 0].plot(nc_vector, memory_vector, linewidth=1.6)
    ax[0, 0].grid()
    ax[0, 0].set_title('Number of Checkpoints Relation to Memory used')
    ax[0, 0].set_xlabel('Number of Checkpoints')
    ax[0, 0].set_ylabel('Memory used (bytes)')

    ax[0, 1].plot(nc_vector, duration_vector, linewidth=1.6)
    ax[0, 1].grid()
    ax[0, 1].set_title('Number of Checkpoints Relation to run time')
    ax[0, 1].set_xlabel('Number of Checkpoints')
    ax[0, 1].set_ylabel('Duration (seconds)')

    ax[1, 0].scatter(memory_vector, duration_vector, s=1.5)
    ax[1, 0].grid()
    ax[1, 0].set_title('Duration Relation to Memory used')
    ax[1, 0].set_xlabel('Memory used (bytes)')
    ax[1, 0].set_ylabel('Duration (seconds)')

    ax[1, 1].plot(nc_vector,
                  abs(derivative_norm_vector-derivative_norm_vector[0]), linewidth=1.6)
    ax[1, 1].plot(nc_vector, abs(value_vector -
                                 value_vector[0]), linewidth=1.6)
    ax[1, 1].grid()
    ax[1, 1].set_title('Output Error')
    ax[1, 1].set_xlabel('Number of Checkpoints')
    ax[1, 1].set_ylabel('Value')
    ax[1, 1].legend(['Value', 'Derivative'])

    # Average:
    memory_vector_avg += memory_vector
    duration_vector_avg += duration_vector
    derivative_norm_vector_avg += derivative_norm_vector
    value_vector_avg += value_vector
plt.tight_layout()

memory_vector_avg = memory_vector_avg/num_trials
duration_vector_avg = duration_vector_avg/num_trials
derivative_norm_vector_avg = derivative_norm_vector_avg/num_trials
value_vector_avg = value_vector_avg/num_trials

# Creating plots/figures:
fig, ax2 = plt.subplots(2, 2)
plt.suptitle(
    'Memory comparison average for size 500 state with 300 timesteps')
ax2[0, 0].plot(nc_vector, memory_vector_avg, linewidth=1.6, color='red')
ax2[0, 0].grid()
ax2[0, 0].set_title('Number of Checkpoints Relation to Memory used')
ax2[0, 0].set_xlabel('Number of Checkpoints')
ax2[0, 0].set_ylabel('Memory used (bytes)')

ax2[0, 1].plot(nc_vector, duration_vector_avg, linewidth=1.6, color='red')
ax2[0, 1].grid()
ax2[0, 1].set_title('Number of Checkpoints Relation to run time')
ax2[0, 1].set_xlabel('Number of Checkpoints')
ax2[0, 1].set_ylabel('Duration (seconds)')

ax2[1, 0].scatter(memory_vector_avg, duration_vector_avg, s=1.5, color='red')
ax2[1, 0].grid()
ax2[1, 0].set_title('Duration Relation to Memory used')
ax2[1, 0].set_xlabel('Memory used (bytes)')
ax2[1, 0].set_ylabel('Duration (seconds)')

ax2[1, 1].plot(nc_vector,
               abs(derivative_norm_vector_avg-derivative_norm_vector_avg[0]), linewidth=1.6)
ax2[1, 1].plot(nc_vector, abs(value_vector_avg -
                              value_vector_avg[0]), linewidth=1.6, color='red')
ax2[1, 1].grid()
ax2[1, 1].set_title('Output Error')
ax2[1, 1].set_xlabel('Number of Checkpoints')
ax2[1, 1].set_ylabel('Value')
ax2[1, 1].legend(['Value', 'Derivative'])

plt.tight_layout()


memory_vector_avg = memory_vector_avg[1:]
duration_vector_avg = duration_vector_avg[1:]
derivative_norm_vector_avg = derivative_norm_vector_avg[1:]
value_vector_avg = value_vector_avg[1:]
nc_vector = nc_vector[1:]

# Creating plots/figures:
fig, ax3 = plt.subplots(2, 2)
plt.suptitle(
    'Memory comparison average for size 500 state with 300 timesteps EXCLUDING STANDARD TIMEMARCHING')
ax3[0, 0].plot(nc_vector, memory_vector_avg, linewidth=1.6, color='red')
ax3[0, 0].grid()
ax3[0, 0].set_title('Number of Checkpoints Relation to Memory used')
ax3[0, 0].set_xlabel('Number of Checkpoints')
ax3[0, 0].set_ylabel('Memory used (bytes)')

ax3[0, 1].plot(nc_vector, duration_vector_avg, linewidth=1.6, color='red')
ax3[0, 1].grid()
ax3[0, 1].set_title('Number of Checkpoints Relation to run time')
ax3[0, 1].set_xlabel('Number of Checkpoints')
ax3[0, 1].set_ylabel('Duration (seconds)')

ax3[1, 0].scatter(memory_vector_avg, duration_vector_avg, s=1.5, color='red')
ax3[1, 0].grid()
ax3[1, 0].set_title('Duration Relation to Memory used')
ax3[1, 0].set_xlabel('Memory used (bytes)')
ax3[1, 0].set_ylabel('Duration (seconds)')

ax3[1, 1].plot(nc_vector,
               abs(derivative_norm_vector_avg-derivative_norm_vector_avg[0]), linewidth=1.6)
ax3[1, 1].plot(nc_vector, abs(value_vector_avg -
                              value_vector_avg[0]), linewidth=1.6, color='red')
ax3[1, 1].grid()
ax3[1, 1].set_title('Output Error')
ax3[1, 1].set_xlabel('Number of Checkpoints')
ax3[1, 1].set_ylabel('Value')
ax3[1, 1].legend(['Value', 'Derivative'])

plt.tight_layout()


plt.show()

import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
files = ['rmse_21_04_2023_16_15_38.npy', 'rmse_V1_VINS_21_04_2023_17_00_55.npy', 'rmse_V1_VO_21_04_2023_17_22_56.npy']

plt.figure()
for f in files:
    data = np.load(os.path.join(dir_path, f))
    plt.plot(data)

plt.legend(['robot_localization', 'VINS_fusion', 'VO'])
plt.title('RMSE with Ground Truth Trajectory')
plt.show()

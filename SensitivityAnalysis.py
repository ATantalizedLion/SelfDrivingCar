import os
import numpy as np
import matplotlib.pyplot as plt

dir = 'Models/'

subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
labels = []


for folder in subfolders:
    if folder[7:9] == 'SO':  # simple env, optimizer
        path = folder+'/Progress.csv'
        progress = np.genfromtxt(path)
        progress = progress[:, :-1] # throw away the last, often interrupted datapoint
        if progress[0].size > 2000:
            progress = progress[:, :2000]

        plt.plot(np.arange(progress[0].size), progress[0, :], linewidth=0.7)
        labels.append(folder[14:])

plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.legend(labels)
plt.show()
plt.savefig('Optimsers-SensA')
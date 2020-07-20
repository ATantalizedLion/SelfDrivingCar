import os
import numpy as np
import matplotlib.pyplot as plt

dir = 'Models/'

subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
# iCount = 0
# oCount = 0
# for folder in subfolders:
#     if folder[7:9] == 'SI':  # simple env, initialiser
#         iCount += 1
#     if folder[7:9] == 'SO':  # simple env, optimizer
#         oCount += 1

labels = []
for folder in subfolders:
    if folder[7:9] == 'SO':  # simple env, optimizer
        path = folder + '/Progress.csv'
        progress = np.genfromtxt(path)
        progress = progress[:, :-1]  # throw away the last data point since it is often interrupted by termination.
        if progress[0].size > 2000:
            progress = progress[:, :2000]

        plt.plot(np.arange(progress[0].size), progress[0, :], linewidth=0.7)
        labels.append(folder[14:])
plt.xlabel('Iteration')
plt.ylabel('Average of the Discounted Rewards')
plt.legend(labels)
plt.show()


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7), sharex=True, sharey=True)
i = 0
j = 0
for folder in subfolders:
    if folder[7:9] == 'SO':  # simple env, optimizer
        path = folder + '/Progress.csv'
        progress = np.genfromtxt(path)
        progress = progress[:, :-1]  # throw away the last data point since it is often interrupted by termination.
        if progress[0].size > 2000:
            progress = progress[:, :2000]
        axes[i, j].plot(np.arange(progress[0].size), progress[0, :], linewidth=0.7)
        axes[i, j].set_title(folder[14:])
        i += 1
        if i == 3:
            j += 1
            i = 0

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.xlabel('Iteration')
plt.ylabel('Average of the Discounted Rewards')
plt.show()






labels = []
for folder in subfolders:
    if folder[7:9] == 'SI':  # simple env, initialiser
        path = folder + '/Progress.csv'
        progress = np.genfromtxt(path)
        progress = progress[:, :-1]  # throw away the last data point since it is often interrupted by termination.
        if progress[0].size > 300:
            progress = progress[:, :300]
        plt.plot(np.arange(progress[0].size), progress[0, :], linewidth=0.7)
        if folder[14:23] == 'Truncated':
            labels.append(folder[14:23] + ' - 0.' + folder[23:])
        else:
            labels.append(folder[14:])
plt.xlabel('Iteration')
plt.ylabel('Average of the Discounted Rewards')
plt.legend(labels)
plt.show()



fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7), sharex=True, sharey=True)
i = 0
j = 0
for folder in subfolders:
    if folder[7:9] == 'SI':  # simple env, initialiser
        path = folder + '/Progress.csv'
        progress = np.genfromtxt(path)
        progress = progress[:, :-1]  # throw away the last data point since it is often interrupted by termination.
        if progress[0].size > 300:
            progress = progress[:, :300]
        if folder[14:23] == 'Truncated':
            axes[i, j].set_title(folder[14:23] + ' - 0.' + folder[23:])
        else:
            axes[i, j].set_title(folder[14:])
        axes[i, j].plot(np.arange(progress[0].size), progress[0, :], linewidth=0.7)
        i += 1
        if i == 3:
            j += 1
            i = 0

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.xlabel('Iteration')
plt.ylabel('Average of the Discounted Rewards')
plt.show()

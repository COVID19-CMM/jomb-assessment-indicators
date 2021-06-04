import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'font.size': 20})
filenames = [
    'data_1a_500',
    'data_1b_500',
    'data_2a_500',
    'data_2b_500',
]

umbrales = [None] * len(filenames)
dias = [None] * len(filenames)
max_infectados = [None] * len(filenames)
max_ucis = [None] * len(filenames)
max_muertos = [None] * len(filenames)
last_deconfinement = 0

for i, filename in enumerate(filenames):
    try:
        with open('out/' + filename + '.npy', 'rb') as f:
            umbrales[i] = np.load(f)
            dias[i] = np.load(f)
            max_infectados[i] = np.load(f)
            max_ucis[i] = np.load(f)
            max_muertos[i] = np.load(f)
            last_deconfinement = max(last_deconfinement, np.load(f))
    except IOError:
        pass
styles = ['-', '-', '-.','-.']
colors = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange']
labels = [ 'Mean of infected (detected)', 'Difference of infected (detected)','Mean of hospitalizations', 'Difference of hospitalizations']
axes = ['Peak of hospitalization demand']#, 'Max. active', 'Max. deceased']
datas = [max_ucis]#, max_infectados, max_muertos]
fig, ax = plt.subplots(figsize=(15,9), nrows=1, ncols=1, squeeze=False)
for i, label in enumerate(labels):
    if umbrales[i] is not None:
        for j, (axis, data) in enumerate(zip(axes, datas)):
            ax[j,0].set_xlabel(axis)
            ax[j,0].set_ylabel("% of total days in lockdown")
            ax[j,0].plot(data[i], 100*dias[i]/last_deconfinement, color=colors[i], ls=styles[i], label=label)


for j in range(len(datas)):
    ax[j,0].legend()
for i, label in enumerate(labels):
    y = np.interp(7000000, datas[0][i], 100*dias[i]/last_deconfinement)
    umbral = np.interp(7000000, datas[0][i], umbrales[i])
    print('%s: %.2g%% with threshold %g' % (labels[i], y, umbral))

ax[0,0].set_xlim(0, 14000000)
ax[0,0].set_title('Trade-off curves: China; t_0= 29 March 2020' )
ax[0,0].axvline(7000000, 0, 100, color='black', ls=':')
ax[0,0].annotate('Objective peak ICU', xy=(7000000, 5), xytext=(40,20), textcoords='offset points', arrowprops={'arrowstyle': '->', 'connectionstyle': 'angle'}, ha='left')

fig.savefig('figs/indicators.png')
#plt.show()

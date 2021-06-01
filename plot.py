import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'figure.autolayout': True})

filenames = [
    'data_2a_500',
    'data_2b_500',
    'data_1a_10',
    'data_1b_500',
]

styles = ['-', '-', '-.', '-.']
colors = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange']
labels = ['Mean of infected (detected)', 'Difference of infected (detected)', 'Mean of hospitalizations', 'Difference of hospitalizations']

umbrales = [None] * len(filenames)
dias = [None] * len(filenames)
max_infectados = [None] * len(filenames)
max_ucis = [None] * len(filenames)
max_muertos = [None] * len(filenames)
last_deconfinement = 0

for i, filename in enumerate(filenames):
    try:
        with open('simulations/' + filename + '.npy', 'rb') as f:
            umbrales[i] = np.load(f)
            dias[i] = np.load(f)
            max_infectados[i] = np.load(f)
            max_ucis[i] = np.load(f)
            max_muertos[i] = np.load(f)
            last_deconfinement = max(last_deconfinement, np.load(f))
    except IOError:
        pass

for data in max_ucis:
    for i in range(1,len(data)):
        if data[i] < data[i-1]:
            data[i] = data[i-1]

print(last_deconfinement)
print("Last deconfinement: %s" % (np.datetime64('2020-09-21') + last_deconfinement*np.timedelta64(1,'D'),))

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

ax[0,0].axvline(1200, 0, 10, color='black', ls=':')
ax[0,0].annotate('Objective peak hospitalization demand', xy=(1200, 5), xytext=(40,40), textcoords='offset points', arrowprops={'arrowstyle': '->', 'connectionstyle': 'angle'}, ha='left')

#offsets = [(20,20),(20,-30),(20,10),(-20,10),(-20,30),(-20,5)]
for i, label in enumerate(labels):
    y = np.interp(1200, datas[0][i], 100*dias[i]/last_deconfinement)
    umbral = np.interp(1200, datas[0][i], umbrales[i])
    print('%s: %.4g%% with threshold %g' % (labels[i], y, umbral))
#    ha = 'left'
#    if offsets[i][0] < 0:
#        ha = 'right'
#    ax[0,0].annotate('%s: %.2f%%' % (labels[i], y,), xy=(1200, y), xytext=offsets[i], textcoords='offset points', arrowprops={'arrowstyle': '->', 'connectionstyle': 'angle'}, ha=ha)

ax[0,0].set_xlim(500, 5000)
ax[0,0].set_ylim(0, 50)
#ax[1].set_ylim(0, 20)
#ax[2].set_ylim(0, 20)
#ax[2].set_xlim(18000, None)

ax[0,0].set_title('Trade-off curves: Metropolitan Region (Chile); t_0 = 21 sep. 2020')

fig.savefig('figs/indicators.png')
plt.show()

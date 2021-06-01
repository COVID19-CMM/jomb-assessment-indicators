import os
import sys
import numpy as np
import argparse as argp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'figure.autolayout': True})

parser = argp.ArgumentParser(description="Correr simulaciones de COVID19")
parser.add_argument("--indicador", help="indicador", type=str, choices=['1', '2'], required=True)
parser.add_argument("--metodo", help="método", type=str, choices=['a', 'b', 'c', 'd'], required=True)
parser.add_argument("--umbral", help="rango para el umbral", type=float, nargs=2, required=True)
parser.add_argument("-n", help="número de umbrales", type=int, default=10)
args = parser.parse_args()

indicator_names = {'1': 'ICU', '2': 'Active'}
method_names = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}

print('Indicador: %s, Método: %s, Umbrales: %f -- %f' % (indicator_names[args.indicador], method_names[args.metodo], args.umbral[0], args.umbral[1]))

dt = np.timedelta64(1,'h')
t0 = np.datetime64('2020-09-21')
tmax = np.datetime64('2025-03-21')
y0 = {
    'S': 6.7e6,
    'E': 1697.2,
    'Im': 1723.3,
    'I': 2539.6,
    'H': 1156.4,
    'Hc': 433.3,
    'R': 4.2e5,
    'D': 1.2e4,
}
params = {
    'βE': 0.04/0.2,
    'βIm': 0.04/0.2,
    'βI': 0.2,
    'βH': 0.0,
    'βHc': 0.0,
    'γE': 0.39,
    'γIm': 0.17,
    'γI': 0.17,
    'γH': 0.17,
    'γHc': 0.14,
    'μb': 0.0, #3.57e-5,
    'μd': 0.0, #1.57e-5,
    'φEI': 0.60,
    'φIR': 0.61,
    'φHR': 0.61,
    'φHD': 0.12,
    'φHcD': 0.12,
}

class Epidemic:
    def __init__(self, t0, tmax, y0, p, indicador, metodo, umbral, dt=np.timedelta64(1,'D')):
        self.col_ids = {'S': 0, 'E': 1, 'Im': 2, 'I': 3, 'H': 4, 'Hc': 5, 'R': 6, 'D': 7,
                        'r_effective': 8, 'alpha': 9, 'indicator': 10}

        # número de días para ver atrás para calcular el promedio del indicador
        Δ = 14

        # mínimo número de días en cuarentena / desconfinado
        δ = 14

        alpha_normal = 1.0
        alpha_cuarentena = 0.2

        T = np.arange(t0, tmax, dt)
        B = np.array([
            [0, -p['βE'], -p['βIm'], -p['βI'], 0, 0, 0, 0],
            [0, p['βE'], p['βIm'], p['βI'], 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        M = np.array([
            # S, E, Im, I
            [p['μb']-p['μd'], p['μb'], p['μb'], p['μb'], p['μb'], p['μb'], p['μb'], 0],
            [0, -p['γE']-p['μd'], 0, 0, 0, 0, 0, 0],
            [0, (1.0-p['φEI'])*p['γE'], -p['γIm']-p['μd'], 0, 0, 0, 0, 0],
            [0, p['φEI']*p['γE'], 0, -p['γI']-p['μd'], 0, 0, 0, 0],

            # H, Hc, R, D
            [0, 0, 0, (1.0-p['φIR'])*p['γI'], -p['γH']-p['μd'], 0, 0, 0],
            [0, 0, 0, 0, (1.0-p['φHR']-p['φHD'])*p['γH'], -p['γHc']-p['μd'], 0, 0],
            [0, 0, p['γIm'], p['φIR']*p['γI'], p['φHR']*p['γH'], (1-p['φHcD'])*p['γHc'], -p['μd'], 0],
            [0, 0, 0, 0, p['φHD']*p['γH'], p['φHcD']*p['γHc'], 0, 0],
        ])
        μb = p['μb']

        # see O. Diekmann, J.A.P. Heesterbeek, and M.G. Roberts, "The construction of next-generation matrices for compartmental epidemic models", J R Soc Interface, 2010
        # K = -B * M^-1  =>  (M.T) * (K.T) = -B.T
        K = np.linalg.solve(M[1:6,1:6].T, -B[1:6,1:6].T).T
        if np.linalg.det(K) != 0.0:
            logging.warn('det(K) = %g != 0' % np.linalg.det(K))
        R = K.trace()

        f = dt/np.timedelta64(1,'D')
        Δ_orig = Δ
        Δ = int(Δ/f + 0.5)
        δ = int(δ/f + 0.5)
        
        days = 0
        alpha_src = alpha_cuarentena
        alpha_dst = alpha_cuarentena
        cuarentena = True
        last_transition = -1000
        last_deconfinement = len(T)
        D = 15.0

        data = np.empty((len(T),11))
        data[:] = np.NaN
        data[0,:8] = np.array([y0['S'], y0['E'], y0['Im'], y0['I'], y0['H'], y0['Hc'], y0['R'], y0['D']])
        N0 = np.sum(data[0,:7])
        data[0,8] = R * data[0,0]/N0
        data[0,9] = alpha_cuarentena if cuarentena else alpha_normal
        for i in range(1, len(T)):
            if Δ < i:
                if indicador == '1':
                    I = data[i-Δ-1:i,5]
                elif indicador == '2':
                    I = (data[i-Δ-1:i,3] + data[i-Δ-1:i,4] + data[i-Δ-1:i,5]) * 100000/N0

                if metodo == 'a':
                    val = np.mean(I)
                elif metodo == 'b':
                    val = (I[-1]-I[0])/(Δ_orig+1)
                elif metodo == 'c':
                    val = (I[-1]-I[0])/I[0]
                elif metodo == 'd':
                    val = (I[-1]-I[-2])/(I[1]-I[0])

                data[i,10] = val
                if last_transition+δ <= i:
                    if not cuarentena and val > umbral:
                        cuarentena = True
                        last_transition = i
                        last_transition_alpha = alpha
                        alpha_src = alpha
                        alpha_dst = alpha_cuarentena
                        D = 3.0
                    elif cuarentena and val < umbral:
                        cuarentena = False
                        last_transition = i
                        last_deconfinement = i
                        alpha_src = alpha
                        alpha_dst = alpha_normal
                        D = 15.0

            state = data[i-1,:8]

            #alpha = alpha_dst + (alpha_src-alpha_dst)*np.exp(-np.log(2)*(1.0/D)*float(i-last_transition)*f)
            alpha = alpha_dst + (alpha_src-alpha_dst)*np.clip(1.0-float(i-last_transition)*f/14.0,0.0,1.0)
            alphav = np.array([1, alpha, alpha, 1, 1, 1, 1, 1])

            data[i,:8] = state + f*(np.dot(M, state) + alphav*np.dot(B, state) * state[0]/np.sum(state[:7]))
            data[i,8] = alpha * R * data[i,0]/np.sum(data[i,:7])
            data[i,9] = alpha
            days += f*cuarentena

        self.T = T
        self.data = data
        self.days = days
        self.last_deconfinement = int(f*last_deconfinement + 0.5)

    def max(self, col):
        return np.max(self.data[:,self.col_ids[col]])

    def plot(self, cols=['I', 'H', 'Hc', 'D'], title=None, filename=None):
        col_names = {'r_effective': '$R_e$', 'alpha': 'Alpha', 'indicator': 'Indicator'}
        col_styles = {
            'S': '-.', 'E': '-.', 'Im': '-.', 'I': '-.',
            'H': '-.', 'Hc': '-.', 'R': '-.', 'D': '-.',
        }
        col_colors = {
            'S': 'tab:gray', 'E': 'tab:cyan', 'Im': 'tab:purple', 'I': 'tab:blue',
            'H': 'tab:olive', 'Hc': 'tab:orange', 'R': 'tab:green', 'D': 'tab:red',
            'r_effective': 'tab:gray', 'alpha': 'tab:blue', 'indicator': 'black',
        }

        fig, ax = plt.subplots(figsize=(15,21), nrows=3, ncols=1, sharex=True, squeeze=False)
        fig.autofmt_xdate()

        ax[0,0].get_xaxis().tick_bottom()
        ax[0,0].get_yaxis().tick_left()

        if title is not None:
            ax[0,0].set_title(title)

        for col in cols:
            name = '$' + col + '$'
            style = '-'
            if col in col_names:
                name = col_names[col]
            if col in col_styles:
                style = col_styles[col]
            if col in col_colors:
                color = col_colors[col]
            ax[0,0].plot(self.T, self.data[:,self.col_ids[col]], color=color, ls=style, label=name)

        ax[1,0].plot(self.T, self.data[:,8], color='black', label='$R_e$')
        ax[1,0].plot(self.T, self.data[:,9], color='tab:blue', label='Alpha')
        ax[2,0].plot(self.T, self.data[:,10], color='black', label='Indicator')
        
        ax[0,0].set_xlim(np.min(self.T), np.max(self.T))
        ax[0,0].set_ylim(0, None)
        ax[0,0].set_ylabel("# of people")

        for i in [0,1,2]:
            ax[i,0].tick_params(bottom=False, top=False, left=False, right=False)
            ax[i,0].legend(frameon=False)
            ax[i,0].grid()
  
        #plt.tight_layout()
        if filename is not None:
            fig.savefig(filename)
        #plt.show()

for f in [0.0]:
    umbral = args.umbral[0]+f*(args.umbral[1]-args.umbral[0])
    print('pred:', umbral)
    epidemic = Epidemic(t0, tmax, y0, params, args.indicador, args.metodo, umbral, dt=dt)
    epidemic.plot(filename='figs/pred_indicador_%s%s_umbral_%g.png' % (args.indicador, args.metodo, umbral), title='Indicator=%s %s Threshold=%g' % (indicator_names[args.indicador], method_names[args.metodo], umbral), cols=['Hc'])

umbrales = np.linspace(args.umbral[0], args.umbral[1], args.n)
dias = np.empty((len(umbrales),))
max_infectados = np.empty((len(umbrales),))
max_ucis = np.empty((len(umbrales),))
max_muertos = np.empty((len(umbrales),))
last_deconfinement = 0

for i in range(len(umbrales)):
    if i%10 == 0:
        print('iter: %4d/%d' % (i+1,len(umbrales)))

    epidemic = Epidemic(t0, tmax, y0, params, args.indicador, args.metodo, umbrales[i], dt=dt)

    dias[i] = epidemic.days
    max_infectados[i] = epidemic.max('I')
    max_ucis[i] = epidemic.max('Hc')
    max_muertos[i] = epidemic.max('D')
    last_deconfinement = max(last_deconfinement, epidemic.last_deconfinement)

if not os.path.exists('simulations'):
    os.makedirs('simulations')
with open('simulations/data_%s%s_%d.npy' % (args.indicador, args.metodo, args.n), 'wb') as f:
    np.save(f, umbrales)
    np.save(f, dias)
    np.save(f, max_infectados)
    np.save(f, max_ucis)
    np.save(f, max_muertos)
    np.save(f, last_deconfinement)

labels = ['Max. ICU', 'Max. active', 'Max. deceased']
datas = [max_ucis, max_infectados, max_muertos]
fig, ax = plt.subplots(figsize=(15,21), nrows=3, ncols=1)
for i, (label, data) in enumerate(zip(labels, datas)):
    ax2 = ax[i].twinx()
    ax2.plot(data, umbrales, color='gray', ls=':')
    ax2.set_ylabel("Threshold", color='gray')

    ax[i].set_xlabel(label)
    ax[i].set_ylabel("% days in quarantine")
    ax[i].plot(data, 100*dias/last_deconfinement, color='black')

ax[0].set_title('Indicator=%s %s Threshold=[%g,%g]' % (indicator_names[args.indicador], method_names[args.metodo], args.umbral[0], args.umbral[1]))

plt.tight_layout()
if not os.path.exists('figs'):
    os.makedirs('figs')
fig.savefig('figs/fig_indicador_%s%s_umbrales_%g-%g_n_%d.png' % (args.indicador, args.metodo, args.umbral[0], args.umbral[1], args.n))
#plt.show()

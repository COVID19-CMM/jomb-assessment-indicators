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

indicator_names = {'1': 'I', '2': 'I acumulado', '3':'Hr+Hd'}
method_names = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}

dt = np.timedelta64(1,'h')
t0 = np.datetime64('2020-09-10')
tmax = np.datetime64('2025-09-10')
y0 = {
    'S': 1.39e9,
    'E': 4310,
    'I': 2586,
    'Iu': 431,
    'Hr': 68.96000000000001,
    'Hd': 21.55,
    'Ru': 28,#7.4*10000
    'Rd': 28,
    'D': 17
}
ivorra29M = [0.2834, 0.3806, 0.4001, 7.0000, 0.0195, 0.0161, 0.1090]
def ivorra_x_params_constante(βI, CE, Cu, δR, δω, ωc, κ1):
    #λ1 = np.datetime64('2020-01-23')
    #λ2 = np.datetime64('2020-03-29')
    m =1.0# lambda t: 1.0 if t < λ1 else (np.exp(-κ1*(t-λ1)/np.timedelta64(1,'D')) if t<λ2 else 1.0)
    ωnc = ωc + δω
    ω =ωnc # lambda t: m(t)*ωnc + (1-m(t))*ωc
    #θt0 = np.datetime64('2020-01-24')
    #θt1 = np.datetime64('2020-02-08')
    θ =(1-0.86)# lambda t: (1-0.86) if t < θt0 else ((1-0.35) if t > θt1 else (1-0.86)-(0.35-0.86)*(t-θt0)/(θt1-θt0))
    dg  = 6 #5.7
    dE  = 5.5
    dI  = 6.7
    dIu = 14-dI
    g   =0.0 #lambda t: dg*(1-m(t))
    γE = 1/ dE
    #γE=0.1818
    γIu = 1/ dIu
    γI = 1/dI #lambda t: 1 / (dI - g(t))
    γIu =1/dIu # lambda t: 1 / (dIu + g(t))
    γHr =1/dIu  #lambda t: 1 / (dIu + g(t))
    γHd =1/(dIu+δR)# lambda t: 1 / (dIu + g(t) + δR)   
    βE = CE*βI
    βImin=Cu*βI
    βIu =βImin+((βI-βImin)/(1-ω))*(1- θ) #lambda t: βImin+((βI-βImin)/(1-ω(t)))*(1- θ(t))
    CH  = 0.0275*(βI/γI + βE/γE+(1-θ)*βIu/γIu)/((1-0.0275)*βI*θ*((1-ω/θ)/γHr + (ω/θ)/γHd))
    βHr =CH*βI #lambda t: CH(t)*βI
    βHd =CH*βI # lambda t: CH(t)*βI      
    return {
        'βE': βE, 
        'βIu': βIu,
        'βI': βI,
        'βHr': βHr,
        'βHd': βHd,
        'γE': γE,
        'γI': γI,
        'γIu': γIu,
        'γHr': γHr,
        'γHd': γHd,
        'μm': 0.0,
        'μn': 0.0,
        'θ': θ,
        'ω': ω,
        '𝛕1': 0.0, 
        '𝛕2':0.0
        
    }
params=ivorra_x_params_constante(*ivorra29M)

class Epidemic_Ivorra:
    def __init__(self, t0, tmax, y0, p, indicador, metodo, umbral, dt=np.timedelta64(1,'D')):
        self.col_ids = {'S': 0, 'E': 1, 'I': 2, 'Iu': 3, 'Hr': 4, 'Hd': 5, 'Ru': 6,'Rd': 7, 'D': 8,
                        'r_effective': 9, 'alpha': 10, 'indicator': 11}

        # número de días para ver atrás para calcular el promedio del indicador
        Δ = 14

        # mínimo número de días en cuarentena / desconfinado
        δ = 14

        alpha_normal = 1.0
        alpha_cuarentena =  0.25

        T = np.arange(t0, tmax, dt)
        # S, E, I, Iu, Hr, Hd, Ru, Rd, D
        B = np.array([
            [0, -p['βE'], -p['βI'], -p['βIu'], -p['βHr'], -p['βHd'], 0, 0, 0],
            [0, p['βE'], p['βI'], p['βI'], p['βHr'], p['βHd'], 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        # S, E, I, Iu, Hr, Hd, Ru, Rd, D
            
        M = np.array([
            [-p["μm"]+p["μn"], p["μn"] , p["μn"], p["μn"], p["μn"], p["μn"], p["μn"], p["μn"],0],
            [0, -p['γE']-p["μm"], 0, 0, 0, 0, 0, 0, 0],
            [0, p['γE'], -p['γI']-p['μm'], 0, 0, 0, 0, 0, 0],
            [0, 0, (1-p['θ'])*p['γI'],-p['μm']-p['γIu'], 0, 0, 0, 0, 0],
            [0, 0, p['θ']*(1-p['ω']/p['θ'])*p['γI'], 0, -p['γHr'], 0, 0, 0, 0],
            [0, 0, p['ω']*p['γI'], 0, 0, -p['γHd'], 0, 0, 0],
            [0, 0, 0, p['γIu'], 0, 0, -p['μm'], 0, 0],
            [0, 0, 0, 0, p['γHr'], 0, 0, -p['μm'], 0],
            [0, 0, 0, 0, 0, p['γHd'], 0, 0, 0]
        ])
        #μb = p['μb']

        # see O. Diekmann, J.A.P. Heesterbeek, and M.G. Roberts, "The construction of next-generation matrices for compartmental epidemic models", J R Soc Interface, 2010
        # K = -B * M^-1  =>  (M.T) * (K.T) = -B.T
        #print(M.T)
        #print(B.T)
        K = np.linalg.solve(M[1:6,1:6].T, -B[1:6,1:6].T).T
        if np.linalg.det(K) != 0.0:
            logging.warn('det(K) = %g != 0' % np.linalg.det(K))
        R = K.trace()

        f = dt/np.timedelta64(1,'D')
        Δ_orig = Δ
        Δ = int(Δ/f + 0.5)
        δ = int(δ/f + 0.5)
        
        days = 0
        cuarentena = True
        last_transition = -δ
        last_deconfinement = len(T)

        data = np.empty((len(T),11))
        data[:] = np.NaN
        data[0,:9] = np.array([y0['S'], y0['E'], y0['I'], y0['Iu'], y0['Hr'], y0['Hd'], y0['Ru'], y0['Rd'], y0['D']])
        N0 = np.sum(data[0,:9])
        #print("N0=",N0)
        data[0,9] = R * data[0,0]/N0
        data[0,10] = alpha_cuarentena if cuarentena else alpha_normal
        for i in range(1, len(T)):
            if Δ < i:
                if indicador == '1':
                    I = data[i-Δ-1:i,2]#I* 100000/N0
                elif indicador == '2': #Hr+Hd
                    I = ( data[i-Δ-1:i,4]+data[i-Δ-1:i,5]) #* 100000/N0
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
                    elif cuarentena and val < umbral:
                        cuarentena = False
                        last_transition = i
                        last_deconfinement = i

            state = data[i-1,:9]

            if cuarentena:
                if last_transition+δ <= i:
                    alpha =alpha_cuarentena
                else:
                    alpha = alpha_normal + (alpha_cuarentena-alpha_normal)*(i-last_transition)/δ
            else:
                if last_transition+δ <= i:
                    alpha =alpha_normal
                else:
                    alpha = alpha_cuarentena + (alpha_normal-alpha_cuarentena)*(i-last_transition)/δ
                    
            data[i,:9] = state + f*(np.dot(M, state) + alpha*np.dot(B, state) * state[0]/np.sum(state[:9]))
            data[i,9] = alpha * R * data[i,0]/np.sum(data[i,:9])
            data[i,10] = alpha
            days += f*cuarentena

        self.T = T
        self.data = data
        self.days = days
        self.last_deconfinement = int(f*last_deconfinement + 0.5)

    def max(self, col):
        if col=="Hc":
            return np.max(self.data[:,self.col_ids['Hr']]+self.data[:,self.col_ids['Hd']])
        else:
            return np.max(self.data[:,self.col_ids[col]])

    def plot(self, cols=['I', 'Hr', 'Hd', 'D'], title=None, filename=None):
        col_names = {'r_effective': '$R_e$', 'alpha': 'Alpha', 'indicator': 'Indicator'}
        col_styles = {
            'S': '-.', 'E': '-.', 'I': '-.', 'Iu': '-.',
            'Hr': '-.', 'Hd': '-.', 'Ru': '-.','Rd': '-.', 'D': '-.'
        }
        col_colors = {
            'S': 'tab:gray', 'E': 'tab:cyan', 'I': 'tab:purple', 'Iu': 'tab:blue',
            'Hr': 'tab:olive', 'Hd': 'tab:orange', 'Ru': 'tab:green','Rd': 'tab:gray', 'D': 'tab:red',
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

        ax[1,0].plot(self.T, self.data[:,9], color='black', label='$R_e$')
        ax[1,0].plot(self.T, self.data[:,10], color='tab:blue', label='Alpha')
        ax[2,0].plot(self.T, self.data[:,11], color='black', label='Indicator')
        
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


umbrales = np.linspace(args.umbral[0], args.umbral[1], args.n)
dias = np.empty((len(umbrales),))
max_infectados = np.empty((len(umbrales),))
max_ucis = np.empty((len(umbrales),))
max_muertos = np.empty((len(umbrales),))
last_deconfinement = 0

for i in range(len(umbrales)):
    if i%10 == 0:
        print('iter: %4d/%d' % (i,len(umbrales)))

    epidemic = Epidemic_Ivorra(t0, tmax, y0, params, args.indicador, args.metodo, umbrales[i], dt=dt)

    dias[i] = epidemic.days
    max_infectados[i] = epidemic.max('I')
    max_ucis[i] = epidemic.max('Hc')
    max_muertos[i] = epidemic.max('D')
    last_deconfinement = max(last_deconfinement, epidemic.last_deconfinement)


with open('out/data_%s%s_%d.npy' % (args.indicador, args.metodo, args.n), 'wb') as f:
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
fig.savefig('figs/fig_indicador_%s%s_umbrales_%g-%g_n_%d.png' % (args.indicador, args.metodo, args.umbral[0], args.umbral[1], args.n))
#plt.show()

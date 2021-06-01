import sys
import argparse
import os
import hashlib
import pickle
import re
import math
import shutil
import datetime

import numpy as np
import scipy as sp
import pandas as pd
import pystan

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({'figure.autolayout': True})

sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=100, help='Number of iterations per chain (default=100)')
parser.add_argument('--warmup', type=int, help='Number of warmup iterations (default=iters/2)')
parser.add_argument('--chains', type=int, default=4, help='Number of chains, should equal number of cores (default=4)')
parser.add_argument('--fit', action="store_true", help='Force running a new fit (otherwise it is asked)')
parser.add_argument('--figs', action="store_true", help='Produce figures')
args = parser.parse_args()

if args.warmup is None:
    args.warmup = math.ceil(args.iters / 2)
if args.figs and not os.path.exists('figs'):
    os.makedirs('figs')

print('Iters:', args.iters)
print('Warmup:', args.warmup)
print('Chains:', args.chains)
print()

model = """
functions {
    real[] dz_dt(real t, real[] z, real[] theta, real[] x_r, int[] x_i) {
        real betaE = theta[1];
        real betaIm = theta[2];
        real betaI = theta[3];
        real gammaE = theta[4];
        real gammaIm = theta[5];
        real gammaI = theta[6];
        real gammaH = theta[7];
        real gammaHc = theta[8];
        real phiEI = theta[9];
        real phiIR = theta[10];
        real phiHR = theta[11];
        real phiHD = theta[12];
        real phiHcD = theta[13];
        real mub = theta[14];
        real mud = theta[15];

        real S  = z[1];
        real E  = z[2];
        real Im = z[3];
        real I  = z[4];
        real H  = z[5];
        real Hc = z[6];
        real R  = z[7];
        real D  = z[8];

        real N = S+E+Im+I+R+H+Hc;
        real ROC = (betaE*E + betaIm*Im + betaI*I)/N;

        return {
            /*  1 S  */  mub*N - S*ROC - mud*S,
            /*  2 E  */  S*ROC - gammaE*E - mud*E,
            /*  3 Im */  (1-phiEI)*gammaE*E - gammaIm*Im - mud*Im,
            /*  4 I  */  phiEI*gammaE*E - gammaI*I - mud*I,
            /*  5 H  */  (1-phiIR)*gammaI*I - gammaH*H - mud*H,
            /*  6 Hc */  (1-phiHR-phiHD)*gammaH*H - gammaHc*Hc - mud*Hc,
            /*  7 R  */  gammaIm*Im + phiIR*gammaI*I + phiHR*gammaH*H + (1-phiHcD)*gammaHc*Hc - mud*R,
            /*  8 D  */  phiHD*gammaH*H + phiHcD*gammaHc*Hc,
            /*  9 iI */  phiEI*gammaE*E
        };
    }
}
data {
    int<lower=3> N;
    real T[N];
    
    int<lower=0> N_pred;
    real T_pred[N_pred];
    
    // constant parameters
    real<lower=0> N0;
    real<lower=0> mub;
    real<lower=0> mud;
    real<lower=0.01,upper=0.99> phiEI;

    // data
    real<lower=0> yR0;
    real<lower=0> yHc[N];
    real<lower=0> yD[N];
    real<lower=0> yiI[N];

    //int<lower=0> N_Sobs;
    //int<lower=0> N_Iobs;
    //real<lower=0> y_Sobs[N_Sobs];
    //real<lower=0> y_Iobs[N_Iobs];
    //int<lower=1> ii_Sobs[N_Sobs];
    //int<lower=1> ii_Smis[N-N_Sobs];
    //int<lower=1> ii_Iobs[N_Iobs];
    //int<lower=1> ii_Imis[N-N_Iobs];
}
transformed data {
    real x_r[0];
    int x_i[0];
    real rel_tol = 1e-6;
    real abs_tol = 1e-6;
    real max_num_steps = 1e4;
}
parameters {
    real<lower=0,upper=N0> R0;
    real<lower=0,upper=N0-R0> E0;
    real<lower=0,upper=N0-R0-E0> Im0;
    real<lower=0,upper=N0-R0-E0-Im0> I0;
    real<lower=0,upper=N0-R0-E0-Im0-I0> H0;
    real<lower=0,upper=N0-R0-E0-Im0-I0-H0> Hc0;
    real<lower=0,upper=N0-R0-E0-Im0-I0-H0-Hc0> D0;
    real<lower=0,upper=N0> iI0;

    real<lower=0.01,upper=1.0> betaI;
    real<lower=1/15.0,upper=1/2.0> gammaE;
    real<lower=1/30.0,upper=1/5.0> gammaIm;
    real<lower=1/30.0,upper=1/5.0> gammaI;
    real<lower=1/30.0,upper=1/5.0> gammaH;
    real<lower=1/30.0,upper=1/5.0> gammaHc;
    real<lower=0.50,upper=0.99> phiIR; // 0.60, 0.90
    real<lower=0.50,upper=0.99> phiHR; // 0.60, 0.90
    real<lower=0.01,upper=fmin(0.50,1-phiHR)> phiHD; // 0.05, 0.15
    real<lower=0.01,upper=0.50> phiHcD; // 0.10, 0.40

    real<lower=0,upper=1> sigma;

    //real<lower=0> y_Smis[N-N_Sobs];
    //real<lower=0> y_Imis[N-N_Iobs];
    //real<lower=0> y_Rmis[N];
    //real<lower=0> S[N];
    //real<lower=0> R[N];
}
transformed parameters {
    real betaE = 0.2*betaI;
    real betaIm = 0.2*betaI;
    real theta[15] = {betaE, betaIm, betaI, gammaE, gammaIm, gammaI, gammaH, gammaHc, phiEI, phiIR, phiHR, phiHD, phiHcD, mub, mud};

    real S0 = N0-E0-Im0-I0-H0-Hc0-R0-D0;
    real z[N,9];
    z[1] = {S0, E0, Im0, I0, H0, Hc0, R0, D0, iI0};
    z[2:,] = integrate_ode_rk45(dz_dt, z[1], 0, T[2:], theta, x_r, x_i, rel_tol, abs_tol, max_num_steps);

    //real y_obs[N,3];
    //y_obs[:,1] = S;
    //y_obs[:,2] = y[:,2];
    //y_obs[:,3] = R;
    //y[ii_Sobs,1] = y_Sobs;
    //y[ii_Smis,1] = y_Smis;
    //y[ii_Iobs,2] = y_Iobs;
    //y[ii_Imis,2] = y_Imis;
    //y[,3] = y_Rmis;
}
model {
    E0/N0 ~ beta(1,4);
    Im0/N0 ~ beta(1,4);
    I0/N0 ~ beta(1,4);
    H0/N0 ~ beta(1,4);
    Hc0 ~ lognormal(log(yHc[1]), sigma);
    R0 ~ lognormal(log(yR0), 0.50); // 0.25
    D0 ~ lognormal(log(yD[1]), sigma);
    iI0 ~ lognormal(log(yiI[1]), sigma);

    betaI ~ normal(0.08, 0.50); // 0.08, 0.10
    1/gammaE ~ normal(5.2, 15.0); // ?, 3.0
    1/gammaIm ~ normal(14.0, 15.0);
    1/gammaI ~ normal(14.0, 15.0);
    1/gammaH ~ normal(12.5, 15.0);
    1/gammaHc ~ normal(10.0, 15.0);
    phiIR ~ normal(0.75, 0.50); // 0.8, 0.1
    phiHR ~ normal(0.75, 0.50); // 0.8, 0.1
    phiHD ~ normal(0.08, 0.50); // 0.08, 0.05
    phiHcD ~ normal(0.15, 0.50); // 0.15, 0.10

    sigma ~ cauchy(0.0, 1.0); // 0, 0.2
    yHc ~ lognormal(log(z[,6]), sigma);
    yD  ~ lognormal(log(z[,8]), sigma);
    yiI ~ lognormal(log(z[,9]), sigma);

    // second derivative of E, Im, and I at the start should be close to zero
    z[3,2] - 2*z[2,2] + z[1,2] ~ cauchy(0, 1);
    z[3,3] - 2*z[2,3] + z[1,3] ~ cauchy(0, 1);
    z[3,4] - 2*z[2,4] + z[1,4] ~ cauchy(0, 1);
}
generated quantities {
    real y_pred[N_pred,9];
    y_pred[1,] = {S0, E0, Im0, I0, H0, Hc0, R0, D0, iI0};
    y_pred[2:,] = integrate_ode_rk45(dz_dt, y_pred[1], 0, T_pred[2:], theta, x_r, x_i, rel_tol, abs_tol, max_num_steps);
}
"""

if not os.path.exists('build'):
    os.makedirs('build')
model_filename = 'build/' + hashlib.sha256(model.encode('utf-8')).hexdigest() + '.stan'
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        sm = pickle.load(f)
else:
    sm = pystan.StanModel(model_name='model', model_code=model)
    with open(model_filename, 'wb') as f:
        pickle.dump(sm, f)

# load data
t0 = '2020-06-20'
data = pd.read_csv('metropolitana.csv', index_col=0, sep=';')
data.index = pd.to_datetime(data.index, dayfirst=True)
data = data['2020-06-20':]

# parameters for MCMC
N0 = 5.624e6
T = np.arange(0, len(data))
T_pred = np.arange(0, len(T)+14)

y = np.zeros((len(T),9))
y[:,5] = data['Pacientes UCI']
y[:,7] = data['Fallecidos']
y[:,8] = data['Casos totales']

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
names = ['S(t)', 'E(t)', 'I_m(t)', 'I(t)', 'H(t)', 'H_c(t)', 'R(t)', 'D(t)', 'I_{cases}(t)']
if args.figs:
    print('Plotting data')
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20,20), squeeze=False)
    ks = [5,7,8]
    for i in range(3):
        ax = axs[i,0]
        #ax.plot(t_true, y_true[:,k], color=colors[k], alpha=0.5)
        k = ks[i]
        ax.plot(T, y[:,k], color=colors[k], marker='o', linestyle='none', ms=3, label='$%s$'%names[k])
        ax.legend()
    plt.savefig('figs/data.png', bbox_inches='tight')
    plt.show()
    plt.close()

#ii_Sobs = np.where(~np.isnan(y[:,0]))[0]
#ii_Smis = np.where(np.isnan(y[:,0]))[0]
#ii_Iobs = np.where(~np.isnan(y[:,1]))[0]
#ii_Imis = np.where(np.isnan(y[:,1]))[0]
#N_Dobs = np.sum(~np.isnan(y[:,7]))
data = {
    'N': len(T),
    'T': T,
    'N_pred': len(T_pred),
    'T_pred': T_pred,
    'N0': N0,
    'mub': mub,
    'mud': mud,
    'phiEI': phiEI,
    'yR0': 0.75*y[0,8]/phiEI,
    'yHc': y[:,5],
    'yD': y[:,7],
    'yiI': y[:,8],
    #'N_Dobs': N_Dobs,
    #'N_Sobs': len(ii_Sobs),
    #'N_Iobs': len(ii_Iobs),
    #'y_Sobs': y[ii_Sobs,0],
    #'y_Iobs': y[ii_Iobs,1],
    #'ii_Sobs': ii_Sobs+1,
    #'ii_Smis': ii_Smis+1,
    #'ii_Iobs': ii_Iobs+1,
    #'ii_Imis': ii_Imis+1,
}

run_sampling = True
fit_filename = 'build/' + hashlib.sha256(model.encode('utf-8')).hexdigest() + '_' + datetime.now().strftime('%Y%m%d') + '.stan'
if os.path.exists(fit_filename) and not args.fit:
    do_fit = input("Run a new fit? (y/N):")
    run_sampling = do_fit == 'y' or do_fit == 'Y'

if run_sampling:
    fit = sm.sampling(data=data, iter=args.iters, warmup=args.warmup, chains=args.chains, check_hmc_diagnostics=False, control={'max_treedepth': 15})
    with open(fit_filename, 'wb') as f:
        pickle.dump(fit, f)
else:
    print('Loading samples')
    with open(fit_filename, 'rb') as f:
        fit = pickle.load(f)

pystan.check_hmc_diagnostics(fit, verbose=True)

summary = fit.stansummary()
z_last = re.findall("z\[%d,.*" % (len(T),), summary)
summary = re.sub("(y_pred|z|theta)\[.*\n?", "", summary)
print(summary)

print()
for z in z_last:
    print(z)
print()

def plot_ode(t_obs, obs, t_pred, param):
    pred = np.quantile(param, [0.025, 0.5, 0.975], axis=0)
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(20,20), squeeze=False)
    fig.suptitle('MCMC fit for Región Metropolitana (2020-06-20 – 2020-09-07)')
    for k in range(9):
        ax = axs[int(k/3),k%3]
        if k in [5,7,8]:
            ax.plot(t_obs, obs[:,k], color=colors[k], marker='o', linestyle='none', ms=3)
        ax.plot(t_pred, pred[0,:,k], color=colors[k], ls='--')
        ax.plot(t_pred, pred[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
        ax.plot(t_pred, pred[2,:,k], color=colors[k], ls='--')
        ax.legend()

def plot_trace(param, name='parameter'):
  """Plot the trace and posterior of a parameter."""
  
  # Summary statistics
  mean = np.mean(param)
  median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
  
  # Plotting
  fig, axs = plt.subplots(2, 1, figsize=(10,10))
  axs[0].plot(param)
  axs[0].set_xlabel('samples')
  axs[0].set_ylabel(name)
  axs[0].axhline(mean, color='r', lw=2, linestyle='--')
  axs[0].axhline(median, color='c', lw=2, linestyle='--')
  axs[0].axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  axs[0].axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  axs[0].set_title('Trace and Posterior Distribution for {}'.format(name))

  axs[1].hist(param, 30, density=True)
  sns.kdeplot(param, shade=True, ax=axs[1])
  axs[1].set_xlabel(name)
  axs[1].set_ylabel('density')
  axs[1].axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  axs[1].axvline(median, color='c', lw=2, linestyle='--',label='median')
  axs[1].axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  axs[1].axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  
  fig.tight_layout()
  axs[1].legend()

def plot_parameters(fit, params):
    fig, axs = plt.subplots(4, 4, figsize=(25,25))
    for j in range(len(params)):
        name = params[j]
        param = fit[name]
        mean = np.mean(param)
        median = np.median(param)
        cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

        ax = axs[int(j/4),j%4]
        ax.hist(param, 30, density=True)
        sns.kdeplot(param, shade=True, ax=ax)
        ax.set_title('{}'.format(name), fontsize=18)
        ax.set_xlabel(name)
        ax.set_ylabel('density')
        ax.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
        ax.axvline(median, color='c', lw=2, linestyle='--',label='median')
        ax.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
        ax.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
        ax.legend()

if args.figs:
    params = ['betaE', 'betaIm', 'betaI', 'gammaE', 'gammaIm', 'gammaI', 'gammaH', 'gammaHc', 'phiIR', 'phiHR', 'phiHD', 'phiHcD', 'sigma', 'lp__']

    plot_parameters(fit, params)
    plt.savefig('figs/parametros.png', bbox_inches='tight')
    plt.close()

    for param in params:
        print('Plotting %s trace'%param)
        plot_trace(fit[param], param)
        plt.savefig('figs/params_%s.png'%param, bbox_inches='tight')
        plt.close()

    print('Plotting prediction')
    y_pred = fit.extract(['y_pred'])['y_pred']
    plot_ode(T, y, T_pred, y_pred)
    plt.savefig('figs/pred.png', bbox_inches='tight')
    plt.show()

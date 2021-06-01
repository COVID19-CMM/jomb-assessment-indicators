# Assessment of event-triggered policies of nonpharmaceutical interventions based on epidemiological indicators

Nonpharmaceutical interventions (NPI) such as banning public events or instituting lockdowns have been widely applied around the world to control the current COVID-19 pandemic. Typically, this type of intervention is imposed when an epidemiological indicator in a given population exceeds a certain threshold. Then, the nonpharmaceutical intervention is lifted when the levels of the indicator used have decreased sufficiently. What is the best indicator to use? In this paper, we propose a mathematical framework to try to answer this question. More specifically, the proposed framework permits to assess and compare different event-triggered controls based on epidemiological indicators. Our methodology consists of considering some outcomes that are consequences of the nonpharmaceutical interventions that a decision maker aims to make as low as possible. The peak demand for intensive care units (ICU) and the total number of days in lockdown are examples of such outcomes. If an epidemiological indicator is used to trigger the interventions, there is naturally a trade-off between the outcomes that can be seen as a curve parameterized by the trigger threshold to be used. The computation of these curves for a group of indicators then allows the selection of the best indicator the curve of which dominates the curves of the other indicators. This methodology is illustrated using indicators in the context of COVID-19 using deterministic compartmental models in discrete-time, although the framework can be adapted for a larger class of models.

## Usage
Make sure to have the following dependencies installed:

- [Python 3.8](https://www.python.org/downloads/) or newer
- [Anaconda](https://docs.anaconda.com/anaconda/install/)
- [Git](https://git-scm.com/downloads)

Then execute get the Git repository:
```bash
$ git clone https://github.com/COVID19-CMM/jomb-assessment-indicators
$ cd jomb-assessment-indicators
```

Create a virtual environment and install the dependencies:

```bash
$ conda env create -f environment.yml
$ conda activate corona
```

### Estimate parameters
The `estimate.py` file contains the MCMC paramter estimator for our model using Stan. The model gets compiled to `build/` and the figures are saved to `figs/`. You can execute parameter estimation by using for example:

    python estimate.py --iters 200 --chains 4 --figs

This is using 200 iterations for every chain (very little), with four chains running in parallel concurrently. It is recommended to set the number of chains to the number of CPUs. The model uses `metropolitana.csv` to train, which contains the number of ICU patients, total cases, and deceased of the metropolitan area of Santiago, Chile.

The parameter estimations are printed and plotted in `figs/`.

### Simulate for different indicators
The parameters estimated using MCMC are put into `simulate.py`, which runs a simulation of the pandemic. The conditions for lockdown are based on an indicator (1 or 2) and a method of measuring it (a, b, c, or d). See our paper for an explanation of each indicator and method.

The simulation data is saved to `simulations/` and the figures to `figs/`. To run a simulation with the indicator in the range of 0 to 1200, with steps of a 120 (n=11), using indicator 1 and method A, run:

    python simulate.py --indicador 1 --metodo a --umbral 0 1200 -n 11

To plot the figure as used in the paper that compares the different indicators, run:

    python plot.py


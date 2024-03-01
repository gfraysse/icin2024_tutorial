![Orange logo](assets/logo_orange.jpg) ![ICIN logo](assets/logo_icin.png)

Implementation of a Reinforcement Learning pipeline for autoscaling of network functions.

This code is made available for the "Setting up a Reinforcement Learning pipeline for a Telco Core Network" tutorial taking place at the ICIN 2024 conference.
This code will eventually move to an official Orange-backed public GitHub repository, when this is the case this repo will be updated and include a link to the new location.

The work described in this tutorial and the code provided here were used in the following publications:
* J. Singh, S. Verma, Y. Matsuo, F. Fossati and G. Fraysse, [[Autoscaling Packet Core Network Functions with Deep Reinforcement Learning]](https://ieeexplore.ieee.org/document/10154301) NOMS 2023-2023 IEEE/IFIP Network Operations and Management Symposium, Miami, FL, USA, 2023, pp. 1-6, doi: 10.1109/NOMS56928.2023.10154312 
* Y. Matsuo, J. Singh, S. Verma and G. Fraysse, [[Integrating state prediction into the Deep Reinforcement Learning for the Autoscaling of Core Network Functions]](https://ieeexplore.ieee.org/document/10154312) NOMS 2023-2023 IEEE/IFIP Network Operations and Management Symposium, Miami, FL, USA, 2023, pp. 1-5, doi: 10.1109/NOMS56928.2023.10154301

# Installation 

First clone this repository.

Then, there are multiple ways to have a working environment to run this code, here we describe how to do it using pip under Linux, macOS or WSL under Windows.

This has been tested with Python 3.11+. There is no guarantee it works with previous versions.

Create python virtualenv 
```bash
$ python3 -m venv {env_name}
$ source {env_name}/bin/activate
```

Install all dependecies, the full virtual env requires around 5.5GB for the binary packages but using pip to install the packages require to have around 10GB available for their compilation
```bash
$ pip install -r requirements.txt
```

# Running an experiment
```bash
$ python pipeline-exp_d3qn_per_sim.py 
```

The first time you run it the [Weights & Biases](https://wandb.ai) library will ask you to either create an account or use your existing account. If you already have an account you just need to input your API Key (under User Setting on the site). If you don't simply follow the steps.

# Analyzing the results of an experiment
The code output metrics to two tools.

You can see the results locally using Tensorboard:
```bash
$ tensorboard serve --logdir backups/tensorboard_runs/d3qn_sine_workload_pre_mainNetwork_2024-03-01-11-09/
```

or on the web based [Weights & Biases](https://wandb.ai), go to the https://wandb.ai/home site and login with your account.
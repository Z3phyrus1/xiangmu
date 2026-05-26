# Dataset & analysis code

This repository contains the data and code supporting the article _"Discrete confidence level revealed by sequential decisions"_, by Matteo Lisi, Gianluigi Mongillo, Georgia Milne, Tessa Dekker and Andrei Gorea. The article is published in the journal Nature Human Behaviour (DOI:10.1038/s41562-020-00953-1 , URL: https://www.nature.com/articles/s41562-020-00953-1).

------
The data (and any intermediate product of the analyses) is in the 'data' folder. In particular, the file `./data/data_src.txt` provides the 'raw' dataset.  

The data of the additional experiment (with judgments about temporal duration) that is reported in the supplementary material is `./data/duration_src.txt`.

The 'R' folder contains R code used in the analyses script (e.g. functions that calculate model's likelihood and do the optimization of parameters, see `./R/code_all_models.R`). 

The operations done by each script should be clear from each script's name. The most important are probably the following:
- `estimate_noise.R`, estimate participant sensitivities and transform stimuli in units of internal noise  
- `model_free.R`, perform model-agnostic analyses (see Supplemental Information for details)  
- `model_comparison.R`, perform statistical comparison between fitted models  
- `compute_predictions.R`, calculate trial-by-trial predictions for all models, and save them for plots  
- `model_recovery_analysis.R`, simulate data and check that the analysis method correctly retrieve the generative model  
- `simulate_sampling_model.R`, this script does the simulation of the Bayesian sampler model described in the Supplemental Information  

All scripts that begin with `plot_[...]` generate the plots presented in the paper (see the heading of each script for info).

Note that before running the analyses make sure you have the files `./R/tabplus.rds`, `./R/tabminus.rds`, `./R/tabindex.rds` (if not you should run the script `make_lookup_table.R`). The likelihood function of the Bayesian model is not available in a closed-form expression, so we need to use numerical integration. These files implement a lookup table that is then used to save some time.

ALl R packages used in the analyses are available on CRAN, with 2 exceptions:  
- `mlisi`, which is a package with some convenient helper functions (e.g. for BCa bootstrap confidence intervals) that is available at [github.com/mattelisi/mlisi](https://github.com/mattelisi/mlisi);  
- `bmsR`, a package that implement Bayesian group-level model comparisons (a direct R translation of the code in SPM 12), available here: [github.com/mattelisi/bmsR](https://github.com/mattelisi/bmsR).  

------

_The senior author on this study, Andrei Gorea, died from stomach cancer on the 7th of May of 2019. He will be missed. If you wish to know more about Andrei, please visit [his website](http://andrei.gorea.free.fr/) which, in addition to his scientific work contains a selection of his poetry and writings, or the website of the workshop that took place in his honor in January 2020 in Paris ([link](https://sites.google.com/view/andrei-gorea-tribute/)). See also the obituary on the journal Perception [here](https://journals.sagepub.com/doi/abs/10.1177/0301006619835409)._

------

Contact: m.lisi [at] essex.ac.uk

------



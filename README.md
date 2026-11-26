# TensorRegressionOptimism (Biometrika)
**Content**
This is the code repository for the paper "Asymptotic optimism for tensor regression models" by Haoming Shi, Eric C. Chi, and Hengrui Luo.

 - We provide the code for the numerical studies and the real data study in the paper (Sections 5 and 6, and Sections S5 and S6 of the Supplementary Material), together with the simulation results and the scripts that reproduce every figure and table.
 - Low-rank CP and Tucker regression models are fitted with `tensorly` ([Kossaifi et al. (2019)](https://jmlr.org/papers/v20/18-277.html)), followed by a Newton polish on an identified chart, at which the plug-in optimism estimators are computed.
 - Reproducible Code
	 - in TensorRegression, it includes the simulation codes for CP (`cp_regression.py`) and Tucker (`tucker_regression.py`) regression with 10 x 8 x 12 Gaussian covariates, under the autoregressive covariance Σ_ab = 0.5^|a-b| on vec(X) (`--design ar`) and the identity covariance (`--design identity`). We provide the simulation results (`results/`) and the codes for reproducing Figures 1 and 2 in the main manuscript and Figures S1-S6 in the supplement (`plot_optimism.py`).
	 - in ShapeRecovery, it includes the simulation codes for CP and Tucker regression on the 64 x 64 x 3 colour versions of the six shape signals of [Zhou, Li and Zhu (2013)](https://doi.org/10.1080/01621459.2013.776499) (`shapes_regression.py`, signals in `data/zhou_shapes_64.npz`). We provide the simulation results and the codes for reproducing Figure 3 in the main manuscript, Figure S7 (`plot_shape_estimates.py`) and Tables S1 and S2 (`make_tables.py`) in the supplement.
	 - in Real_Data_Exp, it includes the codes for Tucker regression on the age prediction task with the UTKFace dataset ([Zhang, Song and Qi (2017)](https://susanqq.github.io/UTKFace/)) (`prepare_utkface.py`, `utkface_regression.py`). We provide the results and the codes for reproducing Figure 4 in the main manuscript (`plot_utkface_criteria.py`) and Table S3 in the supplement (`make_table.py`).

**Reproducing the figures and tables**
The provided results reproduce all figures and tables directly (run inside each folder):
```
cd TensorRegression
python plot_optimism.py --results results/cp_ar --noise-ratios 0.05   # Figures 1, 2
python plot_optimism.py --results results/cp_ar                       # Figures S1, S2
python plot_optimism.py --results results/tucker_ar                   # Figures S3, S4
python plot_optimism.py --results results/cp_identity                 # Figure S5
python plot_optimism.py --results results/tucker_identity             # Figure S6
cd ../ShapeRecovery
python plot_shape_estimates.py --model tucker                         # Figure 3
python plot_shape_estimates.py --model cp                             # Figure S7
python make_tables.py --model cp                                      # Table S1
python make_tables.py --model tucker                                  # Table S2
cd ../Real_Data_Exp
python plot_utkface_criteria.py                                       # Figure 4
python make_table.py                                                  # Table S3
```
In the AR figures the bars use the columns `est_optimism_score` and `est_test_error_score`, and the optimism selector uses `sel_logn_sigma`.

**Rerunning the experiments**
```
cd TensorRegression
python cp_regression.py --design ar --jobs 48            # -> results/cp_ar
python tucker_regression.py --design ar --jobs 48        # -> results/tucker_ar
python cp_regression.py --design identity --jobs 48      # -> results/cp_identity
python tucker_regression.py --design identity --jobs 48  # -> results/tucker_identity
python summarize.py --output results/cp_ar               # likewise for the other runs
cd ../ShapeRecovery
python shapes_regression.py --model cp --jobs 48         # -> results/cp
python shapes_regression.py --model tucker --jobs 48     # -> results/tucker
python summarize_shapes.py --output results/cp           # likewise for tucker
cd ../Real_Data_Exp
python prepare_utkface.py --images <UTKFace image folder>
python utkface_regression.py --jobs 48                   # -> results/tucker
python summarize_utkface.py --output results/tucker
```
The default arguments are the settings used in the paper; `settings.json` in each results folder records them. Each task writes its own file under `raw/`, and finished tasks are skipped, so an interrupted run can be resumed. The full runs take several hours on a 48-core node. The UTKFace runs need about 2.5 GB of memory per worker.

The UTKFace images are for non-commercial research use only and are not redistributed here. Download the aligned and cropped faces (the archive used for the paper came from the Hugging Face mirror `py97/UTKFace-Cropped`), then run `prepare_utkface.py`, which draws the fixed 10,000-image subsample used in the paper. The ShapeRecovery results include the fitted coefficients only for the replications shown in Figure 3 and Figure S7 (listed in `figure_seeds.json`).

**Requirements**
Python 3.11 with the packages in `requirements.txt` (numpy 1.26.4 and tensorly 0.9.0 were used for all runs).

Thank you again for the interest and please reach out if you have further questions.

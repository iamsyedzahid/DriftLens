# Annotated literature matrix (O1)

> **Status: draft.** Entries [1]–[6] are the proposal's references. [7]–[12] were added as
> standard references. Before submission, each entry must be checked against the original paper
> (author list, venue, pages and DOI).

| # | Reference | Topic | Method / claim | Assumptions | Limitations | How DriftLens uses it |
|---|---|---|---|---|---|---|
| 1 | Gama, Žliobaitė, Bifet, Pechenizkiy, Bouchachia (2014), *A Survey on Concept Drift Adaptation*, ACM Computing Surveys 46(4) | Concept drift | Taxonomy of drift (sudden, gradual, recurring) and of adaptation (blind vs informed, windowing, forgetting) | Data arrive sequentially; labels arrive eventually | Mostly a streaming setting; little on calibration | Motivates sequential windows; rolling retraining is a "blind" adaptation strategy |
| 2 | Guo, Pleiss, Sun, Weinberger (2017), *On Calibration of Modern Neural Networks*, ICML, PMLR 70 | Calibration | Modern networks are over-confident; temperature scaling fixes much of it; uses ECE and reliability diagrams | i.i.d. test data | Neural nets only; ECE depends on binning | ECE, reliability tables, post-hoc calibration on held-out data |
| 3 | scikit-learn developers, *Probability calibration* (User Guide) | Calibration practice | Sigmoid (Platt) vs isotonic calibration; the calibrator must be fitted on data independent of training | Enough calibration data (isotonic needs more) | Documentation, not peer-reviewed | Sigmoid calibrator fitted on a window disjoint from the model-fit rows |
| 4 | Moro, Cortez, Rita (2012), *Bank Marketing* dataset, UCI, doi:10.24432/C5K306 | Data | 41,188 calls, 20 inputs, ordered by date May 2008 – Nov 2010 | Row order = time order | No per-row timestamp; `duration` is known only after the call | Primary case study; order-based proxy for time; `duration` excluded |
| 5 | Moro, Cortez, Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems 62 | Domain / evaluation | Compares models with rolling-window evaluation; finds macro-economic context features important | Labels are available after each window | Focuses on accuracy and lift, not calibration or abstention | Basis for rolling evaluation; explains why macro features drift strongly |
| 6 | Saito, Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot…*, PLOS ONE 10(3) | Metrics | Under class imbalance, PR curves are more informative than ROC | Fixed class prior | PR-AUC baseline equals prevalence, so it cannot be compared across different priors | PR-AUC is the headline metric; the report compares it against block prevalence |
| 7 | Platt (1999), *Probabilistic Outputs for Support Vector Machines…*, Advances in Large Margin Classifiers | Calibration | Fits a logistic sigmoid to classifier scores | Monotone score–probability relationship | The fitted slope can be negative if the score–label relationship reverses on the calibration set | Our calibrator; its negative slopes are a key finding (report §4.2) |
| 8 | Geifman, El-Yaniv (2017), *Selective Classification for Deep Neural Networks*, NeurIPS | Selective prediction | Risk-coverage trade-off; a reject option guarantees risk at a target coverage | Confidence ranks errors well | Guarantees need i.i.d. calibration data | Risk-coverage curves; confidence = max(p, 1−p) |
| 9 | Quiñonero-Candela, Sugiyama, Schwaighofer, Lawrence (eds.) (2009), *Dataset Shift in Machine Learning*, MIT Press | Shift taxonomy | Covariate shift, prior-probability shift, concept shift | — | Mostly theory | Terminology: block 3 shows strong prior shift (positive rate 0.10 → 0.46) |
| 10 | Rabanser, Günnemann, Lipton (2019), *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*, NeurIPS | Drift detection | Compares univariate KS tests (with multiple-testing correction) and multivariate tests | Unlabelled target data | Detecting shift says nothing about its effect on accuracy | Per-feature KS/JS; supports RQ4's caution that drift is not failure |
| 11 | Lu, Liu, Dong, Gu, Gama, Zhang (2018), *Learning under Concept Drift: A Review*, IEEE TKDE | Concept drift | Framework of drift detection, understanding and adaptation | — | Survey | Places our three maintenance strategies within the adaptation literature |
| 12 | Ovadia et al. (2019), *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS | Calibration under shift | Calibration degrades as shift grows; post-hoc calibration on in-distribution data does not transfer | Synthetic and natural shifts | Image/text deep models | Supports measuring calibration per window rather than once |

## Novelty boundary (proposal §3.2)
Each component already exists in the literature: windowed retraining [1, 5, 11], post-hoc calibration
[2, 3, 7], selective prediction [8] and univariate drift tests [10]. The claimed contribution is
therefore **only** a joint, leakage-audited, reproducible comparison of these strategies on one
public tabular dataset, with an open tool. It is not a new method. A full search (e.g. "calibration
under temporal shift tabular", "recalibration vs retraining") is still needed before making any
stronger novelty claim.

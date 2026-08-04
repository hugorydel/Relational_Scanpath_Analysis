# Relational Gaze During Encoding Predicts Episodic Recall of Naturalistic Scenes

Data, stimuli, and analysis code for the manuscript by Hugo Rydel and Alex Kafkas
(Division of Psychology, Communication and Human Neuroscience, University of Manchester).

Participants viewed 30 naturalistic scenes while their eye movements were recorded, then
recalled each scene from memory. Using scene-graph annotations from the Synthetic Visual
Genome dataset, we measured whether gaze moved between objects that were *meaningfully
related*, and tested whether this relational scanning predicted subsequent free recall.

## Quick start

```bash
conda env create -f environment.yml
conda activate relational-scanpath

cd data_analysis
python run_pipeline.py --modules 4    # reproduce all statistics, figures and tables
```

Module 4 reads the committed intermediate outputs, so it runs in under a minute without
re-processing raw eye-tracking data. To rebuild everything from raw files:

```bash
python run_pipeline.py                # all modules, all participants
python run_pipeline.py --modules 3 4  # re-run feature extraction and analysis
python run_pipeline.py --subjects 1 3 # specific participants only
```

## Pipeline

| Module | Purpose |
|---|---|
| 1. Behavioral | Parses E-Prime logs into typed per-participant CSVs (encoding, distractor, retrieval) |
| 2. Eye-tracking | Extracts fixations and saccades from EyeLink `.edf` files |
| 3. Features | Assigns fixations to object AOIs, builds object sequences, computes relational scores |
| 4. Analysis | Fits the models, writes tables and figures |

### How the relational score works

Fixations are assigned to SVG-derived object masks and collapsed into a sequence of
distinct object visits. A transition between two successive objects counts as *relational*
if those objects are linked in that image's scene graph. The observed proportion of
relational transitions is then compared against a trial-specific null distribution built
from 1,000 random sequences of the same length drawn from the objects actually visited on
that trial. The relational score is the z-scored difference between observed and null, so
0 means chance-level scanning.

## Repository layout

```
data_analysis/
  run_pipeline.py            orchestrator
  config.py                  paths and constants
  pipeline/
    module1_behavioral.py    E-Prime log parsing
    module2_eyetracking.py   EDF extraction
    module3_features.py      AOI assignment, sequences, relational scores
    module4_analysis.py      models, tables, figures
    module_3/                AOI, metrics, scene-graph helpers
    module_4/                loader, models, output
    meaning/                 meaning-map computation
    salience/                spectral-residual salience maps
    scoring/                 automated recall scoring and validation
  data_behavioral/           raw E-Prime logs
  data_eyetracking/          raw EyeLink .edf files
  data_metadata/             stimulus images and scene-graph annotations
  output/analysis/           results, figures, figure-level data
  tests/                     robustness and sensitivity checks

stimuli_generation/          stimulus selection from Synthetic Visual Genome
```

## Key outputs

All in `data_analysis/output/analysis/`:

| File | Contents |
|---|---|
| `results_summary.csv` | Headline result for every model |
| `appendix_table2_h2_params.csv` | Full fixed effects, encoding and retrieval models |
| `appendix_table3_exploratory_params.csv` | Exploratory dissociation model |
| `descriptives.csv` | Participant-level descriptive statistics |
| `assumption_checks.csv` | Variance inflation factors |
| `model_summaries.txt` | Full model summaries as text |
| `figures/` | Main figures as PNG, SVG and PDF |
| `figure_data/` | Underlying data for each figure |

## Analysis notes

- **Models.** Linear mixed-effects models estimated with REML and a Nelder-Mead
  optimizer, with crossed random intercepts for participant and image. All continuous
  predictors z-scored. Alpha = .05; t-tests are two-tailed.
- **Effect coding.** In the exploratory dissociation model, `memory_type` is effect-coded
  (objects = −0.5, relations = +0.5) so the relational-score term is the average slope
  across memory types rather than a simple slope for one of them.
- **Retrieval sample size.** A relational score needs at least two successive object
  visits. Blank-screen retrieval produces sparser fixation sequences, so more trials fall
  below that threshold: 700 scored observations at encoding versus 372 at retrieval.
- **Automated recall scoring.** Free-recall responses were scored against image-specific
  codebooks using a large language model, validated against manual scoring
  (ICC(2,1) = .75–.77, Pearson's r = .88; 98.7% agreement across two automated runs,
  Cohen's κ = .96). Scored outputs are committed, so the OpenAI dependency is only needed
  to regenerate them.
- **Figures.** Main figures are written as vector (SVG and PDF) with TrueType fonts
  embedded, alongside 300 dpi PNG.

## Data

Participant data are anonymised: identifiers are sequential numbers with no link to
personal information. Stimulus images come from the Synthetic Visual Genome dataset
(Park et al., 2025) and remain subject to their original licences.

## Ethics

Approved by the University of Manchester Research Ethics Committee (Ref:
2025-24187-44486). All participants gave written informed consent. Procedures followed
the Declaration of Helsinki.

## Contact

Hugo Rydel — hugo.rydel-johnston@manchester.ac.uk

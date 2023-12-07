# ensemble-square
Repository for the ensemble-square project

## Installation

First, clone the repository. This can be done in the terminal using the following command
```
git clone git@github.com:gstonge/ensemble-square.git
```

Second, check the requirements in `requirements.txt`. If some are missing, install them. You can
install all requirements in the terminal using:
```
pip install -r requirements.txt
```
in the `ensemble-square` directory.

Also, make sure the [scorepi](https://github.com/gstonge/scorepi) package is installed (not in the requirement list), as well as [jupyter lab or jupyter notebook](https://jupyter.org/install) (whichever is prefered).
The ```scorepi`` package is also included as a directory, and can be installed directly using
```
pip install ./scorepi
```

## Notebook descriptions

pull_data.ipynb - used to pull model projections from the COVID-19 Scenario Modeling Hub, COVID-19 Forecast Hub, and surveillance data for cases, deaths, and hospitalizations.

performance_code.ipynb - used to compute scenario ensemble, use different scoring methods to analyze scenario projections, and generate figures related to the scores.

single_model_viz.ipynb - generate figures of projections for a given model, including the scenario projections and scenario ensemble. 

ensemble_order.ipynb - analyze the impact of the ordering of ensembling in the scenario ensemble (ensemble over models then scenarios and vice versa). 

scoring data folder - scoring output used for analysis in this project.

<!-- MarkdownTOC -->

- [2019 ADAS Validation Code](#2019-adas-validation-code)
    - [Git It](#git-it)
    - [Usage](#usage)
    - [Clustering](#clustering)
        - [Clustering Visualization](#clustering-visualization)

<!-- /MarkdownTOC -->

# 2019 ADAS Validation Code

This repository supports the paper [Critical Scenario Clusters for Accelerated Automotive Safety Validation](https://www.overleaf.com/5844667qvyvfx#/19448268/) by Tim Wheeler and Mykel Kochenderfer.

## Git It

- install Julia 0.5
- pull this repo
- run `Pkg.add("IJulia")`
- run `Pkg.clone("https://github.com/tawheeler/Records.jl.git")`
- run `Pkg.clone("https://github.com/tawheeler/Vec.jl.git")`
- run `Pkg.clone("https://github.com/tawheeler/AutomotiveDrivingModels.jl.git")`
- run `Pkg.clone("https://github.com/tawheeler/AutoViz.jl.git")`
- run `Pkg.clone("https://github.com/tawheeler/AutoRisk.jl.git")`
- switch AutomotiveDrivingModels.jl to the _records_ branch
- switch AutoRisk.jl to the _mobius_ branch

## Usage

The file `mine_collision_scenarios.jl` will generate 1000 mobius simulations with collisions.

The file `export_collisions_estimations.jl` will produce beta distributions for each frame of each simulation record.

There are example Julia notebooks in the jnotebooks directory.

## Clustering 

The clustering code assumes that both `mine_collision_scenarios.jl` and `export_collisions_estimations.jl` have already been run, which will have generated and placed files in `data/collision_scenarios`. 

Given that initial state, to run clustering you need have python 3.5 installed, and to run from the clustering directory `python run_pipeline.py`. This will do three things

First, it will find the critical frames for each scenario. This is performed for each scenario by starting at the final collision frame and backtracking. At each timestep, it computes the beta distribution (gvien from `export_collisions_estimations.jl`), and then computes the fraction of probability mass above a constant threshold. For example, by default it backtracking until the lower bound of the 95% mass interval falls below 90% likelihood of collision. This is all performed in `find_critical_frames.py`, and the output of this process is stored as a csv file called `critical_frames.txt` in `data/clustering`.

Second, it selects this frame from each scenario along with the preceeding N frames, and computes features. This is performed by a feature extractor in `scenario_feature_extractor.jl` and a script in `feature_extraction.jl`. The output of this process is another csv file `data/clustering/critical_features.txt`.

Third, it clusters these features. Currently it uses some basic clustering methods from sklearn in `clustering.py`. The output of this process is a final csv `data/clustering/classes.txt`, which contains the filename and class for the different scenarios. The filename contains the iteration of `mine_collision_scenarios.jl` that generated that scenario.

### Clustering Visualization
After running `python run_pipeline.py`, you can visualize the different classes / critical scenario clusters by navigating to the `jnotebooks` directory, running `jupyter notebook`, opening up `VisualizeCriticalScenarioClusters.ipynb`, and executing all the cells. This will allow you to look through the scenarios categorized by their class.


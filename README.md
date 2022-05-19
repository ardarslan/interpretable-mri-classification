# Environment

In the root of this repository, run
```sh
conda env create -f environment.yml
```

# Task 1

First, activate the environment:

```sh
conda activate ml4hc_project3
```

then navigate to the `src` folder to run the remaining commands.

To train the XGBoost classifier with the best hyperparameter settings, run:

```sh
python task1.py --retrain
```

To perform the grid search used to find the best hyperparameter settings for the
XGBoost classifier, run:

```bash
python task1.py --retrain --grid_search
```

# Task 2

```bash
bsub -n 3 -W 24:00 -R "rusage[mem=8192]" python task2.py --model_name baseline_cnn
```

# Task 3

```bash
python task3.py
```

Run the following command for the `RuleFit` experiment:
N.B.: We encountered some issues with non-reproducibility in the rulefit package. We have hence provided a trained model so that our results can be reproduced.

```sh
python task3_rules.py
```

# Optional Task

```bash
chmod +x optional_task_commands.sh
./optional_task_commands.sh
```

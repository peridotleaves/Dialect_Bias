## 1. Create Grammatically Transformed Datasets
environment file: multivalue.yml (also used for analysis)

make_transformed_datasets.py
## 2. Run Eval-Harness on Grammatically Transformed Datasets
environment file: eval_harness.yml

Eval-Harness prompts and files can be found at 
https://github.com/peridotleaves/lm-evaluation-harness/tree/ed5ca052abdc6c02cdc0c46aff7611767919dda5 

under lm_eval/tasks/[dataset_name]
## 3. Grammar Perturbation Performance Analysis
All statistical analysis performed in grammar_rule_effects.ipynb. Necessary eval-harness files are expected in default folder path. 

## Other: 
### Perplexity analysis:
perplexity.py 

### Grammar Effects Regression 
regression.py

### Exploratory Grammatical Transformations
process_qa_datasets.ipynb 

import argparse
import json
import pandas as pd
import numpy as np
import re
import os
import glob
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze dialectical biases across multiple LLMs and datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names (e.g., sciq boolq mmlu)')
    parser.add_argument('--llms', type=str, nargs='+', required=True,
                        help='LLM names (e.g., gemma gpt mistral)')
    parser.add_argument('--data-dir', type=str, default='dialect_data',
                        help='Base directory containing all datasets (default: dialect_data)')
    parser.add_argument('--output', '-o', type=str, 
                        help='Path to output file for regression results')
    parser.add_argument('--output-dir', '-d', type=str, default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--feature-map', '-f', type=str, default='dialect_data/feature_id_to_function_name.json',
                        help='Path to feature ID to function name mapping file')
    parser.add_argument('--include-dialect', action='store_true',
                        help='Include dialect indicators in the regression model')
    return parser.parse_args()

def find_files(data_dir, dataset, llm):
    llm_dir_mapping = {
        'gemma': 'google__gemma-2b',
        'gpt': 'gpt-4o-mini',
        'mistral': 'mistralai__Mistral-7B-Instruct-v0.3'
    }
    
    llm_dir = llm_dir_mapping.get(llm, llm)
    
    if dataset == 'mmlu':
        dialect_patterns = [
            f"{data_dir}/{dataset}/dialect_transforms/lm_eval_results/{llm_dir}/samples_{dataset}*_dialect*.jsonl",
            f"{data_dir}/{dataset}/dialect_transform/lm_eval_results/{llm_dir}/samples_{dataset}*_dialect*.jsonl"
        ]
        
        sae_pattern = f"{data_dir}/{dataset}/original_subset/lm_eval_results/{llm_dir}/samples_{dataset}*_orig*.jsonl"
        
        all_dialect_files = []
        for pattern in dialect_patterns:
            matches = glob.glob(pattern)
            all_dialect_files.extend(matches)
        
        if not all_dialect_files:
            raise FileNotFoundError(f"Could not find any dialect files for dataset={dataset}, llm={llm}")
        
        all_sae_files = glob.glob(sae_pattern)
        
        if not all_sae_files:
            raise FileNotFoundError(f"Could not find any SAE files for dataset={dataset}, llm={llm}")
        
        return all_dialect_files, all_sae_files
    
    else:
        dialect_patterns = [
            f"{data_dir}/{dataset}/dialect_transforms/lm_eval_results/{llm_dir}/samples_{dataset}_dialect*.jsonl",
            f"{data_dir}/{dataset}/dialect_transform/lm_eval_results/{llm_dir}/samples_{dataset}_dialect*.jsonl"
        ]
        
        dialect_file = None
        for pattern in dialect_patterns:
            matches = glob.glob(pattern)
            if matches:
                dialect_file = matches[0]
                break
        
        if not dialect_file:
            raise FileNotFoundError(f"Could not find dialect file for dataset={dataset}, llm={llm}")
        
        sae_pattern = f"{data_dir}/{dataset}/original_subset/lm_eval_results/{llm_dir}/samples_{dataset}_orig*.jsonl"
        sae_matches = glob.glob(sae_pattern)
        
        if not sae_matches:
            raise FileNotFoundError(f"Could not find SAE file for dataset={dataset}, llm={llm}")
        
        sae_file = sae_matches[0]
        
        return dialect_file, sae_file

def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")
    return data

def load_feature_id_map(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def extract_grammar_rules(rules_executed_str):
    if not rules_executed_str or rules_executed_str == "{}":
        return []
    
    try:
        if isinstance(rules_executed_str, dict):
            rules_dict = rules_executed_str
        else:
            cleaned_str = rules_executed_str.replace("'", '"')
            rules_dict = json.loads(cleaned_str)
        
        rule_types = []
        for entry in rules_dict.values():
            if isinstance(entry, dict) and 'type' in entry:
                rule_types.append(entry['type'])
        
        return rule_types
    
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        try:
            pattern = r"'type': '([^']*)',"
            matches = re.findall(pattern, rules_executed_str)
            return matches
        except Exception as regex_error:
            return []

def categorize_grammar_rules(rule_types, feature_id_map):
    intervals = {
        'pronouns': (1, 48),
        'noun_phrase': (48, 88),
        'tense+aspect': (88, 121),
        'modal_verbs': (121, 128),
        'verb_morphology': (128, 154),
        'negation': (154, 170),
        'agreement': (170, 185),
        'relativization': (185, 200),
        'complementation': (200, 211),
        'adverb_subordination': (211, 216),
        'adverbs+prepositions': (216, 223),
        'discourse+word_order': (223, 236)
    }
    
    def categorize_by_id(feature_id):
        feature_id = int(feature_id)
        for category, (start, end) in intervals.items():
            if start <= feature_id < end:
                return category
        return 'other'
    
    categories = []
    
    for rule in rule_types:
        feature_id = None
        if feature_id_map:
            for id_str, functions in feature_id_map.items():
                if rule in functions:
                    feature_id = id_str
                    break
        
        if feature_id:
            categories.append(categorize_by_id(feature_id))
        else:
            categories.append('other')
    
    return list(dict.fromkeys(categories))

def extract_question_id(entry):
    doc_id = entry.get('doc_id', '')
    doc_id = str(doc_id)
    
    if 'question_id' in entry:
        return str(entry['question_id'])
    
    dialect_markers = [
        '_AppalachianDialect', '_Appalachian English',
        '_ChicanoDialect', '_Chicano English',
        '_ColloquialSingaporeDialect', '_Colloquial Singapore English', '_Singlish',
        '_IndianDialect', '_Indian English',
        '_SoutheastAmericanEnclaveDialect', '_Southeast American enclave dialects',
        '_UrbanAfricanAmericanVernacularEnglish', '_Urban African American Vernacular English',
        '_StandardAmericanEnglish'
    ]
    
    base_id = doc_id
    for marker in dialect_markers:
        if marker in doc_id:
            base_id = doc_id.replace(marker, '')
            break
    
    if base_id == doc_id and '_' in doc_id:
        parts = doc_id.split('_')
        base_id = '_'.join(parts[:-1])
    
    return base_id

def process_llm_dataset_pair(dialect_files, sae_files, llm_name, dataset_name, feature_id_map):
    
    if not isinstance(dialect_files, list):
        dialect_files = [dialect_files]
    if not isinstance(sae_files, list):
        sae_files = [sae_files]
    
    dialect_mapping = {
        'Appalachian English': 'AppalachianDialect',
        'AppalachianDialect': 'AppalachianDialect',
        'Chicano English': 'ChicanoDialect',
        'ChicanoDialect': 'ChicanoDialect',
        'Colloquial Singapore English (Singlish)': 'ColloquialSingaporeDialect',
        'ColloquialSingaporeDialect': 'ColloquialSingaporeDialect',
        'Indian English': 'IndianDialect',
        'IndianDialect': 'IndianDialect',
        'Southeast American enclave dialects': 'SoutheastAmericanEnclaveDialect',
        'SoutheastAmericanEnclaveDialect': 'SoutheastAmericanEnclaveDialect',
        'Urban African American Vernacular English': 'UrbanAfricanAmericanVernacularEnglish',
        'UrbanAfricanAmericanVernacularEnglish': 'UrbanAfricanAmericanVernacularEnglish'
    }
    
    sae_accuracy = {}
    sae_data_count = 0
    
    for sae_file in sae_files:
        sae_data = load_jsonl_data(sae_file)
        sae_data_count += len(sae_data)
        
        for entry in tqdm(sae_data, desc=f"Processing SAE data from {os.path.basename(sae_file)}"):
            if not isinstance(entry, dict):
                continue
                
            question_id = extract_question_id(entry)
            
            if llm_name == 'gpt':
                acc = float(entry.get('exact_match', 0))
            else:
                acc = float(entry.get('acc', 0))
                
            acc_binary = 1 if acc > 0 else 0
            
            sae_accuracy[question_id] = acc_binary
    
    processed_rows = []
    dialect_data_count = 0
    
    for dialect_file in dialect_files:
        dialect_data = load_jsonl_data(dialect_file)
        dialect_data_count += len(dialect_data)
        
        for entry in tqdm(dialect_data, desc=f"Processing dialect data from {os.path.basename(dialect_file)}"):
            if not isinstance(entry, dict) or 'doc' not in entry:
                continue
                
            doc = entry.get('doc', {})
            dialect = doc.get('rule_transform', 'Unknown')
            
            dialect = dialect_mapping.get(dialect, dialect)
            
            if dialect == 'StandardAmericanEnglish':
                continue
            
            question_id = extract_question_id(entry)
            
            if llm_name == 'gpt':
                acc = float(entry.get('exact_match', 0))
            else:
                acc = float(entry.get('acc', 0))
                
            acc_binary = 1 if acc > 0 else 0
            
            standard_accuracy = sae_accuracy.get(question_id, 0)
            
            rules_executed_str = doc.get('rules_executed', '{}')
            rule_types = extract_grammar_rules(rules_executed_str)
            rule_categories = categorize_grammar_rules(rule_types, feature_id_map)
            
            row = {
                'question_id': question_id,
                'doc_id': entry.get('doc_id', ''),
                'dialect': dialect,
                'llm': llm_name,
                'dataset': dataset_name,
                'accuracy': acc_binary,
                'standard_accuracy': standard_accuracy,
                'rule_count': len(rule_types)
            }
            
            all_categories = ['pronouns', 'noun_phrase', 'tense+aspect', 'modal_verbs', 
                              'verb_morphology', 'negation', 'agreement', 'relativization', 
                              'complementation', 'adverb_subordination', 'adverbs+prepositions', 
                              'discourse+word_order', 'other']
            
            for category in all_categories:
                row[category] = 1 if category in rule_categories else 0
            
            row['transformed_text'] = doc.get('transformed_text', '')
            
            processed_rows.append(row)
    
    return processed_rows

def check_multicollinearity(df):
    grammar_features = ['pronouns', 'noun_phrase', 'tense+aspect', 'modal_verbs', 
                      'verb_morphology', 'negation', 'agreement', 'relativization', 
                      'complementation', 'adverb_subordination', 'adverbs+prepositions', 
                      'discourse+word_order', 'other']
    
    X = df[grammar_features]
    X = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data.sort_values("VIF", ascending=False))
    print("\nInterpretation:")
    print("VIF > 10: Severe multicollinearity")
    print("5 < VIF < 10: Moderate to high multicollinearity")
    print("1 < VIF < 5: Low multicollinearity")
    
    return vif_data

def correlation_analysis(df):
    grammar_features = ['pronouns', 'noun_phrase', 'tense+aspect', 'modal_verbs', 
                      'verb_morphology', 'negation', 'agreement', 'relativization', 
                      'complementation', 'adverb_subordination', 'adverbs+prepositions', 
                      'discourse+word_order', 'other']
    
    corr_matrix = df[grammar_features].corr()
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    print("\nCorrelation Matrix for Grammar Features:")
    print(corr_matrix.round(2))
    
    if high_corr:
        print("\nHigh correlations (|r| > 0.7):")
        for var1, var2, corr in high_corr:
            print(f"{var1} & {var2}: {corr:.2f}")
    else:
        print("\nNo high correlations found among grammar features.")
    
    return corr_matrix

def run_combined_logistic_regression(df, include_dialect=False):
    
    grammar_features = ['pronouns', 'noun_phrase', 'tense+aspect', 'modal_verbs', 
                      'verb_morphology', 'negation', 'agreement', 'relativization', 
                      'complementation', 'adverb_subordination', 'adverbs+prepositions', 
                      'discourse+word_order', 'other']
    
    llm_columns = [col for col in df.columns if col.startswith('llm_')]
    dataset_columns = [col for col in df.columns if col.startswith('dataset_')]
    
    X_columns = grammar_features + llm_columns + dataset_columns + ['standard_accuracy']
    
    if include_dialect:
        df_with_dialect_dummies = pd.get_dummies(df, columns=['dialect'], drop_first=True)
        dialect_columns = [col for col in df_with_dialect_dummies.columns if col.startswith('dialect_')]
        X_columns = X_columns + dialect_columns
        df_for_regression = df_with_dialect_dummies
    else:
        df_for_regression = df
    
    zero_var_features = []
    for col in X_columns:
        if col in df_for_regression.columns and df_for_regression[col].nunique() <= 1:
            zero_var_features.append(col)
    
    if zero_var_features:
        X_columns = [col for col in X_columns if col not in zero_var_features]
    
    X = df_for_regression[X_columns]
    X = sm.add_constant(X)
    y = df_for_regression['accuracy']
    
    try:
        model = Logit(y, X)
        result = model.fit(disp=0, method='bfgs', maxiter=1000)
        
        print("\nLogistic Regression Results:")
        print(result.summary())
        
        params = result.params
        pvalues = result.pvalues
        conf_int = result.conf_int()
        
        results_df = pd.DataFrame({
            'Variable': params.index,
            'Coefficient': params.values,
            'p-value': pvalues.values,
            'Lower CI': conf_int[0].values,
            'Upper CI': conf_int[1].values,
            'Odds Ratio': np.exp(params.values),
            'OR Lower CI': np.exp(conf_int[0].values),
            'OR Upper CI': np.exp(conf_int[1].values)
        })
        
        return result, results_df
    except Exception as e:
        
        try:
            model = Logit(y, X)
            result = model.fit_regularized(method='l1', alpha=0.01, disp=0, maxiter=1000)
            
            print("\nLogistic Regression Results (with L1 regularization):")
            print(result.summary())
            
            params = result.params
            
            results_df = pd.DataFrame({
                'Variable': params.index,
                'Coefficient': params.values,
                'Odds Ratio': np.exp(params.values)
            })
            
            return result, results_df
        except Exception as e2:
            return None, pd.DataFrame()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(args.output_dir, f"regression_results.txt")

    feature_id_map = load_feature_id_map(args.feature_map)
    
    all_rows = []
    
    for dataset in args.datasets:
        for llm in args.llms:
            try:
                dialect_file, sae_file = find_files(args.data_dir, dataset, llm)
                rows = process_llm_dataset_pair(dialect_file, sae_file, llm, dataset, feature_id_map)
                all_rows.extend(rows)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"Skipping LLM: {llm} for dataset: {dataset}")
    
    if not all_rows:
        return
    
    df = pd.DataFrame(all_rows)
    
    dialect_mapping = {
        'Appalachian English': 'AppalachianDialect',
        'Chicano English': 'ChicanoDialect',
        'Colloquial Singapore English (Singlish)': 'ColloquialSingaporeDialect',
        'Indian English': 'IndianDialect',
        'Southeast American enclave dialects': 'SoutheastAmericanEnclaveDialect',
        'Urban African American Vernacular English': 'UrbanAfricanAmericanVernacularEnglish'
    }
    
    df['dialect'] = df['dialect'].replace(dialect_mapping)

    df_with_dummies = pd.get_dummies(df, columns=['llm', 'dataset'], drop_first=True)
    
    vif_data = check_multicollinearity(df_with_dummies)
    corr_matrix = correlation_analysis(df_with_dummies)
    
    vif_csv = os.path.join(args.output_dir, f"vif_analysis.csv")
    corr_csv = os.path.join(args.output_dir, f"correlation_matrix.csv")
    
    vif_data.to_csv(vif_csv, index=False)
    corr_matrix.to_csv(corr_csv)
        
    logit_model, results_df = run_combined_logistic_regression(df_with_dummies, args.include_dialect)
    
    with open(output_file, 'w') as f:
        model_type = "WITH" if args.include_dialect else "WITHOUT"
        f.write(f"=== COMBINED DATASETS ANALYSIS ({model_type} DIALECT INDICATORS) ===\n\n")
        f.write("=== COMBINED DATASET SUMMARY ===\n")
        f.write(f"Total entries: {df.shape[0]}\n")
        f.write(f"Accuracy mean: {df['accuracy'].mean():.4f}\n")
        f.write(f"Standard dialect accuracy mean: {df['standard_accuracy'].mean():.4f}\n\n")
        
        f.write("=== GRAMMAR RULE CATEGORY DISTRIBUTION ===\n")
        category_columns = ['pronouns', 'noun_phrase', 'tense+aspect', 'modal_verbs', 
                          'verb_morphology', 'negation', 'agreement', 'relativization', 
                          'complementation', 'adverb_subordination', 'adverbs+prepositions', 
                          'discourse+word_order', 'other']
        for col in category_columns:
            f.write(f"{col}: {df[col].sum()} ({df[col].mean()*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=== MULTICOLLINEARITY ANALYSIS ===\n")
        f.write("Variance Inflation Factors (VIF):\n")
        f.write(vif_data.sort_values("VIF", ascending=False).to_string() + "\n\n")
        
        f.write("=== CORRELATION MATRIX ===\n")
        f.write(corr_matrix.round(2).to_string() + "\n\n")
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            f.write("High correlations (|r| > 0.7):\n")
            for var1, var2, corr in high_corr:
                f.write(f"{var1} & {var2}: {corr:.2f}\n")
            f.write("\n")
        
        f.write("=== DIALECT DISTRIBUTION ===\n")
        dialect_counts = df['dialect'].value_counts()
        for dialect, count in dialect_counts.items():
            f.write(f"{dialect}: {count} ({count/df.shape[0]*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=== LLM DISTRIBUTION ===\n")
        llm_counts = df['llm'].value_counts()
        for llm, count in llm_counts.items():
            f.write(f"{llm}: {count} ({count/df.shape[0]*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=== DATASET DISTRIBUTION ===\n")
        dataset_counts = df['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            f.write(f"{dataset}: {count} ({count/df.shape[0]*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=== ACCURACY BY DIALECT ===\n")
        accuracy_by_dialect = df.groupby('dialect')['accuracy'].mean().sort_values(ascending=False)
        for dialect, acc in accuracy_by_dialect.items():
            f.write(f"{dialect}: {acc:.4f}\n")
        f.write("\n")
        
        f.write("=== ACCURACY BY LLM ===\n")
        accuracy_by_llm = df.groupby('llm')['accuracy'].mean().sort_values(ascending=False)
        for llm, acc in accuracy_by_llm.items():
            f.write(f"{llm}: {acc:.4f}\n")
        f.write("\n")
        
        f.write("=== ACCURACY BY DATASET ===\n")
        accuracy_by_dataset = df.groupby('dataset')['accuracy'].mean().sort_values(ascending=False)
        for dataset, acc in accuracy_by_dataset.items():
            f.write(f"{dataset}: {acc:.4f}\n")
        f.write("\n")
        
        f.write("=== ACCURACY BY DATASET AND DIALECT ===\n")
        accuracy_by_dataset_dialect = df.groupby(['dataset', 'dialect'])['accuracy'].mean().sort_values(ascending=False)
        for (dataset, dialect), acc in accuracy_by_dataset_dialect.items():
            f.write(f"{dataset} - {dialect}: {acc:.4f}\n")
        f.write("\n")
        
        f.write("=== REGRESSION ANALYSIS ===\n")
        
        if logit_model is not None:
            f.write("=== LOGISTIC REGRESSION RESULTS ===\n")
            f.write(str(logit_model.summary()))
            f.write("\n\n")
            
            f.write("=== ODDS RATIOS ===\n")
            odds_ratios = results_df[['Variable', 'Coefficient', 'p-value', 'Odds Ratio']]
            if 'OR Lower CI' in results_df.columns:
                odds_ratios['OR Lower CI'] = results_df['OR Lower CI']
                odds_ratios['OR Upper CI'] = results_df['OR Upper CI']
            
            odds_ratios = odds_ratios.sort_values('Coefficient', ascending=False)
            f.write(odds_ratios.to_string())
            f.write("\n\n")
    
    csv_output = os.path.splitext(output_file)[0] + "_coefficients.csv"
    results_df.to_csv(csv_output, index=False)
    
if __name__ == "__main__":
    main()
from multivalue import Dialects 
import torch
import pandas as pd
from datasets import load_dataset, Dataset, load_from_disk
from datasets import disable_caching
from tqdm import tqdm
import os 
import argparse
import json 
import yaml
import ast


PROJ_DIR = "~/Dialect_Bias/" 

metadata_file = "~/Dialect_Bias/metadata.yml"


def load_metadata_yaml(dataset_name): 
    with open(metadata_file, 'r') as file:
        metadata_dict = yaml.safe_load(file)
        return metadata_dict[dataset_name]

def safe_save(save_df, save_dir, file_name): 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name)
    save_df.to_csv(file_path, index=False)
    
def convert_rule_dict_to_list(rules_executed):
    return [v["type"] for _ , v in rules_executed.items()] if rules_executed else []
    
def generate_rule_transformed_dataset(df, dialect, dialect_name=None, col_to_transform="question", cols_to_save = ["id", "context", "answers"], save_dir="data/oblig_rule_transforms", rule_already_applied=None):
    tranformed_texts = []
    rules_executed = []
    ids_so_far = []
    failed_ids = set()
    if dialect_name is None: 
        dialect_name = dialect.dialect_name

    def save_helper(ids_so_far, tranformed_texts, rules_executed):
        save_df = pd.DataFrame({"id": ids_so_far, "transformed_text": tranformed_texts, "rules_executed": rules_executed})
        save_df = save_df.merge(df[cols_to_save], on="id")
        save_df["rule_transform"] = dialect_name
        safe_save(save_df, save_dir, f"{dialect_name}.csv")
        return save_df

    iterator = tqdm(df.iterrows(), total=df.shape[0])
    for i, row in iterator: 

        try: 
            # get dialect trasnformations 
            transformed_text = dialect.transform(row[col_to_transform])
            # only save non-0 rules
            if len(dialect.executed_rules) > 0 or rule_already_applied: 
                tranformed_texts.append(transformed_text)
                ids_so_far.append(row["id"])
                if rule_already_applied: 
                    
                    executions = dialect.executed_rules
                    print("rule", row["rule_transform"])
                    old_executions = ast.literal_eval(row["rules_executed"])
                    print("old_executions", old_executions)
                    executions.update(old_executions)
                    print(executions)
                    rules_executed.append(executions) 

                else: 
                    rules_executed.append(dialect.executed_rules) 

        except Exception as e: 
            print(e)
            failed_ids.add(row["id"])
        
        if i % 10:
            save_df = save_helper(ids_so_far, tranformed_texts, rules_executed)

    save_df = save_helper(ids_so_far, tranformed_texts, rules_executed)
    return save_df, failed_ids

def transforms_for_each_rule(rule_list, df, col_to_transform="question", cols_to_save = ["id", "context", "answers"], save_dir="data/oblig_rule_transforms"): 
    dialect_df_list = []
    all_failed_ids = [] 

    for r in tqdm(rule_list): 
        dialect =  Dialects.DialectFromFeatureList(feature_list=[r], dialect_name=r)
        dialect_df, failed_ids = generate_rule_transformed_dataset( df, dialect, col_to_transform=col_to_transform, save_dir=save_dir, cols_to_save=cols_to_save)
        all_failed_ids.extend(failed_ids)
        dialect_df_list.append(dialect_df)

    return dialect_df_list, all_failed_ids

def transforms_for_each_rule_pair(rule_list, df, col_to_transform="question", cols_to_save = ["id", "context", "answers"], save_dir="data/oblig_rule_transforms"): 
    dialect_df_list = []
    all_failed_ids = [] 
    
    for r_list in tqdm(rule_list): 
        print(r_list) 
        rule_pair_name = "+".join(r_list)
        dialect =  Dialects.DialectFromFeatureList(feature_list=r_list, dialect_name=rule_pair_name)
        dialect_df, failed_ids = generate_rule_transformed_dataset( df, dialect, col_to_transform=col_to_transform, save_dir=save_dir, cols_to_save=cols_to_save)
        all_failed_ids.extend(failed_ids)
        dialect_df_list.append(dialect_df)

    return dialect_df_list, all_failed_ids

def transforms_for_each_dialect(dialect_list, df, col_to_transform="question", cols_to_save = ["id", "context", "answers"], save_dir="data/oblig_rule_transforms", rule_already_applied=None): 
    dialect_df_list = []
    all_failed_ids = [] 

    dialect_objs = {
        "AfricanAmericanVernacular": Dialects.AfricanAmericanVernacular(),
        "SoutheastAmericanEnclaveDialect" : Dialects.SoutheastAmericanEnclaveDialect(), 
        "ChicanoDialect": Dialects.ChicanoDialect(), 
        "AppalachianDialect": Dialects.AppalachianDialect(), 
        "IndianDialect": Dialects.IndianDialect(), 
        "ColloquialSingaporeDialect": Dialects.ColloquialSingaporeDialect()
    }

    for d in tqdm(dialect_list): 
        dialect =  dialect_objs[d]
        dialect_df, failed_ids = generate_rule_transformed_dataset( df, dialect, dialect_name = d, col_to_transform=col_to_transform, save_dir=save_dir, cols_to_save=cols_to_save, rule_already_applied=rule_already_applied)
        all_failed_ids.extend(failed_ids)
        dialect_df_list.append(dialect_df)

    return dialect_df_list, all_failed_ids

def transforms_for_each_dialect_oblig_only(dialect_list, df, col_to_transform="question", cols_to_save = ["id", "context", "answers"], save_dir="data/oblig_rule_transforms", rule_already_applied=None): 
    dialect_df_list = []
    all_failed_ids = [] 

    with open('data/dialect_oblig_rules.json', 'r') as f: 
        dialect_oblig_rules = json.load(f)

    for d in tqdm(dialect_list): 
        rule_list = dialect_oblig_rules[d]
        dialect =  Dialects.DialectFromFeatureList(feature_list=rule_list, dialect_name=d)
        dialect_df, failed_ids = generate_rule_transformed_dataset( df, dialect, col_to_transform=col_to_transform, save_dir=save_dir, cols_to_save=cols_to_save, rule_already_applied=rule_already_applied)
        all_failed_ids.extend(failed_ids)
        dialect_df_list.append(dialect_df)

    return dialect_df_list, all_failed_ids

def load_in_rules(rule_list, save_dir, pair=False): 
    df_list = [] 
    for r in rule_list:
        filepath = os.path.join(save_dir, f"{r}.csv")
        if pair: 
            rule_pair_name = "+".join(r)
            filepath = os.path.join(save_dir, f"{rule_pair_name}.csv")
        df = pd.read_csv(filepath)
        df_list.append(df)
    return pd.concat(df_list)

def get_slice_with_exec(rules_df, exec_col = "rules_executed"): 
    return rules_df[rules_df["rules_executed"] != "{}"]


def get_rule_list(args): 

    if args.rules == "A": 
        with open('data/attestA_rules.json', 'r') as f:
            attest_a_rules = json.load(f)

        rule_list = list(attest_a_rules.keys())
    elif args.rules == "pair": 
        rule_list = args.rule_pair
        print("pair of rules", rule_list)
    else: 
        rule_list = ["AfricanAmericanVernacular", 
                        "SoutheastAmericanEnclaveDialect", 
                        "ChicanoDialect", 
                        "AppalachianDialect", 
                        "IndianDialect", 
                        "ColloquialSingaporeDialect"
                        ]
    
    return rule_list

def find_failing_ids(args, df, short_rule_list):
    failed_id_path = os.path.join(f"data/{args.dataset_name}", "failed_ids.json")
    
    if os.path.isfile(failed_id_path): 
        print("loading existing failed_ids")
        with open(failed_id_path, "r") as file: 
            failed_ids = json.load(file)
            return failed_ids
    
    # otherwise, use a list of one rule to quickly identify failing ids
    _ , failed_ids = transforms_for_each_rule(
        short_rule_list, 
        df, 
        col_to_transform="question", 
        cols_to_save=metadata["cols_to_save"], 
        save_dir=args.save_dir
        )
    # save failed ids 
    print(f"failed_ids for {args.dataset_name} are", failed_ids)

    with open(failed_id_path, 'w+') as file:
        json.dump(failed_ids, file)
    
    return failed_ids


def generate_transformations(args): 

    # if this is our first time working with a dataset, save the original subset we work with
    original_subset_dir = f"data/{args.dataset_name}/original_subset"
    original_subset_file = f"{original_subset_dir}/{metadata['split']}.csv"
    if not os.path.exists(original_subset_file): 

        if args.indiv_rules: 
            # take the rule-transfomed dataset
            df = pd.read_csv(f"data/{args.dataset_name}/A_transforms/combined/test.csv")
                # subselect the rules listed
            working_df = df[df["rule_transform"].isin(args.indiv_rules)]

        else: 
            if args.dataset_name == "mmlu": 
                ds = load_dataset(metadata["hf_file_path"], "all")
            elif args.dataset_name == "fineweb-edu":
                ds = load_dataset(metadata["hf_file_path"], "sample-350BT", streaming=True)
                ds = ds.shuffle(seed=42, buffer_size=1000000)
                # very large dataset, so we sample a subset and filter out long examples
                subset_len = 20000
                ds_subset = ds.take(subset_len)
                ds = ds_subset.filter(lambda example: len(example["text"]) < 1000)

            else: 
                ds = load_dataset(metadata["hf_file_path"])

            df = ds[metadata["split"]].to_pandas()
            if "id" in df.columns:
                df["orig_id"] = df["id"]
            df["id"] = df.index

            # exclude ids where multivalue fails 
            failed_ids = find_failing_ids(args, df, rule_list[:1])
            working_df = df.drop(index=failed_ids).reset_index(drop=True)
            working_df["id"] = working_df.index 

            safe_save(working_df, original_subset_dir, f"{metadata['split']}.csv")
    else:
        print("loading existing processed subset")
        working_df = pd.read_csv(original_subset_file)

    print("data length excluding failures", len(working_df))
    print(working_df.tail() )

    rule_already_applied = args.indiv_rules is not None
    print("rule already applied", rule_already_applied)
    kwargs = {
        "col_to_transform" : "transformed_text" if rule_already_applied else metadata["col_to_transform"],
        "cols_to_save" : metadata["cols_to_save"], 
        "save_dir" : args.save_dir, 
    }

    if args.rules == "dialect": 
        dialect_df_list, failed_ids = transforms_for_each_dialect(
        rule_list, 
        working_df, 
        rule_already_applied=rule_already_applied, 
        **kwargs,
        )
        
    elif args.rules == "dialect_oblig": 
        dialect_df_list, failed_ids = transforms_for_each_dialect_oblig_only(
        rule_list, 
        working_df, 
        rule_already_applied=rule_already_applied, 
        **kwargs,
        )
    elif args.rules == "pair": 
        dialect_df_list, failed_ids = transforms_for_each_rule_pair(
            rule_list, 
            working_df, 
            **kwargs,
            )
    else: 
        dialect_df_list, failed_ids = transforms_for_each_rule(
            rule_list, 
            working_df, 
            **kwargs,
            )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rules', choices=["A", "pair", "dialect", "dialect_oblig"]) 
    parser.add_argument('--rule_pair', action='append', nargs='+',) 
    parser.add_argument('--indiv_rules', nargs='+', default=None) 
    parser.add_argument('-d', '--dataset_name', choices=["boolq", "mmlu", "sciq", "fineweb-edu"]) 
    parser.add_argument('-s', '--save_dir', default=None) 

    args = parser.parse_args() 

    rule_list = get_rule_list(args)
    metadata = load_metadata_yaml(args.dataset_name)
    args.save_dir = f"data/{args.dataset_name}/{args.rules}_transforms"
    if args.indiv_rules: 
        args.save_dir = f"data/{args.dataset_name}/rule+{args.rules}_transforms"
    print("save_dir",args.save_dir )

    generate_transformations(args)
    # load in rules and combine 
    if args.rules == "pair": 
        loaded_in_rules = load_in_rules(rule_list, save_dir=args.save_dir, pair=True)
    else: 
        loaded_in_rules = load_in_rules(rule_list, save_dir=args.save_dir)
    slice_with_exec = get_slice_with_exec(loaded_in_rules, exec_col = "rules_executed") 
    combined_dir = os.path.join(args.save_dir, "combined")

    if args.dataset_name == "boolq": 
        # additional post-processing step convert to yes and no 
        slice_with_exec["answer"] = slice_with_exec["answer"].apply(lambda x: "yes" if x == True else "no")

    safe_save(slice_with_exec, combined_dir, "test.csv")


       




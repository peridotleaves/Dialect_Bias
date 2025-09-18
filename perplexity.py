import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from scipy import stats
import argparse
import logging
import gc
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("perplexity_calculation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

BENCHMARKS = ["sciq", "boolq", "mmlu"]
ORIGINAL_PATH_TEMPLATE = "dialect_data/{benchmark}/original_subset/combined/test.csv"
DIALECT_PATH_TEMPLATE = "dialect_data/{benchmark}/dialect_transforms/combined/test.csv"
OUTPUT_DIR = "perplexity_results"

def memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"GPU Memory: Allocated = {allocated:.2f} GB, Reserved = {reserved:.2f} GB")
    
    import psutil
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)
    logger.info(f"RAM Usage: {ram_usage:.2f} GB")

def load_model_and_tokenizer():
    model_name = "HuggingFaceFW/ablation-model-fineweb-edu"
    logger.info(f"Loading model: {model_name}")
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    memory_stats()
    return model, tokenizer

def calculate_perplexity(text, model, tokenizer, max_length=512):
    if not text or pd.isna(text) or not isinstance(text, str):
        return float('nan')
    
    try:
        encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss.item()
            
        perplexity = np.exp(neg_log_likelihood)
        
        del encodings, input_ids, outputs
        torch.cuda.empty_cache()
        
        return perplexity
    except Exception as e:
        logger.error(f"Error calculating perplexity: {str(e)}")
        logger.error(f"Problematic text: {text[:100]}...")
        return float('nan')

def process_benchmark(benchmark, model, tokenizer, batch_size=32):
    original_path = ORIGINAL_PATH_TEMPLATE.format(benchmark=benchmark)
    dialect_path = DIALECT_PATH_TEMPLATE.format(benchmark=benchmark)
    
    logger.info(f"Processing {benchmark}...")
    logger.info(f"Original path: {original_path}")
    logger.info(f"Dialect path: {dialect_path}")

    try:
        original_df = pd.read_csv(original_path)
        dialect_df = pd.read_csv(dialect_path)
        
        logger.info(f"Loaded {len(original_df)} samples from original dataset")
        logger.info(f"Loaded {len(dialect_df)} samples from dialect dataset")
        
        if 'question' not in original_df.columns:
            logger.error(f"Original dataset missing 'question' column. Available columns: {original_df.columns.tolist()}")
            return pd.DataFrame()
            
        if 'transformed_text' not in dialect_df.columns:
            logger.error(f"Dialect dataset missing 'transformed_text' column. Available columns: {dialect_df.columns.tolist()}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        return pd.DataFrame()
    
    original_texts = original_df['question'].tolist()
    dialect_texts = dialect_df['transformed_text'].tolist()
    
    if len(original_texts) != len(dialect_texts):
        logger.warning(f"Mismatch in dataset sizes: original={len(original_texts)}, dialect={len(dialect_texts)}")
        min_len = min(len(original_texts), len(dialect_texts))
        original_texts = original_texts[:min_len]
        dialect_texts = dialect_texts[:min_len]
    
    results = []
    total_samples = len(original_texts)

    for i in tqdm(range(0, total_samples, batch_size), desc=f"Processing {benchmark}"):
        batch_end = min(i + batch_size, total_samples)
        batch_original = original_texts[i:batch_end]
        batch_dialect = dialect_texts[i:batch_end]
        
        batch_results = []
        for j, (orig_text, dial_text) in enumerate(zip(batch_original, batch_dialect)):
            idx = i + j
            
            orig_perplexity = calculate_perplexity(orig_text, model, tokenizer)
            dial_perplexity = calculate_perplexity(dial_text, model, tokenizer)
            
            perplexity_diff = dial_perplexity - orig_perplexity
            perplexity_ratio = dial_perplexity / orig_perplexity if orig_perplexity > 0 else float('nan')
            
            if (idx + 1) % 100 == 0 or idx == 0 or idx == total_samples - 1:
                logger.info(f"Sample {idx+1}/{total_samples}: orig_ppl={orig_perplexity:.2f}, dial_ppl={dial_perplexity:.2f}")
                memory_stats()
            
            batch_results.append({
                'benchmark': benchmark,
                'index': idx,
                'original_text': orig_text,
                'dialect_text': dial_text,
                'original_perplexity': orig_perplexity,
                'dialect_perplexity': dial_perplexity,
                'perplexity_diff': perplexity_diff,
                'perplexity_ratio': perplexity_ratio
            })
        
        results.extend(batch_results)
        
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(f"{OUTPUT_DIR}/{benchmark}_perplexity_results_partial.csv", index=False)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/{benchmark}_perplexity_results.csv", index=False)
    
    return results_df

def analyze_results(results_df):
    
    benchmark_stats = results_df.groupby('benchmark').agg({
        'original_perplexity': ['mean', 'std', 'median', 'count'],
        'dialect_perplexity': ['mean', 'std', 'median', 'count'],
        'perplexity_diff': ['mean', 'std', 'median'],
        'perplexity_ratio': ['mean', 'std', 'median']
    }).reset_index()
    
    for benchmark in BENCHMARKS:
        if benchmark not in results_df['benchmark'].unique():
            continue
            
        bench_data = results_df[results_df['benchmark'] == benchmark]
        orig_mean = bench_data['original_perplexity'].mean()
        dial_mean = bench_data['dialect_perplexity'].mean()
        
        pct_increase = ((dial_mean - orig_mean) / orig_mean) * 100
        logger.info(f"{benchmark}: Original PPL={orig_mean:.2f}, Dialect PPL={dial_mean:.2f}, Increase={pct_increase:.2f}%")
    
    stat_results = {}
    for benchmark in BENCHMARKS:
        if benchmark not in results_df['benchmark'].unique():
            continue
            
        benchmark_data = results_df[results_df['benchmark'] == benchmark]
        
        paired_data = benchmark_data.dropna(subset=['original_perplexity', 'dialect_perplexity'])
        
        if len(paired_data) > 1:
            t_stat, p_value = stats.ttest_rel(
                paired_data['dialect_perplexity'],
                paired_data['original_perplexity']
            )
            
            stat_results[benchmark] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'sample_size': len(paired_data)
            }
            
            logger.info(f"{benchmark} t-test: t={t_stat:.3f}, p={p_value:.5f}, n={len(paired_data)}")
    
    return benchmark_stats, stat_results

def main():

    global BENCHMARKS, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="Calculate perplexity for original and dialectal text variants")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, help="Benchmarks to process")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    BENCHMARKS = args.benchmarks
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Starting perplexity calculation for benchmarks: {', '.join(BENCHMARKS)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model, tokenizer = load_model_and_tokenizer()
    
    all_results = []

    for benchmark in BENCHMARKS:
        try:
            results_df = process_benchmark(benchmark, model, tokenizer)
            if not results_df.empty:
                all_results.append(results_df)
        except Exception as e:
            logger.error(f"Error processing benchmark {benchmark}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{OUTPUT_DIR}/all_benchmarks_perplexity_results.csv", index=False)
        
        benchmark_stats, stat_results = analyze_results(combined_results)
        
        benchmark_stats.to_csv(f"{OUTPUT_DIR}/benchmark_perplexity_stats.csv", index=False)

        logger.info("\n===== SUMMARY STATISTICS =====")
        logger.info(benchmark_stats)
        logger.info("\n===== STATISTICAL TESTS =====")
        for benchmark, stats in stat_results.items():
            logger.info(f"{benchmark}: t={stats['t_statistic']:.3f}, p={stats['p_value']:.5f} {'(significant)' if stats['significant'] else ''}")
    else:
        logger.error("No results were generated for any benchmark")

if __name__ == "__main__":
    main()
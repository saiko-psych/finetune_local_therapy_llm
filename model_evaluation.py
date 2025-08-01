#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
=====================================
Vergleicht Original vs Finetuned Model mit/ohne Filter
Mit Checkpoints, Error-Handling und automatischen Reports
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from datasets import load_dataset
import random
import json
import os
import gc
import time
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_MODEL = "google/gemma-3-4b-it"
FINETUNED_PATH = "./best-finetuned-gemma"

# Settings
NUM_SAMPLES = 100  
CHECKPOINT_INTERVAL = 10 
MAX_RETRIES = 3  

# Output directory
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = OUTPUT_DIR / f"evaluation_log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Disable CUDA for stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, data, filename):
        """Save checkpoint data"""
        try:
            checkpoint_path = self.checkpoint_dir / f"{filename}.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f" Checkpoint gespeichert: {checkpoint_path}")
        except Exception as e:
            logger.error(f" Checkpoint-Fehler: {e}")
    
    def load_checkpoint(self, filename):
        """Load checkpoint data"""
        try:
            checkpoint_path = self.checkpoint_dir / f"{filename}.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f" Checkpoint geladen: {checkpoint_path}")
                return data
            return None
        except Exception as e:
            logger.error(f" Checkpoint-Load-Fehler: {e}")
            return None
    
    def checkpoint_exists(self, filename):
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f"{filename}.json").exists()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_cleanup():
    """Safe memory cleanup"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")

def retry_on_failure(func, max_retries=MAX_RETRIES, *args, **kwargs):
    """Retry function on failure"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All attempts failed for {func.__name__}")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# =============================================================================
# DATA LOADING
# =============================================================================

def load_mental_health_dataset(num_samples, use_filters=True):
    """Load and prepare dataset samples"""
    logger.info(" Lade Mental Health Counseling Dataset...")
    
    try:
        dataset = load_dataset("Amod/mental_health_counseling_conversations")
        
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        logger.info(f" Dataset geladen: {len(data)} Einträge total")
        
        samples = []
        
        if use_filters:
            logger.info(" Filtere geeignete Einträge (50-1000 chars Context, 50-2000 chars Response)...")
            valid_entries = []
            
            for idx in range(len(data)):
                entry = data[idx]
                context = entry['Context'].strip()
                response = entry['Response'].strip()
                
                if (50 <= len(context) <= 1000 and 50 <= len(response) <= 2000):
                    valid_entries.append((idx, context, response))
            
            logger.info(f" {len(valid_entries)} geeignete Einträge gefunden")
            
            if len(valid_entries) < num_samples:
                logger.warning(f" Nur {len(valid_entries)} geeignete Einträge verfügbar")
                selected_entries = valid_entries
            else:
                selected_entries = random.sample(valid_entries, num_samples)
                
        else:
            logger.info(" Nehme komplett zufällige Einträge (ohne Filter)...")
            indices = random.sample(range(len(data)), min(num_samples, len(data)))
            selected_entries = []
            
            for idx in indices:
                entry = data[idx]
                context = entry['Context'].strip()
                response = entry['Response'].strip()
                selected_entries.append((idx, context, response))
        
        # Convert to standard format
        for idx, context, response in selected_entries:
            samples.append({
                'index': idx,
                'context': context,
                'ground_truth': response,
                'context_length': len(context),
                'response_length': len(response)
            })
        
        logger.info(f" {len(samples)} Samples vorbereitet")
        return samples
        
    except Exception as e:
        logger.error(f" Dataset-Fehler: {e}")
        raise

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_path, model_name):
        """Load model with error handling"""
        try:
            logger.info(f" Lade {model_name}...")
            
            if model_path.startswith("./"):
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            model.eval()
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f" {model_name} erfolgreich geladen")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
            return False
    
    def generate_response(self, model_name, prompt, max_tokens=200):
        """Generate response with robust error handling"""
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Truncate prompt if too long
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Dynamic max_new_tokens
            input_length = inputs['input_ids'].shape[1]
            available_tokens = 1024 - input_length
            max_tokens = min(max_tokens, max(50, available_tokens - 50))
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            full_answer = tokenizer.decode(output[0], skip_special_tokens=True)
            response = full_answer[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation error for {model_name}: {e}")
            return f"[ERROR: {str(e)[:100]}]"
    
    def cleanup(self):
        """Clean up models from memory"""
        for model_name in list(self.models.keys()):
            del self.models[model_name]
            del self.tokenizers[model_name]
        self.models.clear()
        self.tokenizers.clear()
        safe_cleanup()

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def create_therapeutic_prompt(context):
    """Create therapeutic prompt from context"""
    return (
        "Du bist ein einfühlsamer Therapeut, der professionelle und hilfreiche Antworten gibt.\n"
        f"Patient: {context}\n"
        "Therapeut: "
    )

def calculate_similarities(original_responses, finetuned_responses, ground_truth_responses):
    """Calculate three-way similarities"""
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000,
            stop_words='english'
        )
        
        all_texts = original_responses + finetuned_responses + ground_truth_responses
        # Filter empty responses
        all_texts = [text if text and len(text.strip()) > 0 else "empty response" for text in all_texts]
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        n = len(original_responses)
        tfidf_original = tfidf_matrix[:n]
        tfidf_finetuned = tfidf_matrix[n:2*n]
        tfidf_ground_truth = tfidf_matrix[2*n:]
        
        similarities = {
            'original_vs_ground_truth': [],
            'finetuned_vs_ground_truth': [],
            'original_vs_finetuned': []
        }
        
        for i in range(n):
            sim_og = cosine_similarity(tfidf_original[i], tfidf_ground_truth[i])[0][0]
            sim_fg = cosine_similarity(tfidf_finetuned[i], tfidf_ground_truth[i])[0][0]
            sim_of = cosine_similarity(tfidf_original[i], tfidf_finetuned[i])[0][0]
            
            similarities['original_vs_ground_truth'].append(sim_og)
            similarities['finetuned_vs_ground_truth'].append(sim_fg)
            similarities['original_vs_finetuned'].append(sim_of)
        
        return similarities
        
    except Exception as e:
        logger.error(f"❌ Similarity calculation error: {e}")
        # Return dummy similarities
        n = len(original_responses)
        return {
            'original_vs_ground_truth': [0.0] * n,
            'finetuned_vs_ground_truth': [0.0] * n,
            'original_vs_finetuned': [0.0] * n
        }

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(results_filtered, results_unfiltered, output_dir):
    """Create comparison visualizations"""
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Comparison: Filtered vs Unfiltered Data\n({len(results_filtered)} samples each)', 
                     fontsize=16, fontweight='bold')
        
        # Data preparation
        datasets = {'Filtered': results_filtered, 'Unfiltered': results_unfiltered}
        colors = {'Filtered': 'blue', 'Unfiltered': 'red'}
        
        # Plot 1: Original vs Ground Truth
        ax = axes[0, 0]
        for name, data in datasets.items():
            values = [float(x) for x in data['Original_vs_GT']]
            ax.hist(values, alpha=0.6, label=f'{name} (μ={np.mean(values):.3f})', 
                   color=colors[name], bins=20)
        ax.set_title('Original vs Ground Truth Similarity')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Finetuned vs Ground Truth
        ax = axes[0, 1]
        for name, data in datasets.items():
            values = [float(x) for x in data['Finetuned_vs_GT']]
            ax.hist(values, alpha=0.6, label=f'{name} (μ={np.mean(values):.3f})', 
                   color=colors[name], bins=20)
        ax.set_title('Finetuned vs Ground Truth Similarity')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Original vs Finetuned
        ax = axes[0, 2]
        for name, data in datasets.items():
            values = [float(x) for x in data['Original_vs_Finetuned']]
            ax.hist(values, alpha=0.6, label=f'{name} (μ={np.mean(values):.3f})', 
                   color=colors[name], bins=20)
        ax.set_title('Original vs Finetuned Similarity')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Box plot comparison
        ax = axes[1, 0]
        data_for_box = []
        labels_for_box = []
        for name, data in datasets.items():
            orig_vals = [float(x) for x in data['Original_vs_GT']]
            fine_vals = [float(x) for x in data['Finetuned_vs_GT']]
            data_for_box.extend([orig_vals, fine_vals])
            labels_for_box.extend([f'{name}\nOriginal', f'{name}\nFinetuned'])
        
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            color = 'lightblue' if i % 2 == 0 else 'lightgreen'
            patch.set_facecolor(color)
        ax.set_title('Similarity Distributions vs Ground Truth')
        ax.set_ylabel('Cosine Similarity')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Response length comparison
        ax = axes[1, 1]
        for name, data in datasets.items():
            orig_lengths = data['Original_Länge']
            fine_lengths = data['Finetuned_Länge']
            gt_lengths = data['GT_Länge']
            
            ax.scatter(orig_lengths, fine_lengths, alpha=0.6, label=f'{name}', 
                      color=colors[name], s=30)
        
        ax.set_xlabel('Original Response Length')
        ax.set_ylabel('Finetuned Response Length')
        ax.set_title('Response Length Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Performance improvement
        ax = axes[1, 2]
        improvements = {}
        for name, data in datasets.items():
            orig_vals = [float(x) for x in data['Original_vs_GT']]
            fine_vals = [float(x) for x in data['Finetuned_vs_GT']]
            improvement = [f - o for f, o in zip(fine_vals, orig_vals)]
            improvements[name] = improvement
            
            ax.hist(improvement, alpha=0.6, label=f'{name} (μ={np.mean(improvement):.3f})', 
                   color=colors[name], bins=20)
        
        ax.set_xlabel('Improvement (Finetuned - Original)')
        ax.set_ylabel('Frequency')
        ax.set_title('Finetuning Improvement Distribution')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'model_comparison_plots_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f" Visualizations saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"❌ Visualization error: {e}")
        return None

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results_filtered, results_unfiltered, output_dir):
    """Generate comprehensive evaluation report"""
    try:
        report_path = output_dir / f'evaluation_report_{timestamp}.md'
        
        # Calculate statistics
        stats = {}
        for name, data in [('Filtered', results_filtered), ('Unfiltered', results_unfiltered)]:
            orig_gt = [float(x) for x in data['Original_vs_GT']]
            fine_gt = [float(x) for x in data['Finetuned_vs_GT']]
            orig_fine = [float(x) for x in data['Original_vs_Finetuned']]
            improvement = [f - o for f, o in zip(fine_gt, orig_gt)]
            
            stats[name] = {
                'orig_gt_mean': np.mean(orig_gt),
                'orig_gt_std': np.std(orig_gt),
                'fine_gt_mean': np.mean(fine_gt),
                'fine_gt_std': np.std(fine_gt),
                'orig_fine_mean': np.mean(orig_fine),
                'improvement_mean': np.mean(improvement),
                'improvement_std': np.std(improvement),
                'samples_improved': sum(1 for x in improvement if x > 0),
                'total_samples': len(improvement)
            }
        
        # Generate report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Model Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Models:** Original vs Finetuned Gemma-3-4B  
**Dataset:** Mental Health Counseling Conversations  
**Samples per condition:** {NUM_SAMPLES}  

## Executive Summary

This evaluation compares the performance of an original Gemma-3-4B model against its finetuned version using real mental health counseling data, tested both with and without data filtering.

### Key Findings

""")
            
            for name in ['Filtered', 'Unfiltered']:
                s = stats[name]
                winner = "🏆 **Finetuned**" if s['fine_gt_mean'] > s['orig_gt_mean'] else "📉 **Original**"
                improvement_pct = (s['samples_improved'] / s['total_samples']) * 100
                
                f.write(f"""
#### {name} Data Results

- **Winner:** {winner}
- **Original vs GT:** {s['orig_gt_mean']:.3f} ± {s['orig_gt_std']:.3f}
- **Finetuned vs GT:** {s['fine_gt_mean']:.3f} ± {s['fine_gt_std']:.3f}
- **Average Improvement:** {s['improvement_mean']:.3f} ± {s['improvement_std']:.3f}
- **Samples Improved:** {s['samples_improved']}/{s['total_samples']} ({improvement_pct:.1f}%)
""")
            
            # Comparison between filtered and unfiltered
            filtered_advantage = stats['Filtered']['fine_gt_mean'] - stats['Unfiltered']['fine_gt_mean']
            
            f.write(f"""
### Filtered vs Unfiltered Impact

- **Filtered Data Advantage:** {filtered_advantage:.3f}
- **Recommendation:** {"Use filtered data" if filtered_advantage > 0.01 else "Both approaches similar"}

## Detailed Statistics

| Metric | Filtered | Unfiltered | Difference |
|--------|----------|------------|------------|
| Original vs GT | {stats['Filtered']['orig_gt_mean']:.3f} | {stats['Unfiltered']['orig_gt_mean']:.3f} | {stats['Filtered']['orig_gt_mean'] - stats['Unfiltered']['orig_gt_mean']:.3f} |
| Finetuned vs GT | {stats['Filtered']['fine_gt_mean']:.3f} | {stats['Unfiltered']['fine_gt_mean']:.3f} | {stats['Filtered']['fine_gt_mean'] - stats['Unfiltered']['fine_gt_mean']:.3f} |
| Original vs Finetuned | {stats['Filtered']['orig_fine_mean']:.3f} | {stats['Unfiltered']['orig_fine_mean']:.3f} | {stats['Filtered']['orig_fine_mean'] - stats['Unfiltered']['orig_fine_mean']:.3f} |

## Methodology

1. **Data Source:** Amod/mental_health_counseling_conversations from Hugging Face
2. **Filtering Criteria:** 
   - Context: 50-1000 characters
   - Response: 50-2000 characters
3. **Evaluation Metric:** Cosine similarity using TF-IDF vectors
4. **Model Configuration:** CPU-only, temperature=0.7, top_p=0.9

## Technical Details

- **Total Runtime:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Checkpoint Interval:** Every {CHECKPOINT_INTERVAL} samples
- **Error Handling:** {MAX_RETRIES} retries per generation
- **Output Directory:** {output_dir.absolute()}

## Files Generated

- `evaluation_results_filtered_{timestamp}.csv`
- `evaluation_results_unfiltered_{timestamp}.csv`
- `model_comparison_plots_{timestamp}.png`
- `evaluation_log_{timestamp}.log`

## Recommendations

Based on the evaluation results:

""")
            
            # Add recommendations based on results
            if stats['Filtered']['fine_gt_mean'] > stats['Filtered']['orig_gt_mean']:
                f.write("✅ **Finetuning was successful** - the finetuned model shows improved similarity to ground truth responses.\n\n")
            else:
                f.write("⚠️ **Finetuning needs review** - the original model performed better. Consider reviewing training data or hyperparameters.\n\n")
            
            if filtered_advantage > 0.01:
                f.write("📊 **Use filtered data** for more reliable evaluation - extreme data points may skew results.\n\n")
            else:
                f.write("📊 **Both filtered and unfiltered** data show similar patterns - the model is robust across data distributions.\n\n")
        
        logger.info(f" Report generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f" Report generation error: {e}")
        return None

# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_evaluation(use_filters, condition_name, checkpoint_manager, model_manager):
    """Run evaluation for one condition (filtered/unfiltered)"""
    logger.info(f" Starting {condition_name} evaluation...")
    
    # Check for existing checkpoint
    checkpoint_name = f"evaluation_{condition_name.lower()}_{timestamp}"
    existing_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_name)
    
    if existing_checkpoint:
        logger.info(f" Resuming from checkpoint: {len(existing_checkpoint.get('results', []))} completed")
        results = existing_checkpoint.get('results', [])
        samples = existing_checkpoint.get('samples', [])
        start_idx = len(results)
    else:
        logger.info(f" Starting fresh {condition_name} evaluation")
        samples = load_mental_health_dataset(NUM_SAMPLES, use_filters)
        results = []
        start_idx = 0
        
        # Save initial checkpoint with samples
        checkpoint_manager.save_checkpoint({
            'samples': samples,
            'results': results,
            'condition': condition_name,
            'use_filters': use_filters
        }, checkpoint_name)
    
    logger.info(f" Processing {len(samples)} samples, starting from index {start_idx}")
    
    # Process samples
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        sample_start_time = time.time()
        
        try:
            logger.info(f" Processing sample {i+1}/{len(samples)} ({condition_name})")
            
            context = sample['context']
            ground_truth = sample['ground_truth']
            prompt = create_therapeutic_prompt(context)
            
            # Generate responses with retries
            original_response = retry_on_failure(
                model_manager.generate_response, 
                MAX_RETRIES, 
                "original", 
                prompt
            )
            
            finetuned_response = retry_on_failure(
                model_manager.generate_response, 
                MAX_RETRIES, 
                "finetuned", 
                prompt
            )
            
            # Calculate similarities for this sample
            similarities = calculate_similarities(
                [original_response], 
                [finetuned_response], 
                [ground_truth]
            )
            
            # Store result
            result = {
                'sample_id': i,
                'context': context[:100] + "..." if len(context) > 100 else context,
                'context_length': len(context),
                'original_response': original_response,
                'finetuned_response': finetuned_response,
                'ground_truth': ground_truth,
                'original_vs_gt': similarities['original_vs_ground_truth'][0],
                'finetuned_vs_gt': similarities['finetuned_vs_ground_truth'][0],
                'original_vs_finetuned': similarities['original_vs_finetuned'][0],
                'original_length': len(original_response),
                'finetuned_length': len(finetuned_response),
                'gt_length': len(ground_truth),
                'processing_time': time.time() - sample_start_time
            }
            
            results.append(result)
            
            # Progress logging
            elapsed = time.time() - sample_start_time
            logger.info(f" Sample {i+1} completed in {elapsed:.1f}s - "
                       f"Similarities: O→GT={result['original_vs_gt']:.3f}, "
                       f"F→GT={result['finetuned_vs_gt']:.3f}")
            
            # Save checkpoint periodically
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_manager.save_checkpoint({
                    'samples': samples,
                    'results': results,
                    'condition': condition_name,
                    'use_filters': use_filters
                }, checkpoint_name)
                
                avg_orig = np.mean([r['original_vs_gt'] for r in results])
                avg_fine = np.mean([r['finetuned_vs_gt'] for r in results])
                logger.info(f" Progress: {len(results)}/{len(samples)} - "
                           f"Avg similarities: Original={avg_orig:.3f}, Finetuned={avg_fine:.3f}")
        
        except Exception as e:
            logger.error(f" Error processing sample {i+1}: {e}")
            logger.error(traceback.format_exc())
            
            # Add error placeholder
            results.append({
                'sample_id': i,
                'context': sample.get('context', 'ERROR')[:100],
                'context_length': sample.get('context_length', 0),
                'original_response': f"ERROR: {str(e)[:100]}",
                'finetuned_response': f"ERROR: {str(e)[:100]}",
                'ground_truth': sample.get('ground_truth', ''),
                'original_vs_gt': 0.0,
                'finetuned_vs_gt': 0.0,
                'original_vs_finetuned': 0.0,
                'original_length': 0,
                'finetuned_length': 0,
                'gt_length': len(sample.get('ground_truth', '')),
                'processing_time': 0.0
            })
            
            continue
    
    # Final checkpoint save
    checkpoint_manager.save_checkpoint({
        'samples': samples,
        'results': results,
        'condition': condition_name,
        'use_filters': use_filters,
        'completed': True
    }, checkpoint_name)
    
    logger.info(f" {condition_name} evaluation completed: {len(results)} samples processed")
    return results

def main():
    """Main evaluation pipeline"""
    try:

        print("\n" + "="*70)
        print("🧠 Local Therapy Model Evaluation")
        print("-" * 70)
        print("🔍 Vergleich: Original vs Finetuned LLM")
        print("🎯 Fokus: Semantische Ähnlichkeit mit echten Therapeut:innen")
        print("📊 Modus: Mit & ohne Daten-Filterung")
        print("💾 Ergebnisse: Ähnlichkeits-Scores, Visualisierungen, Reports")
        print("🚀 Los geht's!\n")
        print("="*70 + "\n")

        logger.info("🚀 STARTING MAIN EVALUATION")
        
        # Initialize managers
        checkpoint_manager = CheckpointManager(OUTPUT_DIR / "checkpoints")
        model_manager = ModelManager()
        
        # Load models
        success_original = model_manager.load_model(BASE_MODEL, "original")
        success_finetuned = model_manager.load_model(FINETUNED_PATH, "finetuned")
        
        if not (success_original and success_finetuned):
            logger.error(" Model loading failed, aborting.")
            return
        
        # Run evaluations
        results_filtered = run_evaluation(True, "Filtered", checkpoint_manager, model_manager)
        results_unfiltered = run_evaluation(False, "Unfiltered", checkpoint_manager, model_manager)
        
        # Cleanup models from memory
        model_manager.cleanup()
        
        # Convert results to format for visualization
        def results_to_dict(results):
            return {
                "Original_vs_GT": [r["original_vs_gt"] for r in results],
                "Finetuned_vs_GT": [r["finetuned_vs_gt"] for r in results],
                "Original_vs_Finetuned": [r["original_vs_finetuned"] for r in results],
                "Original_Länge": [r["original_length"] for r in results],
                "Finetuned_Länge": [r["finetuned_length"] for r in results],
                "GT_Länge": [r["gt_length"] for r in results]
            }
        
        dict_filtered = results_to_dict(results_filtered)
        dict_unfiltered = results_to_dict(results_unfiltered)
        
        # Create visualizations
        create_visualizations(dict_filtered, dict_unfiltered, OUTPUT_DIR)
        
        # Save raw results
        pd.DataFrame(results_filtered).to_csv(OUTPUT_DIR / f"evaluation_results_filtered_{timestamp}.csv", index=False)
        pd.DataFrame(results_unfiltered).to_csv(OUTPUT_DIR / f"evaluation_results_unfiltered_{timestamp}.csv", index=False)
        
        # Generate final report
        generate_report(dict_filtered, dict_unfiltered, OUTPUT_DIR)
        
        logger.info(" Evaluation complete")
        
    except Exception as e:
        logger.error(f" Fatal error in main: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

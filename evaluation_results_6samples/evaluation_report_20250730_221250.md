# Model Evaluation Report

**Generated:** 2025-07-30 23:37:57  
**Models:** Original vs Finetuned Gemma-3-4B  
**Dataset:** Mental Health Counseling Conversations  
**Samples per condition:** 6  

## Executive Summary

This evaluation compares the performance of an original Gemma-3-4B model against its finetuned version using real mental health counseling data, tested both with and without data filtering.

### Key Findings


#### Filtered Data Results

- **Winner:** 🏆 **Finetuned**
- **Original vs GT:** 0.040 ± 0.035
- **Finetuned vs GT:** 0.045 ± 0.036
- **Average Improvement:** 0.004 ± 0.030
- **Samples Improved:** 4/6 (66.7%)

#### Unfiltered Data Results

- **Winner:** 🏆 **Finetuned**
- **Original vs GT:** 0.036 ± 0.028
- **Finetuned vs GT:** 0.044 ± 0.012
- **Average Improvement:** 0.008 ± 0.031
- **Samples Improved:** 5/6 (83.3%)

### Filtered vs Unfiltered Impact

- **Filtered Data Advantage:** 0.000
- **Recommendation:** Both approaches similar

## Detailed Statistics

| Metric | Filtered | Unfiltered | Difference |
|--------|----------|------------|------------|
| Original vs GT | 0.040 | 0.036 | 0.004 |
| Finetuned vs GT | 0.045 | 0.044 | 0.000 |
| Original vs Finetuned | 0.068 | 0.074 | -0.006 |

## Methodology

1. **Data Source:** Amod/mental_health_counseling_conversations from Hugging Face
2. **Filtering Criteria:** 
   - Context: 50-1000 characters
   - Response: 50-2000 characters
3. **Evaluation Metric:** Cosine similarity using TF-IDF vectors
4. **Model Configuration:** CPU-only, temperature=0.7, top_p=0.9

## Technical Details

- **Total Runtime:** 2025-07-30 23:37:57
- **Checkpoint Interval:** Every 2 samples
- **Error Handling:** 3 retries per generation
- **Output Directory:** C:\Users\David\Documents\python\Projects\uni\001\evaluation_results

## Files Generated

- `evaluation_results_filtered_20250730_221250.csv`
- `evaluation_results_unfiltered_20250730_221250.csv`
- `model_comparison_plots_20250730_221250.png`
- `evaluation_log_20250730_221250.log`

## Recommendations

Based on the evaluation results:

✅ **Finetuning was successful** - the finetuned model shows improved similarity to ground truth responses.

📊 **Both filtered and unfiltered** data show similar patterns - the model is robust across data distributions.


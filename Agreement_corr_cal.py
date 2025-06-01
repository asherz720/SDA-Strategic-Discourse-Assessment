import scipy
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy import stats
from statsmodels.stats.inter_rater import fleiss_kappa


def compare_human_llm_corrf (llm_df, human_df, value):
    all_annotators_names = human_df['annotator'].unique()

    print(f"Correlations between GPT and human annotators for {value} scores:")
    correlations = []
    for annotator in all_annotators_names:
        annotator_df = human_df[human_df['annotator'] == annotator]
        
        # Merge GPT annotations with current annotator's data
        merged_df = pd.merge(llm_df[['question', f'{value}']], 
                            annotator_df[['question', f'{value}']], 
                            on='question', 
                            suffixes=('_gpt', '_human'))
        
        correlation, p_value = scipy.stats.spearmanr(merged_df[f'{value}_gpt'], merged_df[f'{value}_human'])
        correlations.append(correlation)
        print(f"Correlation with {annotator}: {correlation:.3f} (p={p_value:.10f})")
    
    mean_correlation = sum(correlations) / len(correlations)
    # print(f"\nMean correlation across annotators: {mean_correlation:.3f}")
    return mean_correlation


# Calculate agreement between human annotators and LLM using Cohen's kappa
def cal_agreement_cohen_llm(llm_df, human_df, value):

    all_annotators_names = human_df['annotator'].unique()
    agreements = []
    kappas = []

    print(f"\nAgreement between GPT and human annotators for {value}:")
    
    for annotator in all_annotators_names:
        annotator_df = human_df[human_df['annotator'] == annotator]
        
        # Merge GPT annotations with current annotator's data
        merged_df = pd.merge(llm_df[['question', f'{value}']], 
                           annotator_df[['question', f'{value}']], 
                           on='question',
                           suffixes=('_gpt', '_human'))
        
        # For outcome_value, convert to binary (1 for Witness, 0 for Questioner)
        if value == 'outcome_value':
            merged_df[f'{value}_gpt'] = (merged_df[f'{value}_gpt'] == 'Witness').astype(int)
            merged_df[f'{value}_human'] = (merged_df[f'{value}_human'] == 'Witness').astype(int)
        
        # Calculate percentage agreement
        agreement = (merged_df[f'{value}_gpt'] == merged_df[f'{value}_human']).mean() * 100
        agreements.append(agreement)
        
        # Calculate Cohen's kappa
        kappa = cohen_kappa_score(merged_df[f'{value}_gpt'], merged_df[f'{value}_human'])
        kappas.append(kappa)
        
        # print(f"\nAnnotator {annotator}:")
        # print(f"Agreement percentage: {agreement:.1f}%")
        # print(f"Cohen's kappa: {kappa:.3f}")

    mean_agreement = np.mean(agreements)
    mean_kappa = np.mean(kappas)
    
    # print(f"\nOverall metrics:")
    # print(f"Mean agreement percentage: {mean_agreement:.1f}%")
    # print(f"Mean Cohen's kappa: {mean_kappa:.3f}")
    return mean_kappa

# Calculate agreement between human annotators and LLM using Randolph's kappa
def cal_agreement_randolf_llm(llm_df, human_df, value):


    all_annotators_names = human_df['annotator'].unique()
    agreements = []
    kappas = []

    print(f"\nAgreement between GPT and human annotators for {value}:")
    
    for annotator in all_annotators_names:
        annotator_df = human_df[human_df['annotator'] == annotator]
        
        # Merge GPT annotations with current annotator's data
        merged_df = pd.merge(llm_df[['question', f'{value}']], 
                           annotator_df[['question', f'{value}']], 
                           on='question',
                           suffixes=('_gpt', '_human'))
        
        # Calculate percentage agreement
        agreement = (merged_df[f'{value}_gpt'] == merged_df[f'{value}_human']).mean() * 100
        agreements.append(agreement)
        
        # Create ratings matrix for Fleiss' kappa
        n_categories = len(np.unique(merged_df[[f'{value}_gpt', f'{value}_human']].values))
        ratings = np.zeros((len(merged_df), n_categories))
        
        # Fill ratings matrix - 1 for the category chosen by each rater
        for i, row in merged_df.iterrows():
            ratings[i, int(row[f'{value}_gpt'])-1] += 1
            ratings[i, int(row[f'{value}_human'])-1] += 1
            
        # Calculate Randolph's kappa using fleiss_kappa
        kappa = fleiss_kappa(ratings, method='randolph')
        kappas.append(kappa)
        
        print(f"\nAnnotator {annotator}:")
        print(f"Agreement percentage: {agreement:.1f}%")
        print(f"Randolph's kappa: {kappa:.3f}")
        
        # Calculate standard error and p-value for kappa
        n = len(merged_df)  # number of subjects
        se = np.sqrt((2 * (1 - kappa)) / n)
        z = kappa / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"p-value: {p_value:.3f}")

    mean_agreement = np.mean(agreements)
    mean_kappa = np.mean(kappas)
    
    # print(f"\nOverall metrics:")
    # print(f"Mean agreement percentage: {mean_agreement:.1f}%")
    # print(f"Mean Randolph's kappa: {mean_kappa:.3f}")
    return mean_kappa

def cal_pairwise_accuracy(llm_df, human_df, value):
    """
    Calculate true positive rate between LLM and human annotations.
    True positives are cases where both LLM and human annotated as 1.
    
    Args:
        llm_df: DataFrame containing LLM annotations
        human_df: DataFrame containing human annotations 
        value: Column name of the value to compare
        
    Returns:
        DataFrame with true positive rate results
    """
    
    results = []
    
    # Get unique annotators from human dataframe
    all_annotators = human_df['annotator'].unique()
    
    for annotator in all_annotators:
        # Get data for current annotator
        annotator_df = human_df[human_df['annotator'] == annotator]
        
        # Merge GPT annotations with current annotator's data
        merged_df = pd.merge(llm_df[['question', f'{value}']], 
                           annotator_df[['question', f'{value}']], 
                           on='question',
                           suffixes=('_gpt', '_human'))
        
        # Skip if no overlapping data
        if len(merged_df) == 0:
            continue
            
        # Calculate true positive rate
        true_positives = ((merged_df[f'{value}_gpt'] == 1) & (merged_df[f'{value}_human'] == 1)).sum()
        actual_positives = (merged_df[f'{value}_human'] == 1).sum()
        
        if actual_positives > 0:
            true_positive_rate = true_positives / actual_positives
        else:
            true_positive_rate = 0
        
        results.append({
            'annotator': annotator,
            'true_positive_rate': true_positive_rate,
            'true_positives': true_positives,
            'actual_positives': actual_positives,
            'n_samples': len(merged_df)
        })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    # print(f"\nTrue Positive Rate Analysis for {value}:")
    # print(f"Mean TPR: {results_df['true_positive_rate'].mean():.3f}")
    # print(f"Std TPR: {results_df['true_positive_rate'].std():.3f}")
    # print("\nPer-annotator results:")
    # print(results_df)
    
    return results_df['true_positive_rate'].mean()


def compare_human_llm(Trial, all_annotators):
    bat = compare_human_llm_corrf(Trial, all_annotators,'bat')
    print("----")
    pat = compare_human_llm_corrf(Trial, all_annotators,'pat')
    print("----")
    nrbat = compare_human_llm_corrf(Trial, all_annotators,'net_ZNRBaT')
    print("----")
    compare_human_llm_corrf(Trial, all_annotators,'NRA')

    print("\n=== Agreement Analysis for outcome ===") 
    outcome = cal_agreement_cohen_llm(Trial, all_annotators, 'outcome_value')

    # print("\n=== Corr Fisher ===") if want to do significance test over three annotators

    # bat = compare_human_llm_corrf_fisher(Trial, all_annotators,'bat')
    # print("----")
    # pat = compare_human_llm_corrf_fisher(Trial, all_annotators,'pat')
    # print("----")
    # nrbat = compare_human_llm_corrf_fisher(Trial, all_annotators,'net_ZNRBaT')
    # print("----")
    # compare_human_llm_corrf_fisher(Trial, all_annotators,'NRA')

    print("\n=== Agreement Analysis for Commitment ===") 
    commit = cal_agreement_cohen_llm(Trial, all_annotators, 'Committment_value')

    print("\n=== Agreement Analysis for Relevance Binary ===") 
    relevance = cal_agreement_randolf_llm(Trial, all_annotators, 'relevance_binary')
    print("\n=== Agreement Analysis for Manner Binary ===")
    manner = cal_agreement_randolf_llm(Trial, all_annotators, 'manner_binary')
    print("\n=== Agreement Analysis for Quality Binary ===")
    quality = cal_agreement_randolf_llm(Trial, all_annotators, 'quality_binary')
    
    print("\n=== Agreement Analysis for CONSISTENCY Binary ===") 
    consistency = cal_pairwise_accuracy(Trial, all_annotators, 'consistency_value')
    return bat, pat, nrbat, outcome, commit, relevance, manner, quality, consistency

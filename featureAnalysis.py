import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


data = pd.read_csv("Final_Balanced_Data.csv" )

winners = data[data["award_status"] == "winner"].drop(columns=["award_status"])
nominees = data[data["award_status"] == "nominee"].drop(columns=["award_status"])

fisherScores = {}
t_test_results = {}

for feature in winners.columns:
    
    mu_winner = winners[feature].mean()
    mu_nominee = nominees[feature].mean()
    var_winner = winners[feature].var()
    var_nominee = nominees[feature].var()
    
    # handle edge cases where variance is zero to avoid division by zero
    if var_winner + var_nominee == 0:
        fisher_score = 0  
    else:
        fisher_score = (mu_winner - mu_nominee)**2 / (var_winner + var_nominee)
    
   
    fisherScores[feature] = fisher_score

    t_stat, p_value = ttest_ind(winners[feature], nominees[feature], equal_var=False)
    t_test_results[feature] = p_value

fisherScores_df = pd.DataFrame(list(fisherScores.items()), columns=["Feature", "Fisher_Score"])
fisherScores_df = fisherScores_df.sort_values(by="Fisher_Score", ascending=False)

t_test_results_df = pd.DataFrame(list(t_test_results.items()), columns=["Feature", "P_Value"])
t_test_results_df = t_test_results_df.sort_values(by="P_Value")

results_df = fisherScores_df.merge(t_test_results_df, on="Feature")

output_path = "feature_analysis_results.csv"
results_df.to_csv(output_path, index=False)
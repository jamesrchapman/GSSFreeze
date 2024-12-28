import json
import pandas as pd
import numpy as np
import os
import ast
from statsmodels.stats.multitest import multipletests

from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to the CSV file
gss_csv_file = 'GSS_cumulative_data.csv'

variables=['year', 'id_', 'hrs1', 'hrs2', 'wrkslf', 'occ10', 'sphrs1', 'sphrs2', 'happy', 'hapmar', 'joblose', 'satjob', 'class_', 'satfin', 'finalter', 'tvhours', 'wrktype', 'yearsjob', 'waypaid', 'wrksched', 'moredays', 'mustwork', 'wrkhome', 'whywkhme', 'famwkoff', 'wkvsfam', 'famvswk', 'hrsrelax', 'secondwk', 'learnnew', 'workfast', 'overwork', 'respect', 'trustman', 'proudemp', 'supcares', 'condemnd', 'promtefr', 'cowrkint', 'jobsecok', 'manvsemp', 'trynewjb', 'health1', 'mntlhlth', 'spvtrfair', 'slpprblm', 'satjob1', 'hyperten', 'stress', 'realinc', 'ballot', 'sei10']
# predictor=['hrs1', 'hrs2', 'wrkslf', 'occ10', 'sphrs1', 'sphrs2', 'hapmar', 'joblose', 'satjob', 'class_', 'satfin', 'finalter', 'tvhours', 'wrktype', 'yearsjob', 'waypaid', 'wrksched', 'moredays', 'mustwork', 'wrkhome', 'whywkhme', 'famwkoff', 'wkvsfam', 'famvswk', 'hrsrelax', 'secondwk', 'learnnew', 'workfast', 'overwork', 'respect', 'trustman', 'proudemp', 'supcares', 'condemnd', 'promtefr', 'cowrkint', 'jobsecok', 'manvsemp', 'trynewjb', 'health1', 'mntlhlth', 'spvtrfair', 'slpprblm', 'satjob1', 'hyperten', 'stress', 'realinc', 'ballot', 'sei10']
dependent=['happy', 'health1', 'mntlhlth', 'slpprblm', 'hyperten', 'stress']
def main():
        
    # Load the CSV file into a pandas DataFrame
    try:
        print("Attempting to load the CSV file...")
        df = pd.read_csv(gss_csv_file)
        print("Successfully loaded GSS cumulative data.")
        print("DataFrame preview:")
        print(df.head())
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        df = None


    # Analyze intersectional variables if DataFrame is available
    if df is not None:
        try:
            print("Selecting key variables for analysis...")
            # Pick 3 key variables to examine based on potential insights into freeze and stress
            variables = ['stress', 'joblose', 'satfin']

            # Define the correct order for each ordinal variable
            stress_order = ["Never", "Hardly ever", "Sometimes", "Often", "Always"]
            joblose_order = ["Not likely", "Not too likely", "Fairly likely", "Very likely"]
            satfin_order = ["Not satisfied at all", "More or less satisfied", "Pretty well satisfied"]

            # Replace inapplicable and non-answer values with NaN for analysis
            df_subset = df[variables].replace({
                r'\.i:.*': np.nan, 
                r'\.n:.*': np.nan, 
                r'\.d:.*': np.nan, 
                r'\.s:.*': np.nan,
                r'\.y:.*': np.nan
            }, regex=True)

            # Convert ordinal columns to categorical types with the defined order
            df_subset['stress'] = pd.Categorical(df_subset['stress'], categories=stress_order, ordered=True)
            df_subset['joblose'] = pd.Categorical(df_subset['joblose'], categories=joblose_order, ordered=True)
            df_subset['satfin'] = pd.Categorical(df_subset['satfin'], categories=satfin_order, ordered=True)

            # Drop rows with any NaN values to only keep rows with complete data
            df_filtered = df_subset.dropna()

            # Display the filtered data
            print("Filtered DataFrame preview:")
            print(df_filtered.head())

            # Convert categorical data to numeric rankings for correlation analysis
            df_numeric = df_filtered.apply(lambda x: x.cat.codes)

            # Perform Spearman correlation analysis on the filtered data
            print("Calculating Spearman correlation between selected variables...")
            spearman_corr, p_values = spearmanr(df_numeric, axis=0) 
            spearman_corr_df = pd.DataFrame(spearman_corr, index=variables, columns=variables)
            print("Spearman Correlation Matrix:")
            print(spearman_corr_df)

            # Save the Spearman correlation matrix to a CSV file for reference
            spearman_corr_df.to_csv('GSS_spearman_correlation.csv')
            print("Spearman correlation matrix saved to 'GSS_spearman_correlation.csv'")

            # Calculate p-values and apply correction for multiple testing
            print("Calculating p-values and adjusting for multiple comparisons...")
            p_values_df = pd.DataFrame(p_values, index=variables, columns=variables)
            p_values_flattened = p_values[np.triu_indices_from(p_values, k=1)]  # Extract upper triangle values
            _, p_values_corrected, _, _ = multipletests(p_values_flattened, method='bonferroni')

            # Convert corrected p-values back into a DataFrame format
            p_values_corrected_matrix = np.zeros_like(p_values)
            p_values_corrected_matrix[np.triu_indices_from(p_values, k=1)] = p_values_corrected
            p_values_corrected_matrix += p_values_corrected_matrix.T
            p_values_corrected_df = pd.DataFrame(p_values_corrected_matrix, index=variables, columns=variables)
            p_values_corrected_df.to_csv('GSS_corrected_p_values.csv')
            print("Corrected p-values matrix saved to 'GSS_corrected_p_values.csv'")

            # Plot the Spearman correlation matrix with p-values indicated
            plt.figure(figsize=(10, 8))
            sns.heatmap(spearman_corr_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
            plt.title('Spearman Correlation Matrix for Selected Variables')
            plt.savefig('GSS_spearman_correlation_heatmap.png')
            plt.show()
            print("Spearman correlation heatmap saved to 'GSS_spearman_correlation_heatmap.png'")

            # Display p-values as a heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(p_values_corrected_df, annot=True, cmap='viridis', linewidths=0.5, fmt='.2e')
            plt.title('Corrected p-values for Spearman Correlation')
            plt.savefig('GSS_corrected_p_values_heatmap.png')
            plt.show()
            print("Corrected p-values heatmap saved to 'GSS_corrected_p_values_heatmap.png'")

        except KeyError as e:
            print(f"Error: Some variables are not found in the DataFrame: {e}")
        except Exception as e:
            print(f"Error analyzing selected variables: {e}")

if __name__ == '__main__':
    main()

    """
I'm looking at a bunch of variables from a survey and how they're going to predict these dependent variables which are also survey questions. 
So, the issue I'm running into is that the survey questions are different formats right - like what I want to do is find all of the variables that are convincingly associated with variation in the dependent variables (independent of each other). 
So I need help figuring out how to find significance between different types of survey questions and the predictor variables - e.g. how do I associate nominal independent with ordinal dependent. How do I associate ordinal with nominal, ratio with ordinal, all combinations. 

Then I want to hold onto those variables for each predictor. 
Then the real kicker is looking to associate each validated predictor variable with the dependent variables in all pairs over those variables. I want to look at some kind of regression model and to examine the bayes information criterion or AIC or whatever. Something like that, to see if the interaction of the pairs explains a good deal more of the variance than independently. 
I don't know what kind of regression to do for the different question formats again! how do you predict ordinal variables with a pair of nominal and ordinal inputs? etc. 

and then there's a validation issue where I should probably look at random data to make sure it's not just an artifact of the model. 




Regression by Dependent Variable Type
Ordinal Dependent Variable: Use Ordinal Logistic Regression. This can handle both nominal and ratio predictors, as well as interaction terms.
Nominal Dependent Variable: Use Multinomial Logistic Regression.
Ratio Dependent Variable: Use Linear Regression or Generalized Linear Models (GLMs) for non-normal distributions.

Including Interaction Terms
To evaluate interactions:

Create interaction terms by multiplying predictors
Use a regression model suited to your dependent variable (e.g., ordinal logistic regression for ordinal outcomes).
Compare model fits using Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC). Models with lower AIC/BIC are preferred, but ensure they’re not overfitting.



    """

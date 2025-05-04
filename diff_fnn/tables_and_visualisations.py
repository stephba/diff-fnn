import torch
import pandas as pd
import numpy as np
import os
from diff_fnn.utils import logging_decorator, Config, highlight_values, store_df_as_html, store_df_as_latex, store_flowchart
from scipy.stats import wilcoxon

P_VALUE_THRESHOLD = 0.01

@logging_decorator("Generate tables and visualisations")
def generate_tables_and_visualisations(config: Config, n_repeat_exp):
    for i in range(n_repeat_exp):
        final_fuzzy_weights_df = pd.read_csv(os.path.join(config.results_path, f"fuzzy_weights_run_{i}.csv"), index_col=0)
        final_fuzzy_weights_styler = final_fuzzy_weights_df.style.apply(highlight_values)
        store_df_as_html(
            final_fuzzy_weights_styler, 
            os.path.join(config.results_path, f"fuzzy_weights_run_{i}.html")
        )
        store_df_as_latex(
            final_fuzzy_weights_styler, 
            os.path.join(config.results_path, f"fuzzy_weights_run_{i}.tex")
        )

        # Rule extraction:
        extraction_threshold = 0.1
        variables_names = final_fuzzy_weights_df.columns
        model = torch.load(os.path.join(config.results_path, f"model_run_{i}.pth"), weights_only=False)
        model.eval()
        final_fuzzy_weights_extracted_rules = model.get_fuzzy_weights(extraction_threshold=extraction_threshold).detach().cpu()
        rules = [variables_names] * final_fuzzy_weights_extracted_rules.shape[0]
        weights = np.array(final_fuzzy_weights_extracted_rules)
        rules = []
        weights = []
        for rule_weights in final_fuzzy_weights_extracted_rules:
            n = np.array(variables_names)[rule_weights > 0.0]
            w = rule_weights[rule_weights > 0.0]
            sort_indices = torch.flip(torch.argsort(w), dims=[0])  # sort by weights decreasing
            n = np.atleast_1d(n[sort_indices])
            w = w[sort_indices]
            rules.append(n)
            weights.append(np.array(w))
        store_flowchart(rules, weights, os.path.join(config.results_path, f"extracted_rules_threshold_{extraction_threshold}_run_{i}.pdf"))

    generate_results_table(config.results_path)

def generate_results_table(results_path):
    results_df = pd.read_csv(os.path.join(results_path, f"results.csv"), index_col=0)
    results_df = results_df.map(lambda x: x.replace('nan', 'np.nan') if isinstance(x, str) else x)  # replace nans with np.nan
    results_df = results_df.map(lambda values: np.array(eval(values)))  # convert the strings to numpy arrays

    mean_df = results_df.map(lambda values: np.mean(values))
    best_models_df = mean_df.eq(mean_df.max())
    second_highest_values = mean_df.apply(lambda col: col.nlargest(2).iloc[-1])
    second_best_models_df = mean_df.apply(lambda col: col == second_highest_values[col.name])
    our_approach_results = results_df.loc['Our Model']
    def wilcoxon_rowwise(row1, row2):
        # avoid division by zero
        if row1.equals(row2):
            return pd.Series([1.0] * len(row1), index=row1.index)
        p_values = [wilcoxon(row1[col], row2[col], alternative='two-sided').pvalue for col in row1.index]
        return pd.Series(p_values, index=row1.index)
    significant_difference_df = results_df.apply(
        lambda row: wilcoxon_rowwise(row, our_approach_results), 
        axis=1
    )
    significant_difference_df = significant_difference_df < P_VALUE_THRESHOLD

    latex_df = results_df.copy()
    latex_df = latex_df.map(lambda values: f'{np.mean(values):.3f}_{{({np.std(values):.3f})}}'.replace('0.', '.'))
    latex_df = latex_df.where(~best_models_df, latex_df.map(lambda string: f'\\mathbf{{{string}}}'))
    latex_df = latex_df.where(~second_best_models_df, latex_df.map(lambda string: f'\\underline{{{string}}}'))
    latex_df = latex_df.map(lambda string: f'${string}$')
    latex_df = latex_df.where(~significant_difference_df, latex_df.map(lambda string: f'{string}*'))

    html_df = results_df.copy()
    html_df = html_df.map(lambda values: f'{np.mean(values):.3f}<sub>({np.std(values):.3f})</sub>'.replace('0.', '.'))
    html_df = html_df.where(~best_models_df, html_df.map(lambda string: f'<b>{string}</b>'))
    html_df = html_df.where(~second_best_models_df, html_df.map(lambda string: f'<u>{string}</u>'))
    html_df = html_df.where(~significant_difference_df, html_df.map(lambda string: f'{string}*'))

    store_df_as_html(html_df, os.path.join(results_path, f"results.html"))
    store_df_as_latex(latex_df, os.path.join(results_path, f"results.tex"))

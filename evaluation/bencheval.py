"""
Clembench Evaluation

This script produces the main table with benchmark results, for all models
and games in the results/ directory structure.

"""
from pathlib import Path

import pandas as pd

import evaluation.evalutils as utils
import clemgame.metrics as clemmetrics

PATH = Path(utils.RESULTS_DIR)

scores = utils.load_scores()

df_episode_scores = utils.build_df_episode_scores(scores)

# Create the PLAYED variable
aux = df_episode_scores[df_episode_scores["metric"] == "Aborted"].copy()
aux["metric"] = clemmetrics.METRIC_PLAYED
aux["value"] = 1 - aux["value"]
# We need ignore_index=True to reset the indices (otherwise we have duplicates)
df_episode_scores = pd.concat([df_episode_scores, aux], ignore_index=True)

def save_clem_table(df):
    """Create table with % played and main score."""
    df_aux = df[df['metric'].isin(utils.MAIN_METRICS)]
    categories = ['game', 'model', 'metric']
    # mean over all experiments
    df_mean = (df_aux.groupby(categories)
                     .mean(numeric_only=True)
                     .rename({'value': 'mean'}, axis=1)
                     .reset_index())
    df_mean.loc[df_mean.metric == clemmetrics.METRIC_PLAYED, 'mean'] *= 100
    df_mean = df_mean.round(2)
    # standard deviation over all experiments
    df_std = (df_aux.groupby(categories)
                    .std(numeric_only=True)
                    .rename({'value': 'std'}, axis=1)
                    .reset_index()
                    .round(2))
    df_std.loc[df_std.metric == clemmetrics.METRIC_PLAYED, 'std'] = '-'

    # average across all activities (i.e. mean by row)
    df_mean = df_mean.pivot(columns=['game'], index=['model', 'metric'])
    df_mean['all'] = df_mean.mean(numeric_only=True, axis=1)
    df_std = df_std.pivot(columns=['game'], index=['model', 'metric'])

    # double check the order
    assert all(df_mean.index == df_std.index)
    pairs = zip(df_mean.columns[:-1], df_std.columns)
    assert all(mean_col[1] == std_col[1] for mean_col, std_col in pairs)

    # merge both, putting std in parenthesis
    df_aux = df_mean['mean'].astype(str) + ' (' + df_std['std'].astype(str)+')'
    df_clem = pd.concat([df_mean['all'].round(2), df_aux], axis=1)

    df_clem.to_csv(PATH / 'bench-results-table.csv')
    df_clem.to_html(buf=PATH / 'bench-results-table.html')
    df_clem.to_latex(buf=PATH / 'bench-results-table.tex',
                     float_format=utils.FLOAT_FORMAT, na_rep='n/a')

    return df_clem


def save_clem_score_table(df_paper: pd.DataFrame) -> None:
    """Create a table with the clem score for each model."""
    df_aux = (df_paper['all'].to_frame()
                             .reset_index()
                             .pivot(index=['model'], columns=['metric']))
    df_aux['clemscore'] = ((df_aux[('all', 'Played')] / 100)
                           * df_aux[('all', 'Main Score')])
    df_aux = df_aux['clemscore'].to_frame().reset_index()

    df_aux.to_csv(PATH / 'clem-table.csv')
    df_aux.to_html(buf=PATH / 'clem-table.html')
    df_aux.to_latex(buf=PATH / 'clem-table.tex',
                     float_format=utils.FLOAT_FORMAT, na_rep='n/a')

    return df_aux

df_clem = save_clem_table(df_episode_scores)
_ = save_clem_score_table(df_clem)

print(f'\n Saved tables into {PATH}/.')
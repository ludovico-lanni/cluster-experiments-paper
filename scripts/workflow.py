# %%
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from sklearn.ensemble import HistGradientBoostingRegressor

from cluster_experiments import (
    NonClusteredSplitter,
    ClusteredSplitter,
    SwitchbackSplitter,
    ConstantWashover,
    ConstantPerturbator,
    NormalPerturbator,
    ClusteredOLSAnalysis,
    OLSAnalysis,
    PowerAnalysis,
    NormalPowerAnalysis,
    AnalysisPlan,
    SimpleMetric,
    Dimension,
    Variant,
    HypothesisTest
)

# %% [markdown]
# # Data Generation

# %%
def generate_data(sample_size=50_000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)

    N_CUSTOMERS = sample_size
    MIN_DATE = pd.Timestamp('2021-01-01')
    MAX_DATE = MIN_DATE + pd.Timedelta(days=360)
    
    customer_ids = rng.choice(np.arange(1e6, 1e7).astype(int), size=N_CUSTOMERS, replace=False)
    beta_samples = rng.beta(2, 5, size=N_CUSTOMERS)
    customer_mean_time_between_orders = 7 + (60 - 7) * (1 - beta_samples) # Every customer has a different mean time between orders
    customer_average_spend = rng.uniform(20, 200, size=N_CUSTOMERS) # Every customer has a different average spend
    start_times = rng.choice(pd.date_range(MIN_DATE.replace(year=2020), MAX_DATE.replace(year=2020)), size=N_CUSTOMERS, replace=True) # Every customer places their first order sometime in 2020
    orders = []
    for customer_id, start_time, time_between_orders, avg_spend in zip(
        customer_ids,
        start_times,
        customer_mean_time_between_orders,
        customer_average_spend
    ):
        order_time = start_time
        while order_time < MAX_DATE: # Generate orders until MAX_DATE
            order_time = order_time + pd.Timedelta(days=rng.exponential(scale=time_between_orders))
            if order_time >= MIN_DATE: # Only keep orders after MIN_DATE
                order_value = np.round(rng.normal(loc=avg_spend, scale=avg_spend * 0.3), 2)
                orders.append({
                    'customer_id': customer_id,
                    'order_time': order_time,
                    'order_value': max(0, order_value)
                })
    data = (
        pd.DataFrame(orders)
        .sample(frac=1, random_state=seed, replace=False)
        .reset_index(drop=True)
        .assign(
            time_index = lambda df: (df['order_time'] - df['order_time'].min()).dt.days
        )
    )

    return data

# %%
data = generate_data(
    sample_size=50_000,
    seed=42
)

# %%
data.describe(include='all').T

# %% [markdown]
# # Clustered Design

# %% [markdown]
# ### Helper functions

# %%
def get_cupac_df(
    pre_df:pd.DataFrame,
    post_df:pd.DataFrame,
    cluster_cols: list[str]
    ) -> pd.DataFrame:
    agg_pre_df = (
        pre_df
        .groupby(
            by = cluster_cols,
            as_index = False
        )
        .agg(
            pre_n_orders = ('order_time', 'count'),
            pre_aov = ('order_value', 'mean')
        )
    )
    cupac_df = (
        pd.merge(
            left = post_df,
            right = agg_pre_df,
            on = cluster_cols
        )
        .fillna(0)
    )
    return cupac_df

# %% [markdown]
# ### Data definition

# %%
cluster_cols = ['customer_id']
feature_cols = [
    'pre_n_orders',
    'pre_aov'
]
target_col = 'order_value'
time_col = 'order_time'
average_effect = 1

# %%
cupac_training_data = (
    data
    .query('time_index < 90')
    .reset_index(drop=True)
)
print(f'{cupac_training_data.shape=}')

pre_experiment_data = (
    data
    .query('90 <= time_index < 180')
    .reset_index(drop=True)
)
print(f'{pre_experiment_data.shape=}')

experiment_design_data = (
    data
    .query('180 <= time_index < 270')
    .reset_index(drop=True)
)
print(f'{experiment_design_data.shape=}')

# %% [markdown]
# ## Code Chunk: Simulation-based power estimation under a clustered design

# %%
splitter = ClusteredSplitter(
    cluster_cols=cluster_cols,
)
perturbator = ConstantPerturbator(
    target_col=target_col
)
analysis = ClusteredOLSAnalysis(
    cluster_cols=cluster_cols,
    target_col=target_col
)

# %%
sim_power_analysis = PowerAnalysis(
    perturbator=perturbator,
    splitter=splitter,
    analysis=analysis,
    target_col=target_col,
    seed = 42
)
start = time.time()
power_sim = sim_power_analysis.power_analysis(
    df=experiment_design_data,
    average_effect= average_effect,
    n_simulations = 100
)
end = time.time()
duration = end - start

# %%
print(f'Estimated Power (Simulation): {power_sim:.3f} in {duration:.2f} seconds')

# %% [markdown]
# ## Code Chunk: Simulation-based power estimation under a clustered design with CUPAC

# %%
cupac_training_data = get_cupac_df(
    pre_df = cupac_training_data,
    post_df = pre_experiment_data,
    cluster_cols=cluster_cols
)
cupac_experiment_data = get_cupac_df(
    pre_df = pre_experiment_data,
    post_df = experiment_design_data,
    cluster_cols=cluster_cols
)

# %%
analysis = ClusteredOLSAnalysis(
    cluster_cols=cluster_cols,
    target_col=target_col,
    covariates=['estimate_' + target_col]
)

# %%
sim_power_analysis_cupac = PowerAnalysis(
    perturbator=perturbator,
    splitter=splitter,
    analysis=analysis,
    target_col=target_col,
    cupac_model = HistGradientBoostingRegressor(),
    features_cupac_model=feature_cols
)

start = time.time()
power_sim_cupac = sim_power_analysis_cupac.power_analysis(
    df=cupac_experiment_data,
    average_effect= average_effect,
    n_simulations = 100,
    pre_experiment_df=cupac_training_data
)
end = time.time()
duration = end - start

# %%
print(f'Estimated Power (Simulation with CUPAC): {power_sim_cupac:.3f} in {duration:.2f} seconds')

# %% [markdown]
# # Non-Clustered Design

# %% [markdown]
# ### Helper function

# %%
def get_cupac_df(
    pre_df:pd.DataFrame,
    post_df:pd.DataFrame,
    cluster_cols: list[str]
    ) -> pd.DataFrame:
    cupac_df = (
        pd.merge(
            left = post_df,
            right = pre_df.rename(columns={'n_orders': 'pre_n_orders'}),
            on = cluster_cols,
            how='left',
            suffixes = ('', '_pre')
        )
        .fillna(0)
    )
    return cupac_df

# %% [markdown]
# ### Data definitions

# %%
feature_cols = [
    'pre_n_orders'
]
target_col = 'n_orders'
time_col = 'first_order_time'
average_effect = 0.01

# %%
cupac_training_data = (
    data
    .query('time_index < 90')
    .groupby(
        by = ['customer_id'],
        as_index = False
    )
    .agg(
        n_orders = ('order_time', 'count'),
        first_order_time = ('order_time', 'min')
    )
    .astype({'n_orders': float})
    .reset_index(drop=True)
)
print(f'{cupac_training_data.shape=}')

pre_experiment_data = (
    data
    .query('90 <= time_index < 180')
    .groupby(
        by = ['customer_id'],
        as_index = False
    )
    .agg(
        n_orders = ('order_time', 'count'),
        first_order_time = ('order_time', 'min')
    )
    .astype({'n_orders': float})
    .reset_index(drop=True)
)
print(f'{pre_experiment_data.shape=}')

experiment_design_data = (
    data
    .query('180 <= time_index < 270')
    .groupby(
        by = ['customer_id'],
        as_index = False
    )
    .agg(
        n_orders = ('order_time', 'count'),
        first_order_time = ('order_time', 'min')
    )
    .astype({'n_orders': float})
    .reset_index(drop=True)
)
print(f'{experiment_design_data.shape=}')

# %% [markdown]
# ## Code Chunk: Comparing power estimations between simulation and analytical approaches

# %%
splitter = NonClusteredSplitter()
perturbator = ConstantPerturbator(
    target_col=target_col
)
analysis = OLSAnalysis(
    target_col=target_col
)

sim_power_analysis = PowerAnalysis(
    splitter=splitter,
    perturbator=perturbator,
    analysis=analysis,
    target_col=target_col,
    n_simulations=100
)

start = time.time()
sim_power = sim_power_analysis.power_analysis(
    df=experiment_design_data,
    average_effect=average_effect,
    n_simulations=100
)
end = time.time()
sim_duration = end - start

# %%
print(f'Estimated Power (Simulation): {sim_power:.3f} in {sim_duration:.2f} seconds')

# %%
normal_power_analysis = NormalPowerAnalysis(
    splitter=splitter,
    analysis=analysis,
    target_col=target_col,
    time_col=time_col
)
start = time.time()
normal_power = normal_power_analysis.power_analysis(
    df=experiment_design_data,
    average_effect=average_effect,
    n_simulations=1
)
end = time.time()
normal_duration = end - start

# %%
print(f'Estimated Power (Analytical): {normal_power:.3f} in {normal_duration:.2f} seconds')

# %% [markdown]
# ## Code Chunk: Using the `mde_time_line` method

# %%
mde_time_line = normal_power_analysis.mde_time_line(
    df = experiment_design_data,
    experiment_length = np.arange(7, 7*12, 7),
    alpha = 0.05,
    powers=[0.8]
)
mde_time_line_df = pd.DataFrame(mde_time_line)
sns.lineplot(
    data=mde_time_line_df,
    x='experiment_length',
    y='mde',
    marker='o'
)

# %% [markdown]
# # Switchback

# %% [markdown]
# ### Data Definition

# %%
target_col = 'order_value'
time_col = 'order_time'
average_effect = 1

# %%
experiment_design_data = (
    data
    .query('180 <= time_index < 270')
    .reset_index(drop=True)
)
print(f'{experiment_design_data.shape=}')

# %% [markdown]
# ## Code Chunk: Switchback design with simulation-based power estimation

# %%
washover = ConstantWashover(
    washover_time_delta = '30m'
)
splitter = SwitchbackSplitter(
    time_col = time_col,
    switch_frequency='1h',
    cluster_cols = ['order_time'],
    washover=washover
)
perturbator = ConstantPerturbator(
    target_col=target_col
)
analysis = ClusteredOLSAnalysis(
    cluster_cols=['order_time'],
    target_col=target_col
)
switchback_power_analysis = PowerAnalysis(
    splitter=splitter,
    perturbator=perturbator,
    analysis=analysis,
    target_col=target_col
)
switchback_power = switchback_power_analysis.power_analysis(
    df=experiment_design_data,
    average_effect= average_effect,
    n_simulations = 100
)
print(f'Estimated Power (Switchback): {switchback_power:.3f}')

# %% [markdown]
# # Experiment Analysis

# %% [markdown]
# ### Helper function

# %%
def get_cupac_df(
    pre_df:pd.DataFrame,
    post_df:pd.DataFrame,
    cluster_cols: list[str]
    ) -> pd.DataFrame:
    agg_pre_df = (
        pre_df
        .groupby(
            by = cluster_cols,
            as_index = False
        )
        .agg(
            pre_n_orders = ('order_time', 'count'),
            pre_aov = ('order_value', 'mean')
        )
    )
    cupac_df = (
        pd.merge(
            left = post_df,
            right = agg_pre_df,
            on = cluster_cols
        )
        .fillna(0)
    )
    return cupac_df

# %% [markdown]
# ### Data Definition

# %%
cluster_cols = ['customer_id']
feature_cols = [
    'pre_n_orders',
    'pre_aov'
]
target_col = 'order_value'
time_col = 'order_time'
average_effect = 1

# %%
pre_experiment_data = (
    data
    .query('90 <= time_index < 180')
    .reset_index(drop=True)
)
print(f'{pre_experiment_data.shape=}')

experiment_design_data = (
    data
    .query('180 <= time_index < 270')
    .reset_index(drop=True)
)
print(f'{experiment_design_data.shape=}')

_experiment_analysis_data = (
    data
    .query('270 <= time_index < 360')
    .reset_index(drop=True)
)
print(f'{_experiment_analysis_data.shape=}')

# %% [markdown]
# ### Data pre-processing

# %%
city_assigner = ClusteredSplitter(
    cluster_cols=cluster_cols,
    treatment_col='city_code',
    treatments=['MAD', 'BCN'],
    splitter_weights=[0.7, 0.3]
)
treatment_assigner = ClusteredSplitter(
    cluster_cols=cluster_cols,
    treatment_col='variant',
    treatments=['A', 'B'],
)
treatment_perturbator = NormalPerturbator(
    target_col = target_col,
    treatment_col='variant',
    treatment='B',
    average_effect=1,
    scale=1
)
    
experiment_analysis_data = (
    _experiment_analysis_data
    .pipe(city_assigner.assign_treatment_df)
    .pipe(treatment_assigner.assign_treatment_df)
    .pipe(treatment_perturbator.perturbate)
)

# %% [markdown]
# ## Code Chunk: Experiment analysis example

# %%
metric__order_value = SimpleMetric(
    alias = 'AOV',
    name = 'order_value'
)

dimension__city_code = Dimension(
    name = 'city_code',
    values = ['MAD', 'BCN'],
)

variant__control = Variant(
    name = 'A',
    is_control = True
)
variant__treatment = Variant(
    name = 'B',
    is_control = False
)

test__order_value = HypothesisTest(
    metric = metric__order_value,
    dimensions=[dimension__city_code],
    analysis_type = 'clustered_ols',
    analysis_config = {
        'target_col': target_col,
        'cluster_cols': cluster_cols,
        'covariates': ['estimate_' + target_col]
    },
    cupac_config = {
        'cupac_model': HistGradientBoostingRegressor(),
        'target_col': target_col,
        'features_cupac_model': feature_cols
    }
)

analysis_plan = AnalysisPlan(
    tests = [test__order_value],
    variants = [variant__control, variant__treatment],
    variant_col = 'variant',
    alpha = 0.05
)

analysis_results = (
    analysis_plan
    .analyze(
        exp_data = get_cupac_df(
            pre_df = experiment_design_data,
            post_df = experiment_analysis_data,
            cluster_cols=cluster_cols
        ),
        pre_exp_data = get_cupac_df(
            pre_df = pre_experiment_data,
            post_df = experiment_design_data,
            cluster_cols=cluster_cols
        )
    )
    .to_dataframe()
)   

# %%
print(analysis_results)

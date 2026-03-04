# %%
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'serif',
    'lines.marker': 'o',
    'lines.markersize': 5,
    'lines.linewidth': 1,
    'lines.color': 'black',
})

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
    MAX_DATE = pd.Timestamp('2021-12-31')
    
    customer_ids = rng.choice(np.arange(1e6, 1e7).astype(int), size=N_CUSTOMERS, replace=False)
    customer_city_codes = rng.choice(['MAD', 'BCN'], size=N_CUSTOMERS, replace=True, p=[0.7, 0.3])
    beta_samples = rng.beta(2, 5, size=N_CUSTOMERS)
    customer_mean_time_between_orders = 7 + (60 - 7) * (1 - beta_samples) # Every customer has a different mean time between orders
    customer_average_spend = rng.uniform(20, 200, size=N_CUSTOMERS) # Every customer has a different average spend
    MIN_START_TIME = MIN_DATE.replace(year=2020)
    MAX_START_TIME = MAX_DATE.replace(year=2020)
    start_times = rng.choice(pd.date_range(MIN_START_TIME, MAX_START_TIME), size=N_CUSTOMERS, replace=True) # Every customer places their first order sometime in 2020
    orders = []
    for customer_id, customer_city_code, start_time, time_between_orders, avg_spend in zip(
        customer_ids,
        customer_city_codes,
        start_times,
        customer_mean_time_between_orders,
        customer_average_spend
    ):
        order_time = start_time
        while order_time < MAX_DATE: # Generate orders until MAX_DATE
            if order_time >= MIN_DATE: # Only keep orders after MIN_DATE
                order_value = np.round(rng.normal(loc=avg_spend, scale=avg_spend * 0.3), 2)
                orders.append({
                    'customer_id': customer_id,
                    'city_code': customer_city_code,
                    'order_time': order_time,
                    'order_value': max(0, order_value)
                })
            order_time = order_time + pd.Timedelta(days=rng.exponential(scale=time_between_orders))

    data = (
        pd.DataFrame(orders)
        .sample(frac=1, random_state=seed, replace=False) # This just shuffles the data
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
print(data.describe(include='all').T)

# %% [markdown]
# # Clustered Design

# %% [markdown]
# ### Data definition

# %%
data_0_to_90 = (
    data
    .query('time_index < 90')
    .reset_index(drop=True)
)
print(f'{data_0_to_90.shape=}')

data_90_to_180 = (
    data
    .query('90 <= time_index < 180')
    .reset_index(drop=True)
)
print(f'{data_90_to_180.shape=}')

data_180_to_270 = (
    data
    .query('180 <= time_index < 270')
    .reset_index(drop=True)
)
print(f'{data_180_to_270.shape=}')

data_270_to_365 = (
    data
    .query('270 <= time_index < 360')
    .reset_index(drop=True)
)
print(f'{data_270_to_365.shape=}')

# %% [markdown]
# ## Code Chunk: Simulation-based power estimation under a clustered design

# %%

# %%
splitter = ClusteredSplitter(
    cluster_cols=['customer_id'],
)
perturbator = ConstantPerturbator(
    target_col='order_value'
)
analysis = ClusteredOLSAnalysis(
    cluster_cols=['customer_id'],
    target_col='order_value'
)

sim_power_analysis = PowerAnalysis(
    perturbator=perturbator,
    splitter=splitter,
    analysis=analysis,
    target_col='order_value',
    seed=42
)
power_sim = sim_power_analysis.power_analysis(
    df=data_180_to_270,
    average_effect=1,
    n_simulations=100
)

print(f'Estimated Power (Simulation without CUPAC): {power_sim:.3f}')

# >>> splitter = ClusteredSplitter(
# ...     cluster_cols=['customer_id'],
# ... )
# >>> perturbator = ConstantPerturbator(
# ...     target_col='order_value'
# ... )
# >>> analysis = ClusteredOLSAnalysis(
# ...     cluster_cols=['customer_id'],
# ...     target_col='order_value'
# ... )
# >>> 
# >>> sim_power_analysis = PowerAnalysis(
# ...     perturbator=perturbator,
# ...     splitter=splitter,
# ...     analysis=analysis,
# ...     target_col='order_value',
# ...     seed=42
# ... )
# >>> power_sim = sim_power_analysis.power_analysis(
# ...     df=data_180_to_270,
# ...     average_effect=1,
# ...     n_simulations=100
# ... )
# >>> 
# >>> print(f'Estimated Power (Simulation without CUPAC): {power_sim:.3f}')

# %% [markdown]
# ## Code Chunk: Simulation-based power estimation under a clustered design with CUPAC

cupac_training_data = pd.merge(
    left=data_90_to_180,
    right=(
        data_0_to_90
        .groupby('customer_id', as_index=False)
        .agg(
            pre_n_orders = ('order_time', 'count'),
            pre_aov = ('order_value', 'mean')
        )
    ),
    how='left'
)
cupac_experiment_data = pd.merge(
    left=data_180_to_270,
    right=(
        data_90_to_180
        .groupby('customer_id', as_index=False)
        .agg(
            pre_n_orders = ('order_time', 'count'),
            pre_aov = ('order_value', 'mean')
        )
    ),
    how='left'
)

analysis = ClusteredOLSAnalysis(
    cluster_cols=['customer_id'],
    target_col='order_value',
    covariates=['estimate_order_value']
)

sim_power_analysis_cupac = PowerAnalysis(
    perturbator=perturbator,
    splitter=splitter,
    analysis=analysis,
    target_col='order_value',
    cupac_model = HistGradientBoostingRegressor(),
    features_cupac_model=['pre_n_orders', 'pre_aov'],
    seed=42
)

power_sim_cupac = sim_power_analysis_cupac.power_analysis(
    df=cupac_experiment_data,
    average_effect= 1,
    n_simulations = 100,
    pre_experiment_df=cupac_training_data
)

print(f'Estimated Power (Simulation with CUPAC): {power_sim_cupac:.3f}')

# >>> cupac_training_data = pd.merge(
# ...     left=data_90_to_180,
# ...     right=(
# ...         data_0_to_90
# ...         .groupby('customer_id', as_index=False)
# ...         .agg(
# ...             pre_n_orders = ('order_time', 'count'),
# ...             pre_aov = ('order_value', 'mean')
# ...         )
# ...     ),
# ...     how='left'
# ... )
# >>> cupac_experiment_data = pd.merge(
# ...     left=data_180_to_270,
# ...     right=(
# ...         data_90_to_180
# ...         .groupby('customer_id', as_index=False)
# ...         .agg(
# ...             pre_n_orders = ('order_time', 'count'),
# ...             pre_aov = ('order_value', 'mean')
# ...         )
# ...     ),
# ...     how='left'
# ... )
# >>> 
# >>> analysis = ClusteredOLSAnalysis(
# ...     cluster_cols=['customer_id'],
# ...     target_col='order_value',
# ...     covariates=['estimate_order_value']
# ... )
# >>> 
# >>> sim_power_analysis_cupac = PowerAnalysis(
# ...     perturbator=perturbator,
# ...     splitter=splitter,
# ...     analysis=analysis,
# ...     target_col='order_value',
# ...     cupac_model = HistGradientBoostingRegressor(),
# ...     features_cupac_model=['pre_n_orders', 'pre_aov'],
# ...     seed=42
# ... )
# >>> 
# >>> power_sim_cupac = sim_power_analysis_cupac.power_analysis(
# ...     df=cupac_experiment_data,
# ...     average_effect= 1,
# ...     n_simulations = 100,
# ...     pre_experiment_df=cupac_training_data
# ... )
# >>> 
# >>> print(f'Estimated Power (Simulation with CUPAC): {power_sim_cupac:.3f}')

# %% [markdown]
# # Non-Clustered Design

# %% [markdown]
# ### Data definitions

# %%
experiment_design_data = (
    data_180_to_270
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

splitter = NonClusteredSplitter()
perturbator = ConstantPerturbator(
    target_col='n_orders'
)
analysis = OLSAnalysis(
    target_col='n_orders'
)

sim_power_analysis = PowerAnalysis(
    splitter=splitter,
    perturbator=perturbator,
    analysis=analysis,
    target_col='n_orders'
)

start = time.time()
sim_power = sim_power_analysis.power_analysis(
    df=experiment_design_data,
    average_effect=0.01,
    n_simulations=500
)
end = time.time()
sim_duration = end - start
print(f'Estimated Power (Simulation): {sim_power:.3f} in {sim_duration:.2f} seconds')

# >>> experiment_design_data = (
# ...     data_180_to_270
# ...     .groupby(
# ...         by = ['customer_id'],
# ...         as_index = False
# ...     )
# ...     .agg(
# ...         n_orders = ('order_time', 'count'),
# ...         first_order_time = ('order_time', 'min')
# ...     )
# ...     .astype({'n_orders': float})
# ...     .reset_index(drop=True)
# ... )
# >>> 
# >>> splitter = NonClusteredSplitter()
# >>> perturbator = ConstantPerturbator(
# ...     target_col='n_orders'
# ... )
# >>> analysis = OLSAnalysis(
# ...     target_col='n_orders'
# ... )
# >>> 
# >>> sim_power_analysis = PowerAnalysis(
# ...     splitter=splitter,
# ...     perturbator=perturbator,
# ...     analysis=analysis,
# ...     target_col='n_orders'
# ... )
# >>> 
# >>> start = time.time()
# >>> sim_power = sim_power_analysis.power_analysis(
# ...     df=experiment_design_data,
# ...     average_effect=0.01,
# ...     n_simulations=500
# ... )
# >>> end = time.time()
# >>> sim_duration = end - start
# >>> print(f'Estimated Power (Simulation): {sim_power:.3f} in {sim_duration:.2f} seconds')

# %%
normal_power_analysis = NormalPowerAnalysis(
    splitter=splitter,
    analysis=analysis,
    target_col='n_orders',
)

start = time.time()
normal_power = normal_power_analysis.power_analysis(
    df=experiment_design_data,
    average_effect=0.01,
    n_simulations=10
)
end = time.time()
normal_duration = end - start

print(f'Estimated Power (Analytical): {normal_power:.3f} in {normal_duration:.2f} seconds')

# >>> normal_power_analysis = NormalPowerAnalysis(
# ...     splitter=splitter,
# ...     analysis=analysis,
# ...     target_col='n_orders',
# ... )
# >>>
# >>> start = time.time()
# >>> normal_power = normal_power_analysis.power_analysis(
# ...     df=experiment_design_data,
# ...     average_effect=0.01,
# ...     n_simulations=10
# ... )
# >>> end = time.time()
# >>> normal_duration = end - start
# >>>
# >>> print(f'Estimated Power (Analytical): {normal_power:.3f} in {normal_duration:.2f} seconds')

# %% [markdown]
# ## Code Chunk: Using the `mde_time_line` method
normal_power_analysis = NormalPowerAnalysis(
    splitter=splitter,
    analysis=analysis,
    target_col='n_orders',
    time_col='first_order_time'
)
mde_time_line = normal_power_analysis.mde_time_line(
    df=experiment_design_data,
    experiment_length=np.arange(7, 7*12, 7),
    alpha=0.05,
    powers=[0.8]
)
mde_time_line_df = pd.DataFrame(mde_time_line)
fig, ax = plt.subplots()
fig.set_size_inches(9, 5)
ax.plot(
    mde_time_line_df['experiment_length'],
    mde_time_line_df['mde']
)
ax.set_xlabel('Experiment Length (days)')
ax.set_ylabel('Minimum Detectable Effect (MDE)')
ticks = np.round(np.arange(round(mde_time_line_df['mde'].min(), 2), round(mde_time_line_df['mde'].max()+0.01, 2), step=0.01), 2)
ax.set_yticks(ticks)
_ = ax.set_yticklabels([f'{s:1,.2f}' for s in ticks], fontdict={'fontname': 'serif'})
fig.savefig('mde_time_line.png')

# >>> normal_power_analysis = NormalPowerAnalysis(
# ...     splitter=splitter,
# ...     analysis=analysis,
# ...     target_col='n_orders',
# ...     time_col='first_order_time'
# ... )
# >>> mde_time_line = normal_power_analysis.mde_time_line(
# ...     df=experiment_design_data,
# ...     experiment_length=np.arange(7, 7*12, 7),
# ...     alpha=0.05,
# ...     powers=[0.8]
# ... )
# >>> mde_time_line_df = pd.DataFrame(mde_time_line)
# >>>
# >>> fig, ax = plt.subplots()
# >>> fig.set_size_inches(9, 5)
# >>> ax.plot(
# ...     mde_time_line_df['experiment_length'],
# ...     mde_time_line_df['mde']
# ... )
# >>> ax.set_xlabel('Experiment Length (days)')
# >>> ax.set_ylabel('Minimum Detectable Effect (MDE)')
# >>> ticks = np.round(np.arange(round(mde_time_line_df['mde'].min(), 2), round(mde_time_line_df['mde'].max()+0.01, 2), step=0.01), 2)
# >>> ax.set_yticks(ticks)
# >>> _ = ax.set_yticklabels([f'{s:1,.2f}' for s in ticks])
# >>>
# >>> fig.savefig('mde_time_line.png')
raise NotImplementedError()
# %% [markdown]
# # Switchback

# %% [markdown]
# ### Data Definition

# %%
target_col = 'order_value'
time_col = 'order_time'
average_effect = 1

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


# %% [markdown]
# ### Data pre-processing

# %%
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
    experiment_analysis_data
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

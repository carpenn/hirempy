import pandas as pd
import numpy as np
import random
from scipy.stats import beta
from datetime import datetime, timedelta

def simulate_scenario_baseline(seed, n=125000, prob_Type=[0.60, 0.25, 0.15], prob_Hidden=[0.35, 0.45, 0.20],
                               max_rep_delay=2, max_set_delay=20, gen_recoveries=False, prob_Hidden_recov=[0.20, 0.30, 0.50]):
    np.random.seed(seed)
    random.seed(seed)

    date_occ_seq = pd.date_range(start="2010-01-01", end="2020-12-31")
    date_occ = np.random.choice(date_occ_seq, size=n, replace=True)
    base_year = date_occ_seq[0].year + max_rep_delay

    df = pd.DataFrame({'occ.date': date_occ})
    df['occ.year'] = df['occ.date'].dt.year
    df['occ.month'] = df['occ.date'].dt.month
    df['occ.year.fact'] = pd.Categorical(df['occ.year'], categories=sorted(df['occ.year'].unique()))

    x1 = np.random.choice(["T1", "T2", "T3"], size=n, replace=True, p=prob_Type)
    xh = np.random.choice(["L", "M", "H"], size=n, replace=True, p=prob_Hidden)
    df['claim.nr'] = range(1, n + 1)
    df['type'] = pd.Categorical(x1, categories=["T1", "T2", "T3"])
    df['hidden'] = pd.Categorical(xh, categories=["L", "M", "H"])

    ndays = 365.25 * max_rep_delay
    delay_report = (ndays * beta.rvs(a=np.array([1, 2, 3])[df['type'].cat.codes], b=10, size=n)).astype(int)
    df['rep.delay'] = delay_report / 365.25

    date_report = df['occ.date'] + pd.to_timedelta(delay_report, unit='D')
    df['rep.date'] = date_report
    df['rep.year'] = date_report.dt.year - base_year + 1
    df['rep.year.fact'] = pd.Categorical(df['rep.year'], categories=sorted(df['rep.year'].unique()))
    df['rep.month'] = date_report.dt.month

    n_payments = 30
    lijst = [date_report.iloc[x] + pd.to_timedelta(np.cumsum(np.floor(np.concatenate([
        np.random.exponential(scale=1 / rate, size=1),
        np.random.exponential(scale=1 / rate / 2, size=n_payments - 1)
    ]) * 365.25)), unit='D') for x, rate in enumerate(np.array([6, 5, 4])[df['type'].cat.codes])]

    df_pay = pd.DataFrame(lijst).T
    df_pay_copy = df_pay.copy()
    df_pay_copy['Report'] = date_report.values.astype('datetime64[D]').astype(int)
    df_pay_copy = (df_pay_copy.sub(df_pay_copy['Report'], axis=0) / 365.25).drop(columns=['Report']).T
    df_pay = df_pay.T

    delay_rep = ((date_report - pd.Timestamp('2010-01-01')).dt.days / 365.25).repeat(n_payments)
    payments = df_pay_copy.values.flatten()
    meanlog = lambda x1, x2: np.log([100, 200, 400][x1]) + 0.1 * (payments ** [1.50, 1.25, 1.40][x2])
    df_size = np.random.lognormal(mean=meanlog(df['type'].cat.codes, df['hidden'].cat.codes), sigma=1, size=(n, n_payments))
    df_size = pd.DataFrame(df_size.T, columns=df['claim.nr'])

    ndays = 365.25 * max_set_delay
    settlement_delay = (beta.rvs(a=1, b=8 * [1, 0.75, 0.5][df['type'].cat.codes], size=n) * ndays).astype(int)
    date_settlement = date_report + pd.to_timedelta(settlement_delay, unit='D')
    df['settlement.date'] = date_settlement
    df['settlement.year'] = date_settlement.dt.year - base_year + 1

    max_delays = date_settlement.values.astype('datetime64[D]').astype(int)
    df_pay = df_pay.T
    df_size = df_size.T
    ind = np.where(df_pay.values > max_delays[:, None])
    df_pay.values[ind] = np.nan
    df_size.values[ind] = np.nan
    df_pay = df_pay.T
    df_size = df_size.T

    breaks = pd.date_range(start='2009-12-31', end='2040-12-31').astype(int)
    calc_obs_year = lambda k: np.diff(np.searchsorted(breaks, np.concatenate([
        breaks[(df['rep.year'][k] + base_year - 1 - 1):(df['rep.year'][k] + base_year + 8)],
        df_pay.iloc[:, k].dropna().astype(int)
    ])) - 1)[:len(breaks) - 1]
    obs_years = [calc_obs_year(k) for k in range(n)]

    agg_fu = lambda k: np.bincount(obs_years[k], weights=df_size.iloc[:, k].dropna(), minlength=9)
    df_size_agg = np.array([agg_fu(k) for k in range(n)])
    df_size_agg = pd.DataFrame(df_size_agg, columns=[f'size_obs{i + 1}' for i in range(9)])

    create_open = lambda ones: np.pad(np.ones(min(ones + 1, 9)), (0, 9 - min(ones + 1, 9)), 'constant')
    df_open = np.array([create_open(ones) for ones in df['settlement.year'] - df['rep.year']])
    df_open = pd.DataFrame(df_open, columns=[f'open_obs{i + 1}' for i in range(9)])

    create_settlement = lambda zeros: np.pad(np.zeros(min(zeros, 9)), (0, 9 - min(zeros, 9)), 'constant')
    df_settlement = np.array([create_settlement(zeros) for zeros in df['settlement.year'] - df['rep.year']])
    df_settlement = pd.DataFrame(df_settlement, columns=[f'settlement_obs{i + 1}' for i in range(9)])

    df = pd.concat([df, df_open, df_settlement, df_size_agg], axis=1)
    df = df[(df['rep.year'] <= 9) & (df['rep.year'] >= 1)]

    if gen_recoveries:
        df = simulate_recoveries(df, prob_Hidden_recov)

    df_long = df.melt(id_vars=['claim.nr', 'type', 'hidden', 'occ.date', 'occ.year', 'occ.month', 'occ.year.fact', 'rep.delay', 'rep.date', 'rep.year', 'rep.year.fact', 'rep.month', 'settlement.date', 'settlement.year'], 
                      value_vars=[col for col in df.columns if col.startswith(('open_obs', 'settlement_obs', 'size_obs'))], 
                      var_name=['.value', 'obs'], 
                      value_name='value')
    df_long['dev.year.fact'] = df_long['obs'].map({f'obs{i + 1}': i + 1 for i in range(9)})
    df_long['dev.year'] = df_long['dev.year.fact']
    df_long['calendar.year'] = df_long['rep.year'] + df_long['dev.year'] - 1
    df_long['payment'] = (df_long['size'] > 0).astype(int)

    if gen_recoveries:
        df_long['recovery'] = (df_long['size'] < 0).astype(int)
        df_long['size.pay'] = np.where(df_long['size'] > 0, df_long['size'], 0)
        df_long['size.recov'] = np.where(df_long['size'] < 0, -df_long['size'], 0)

    return df_long

def simulate_recoveries(df, prob_Hidden_recov=[0.20, 0.30, 0.50]):
    size_obs = df[[f'size_obs{i + 1}' for i in range(9)]]
    size_recov = pd.DataFrame(0, index=df.index, columns=[f'size_recov{i + 1}' for i in range(9)])

    for i in range(1, 9):
        size = size_obs.iloc[:, i]
        prob_recov = np.array(prob_Hidden_recov)[df['hidden'].cat.codes]
        prob_recov *= (i <= (df['settlement.year'] - df['rep.year'] + 1)) * i / (df['settlement.year'] - df['rep.year'] + 1)
        recov = ((size_obs.iloc[:, i - 1] > size) & (i <= (df['settlement.year'] - df['rep.year'] + 1))) * np.random.binomial(1, prob_recov)
        p_recov = beta.rvs(a=1, b=np.array([1, 0.75, 0.5])[df['type'].cat.codes], size=len(df)) * i / 10 * (i <= (df['settlement.year'] - df['rep.year'] + 1))
        size_recov.iloc[:, i] = recov * (size + p_recov * (size_obs.iloc[:, i - 1] - size))
        size_obs.iloc[:, i] = size - size_recov.iloc[:, i]

    df['ultimate'] = size_obs.sum(axis=1)
    return df

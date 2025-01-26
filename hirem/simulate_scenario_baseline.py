import pandas as pd
import numpy as np
import random
from scipy.stats import beta
from datetime import datetime, timedelta

def simulate_scenario_baseline(seed, n=125000, prob_Type=[0.60, 0.25, 0.15], prob_Hidden=[0.35, 0.45, 0.20],
                               max_rep_delay=2, max_set_delay=20, gen_recoveries=False, prob_Hidden_recov=[0.20, 0.30, 0.50]):
    np.random.seed(seed)
    random.seed(seed)

    # Step 1: Simulating the occurrence date
    date_occ_seq = pd.date_range(start="2010-01-01", end="2020-12-31")
    date_occ = np.random.choice(date_occ_seq, size=n, replace=True)
    
    # Convert the date_occ array to a pandas DatetimeIndex 
    date_occ = pd.DatetimeIndex(date_occ)
    base_year = date_occ_seq[0].year + max_rep_delay    

    df = pd.DataFrame({
        'occ.date': date_occ,
        'occ.year': date_occ.year,
        'occ.month': date_occ.month
    })

    #df = pd.DataFrame({'occ.date': date_occ})
    #df['occ.year'] = df['occ.date'].dt.year
    #df['occ.month'] = df['occ.date'].dt.month
    df['occ.year.fact'] = pd.Categorical(df['occ.year'], categories=sorted(df['occ.year'].unique()))

    # Step 2: Simulating the covariates
    # Covariate 'Type'
    x1 = np.random.choice(['T1', 'T2', 'T3'], size=n, replace=True, p=prob_Type)
    x1 = pd.Categorical(x1, categories=['T1', 'T2', 'T3'])

    # Covariate 'Hidden'
    xh = np.random.choice(['L', 'M', 'H'], size=n, replace=True, p=prob_Hidden)
    xh = pd.Categorical(xh, categories=['L', 'M', 'H'])
    
    df['claim.nr'] = range(1, n + 1)
    df['type'] = x1
    df['hidden'] = xh

    # Step 3: Simulating the reporting delay
    ndays = 365.25 * max_rep_delay
    shape_params = np.array([1, 2, 3])[x1.codes] 
    delay_report = np.floor(ndays * np.random.beta(shape_params, 10)) 

    df['rep.delay'] = delay_report / 365.25
    date_report = df['occ.date'] + pd.to_timedelta(delay_report, unit='D')

    df['rep.date'] = date_report
    df['rep.year'] = date_report.dt.year - base_year + 1
    df['rep.year.fact'] = pd.Categorical(df['rep.year'], categories=sorted(df['rep.year'].unique()))
    df['rep.month'] = date_report.dt.month

    # Step 4: Simulating the payment delays

    num_payments = 30
    total_claims = n

    def payment_arrival_times(claim_nr):
        report_date = date_report.iloc[claim_nr]
        lst_pay_dates = np.floor(np.cumsum(np.concatenate([
            np.random.exponential(1 / [6, 5, 4][x1.codes[claim_nr]], 1),
            np.random.exponential(1 / [2, 1.5, 1][x1.codes[claim_nr]], num_payments - 1)
        ])) * 365.25)
        return report_date + pd.to_timedelta(lst_pay_dates, unit='D')

    claims_df = pd.DataFrame({
        'claim.nr': np.repeat(range(1, total_claims+1), num_payments),
        'type': np.repeat(x1, num_payments),
        'hidden': np.repeat(xh, num_payments),
        'occ.date': np.repeat(date_occ, num_payments),
        'occ.year': np.repeat(date_occ.year, num_payments),
        'occ.year.fact': np.repeat(pd.Categorical(df['occ.year'], categories=sorted(df['occ.year'].unique())), num_payments),
        'occ.month': np.repeat(date_occ.month, num_payments),
        'rep.delay': np.repeat(delay_report, num_payments),
        'rep.date': np.repeat(date_report, num_payments),
        'rep.year': np.repeat(date_report.dt.year - base_year + 1, num_payments),
        'rep.year.fact': np.repeat(pd.Categorical(df['rep.year'], categories=sorted(df['rep.year'].unique())), num_payments),
        'rep.month': np.repeat(date_report.dt.month, num_payments),
    
        'pay.nr': np.tile(range(1, num_payments+1), total_claims),
        'pay.date': np.concatenate([payment_arrival_times(i) for i in range(total_claims)])
    })

    claims_df['pay.delay'] = (claims_df['pay.date'] - claims_df['rep.date']).dt.days / 365.25

    # Step 5: Simulating the payment sizes

    claims_df['base'] = np.log([100, 200, 400])[claims_df['type'].cat.codes]
    claims_df['power'] = np.array([1.50, 1.25, 1.40])[claims_df['hidden'].cat.codes]
    claims_df['meanlog'] = claims_df['base'] + 0.1 * (claims_df['pay.delay'] ** claims_df['power']) 
    claims_df['pay.amount'] = np.random.lognormal(mean=claims_df['meanlog'], sigma=1) 
    claims_df['dev.year'] = claims_df['pay.date'].dt.year - claims_df['rep.date'].dt.year + 1


    # Step 6: Simulating the settlement delay 

    ndays = 365.25 * max_set_delay 
    settlement_delay = np.floor(np.random.beta(1, 8 * np.array([1, 0.75, 0.5])[x1.codes]) * ndays) 
    date_settlement = date_report + pd.to_timedelta(settlement_delay, unit='D') 
    claims_df['settlement.date'] = np.repeat(date_settlement, num_payments) 
    claims_df = claims_df[claims_df['pay.date'] <= claims_df['settlement.date']]

    claims_df.drop(columns=['base', 'power', 'meanlog'], inplace = True)
    # breaks = pd.date_range(start='2009-12-31', end='2040-12-31').astype(int)
    # calc_obs_year = lambda k: np.diff(np.searchsorted(breaks, np.concatenate([
    #     breaks[(df['rep.year'][k] + base_year - 1 - 1):(df['rep.year'][k] + base_year + 8)],
    #     df_pay.iloc[:, k].dropna().astype(int)
    # ])) - 1)[:len(breaks) - 1]
    # obs_years = [calc_obs_year(k) for k in range(n)]

    # agg_fu = lambda k: np.bincount(obs_years[k], weights=df_size.iloc[:, k].dropna(), minlength=9)
    # df_size_agg = np.array([agg_fu(k) for k in range(n)])
    # df_size_agg = pd.DataFrame(df_size_agg, columns=[f'size_obs{i + 1}' for i in range(9)])

    # create_open = lambda ones: np.pad(np.ones(min(ones + 1, 9)), (0, 9 - min(ones + 1, 9)), 'constant')
    # df_open = np.array([create_open(ones) for ones in df['settlement.year'] - df['rep.year']])
    # df_open = pd.DataFrame(df_open, columns=[f'open_obs{i + 1}' for i in range(9)])

    # create_settlement = lambda zeros: np.pad(np.zeros(min(zeros, 9)), (0, 9 - min(zeros, 9)), 'constant')
    # df_settlement = np.array([create_settlement(zeros) for zeros in df['settlement.year'] - df['rep.year']])
    # df_settlement = pd.DataFrame(df_settlement, columns=[f'settlement_obs{i + 1}' for i in range(9)])

    # df = pd.concat([df, df_open, df_settlement, df_size_agg], axis=1)
    # df = df[(df['rep.year'] <= 9) & (df['rep.year'] >= 1)]

    # if gen_recoveries:
    #     df = simulate_recoveries(df, prob_Hidden_recov)

    # df_long = df.melt(id_vars=['claim.nr', 'type', 'hidden', 'occ.date', 'occ.year', 'occ.month', 'occ.year.fact', 'rep.delay', 'rep.date', 'rep.year', 'rep.year.fact', 'rep.month', 'settlement.date', 'settlement.year'], 
    #                   value_vars=[col for col in df.columns if col.startswith(('open_obs', 'settlement_obs', 'size_obs'))], 
    #                   var_name=['.value', 'obs'], 
    #                   value_name='value')
    # df_long['dev.year.fact'] = df_long['obs'].map({f'obs{i + 1}': i + 1 for i in range(9)})
    # df_long['dev.year'] = df_long['dev.year.fact']
    # df_long['calendar.year'] = df_long['rep.year'] + df_long['dev.year'] - 1
    # df_long['payment'] = (df_long['size'] > 0).astype(int)

    # if gen_recoveries:
    #     df_long['recovery'] = (df_long['size'] < 0).astype(int)
    #     df_long['size.pay'] = np.where(df_long['size'] > 0, df_long['size'], 0)
    #     df_long['size.recov'] = np.where(df_long['size'] < 0, -df_long['size'], 0)

    return claims_df

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

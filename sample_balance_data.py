import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp



adhd = pd.read_csv(r'adha.csv')
td = pd.read_csv(r'td.csv')
adhd = adhd.dropna()
td = td.dropna()


sex_counts = adhd["gender"].value_counts()
sex_ratio = adhd["gender"].value_counts(normalize=True).round(3)

sex_df = pd.DataFrame({"sex_counts": sex_counts, "sex_ratio": sex_ratio})
print(sex_df)

# ADHD
adhd_count = adhd.groupby(["age", "gender"]).size().reset_index(name="num_count")
adhd_count["class"] = "ADHD"

# TD
td_count = td.groupby(["age", "gender"]).size().reset_index(name="num_count")
td_count["class"] = "TD"

result = pd.concat([adhd_count, td_count], ignore_index=True)



# ========== tools ==========
def _ratio_bounds(N, ratio, tol):
    r_low, r_high = ratio*(1-tol), ratio*(1+tol)
    m_low  = int(np.floor(N * (r_low/(1+r_low))))
    m_high = int(np.ceil (N * (r_high/(1+r_high))))
    m_low  = max(0, m_low)
    m_high = min(N, m_high)
    if m_low > m_high:
        m_low, m_high = m_high, m_high
    return m_low, m_high

def _max_feasible_N(m_avail, f_avail, ratio, tol, N_start):
    """Given available counts (m_avail, f_avail) for a group, find the largest N ≤ N_start 
    such that there exists m ∈ [m_low, m_high] with m ≤ m_avail and N - m ≤ f_avail."""
    for N in range(N_start, 0, -1):
        m_low, m_high = _ratio_bounds(N, ratio, tol)
        lo = max(m_low, N - f_avail)
        hi = min(m_high, m_avail)
        if lo <= hi:
            return N
    return 0

def _pick_m_target(N, ratio, tol, m_avail, f_avail):
    m_low, m_high = _ratio_bounds(N, ratio, tol)
    lo = max(m_low, N - f_avail)
    hi = min(m_high, m_avail)
    if lo > hi:
        return None
    m_star = int(round(N * (ratio/(1+ratio))))
    return int(np.clip(m_star, lo, hi))

def _sample_fixed_counts(df, sex_col, male_label, n_m, n_f, rng, chosen_idx=None):
    if chosen_idx is None:
        chosen_idx = set()
    males = df[(df[sex_col]==male_label) & (~df.index.isin(chosen_idx))]
    femas = df[(df[sex_col]!=male_label) & (~df.index.isin(chosen_idx))]
    if len(males) < n_m or len(femas) < n_f:
        return None
    idx_m = rng.choice(males.index.to_numpy(), size=n_m, replace=False)
    idx_f = rng.choice(femas.index.to_numpy(), size=n_f, replace=False)
    return np.concatenate([idx_m, idx_f])

def _sex_table(adhd_s, td_s, sex_col, male_label):
    ac = adhd_s[sex_col].value_counts()
    tc = td_s[sex_col].value_counts()
    mA, fA = ac.get(male_label,0), ac.sum() - ac.get(male_label,0)
    mT, fT = tc.get(male_label,0), tc.sum() - tc.get(male_label,0)
    return np.array([[mA, fA],[mT, fT]]), (mA, fA, mT, fT)

def _sex_p(contingency):
    chi2, p, _, _ = chi2_contingency(contingency)
    return p

def _age_p(adhd_s, td_s, age_col):
    return ks_2samp(adhd_s[age_col].to_numpy(), td_s[age_col].to_numpy()).pvalue

def _within_ratio(m, f, ratio, tol):
    if f == 0:
        return np.isinf(ratio)  
    r = m / f
    return (ratio*(1-tol) <= r <= ratio*(1+tol))


# ========== key function ==========
def sample_equal_n_with_ratio_balance(
    adhd, td,
    age_col="age", sex_col="gender",
    male_label="male", female_label="female",
    target_ratio=3.5, tol=0.143,          
    p_threshold=0.05,
    max_restarts=25, max_adjust_iters=800,
    random_state=42
):
    """
    Produce two equal-sized subsets (ADHD_s, TD_s) satisfying:
    - Each group’s overall gender ratio ≈ target_ratio:1 (tolerance = tol)
    - No significant gender-distribution difference between groups (p >= p_threshold)
    - No significant age-distribution difference between groups (p >= p_threshold)
    - Subject to the above constraints, maximize N
    """
    rng = np.random.default_rng(random_state)

    # 1) Determine the maximum feasible N (constrained by group-wise gender supply and total sample size)
    A_m = (adhd[sex_col]==male_label).sum()
    A_f = len(adhd) - A_m
    T_m = (td[sex_col]==male_label).sum()
    T_f = len(td)  - T_m

    N0 = min(len(adhd), len(td))
    N_A = _max_feasible_N(A_m, A_f, target_ratio, tol, N0)
    N_T = _max_feasible_N(T_m, T_f, target_ratio, tol, N0)
    N   = min(N_A, N_T)

    if N == 0:
        raise ValueError(
            "Given the current male/female availability and ratio/tolerance settings, "
            "no feasible equal-sized samples can be constructed. Please relax `tol` or adjust `target_ratio`."
        )

    # 2) Try N from large to small, randomly restart several times to find a solution that meets the test
    for curN in range(N, 0, -1):
        mA_target = _pick_m_target(curN, target_ratio, tol, A_m, A_f)
        mT_target = _pick_m_target(curN, target_ratio, tol, T_m, T_f)
        if mA_target is None or mT_target is None:
            continue
        fA_target = curN - mA_target
        fT_target = curN - mT_target

        for restart in range(max_restarts):
            # 2.1 randomly sample by gender target counts (without age constraint)
            idx_A = _sample_fixed_counts(adhd, sex_col, male_label, mA_target, fA_target, rng)
            idx_T = _sample_fixed_counts(td,   sex_col, male_label, mT_target, fT_target, rng)
            if idx_A is None or idx_T is None:
                break  # Insufficient supply (should not occur in theory), fall back to a smaller N


            adhd_s = adhd.loc[idx_A].copy()
            td_s   = td.loc[idx_T].copy()

            # 2.2 Iterative adjustment: keep the equal quantity premise and let the gender and age tests pass
            for it in range(max_adjust_iters):
                sex_tab, (mA, fA, mT, fT) = _sex_table(adhd_s, td_s, sex_col, male_label)
                p_sex = _sex_p(sex_tab)
                p_age = _age_p(adhd_s, td_s, age_col)

                # condition satisfied
                if (p_sex >= p_threshold) and (p_age >= p_threshold):
                    return adhd_s.reset_index(drop=True), td_s.reset_index(drop=True), {
                        "N_per_group": curN,
                        "male_adhd": int(mA), "female_adhd": int(fA),
                        "male_td":   int(mT), "female_td":   int(fT),
                        "ratio_adhd": (mA/fA) if fA>0 else np.inf,
                        "ratio_td":   (mT/fT) if fT>0 else np.inf,
                        "p_sex": float(p_sex), "p_age": float(p_age),
                        "target_ratio": target_ratio, "tol": tol,
                    }

                # 2.2.a try to fix gender test first (if not passed)
                if p_sex < p_threshold:
                    # Which group has a male surplus relative to the other?
                    # Try swapping one male for a female in that group (while keeping the in-group ratio within tolerance)
                    devA = (mA/(fA if fA>0 else np.inf)) - (mT/(fT if fT>0 else np.inf))
                    if devA > 0:
                        # ADHD male change to female
                        if mA > 0:
                            # randomly pick a male to remove
                            cand_out = adhd_s[adhd_s[sex_col]==male_label]
                            # pick a female to replace (not in current sample)
                            pool_in  = adhd[(adhd[sex_col]!=male_label) & (~adhd.index.isin(adhd_s.index))]
                            if len(cand_out)>0 and len(pool_in)>0:
                                out_idx = rng.choice(cand_out.index.to_numpy(), 1)[0]
                                in_idx  = rng.choice(pool_in.index.to_numpy(), 1)[0]

                                # Check whether the ADHD group still satisfies the gender ratio tolerance after the swap
                                mA2, fA2 = mA-1, fA+1
                                if _within_ratio(mA2, fA2, target_ratio, tol):
                                    adhd_s = adhd_s.drop(index=out_idx)
                                    adhd_s = pd.concat([adhd_s, adhd.loc[[in_idx]]])
                                    continue
                    else:
                        # TD male change to female
                        if mT > 0:
                            cand_out = td_s[td_s[sex_col]==male_label]
                            pool_in  = td[(td[sex_col]!=male_label) & (~td.index.isin(td_s.index))]
                            if len(cand_out)>0 and len(pool_in)>0:
                                out_idx = rng.choice(cand_out.index.to_numpy(), 1)[0]
                                in_idx  = rng.choice(pool_in.index.to_numpy(), 1)[0]
                                mT2, fT2 = mT-1, fT+1
                                if _within_ratio(mT2, fT2, target_ratio, tol):
                                    td_s = td_s.drop(index=out_idx)
                                    td_s = pd.concat([td_s, td.loc[[in_idx]]])
                                    continue

                # 2.2.b try to fix age test next (keeping sample sizes and gender counts the same)
                if p_age < p_threshold:
                    # Calculate age differences (ADHD - TD), positive means more in ADHD, negative means more in TD
                    ages = sorted(set(adhd_s[age_col]).union(set(td_s[age_col])))
                    cntA = adhd_s[age_col].value_counts()
                    cntT = td_s[age_col].value_counts()
                    diff = {a: int(cntA.get(a,0) - cntT.get(a,0)) for a in ages}
                    overA = [a for a in ages if diff[a] > 0]   # ADHD 相对多
                    overT = [a for a in ages if diff[a] < 0]   # TD   相对多

                    moved = False
                    # Try to let the surplus group reallocate some samples from this age to an age it lacks
                    # (within the same group; keep gender unchanged)
                    if overA:
                        a = rng.choice(overA)
                        # From ADHD samples with age == a, randomly pick one (male or female),
                        # and replace it with another ADHD sample of age b (where diff[b] < 0)
                        cand_out = adhd_s[adhd_s[age_col]==a]
                        if len(cand_out)>0 and overT:
                            b_choices = [b for b in overT if b in adhd[age_col].unique()]
                            if b_choices:
                                b = rng.choice(b_choices)
                                row_out = cand_out.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
                                sex_need = row_out[sex_col]
                                pool_in = adhd[(adhd[age_col]==b) & (adhd[sex_col]==sex_need) & (~adhd.index.isin(adhd_s.index))]
                                if len(pool_in)>0:
                                    in_row = pool_in.sample(1, random_state=int(rng.integers(1e9)))
                                    adhd_s = adhd_s.drop(index=row_out.name)
                                    adhd_s = pd.concat([adhd_s, in_row])
                                    moved = True
                    if (not moved) and overT:
                        a = rng.choice(overT)
                        cand_out = td_s[td_s[age_col]==a]
                        if len(cand_out)>0 and overA:
                            b_choices = [b for b in overA if b in td[age_col].unique()]
                            if b_choices:
                                b = rng.choice(b_choices)
                                row_out = cand_out.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
                                sex_need = row_out[sex_col]
                                pool_in = td[(td[age_col]==b) & (td[sex_col]==sex_need) & (~td.index.isin(td_s.index))]
                                if len(pool_in)>0:
                                    in_row = pool_in.sample(1, random_state=int(rng.integers(1e9)))
                                    td_s = td_s.drop(index=row_out.name)
                                    td_s = pd.concat([td_s, in_row])
                                    moved = True

                    # If no move could be made in this iteration, break out of the adjustment loop
                    # (restart once or fall back to a smaller N)
                    if not moved:
                        break

            # If the iteration ends without success, try a new random restart
        # Otherwise, reduce the target N and continue
    raise RuntimeError(
    "No feasible solution found under the given tolerance and test thresholds; "
    "consider relaxing `tol`, lowering `p_threshold`, or increasing "
    "`max_restarts` / `max_adjust_iters`."
    )




adhd_s, td_s, info = sample_equal_n_with_ratio_balance(
    adhd, td,
    age_col="age", sex_col="gender",
    male_label="male", female_label="female",
    target_ratio=4, tol=0.10,     
    p_threshold=0.7,
    random_state=42
)
print(info)

print(td_s.shape)
print(adhd_s.shape)

# If the age column is not purely numeric, safely convert it first
adhd_s["age"] = pd.to_numeric(adhd_s["age"], errors="coerce")
td_s["age"]   = pd.to_numeric(td_s["age"], errors="coerce")

# calculate mean and std
mu_adhd = adhd_s["age"].mean()
sd_adhd = adhd_s["age"].std(ddof=1)

mu_td = td_s["age"].mean()
sd_td = td_s["age"].std(ddof=1)

print(mu_adhd, sd_adhd, mu_td, sd_td)

td_s['class'] = 'TD'
adhd_s['class'] = 'ADHD'


common_cols = td_s.columns.intersection(adhd_s.columns)

# based on common columns to merge (vertical concatenation)
merged_df = pd.concat([td_s[common_cols], adhd_s[common_cols]], ignore_index=True)

merged_df.to_csv('data.csv', index=False)


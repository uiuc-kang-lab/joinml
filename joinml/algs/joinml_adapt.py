from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset, JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import (
    set_up_logging,
    normalize,
    weighted_sample_pd,
    get_ci_bootstrap_ttest,
    defensive_mix,
)
from joinml.percentile_est_utils import cal_weighted_median
from joinml.estimates import Estimates

import itertools
import logging
import numpy as np
from typing import Tuple, List, Dict, Optional


# --------------------------------------------------------------------------- #
# Allocation: finds β ⊂ {1..K} minimising MSE with budget reallocation.
# Stratum 0 is the sampling-only head (low similarity); strata 1..K are
# blocking candidates ordered by ascending similarity. The highest-similarity
# candidate is index K in our ascending-argsort layout.
# --------------------------------------------------------------------------- #

def _alloc_for_subset(blocking, pop, W, sampling_budget):
    K = len(pop) - 1
    blocked_size = sum(pop[i] for i in blocking)
    if blocked_size >= sampling_budget:
        return None
    remaining = sampling_budget - blocked_size
    sampling = [i for i in range(K + 1) if i not in blocking]
    W_rest = sum(W[i] for i in sampling)
    if W_rest <= 0:
        return None
    n = {i: pop[i] for i in blocking}
    for i in sampling:
        n[i] = max(1, int(remaining * W[i] / W_rest))
    return n


def _mse(aggregator, K, sampling, pop, cnt_var, sum_var, cnt_est_total, sum_est_total, n,
         pilot_extremes=None):
    """Allocator MSE proxy.

    For MIN/MAX, paper §5.2 uses Extreme Value Theory: allocate to maximise
    P(true extreme ∈ blocked region). We approximate via a simple ranking on
    per-stratum pilot extremes — strata with the highest pilot-max (for MAX)
    or lowest pilot-min (for MIN) are most valuable to block. The proxy
    objective is "sum of extremes left in sampling strata, signed for direction":
    minimise this == prefer to block strata with the most extreme pilot values.
    """
    if aggregator == "count":
        return sum((pop[i] ** 2) * cnt_var[i] / max(n[i], 1) for i in sampling)
    if aggregator == "sum":
        return sum((pop[i] ** 2) * sum_var[i] / max(n[i], 1) for i in sampling)
    if aggregator in ("max", "min"):
        # pilot_extremes[i] = pilot's max (for MAX) or min (for MIN); None if no
        # matched samples in pilot. Strata with no pilot matches contribute the
        # neutral element (0 — strata that never showed a positive in pilot are
        # both unlikely to contain extremes AND don't pull allocation away from
        # candidates that did).
        if pilot_extremes is None:
            return sum((pop[i] ** 2) * cnt_var[i] / max(n[i], 1) for i in sampling)
        score = 0.0
        for i in sampling:
            ex = pilot_extremes[i]
            if ex is None:
                continue
            if aggregator == "max":
                score += ex            # higher pilot max in sampling stratum is bad
            else:
                score -= ex            # lower pilot min in sampling stratum is bad (subtract → minimise -ex)
        # Add a small count-variance term as a tie-breaker so the allocator
        # converges to the count-variance allocation when no extremes signal exists.
        eps = 1e-12 * sum((pop[i] ** 2) * cnt_var[i] / max(n[i], 1) for i in sampling)
        return score + eps
    if aggregator == "median":
        # Variance of CDF estimate at median: dominated by per-stratum value
        # variance. Use sum_var (variance of g·O/π contributions) as a proxy.
        return sum((pop[i] ** 2) * sum_var[i] / max(n[i], 1) for i in sampling)
    # avg — delta method on ratio sum/count
    total_cnt = sum(cnt_est_total[i] for i in range(K + 1))
    total_sum = sum(sum_est_total[i] for i in range(K + 1))
    if total_cnt <= 0:
        return float("inf")
    cnt_v = sum((pop[i] ** 2) * cnt_var[i] / max(n[i], 1) for i in sampling)
    sum_v = sum((pop[i] ** 2) * sum_var[i] / max(n[i], 1) for i in sampling)
    ratio = total_sum / total_cnt
    sum_denom = max(total_sum ** 2, 1e-30)
    return (ratio ** 2) * (cnt_v / (total_cnt ** 2) + sum_v / sum_denom)


def _evt_min_max_score(
    aggregator, beta, K, pop, n, pilot_matched_values, pilot_extremes
):
    """Proposal C — GEV/Weibull tail-fit allocator score for MIN/MAX.

    For each sampling stratum i with at least 5 matched pilot values, fit a
    Generalized Extreme Value distribution to (negated) values for MIN. Project
    P(stratum_min < observed_global_min) using the fitted CDF. Score β by total
    expected miss probability weighted by stratum population. Lower = better.

    Falls back to empirical CDF for strata with <5 pilot matches, and to a
    population-weighted miss-prob of 1 for strata with no pilot matches at all.
    """
    try:
        from scipy.stats import genextreme
    except ImportError:
        # If scipy unavailable, fall back to the simple pilot-extreme heuristic
        score = 0.0
        for i in range(K + 1):
            if i in beta:
                continue
            ex = pilot_extremes[i]
            if ex is None:
                continue
            score += ex if aggregator == "max" else -ex
        return score

    # Determine the observed global extreme across ALL pilot-matched values,
    # used as the threshold for "did we miss something more extreme?"
    all_vals = [v for vs in pilot_matched_values for v in vs]
    if not all_vals:
        # No pilot matches anywhere — fall back uniformly across sampling strata
        sampling = [i for i in range(K + 1) if i not in beta]
        return float(sum(pop[i] for i in sampling))
    obs_global_extreme = max(all_vals) if aggregator == "max" else min(all_vals)

    score = 0.0
    for i in range(K + 1):
        if i in beta:
            continue
        vals = pilot_matched_values[i]
        n_i = max(int(n[i]), 1)
        if len(vals) == 0:
            # No pilot matches in this stratum — assume a non-trivial miss prob.
            # Population could contain a more-extreme value but we have no signal.
            score += 0.5 * pop[i]
            continue
        if len(vals) >= 5:
            try:
                # GEV for maxima — for MIN we negate values then fit GEV-of-max
                arr = np.asarray(vals, dtype=float)
                if aggregator == "min":
                    arr = -arr
                    target = -obs_global_extreme
                else:
                    target = obs_global_extreme
                shape, loc, scale = genextreme.fit(arr)
                # P(single sample exceeds the observed global extreme)
                cdf_at_target = genextreme.cdf(target, shape, loc, scale)
                p_exceed_one = max(0.0, 1.0 - float(cdf_at_target))
            except Exception:
                p_exceed_one = float(np.mean(np.asarray(vals) > obs_global_extreme
                                             if aggregator == "max"
                                             else np.asarray(vals) < obs_global_extreme))
        else:
            # Empirical fallback for tiny pilot
            arr = np.asarray(vals)
            if aggregator == "max":
                p_exceed_one = float(np.mean(arr > obs_global_extreme))
            else:
                p_exceed_one = float(np.mean(arr < obs_global_extreme))
        # Probability we DON'T find a more-extreme value in n_i additional samples
        miss_prob = (1.0 - p_exceed_one) ** n_i
        # Weight by population that could contain such a value
        score += miss_prob * pop[i]
    return score


def _match_rate_uniform(pilot_match_rates, threshold=5.0):
    """Proposal A — return True when match rates across blocking-candidate strata
    are within `threshold` of stratum 0's match rate. In that case, blocking
    provides no recall advantage, so the allocator should not block at all.
    """
    if not pilot_match_rates or len(pilot_match_rates) < 2:
        return False
    head_rate = pilot_match_rates[0]
    tail_rates = pilot_match_rates[1:]
    if head_rate <= 0:
        # Head saw no matches — but if tail did, blocking probably DOES help.
        return all(r <= 0 for r in tail_rates)
    max_tail = max(tail_rates) if tail_rates else 0.0
    return max_tail / max(head_rate, 1e-30) < threshold


def adaptive_allocation(
    pop: List[int],
    W: List[float],
    cnt_var: List[float],
    sum_var: List[float],
    cnt_est_total: List[float],
    sum_est_total: List[float],
    sampling_budget: int,
    aggregator: str = "count",
    search: str = "prefix",
    pilot_extremes: Optional[List[Optional[float]]] = None,
    pilot_matched_values: Optional[List[List[float]]] = None,
    pilot_match_rates: Optional[List[float]] = None,
) -> Tuple[List[int], List[int], Dict[int, int], Optional[float]]:
    """Return (sampling_strata, blocking_strata, n_per_stratum, best_mse).

    `search` selects allocator strategy. New options for MIN/MAX:
      - "auto"   : Proposal A — refuse to block when match rates are uniform
                   across strata (matches scattered → blocking has no advantage)
      - "evt"    : Proposal C — GEV/Weibull tail-fit allocator that minimises
                   expected probability of missing the true extreme
    """
    K = len(pop) - 1
    best_mse: Optional[float] = None
    best_beta: set = set()
    best_n: Dict[int, int] = {}

    # ---- Proposal A short-circuit: if matches are scattered uniformly across
    # ---- strata and we're computing MIN/MAX, refuse to block. ----
    if (search == "auto" and aggregator in ("min", "max")
            and pilot_match_rates is not None
            and _match_rate_uniform(pilot_match_rates)):
        n = _alloc_for_subset(set(), pop, W, sampling_budget)
        if n is not None:
            return list(range(K + 1)), [], n, 0.0

    use_evt = (search == "evt" and aggregator in ("min", "max"))

    def consider(beta):
        nonlocal best_mse, best_beta, best_n
        n = _alloc_for_subset(beta, pop, W, sampling_budget)
        if n is None:
            return
        sampling = [j for j in range(K + 1) if j not in beta]
        if use_evt and pilot_matched_values is not None:
            mse = _evt_min_max_score(aggregator, beta, K, pop, n,
                                     pilot_matched_values, pilot_extremes)
        else:
            mse = _mse(aggregator, K, sampling, pop, cnt_var, sum_var,
                       cnt_est_total, sum_est_total, n, pilot_extremes=pilot_extremes)
        if best_mse is None or mse < best_mse:
            best_mse, best_beta, best_n = mse, set(beta), n

    consider(set())  # no blocking

    # The β-search topology: "subset" enumerates all subsets (when small) or
    # uses greedy add/remove. Anything else (including "prefix", "auto", "evt")
    # uses the prefix scan over candidate cutoffs.
    if search == "subset" and K <= 12:
        for c in range(1, K + 1):
            for combo in itertools.combinations(range(1, K + 1), c):
                consider(set(combo))
    elif search == "subset":  # greedy when subset-enumeration too big
        current = set()
        improved = True
        while improved:
            improved = False
            for i in set(range(1, K + 1)) - current:
                cand = current | {i}
                consider(cand)
                if best_beta == cand:
                    current, improved = cand, True
            for i in list(current):
                cand = current - {i}
                consider(cand)
                if best_beta == cand:
                    current, improved = cand, True
    else:  # "prefix"
        for c in range(1, K + 1):
            consider(set(range(K - c + 1, K + 1)))

    sampling_strata = sorted([i for i in range(K + 1) if i not in best_beta])
    blocking_strata = sorted(list(best_beta))
    return sampling_strata, blocking_strata, best_n, best_mse


# --------------------------------------------------------------------------- #
# Estimators and variance helpers.
# --------------------------------------------------------------------------- #

def get_estimation(strata_population, strata_cnt_arrs, strata_sum_arrs):
    count_est = 0.0
    sum_est = 0.0
    for cnt_arr, sum_arr, pop in zip(strata_cnt_arrs, strata_sum_arrs, strata_population):
        if len(cnt_arr):
            count_est += float(np.mean(cnt_arr)) * len(pop)
            sum_est += float(np.mean(sum_arr)) * len(pop)
    avg_est = sum_est / count_est if count_est > 0 else 0.0
    return count_est, sum_est, avg_est


def get_avg_correction(pop_size, sample_size, count_mean, avg_mean, count_var):
    if count_mean <= 0 or sample_size <= 1:
        return 0.0
    return ((pop_size - sample_size) / max(pop_size - 1, 1)) * avg_mean * count_var / sample_size / (count_mean ** 2)


def get_stratified_var(strata_population, strata_sample_results):
    pop = sum(len(p) for p in strata_population)
    variance = 0.0
    for s_pop, arr in zip(strata_population, strata_sample_results):
        if len(arr) <= 1:
            continue
        stratum_var = float(np.var(arr, ddof=1))
        variance += ((len(s_pop) / pop) ** 2) * stratum_var / len(arr)
    return variance * (pop ** 2)


def get_avg_var(sample_size, count_var, count_mean, sum_var, sum_mean):
    if sample_size <= 0 or count_mean == 0:
        return 0.0
    return (1.0 / sample_size) * (
        sum_var / (count_mean ** 2) + count_var * (sum_mean ** 2) / (count_mean ** 4)
    )


def stats_func(strata_population, strata_cnt_arrs, strata_sum_arrs):
    count_est, sum_est, avg_est = get_estimation(strata_population, strata_cnt_arrs, strata_sum_arrs)
    count_var = get_stratified_var(strata_population, strata_cnt_arrs)
    pop = sum(len(p) for p in strata_population)
    n = sum(len(a) for a in strata_cnt_arrs)
    if count_est > 0 and n > 1:
        avg_est -= get_avg_correction(pop, n, count_est, avg_est, count_var)
    return count_est, sum_est, avg_est


def get_gt_strata(config, strata_population, dataset_sizes, dataset, oracle):
    """Debug helper; exhaustively labels each blocking-candidate stratum."""
    gts = [-1]
    for stratum_population in strata_population[1:]:
        data_ids = np.array(np.unravel_index(stratum_population, dataset_sizes)).T
        result = []
        for data_id in data_ids:
            if oracle.query(data_id):
                if config.aggregator == "count":
                    result.append(1)
                elif config.aggregator == "sum":
                    result.append(dataset.get_statistics(data_id))
                else:
                    result.append(dataset.get_statistics(data_id) / len(stratum_population))
        gts.append(sum(result))
    return gts


# --------------------------------------------------------------------------- #
# Main per-experiment routine.
# --------------------------------------------------------------------------- #

def _sample_stratum(stratum_proxy_weights, size, replace, scheme):
    """Weighted sample of `size` indices with the requested scheme.

    Uses ``np.random.choice`` directly. Pandas-based sampling allocates the
    weights array twice (Series + weights kwarg) which OOMs on full cross
    products of >1B pairs.
    """
    if scheme == "wor" and size < len(stratum_proxy_weights):
        replace = False
    return np.random.choice(
        len(stratum_proxy_weights),
        size=int(size),
        replace=replace,
        p=stratum_proxy_weights,
    )


def _is_sample_and_query(
    i, count_to_add, strata_weights, strata_population, dataset_sizes,
    eps_mix, scheme, dataset, oracle,
    strata_cnt, strata_sum, strata_stats, strata_probs,
    max_est, min_est,
):
    """Append IS samples to stratum i. Mutates strata_*. Returns updated max/min."""
    if count_to_add <= 0:
        return max_est, min_est
    w = strata_weights[i]
    if eps_mix > 0:
        w = defensive_mix(w, eps_mix)
    idxs = _sample_stratum(w, count_to_add, replace=True, scheme=scheme)
    flat_ids = strata_population[i][idxs]
    mat_ids = np.array(np.unravel_index(flat_ids, dataset_sizes)).T
    N_w = len(w)
    for s_flat, s_mat in zip(idxs, mat_ids):
        pi = float(w[s_flat])
        if oracle.query(s_mat):
            stats_val = dataset.get_statistics(s_mat)
            if stats_val > max_est: max_est = stats_val
            if stats_val < min_est: min_est = stats_val
            strata_cnt[i].append(1.0 / (N_w * pi))
            strata_sum[i].append(stats_val / (N_w * pi))
            strata_stats[i].append(stats_val)
        else:
            strata_cnt[i].append(0.0)
            strata_sum[i].append(0.0)
            strata_stats[i].append(None)
        strata_probs[i].append(pi)
    return max_est, min_est


def _recompute_pilot_stats(num_strata, strata_cnt, strata_sum, strata_stats, pop_sizes,
                            aggregator, var_shrinkage):
    """Recompute the per-stratum allocator inputs from current strata buffers."""
    cnt_unit_var = []
    sum_unit_var = []
    cnt_stratum_total = []
    sum_stratum_total = []
    pilot_extremes: List[Optional[float]] = []
    pilot_matched_values: List[List[float]] = []
    pilot_match_rates: List[float] = []
    for i in range(num_strata):
        c_arr = np.asarray(strata_cnt[i], dtype=float)
        s_arr = np.asarray(strata_sum[i], dtype=float)
        cnt_unit_var.append(float(np.var(c_arr, ddof=1)) if len(c_arr) > 1 else 0.0)
        sum_unit_var.append(float(np.var(s_arr, ddof=1)) if len(s_arr) > 1 else 0.0)
        c_mean = float(np.mean(c_arr)) if len(c_arr) else 0.0
        s_mean = float(np.mean(s_arr)) if len(s_arr) else 0.0
        cnt_stratum_total.append(pop_sizes[i] * c_mean)
        sum_stratum_total.append(pop_sizes[i] * s_mean)
        matched_vals = [float(v) for v in strata_stats[i] if v is not None]
        pilot_matched_values.append(matched_vals)
        n_pilot = max(len(strata_stats[i]), 1)
        pilot_match_rates.append(len(matched_vals) / n_pilot)
        if matched_vals:
            if aggregator == "max":
                pilot_extremes.append(float(max(matched_vals)))
            elif aggregator == "min":
                pilot_extremes.append(float(min(matched_vals)))
            else:
                pilot_extremes.append(None)
        else:
            pilot_extremes.append(None)
    if var_shrinkage > 0:
        ns = [max(len(strata_cnt[i]), 1) for i in range(num_strata)]
        total_n = float(sum(ns))
        cnt_pooled = sum(cnt_unit_var[i] * ns[i] for i in range(num_strata)) / total_n
        sum_pooled = sum(sum_unit_var[i] * ns[i] for i in range(num_strata)) / total_n
        for i in range(num_strata):
            cnt_unit_var[i] = (ns[i] * cnt_unit_var[i] + var_shrinkage * cnt_pooled) / (ns[i] + var_shrinkage)
            sum_unit_var[i] = (ns[i] * sum_unit_var[i] + var_shrinkage * sum_pooled) / (ns[i] + var_shrinkage)
    return (cnt_unit_var, sum_unit_var, cnt_stratum_total, sum_stratum_total,
            pilot_extremes, pilot_matched_values, pilot_match_rates)


def run_once(
    config: Config,
    dataset,
    oracle,
    dataset_sizes,
    count_gt, sum_gt, avg_gt, min_gt, max_gt, median_gt,
    strata_weights,
    strata_W_sums,
    strata,
    strata_population,
    strata_sample_sizes,
    sampling_budget: int,
    seed: Optional[int] = None,
):
    if seed is not None:
        np.random.seed(seed)

    eps_mix = float(getattr(config, "defensive_mix_ratio", 0.0))
    scheme = getattr(config, "sampling_scheme", "wr")
    search = getattr(config, "allocation_search", "prefix")

    num_strata = len(strata_population)

    strata_cnt = [[] for _ in range(num_strata)]   # O(x)/π per matched sample
    strata_sum = [[] for _ in range(num_strata)]   # g(x)*O(x)/π per matched sample
    strata_stats = [[] for _ in range(num_strata)] # raw g(x) or None
    strata_probs = [[] for _ in range(num_strata)] # π(x) per sample

    max_est = float("-inf")
    min_est = float("inf")

    # ---------- Pilot sampling for every stratum ----------
    for i in range(num_strata):
        w = strata_weights[i]
        if eps_mix > 0:
            w = defensive_mix(w, eps_mix)
        n_pilot = int(strata_sample_sizes[i])
        idxs = _sample_stratum(w, n_pilot, replace=True, scheme=scheme)
        flat_ids = strata_population[i][idxs]
        mat_ids = np.array(np.unravel_index(flat_ids, dataset_sizes)).T
        N_w = len(w)
        for s_flat, s_mat in zip(idxs, mat_ids):
            pi = float(w[s_flat])
            if oracle.query(s_mat):
                stats_val = dataset.get_statistics(s_mat)
                if stats_val > max_est: max_est = stats_val
                if stats_val < min_est: min_est = stats_val
                strata_cnt[i].append(1.0 / (N_w * pi))
                strata_sum[i].append(stats_val / (N_w * pi))
                strata_stats[i].append(stats_val)
            else:
                strata_cnt[i].append(0.0)
                strata_sum[i].append(0.0)
                strata_stats[i].append(None)
            strata_probs[i].append(pi)

    # ---------- Pilot variance statistics + Q1 shrinkage ----------
    pop_sizes = [len(p) for p in strata_population]
    W_sums = list(strata_W_sums)
    var_shrinkage = float(getattr(config, "var_shrinkage", 0.0))

    (cnt_unit_var, sum_unit_var, cnt_stratum_total, sum_stratum_total,
     pilot_extremes, pilot_matched_values, pilot_match_rates) = _recompute_pilot_stats(
         num_strata, strata_cnt, strata_sum, strata_stats, pop_sizes,
         config.aggregator, var_shrinkage)

    # ---------- Q2: force-block-when-concentrated ----------
    force_block_concentrated = bool(getattr(config, "force_block_concentrated", False))
    fb_threshold = float(getattr(config, "force_block_threshold", 100.0))

    def _detect_forced_beta(match_rates):
        if not force_block_concentrated or num_strata < 2:
            return None
        head_rate = match_rates[0]
        forced = set()
        for i in range(1, num_strata):
            if head_rate <= 0 and match_rates[i] > 0:
                forced.add(i)
            elif head_rate > 0 and match_rates[i] / head_rate >= fb_threshold:
                forced.add(i)
        return forced or None

    def _allocate(cnt_uv, sum_uv, cnt_st, sum_st, ext, vals, rates):
        forced = _detect_forced_beta(rates)
        if forced is not None:
            n_forced = _alloc_for_subset(forced, pop_sizes, W_sums, sampling_budget)
            if n_forced is not None:
                samp = sorted(i for i in range(num_strata) if i not in forced)
                return samp, sorted(forced), n_forced, None
        return adaptive_allocation(
            pop_sizes, W_sums, cnt_uv, sum_uv, cnt_st, sum_st,
            sampling_budget=sampling_budget,
            aggregator=config.aggregator,
            search=search,
            pilot_extremes=ext if config.aggregator in ("max", "min") else None,
            pilot_matched_values=vals if config.aggregator in ("max", "min") else None,
            pilot_match_rates=rates if config.aggregator in ("max", "min") else None,
        )

    sampling_strata, blocking_strata, n_per_stratum, best_mse = _allocate(
        cnt_unit_var, sum_unit_var, cnt_stratum_total, sum_stratum_total,
        pilot_extremes, pilot_matched_values, pilot_match_rates)

    logging.debug(
        "[BaS Stage1] agg=%s K=%d β=%s sampling=%s n_target=%s best_mse=%s",
        config.aggregator, num_strata - 1, blocking_strata,
        sampling_strata, n_per_stratum, best_mse,
    )

    # ---------- Q5: optional two-stage refinement ----------
    two_stage = bool(getattr(config, "two_stage_allocation", False))
    if two_stage:
        # Stage-1 main: top up sampling strata to 30% of (n_target - current_pilot)
        for i in sampling_strata:
            current_n = len(strata_cnt[i])
            target_n = int(n_per_stratum.get(i, current_n))
            extra_30 = int(0.3 * max(0, target_n - current_n))
            max_est, min_est = _is_sample_and_query(
                i, extra_30, strata_weights, strata_population, dataset_sizes,
                eps_mix, scheme, dataset, oracle,
                strata_cnt, strata_sum, strata_stats, strata_probs,
                max_est, min_est)
        # Recompute stats from extended samples and re-allocate
        (cnt_unit_var, sum_unit_var, cnt_stratum_total, sum_stratum_total,
         pilot_extremes, pilot_matched_values, pilot_match_rates) = _recompute_pilot_stats(
             num_strata, strata_cnt, strata_sum, strata_stats, pop_sizes,
             config.aggregator, var_shrinkage)
        sampling_strata, blocking_strata, n_per_stratum, best_mse = _allocate(
            cnt_unit_var, sum_unit_var, cnt_stratum_total, sum_stratum_total,
            pilot_extremes, pilot_matched_values, pilot_match_rates)
        logging.debug(
            "[BaS Stage2] agg=%s β=%s sampling=%s n_target=%s best_mse=%s",
            config.aggregator, blocking_strata, sampling_strata,
            n_per_stratum, best_mse,
        )

    # ---------- Top up sampling strata to final n_target ----------
    for i in sampling_strata:
        current_n = len(strata_cnt[i])
        target_n = int(n_per_stratum.get(i, current_n))
        max_est, min_est = _is_sample_and_query(
            i, target_n - current_n, strata_weights, strata_population, dataset_sizes,
            eps_mix, scheme, dataset, oracle,
            strata_cnt, strata_sum, strata_stats, strata_probs,
            max_est, min_est)

    # ---------- Deterministic blocking: enumerate every tuple in each β stratum ----------
    for i in blocking_strata:
        strata_cnt[i] = []
        strata_sum[i] = []
        strata_stats[i] = []
        strata_probs[i] = []
        flat_ids = strata_population[i]
        mat_ids = np.array(np.unravel_index(flat_ids, dataset_sizes)).T
        for s_mat in mat_ids:
            if oracle.query(s_mat):
                stats_val = dataset.get_statistics(s_mat)
                if stats_val > max_est: max_est = stats_val
                if stats_val < min_est: min_est = stats_val
                strata_cnt[i].append(1.0)
                strata_sum[i].append(stats_val)
                strata_stats[i].append(stats_val)
            else:
                strata_cnt[i].append(0.0)
                strata_sum[i].append(0.0)
                strata_stats[i].append(None)
            strata_probs[i].append(1.0)

    # ---------- Vectorise for estimation / bootstrap ----------
    cnt_arrs = [np.asarray(a, dtype=float) for a in strata_cnt]
    sum_arrs = [np.asarray(a, dtype=float) for a in strata_sum]

    count_est, sum_est, avg_est = stats_func(strata_population, cnt_arrs, sum_arrs)
    count_var = get_stratified_var(strata_population, cnt_arrs)
    sum_var = get_stratified_var(strata_population, sum_arrs)

    total_sample_size = sum(len(a) for a in cnt_arrs)
    population_size = sum(pop_sizes)
    if count_est > 0:
        avg_var = get_avg_var(
            total_sample_size,
            count_var / population_size ** 2,
            count_est / population_size,
            sum_var / population_size ** 2,
            sum_est / population_size,
        )
    else:
        avg_var = 0.0

    # median via weighted-percentile helper (CDF-based version is a follow-up)
    try:
        median_est = cal_weighted_median(strata_stats, strata_probs)
    except Exception as exc:
        median_est = 0.0
        logging.debug("median estimation failed: %s", exc)

    if config.aggregator == "count":
        point_estimate, point_var, gt = count_est, count_var, count_gt
    elif config.aggregator == "sum":
        point_estimate, point_var, gt = sum_est, sum_var, sum_gt
    elif config.aggregator == "min":
        point_estimate, point_var, gt = min_est if min_est != float("inf") else 0.0, 0.0, min_gt
    elif config.aggregator == "max":
        point_estimate, point_var, gt = max_est if max_est != float("-inf") else 0.0, 0.0, max_gt
    elif config.aggregator == "median":
        point_estimate, point_var, gt = median_est, 0.0, median_gt
    else:  # avg
        point_estimate, point_var, gt = avg_est, avg_var, avg_gt

    if not getattr(config, "ci", False):
        est = Estimates(config.oracle_budget, gt, point_estimate, 0, 0)
        est.log()
        est.save(config.output_file, f"_{config.aggregator}")
        return

    # ---------- CIs ----------
    if config.aggregator in ("min", "max"):
        try:
            gmin, gmax = dataset.get_min_max_statistics()
        except Exception:
            gmin, gmax = point_estimate, point_estimate
        if config.aggregator == "max":
            lb, ub = point_estimate, float(gmax)
        else:
            lb, ub = float(gmin), point_estimate
        est = Estimates(config.oracle_budget, gt, point_estimate, lb, ub)
        est.log()
        est.save(config.output_file, f"_{config.aggregator}")
        return

    if config.aggregator == "median":
        # percentile-based bootstrap CI (paper analytic-variance version is a follow-up)
        ms = []
        for _ in range(config.bootstrap_trials):
            rs_vals = []
            rs_probs = []
            for s_vals, s_probs in zip(strata_stats, strata_probs):
                if not s_vals:
                    rs_vals.append([])
                    rs_probs.append([])
                    continue
                idx_res = np.random.choice(len(s_vals), len(s_vals), replace=True)
                rs_vals.append([s_vals[j] for j in idx_res])
                rs_probs.append([s_probs[j] for j in idx_res])
            try:
                ms.append(cal_weighted_median(rs_vals, rs_probs))
            except Exception:
                continue
        if ms:
            lb = float(np.percentile(ms, 100 * (1 - config.confidence_level) / 2))
            ub = float(np.percentile(ms, 100 * (config.confidence_level + (1 - config.confidence_level) / 2)))
        else:
            lb, ub = point_estimate, point_estimate
        est = Estimates(config.oracle_budget, gt, point_estimate, lb, ub)
        est.log()
        est.save(config.output_file, f"_{config.aggregator}")
        return

    # count / sum / avg → bootstrap-t
    ts = []
    for _ in range(config.bootstrap_trials):
        cnt_rs, sum_rs = [], []
        for i in range(num_strata):
            if len(cnt_arrs[i]) == 0:
                cnt_rs.append(cnt_arrs[i])
                sum_rs.append(sum_arrs[i])
                continue
            idx_res = np.random.choice(len(cnt_arrs[i]), len(cnt_arrs[i]), replace=True)
            cnt_rs.append(cnt_arrs[i][idx_res])
            sum_rs.append(sum_arrs[i][idx_res])
        cnt_b, sum_b, avg_b = stats_func(strata_population, cnt_rs, sum_rs)
        cnt_var_b = get_stratified_var(strata_population, cnt_rs)
        sum_var_b = get_stratified_var(strata_population, sum_rs)
        avg_var_b = (
            get_avg_var(total_sample_size, cnt_var_b, cnt_b, sum_var_b, sum_b)
            if cnt_b > 0 else 0.0
        )
        if config.aggregator == "count":
            denom = np.sqrt(cnt_var_b) if cnt_var_b > 0 else 1.0
            ts.append((cnt_b - count_est) / denom)
        elif config.aggregator == "sum":
            denom = np.sqrt(sum_var_b) if sum_var_b > 0 else 1.0
            ts.append((sum_b - sum_est) / denom)
        else:
            denom = np.sqrt(avg_var_b) if avg_var_b > 0 else 1.0
            ts.append((avg_b - avg_est) / denom)

    lb, ub = get_ci_bootstrap_ttest(point_estimate, ts, point_var, confidence_level=config.confidence_level)
    est = Estimates(config.oracle_budget, gt, point_estimate, lb, ub)
    est.log()
    est.save(config.output_file, f"_{config.aggregator}")


# --------------------------------------------------------------------------- #
# Run entry: stratify, then dispatch per-experiment.
# --------------------------------------------------------------------------- #

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)
    logging.info(config)

    dataset = load_dataset(config)
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = (dataset_sizes[0], dataset_sizes[0])

    proxy_scores = get_proxy_score(config, dataset)
    proxy_rank = get_proxy_rank(config, dataset, proxy_scores)

    oracle = Oracle(config)
    count_gt, sum_gt, avg_gt, min_gt, max_gt, median_gt = dataset.get_gt(oracle)
    logging.info(
        "count_gt=%s sum_gt=%s avg_gt=%s min_gt=%s max_gt=%s median_gt=%s",
        count_gt, sum_gt, avg_gt, min_gt, max_gt, median_gt,
    )

    w_exp = float(getattr(config, "w_exp", 1.0))
    sample_size = int((1 - config.max_blocking_ratio) * config.oracle_budget)
    blocking_size_upperbound = config.oracle_budget - sample_size
    N = int(proxy_rank.shape[0])

    # Chunked sum: avoids materialising `np.power(arr, w_exp)` over multi-GB
    # arrays in one shot. 16M-element chunks → ~128 MB temp per step.
    _CHUNK = 1 << 24

    def _stratum_W(scores_view):
        if w_exp == 1.0:
            return float(np.sum(scores_view, dtype=np.float64))
        s = 0.0
        n = len(scores_view)
        for i in range(0, n, _CHUNK):
            s += float(np.sum(np.power(scores_view[i:i + _CHUNK], w_exp, dtype=np.float64)))
        return s

    # ---- Build strata layout per paper §5.2 stratification ----
    # Stratum 0 (head): sampling-only, holds the low-similarity tail.
    # Strata 1..K (blocking-candidates): equal-size in [N-blocking_cap, N).
    # K is determined by how much pilot budget remains after the head stratum
    # claims its weight-proportional share.
    strata = [[0, N - blocking_size_upperbound]]
    strata_population = [proxy_rank[strata[0][0]:strata[0][1]]]

    head_w = _stratum_W(proxy_scores[strata_population[0]])
    # Coarse total estimate for sizing; refined after layout is fixed.
    full_total = _stratum_W(proxy_scores)
    head_pilot = max(1, int(head_w / full_total * sample_size))

    strata_div = int(getattr(config, "strata_size", 1000))
    strata_div = max(strata_div, 100)
    sample_size_remaining = max(0, sample_size - head_pilot)
    num_strata = max(int(sample_size_remaining / strata_div), 1)
    blocking_stratum_size = max(1, int(blocking_size_upperbound / num_strata))

    for i in range(num_strata):
        if i != num_strata - 1:
            strata.append([strata[i][1], strata[i][1] + blocking_stratum_size])
        else:
            strata.append([strata[i][1], N])
        strata_population.append(proxy_rank[strata[i + 1][0]:strata[i + 1][1]])

    # Final per-stratum weight sums.
    strata_W_sums = [_stratum_W(proxy_scores[p]) for p in strata_population]

    # Pilot sizes per paper §5.2: head gets weight-proportional share; each
    # blocking-candidate stratum gets ≥strata_div pilot samples (target ~1000
    # per stratum) to enable reliable per-stratum variance estimation. Cap at
    # stratum size so we don't over-sample tiny strata. For very small total
    # budgets (e.g. Movie-Q5 at b=1000) the strata_div floor would push total
    # pilot above sample_size — in that case we shrink the head share to
    # restore the budget invariant rather than starving the blocking strata.
    strata_sample_sizes = [head_pilot]
    for i in range(1, num_strata + 1):
        cap = len(strata_population[i])
        proportional = int(strata_W_sums[i] / full_total * sample_size)
        target = min(cap, max(strata_div, proportional))
        strata_sample_sizes.append(target)
    total_pilot = sum(strata_sample_sizes)
    if total_pilot > sample_size:
        # Over-spent: shrink the head pilot to compensate. The blocking-
        # candidate strata keep their floor (paper's variance-estimation
        # guarantee), the head loses the slack.
        slack = total_pilot - sample_size
        strata_sample_sizes[0] = max(1, strata_sample_sizes[0] - slack)

    # Pre-compute normalized weights per stratum once, in float32 chunks. The
    # head stratum on full cross products (e.g. Quora ~2.9B entries,
    # Flickr30k ~5.0B entries) makes a single full materialisation infeasible;
    # we write directly into a pre-allocated float32 output array using small
    # chunks to bound peak temporary use.
    def _stratum_weights(pop, w_sum):
        n = len(pop)
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, _CHUNK):
            sl = proxy_scores[pop[i:i + _CHUNK]]
            if w_exp != 1.0:
                sl = np.power(sl, w_exp, dtype=np.float64)
            else:
                sl = np.asarray(sl, dtype=np.float64)
            sl /= w_sum
            out[i:i + _CHUNK] = sl.astype(np.float32, copy=False)
        # Re-normalize after float32 cast so np.random.choice's sum-check
        # passes (atol ~1e-8). For ~3e9 entries the cumulative cast error is
        # roughly N · eps ~= 3e9 · 1e-7 ≈ 300, dwarfing the unity sum.
        s = float(out.sum(dtype=np.float64))
        if s > 0 and abs(s - 1.0) > 1e-6:
            out *= np.float32(1.0 / s)
        return out

    strata_weights = [_stratum_weights(p, w) for p, w in zip(strata_population, strata_W_sums)]

    # Free the raw proxy_scores once we have the stratum-level weights — large
    # cross-products keep ~tens of GB pinned otherwise.
    del proxy_scores
    import gc
    gc.collect()

    logging.info(
        "K=%d strata_sizes=%s pilot_sizes=%s sample_budget=%s blocking_cap=%s W_sums=%s",
        num_strata, [s[1] - s[0] for s in strata],
        strata_sample_sizes, sample_size, blocking_size_upperbound, strata_W_sums,
    )

    base_seed = int(getattr(config, "seed", 42))
    for exp_id in range(config.internal_loop):
        seed = base_seed + exp_id
        logging.info("[exp=%d] seed=%d", exp_id, seed)
        run_once(
            config, dataset, oracle, dataset_sizes,
            count_gt, sum_gt, avg_gt, min_gt, max_gt, median_gt,
            strata_weights, strata_W_sums,
            strata, strata_population, strata_sample_sizes,
            sampling_budget=int(config.oracle_budget),
            seed=seed,
        )

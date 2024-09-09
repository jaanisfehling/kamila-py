import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


class KAMILA:
    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter_per_init: int = 25,
        tol: float = 1e-4,
        num_weights: np.ndarray = None,
        cat_weights: np.ndarray = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter_per_init = max_iter_per_init
        self.tol = tol
        self.num_weights = num_weights
        self.cat_weights = cat_weights

    def fit_predict(self, X_num, X_cat):
        if X_num.size and X_num.ndim:
            self.has_num = True
            n_samples = X_num.shape[0]
            n_num_features = X_num.shape[1]
        else:
            self.has_num = False
        if X_cat.size and X_cat.ndim:
            self.has_cat = True
            n_samples = X_cat.shape[0]
            n_cat_features = X_cat.shape[1]
        else:
            self.has_cat = False

        if self.num_weights is None and self.has_num:
            self.num_weights = np.ones(n_num_features)
        if self.cat_weights is None and self.has_cat:
            self.cat_weights = np.ones(n_cat_features)

        objective = np.zeros(self.n_init)
        dist = np.zeros((n_samples, self.n_clusters))
        win_dist = np.zeros(self.n_init)
        total_dist = np.sum(np.linalg.norm(X_num * self.num_weights - np.mean(X_num, axis=0) * self.num_weights))

        for init in range(self.n_init):
            # Initialize means and conditional log probabilities
            if self.has_num:
                means = []
                for p in range(X_num.shape[1]):
                    means.append(np.random.uniform(np.min(X_num[:, p]), np.max(X_num[:, p]), self.n_clusters))
                means = np.asarray(means).T  # Shape: k x p
            if self.has_cat:
                log_probs_cond = [
                    np.random.dirichlet(np.ones(len(np.unique(X_cat[:, q]))).T, self.n_clusters)
                    for q in range(n_cat_features)
                ]  # Shape: q x k x L_q

            memb_old = memb_new = np.zeros(n_samples)
            n_iter = 0
            while (n_iter < 3 or not np.all(memb_new == memb_old)) and n_iter < self.max_iter_per_init:
                n_iter += 1

                if self.has_num:
                    # Euclidean distance between numerical features and centroids
                    dist = np.linalg.norm(
                        np.broadcast_to(
                            (X_num * self.num_weights)[:, :, np.newaxis],
                            (n_samples, n_num_features, self.n_clusters),
                        )
                        - np.broadcast_to(
                            (means * self.num_weights).T[np.newaxis, :, :],
                            (n_samples, n_num_features, self.n_clusters),
                        ),
                        axis=1,
                    )

                    # Get minimum distances
                    r = np.min(dist, axis=1)

                    # KDE of minimum distances
                    log_dist_rad_dens = radial_kde(r, dist, n_num_features)  # Shape: n x k

                if self.has_cat:
                    # Log probs of categorical features
                    individual_log_probs = get_individual_log_probs(
                        X_cat, self.cat_weights, log_probs_cond
                    )  # Shape: q x n x k
                    cat_log_liks = np.sum(individual_log_probs, axis=0)  # Shape: n x k

                if self.has_num and self.has_cat:
                    all_log_liks = np.log(log_dist_rad_dens) + cat_log_liks
                elif self.has_num:
                    all_log_liks = np.log(log_dist_rad_dens)
                else:
                    all_log_liks = cat_log_liks

                memb_old = memb_new
                memb_new = np.argmin(all_log_liks, axis=1)

                if self.has_num:
                    means = aggregate_means(X_num, memb_new, self.n_clusters, n_samples, n_num_features)  # Shape: k x p

                if self.has_cat:
                    joint_probs = joint_tab_smoothed(
                        X_cat, memb_new, np.array([len(np.unique(q)) for q in X_cat.T]), 0.025, self.n_clusters
                    )  # Shape: q x k x L_q
                    log_probs_cond = [np.log(q / np.sum(q, axis=1)[:, np.newaxis]) for q in joint_probs]

            win_dist[init] = np.sum(dist[np.arange(n_samples), memb_new])
            win_to_bet_rat = win_dist[init] / (total_dist - win_dist[init])
            if win_to_bet_rat < 0:
                win_to_bet_rat = 100
            objective[init] = win_to_bet_rat * objective[init]

            if init == 0 or (init > 0 and objective[init] > np.max(objective)):
                final_objective = objective[init]
                final_memb = memb_new

        return final_memb


def aggregate_means(X_num, memb_new, k, n_samples, n_num_features):
    means = np.zeros((k, n_num_features))
    count_vec = np.zeros(k)

    for n in range(n_samples):
        means[memb_new[n]] += X_num[n]
        count_vec[memb_new[n]] += 1

    # Avoid division by zero
    count_vec = np.maximum(count_vec, 1)
    return means / count_vec[:, np.newaxis]


def get_individual_log_probs(X_cat, cat_weights, log_probs_cond):
    qq = len(cat_weights)
    nn, _ = X_cat.shape
    kk = log_probs_cond[0].shape[0]

    out_list = []
    for q in range(qq):
        ith_out_mat = np.zeros((nn, kk))
        ith_var_codes = X_cat[:, q]
        ith_log_liks = log_probs_cond[q]

        for n in range(nn):
            for k in range(kk):
                ith_out_mat[n, k] = cat_weights[q] * ith_log_liks[k, ith_var_codes[n]]

        out_list.append(ith_out_mat)

    return out_list


def radial_kde(radii, eval_points, pdim):
    MAX_DENS = 1

    kde = gaussian_kde(radii, bw_method="silverman")
    x = np.linspace(0, np.max(eval_points))
    y = kde(x)

    # Remove any zero and negative density estimates
    nonneg_mask = y > 0
    if not np.all(nonneg_mask):
        min_pos = np.min(y[nonneg_mask])
        y[~nonneg_mask] = min_pos / 100

    # At bottom 5th percentile, replace with line through (0,0) and (q05,f(q05))
    # This removes substantial variability in output near zero
    quant05 = np.percentile(x, 5)
    coords_lt_q05 = x < quant05
    max_pt = np.max(np.where(coords_lt_q05)[0])
    y[coords_lt_q05] = x[coords_lt_q05] * (y[max_pt] / x[max_pt])

    # Radial Jacobian transformation; up to proportionality constant
    rad_y = np.zeros_like(y)
    rad_y[1:] = y[1:] / x[1:] ** (pdim - 1)

    # Replace densities over MAX_DENS with MAX_DENS
    over_max = rad_y > MAX_DENS
    rad_y[over_max] = MAX_DENS

    # Normalize to area 1
    binwidth_x = x[1] - x[0]
    dens_r = rad_y / (binwidth_x * np.sum(rad_y))

    # Create resampling function
    resampler = interp1d(x, dens_r, bounds_error=False, fill_value=(dens_r[0], dens_r[-1]))
    kdes = resampler(eval_points)

    return kdes


def tabulate_two_int_vec(vec1, vec2, nc1, nc2, nn):
    out_mat = np.zeros((nc1, nc2), dtype=int)
    for i in range(nn):
        out_mat[vec1[i] - 1, vec2[i] - 1] += 1
    return out_mat


def smooth_2d_table(input_tab, cat_bw, nn):
    dim1, dim2 = input_tab.shape
    mid_mat = np.zeros((dim1, dim2))
    out_mat = np.zeros((dim1, dim2))

    col_sums = input_tab.sum(axis=0)

    for i in range(dim1):
        for j in range(dim2):
            off_counts1 = col_sums[j] - input_tab[i, j]
            mid_mat[i, j] = (1 - cat_bw) * input_tab[i, j] + cat_bw / (dim1 - 1) * off_counts1

    row_sums = mid_mat.sum(axis=1)

    for i in range(dim1):
        for j in range(dim2):
            off_counts2 = row_sums[i] - mid_mat[i, j]
            out_mat[i, j] = (1 - cat_bw) * mid_mat[i, j] + cat_bw / (dim2 - 1) * off_counts2

    return out_mat


def joint_tab_smoothed(X_cat, memb_new, num_lev, cat_bw, kk):
    nn, qq = X_cat.shape
    out_list = []

    for q in range(qq):
        qth_var = X_cat[:, q]
        qth_tab_raw = tabulate_two_int_vec(memb_new, qth_var, kk, num_lev[q], nn)

        if cat_bw != 0:
            out_list.append(smooth_2d_table(qth_tab_raw, cat_bw, nn))
        else:
            out_list.append(qth_tab_raw)

    return out_list

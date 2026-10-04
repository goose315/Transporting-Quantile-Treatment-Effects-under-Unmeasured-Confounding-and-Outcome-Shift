"""Proposed target-QTE inference: one file, public API and Exp1--3 reproduction.

QUICK START (Python 3.12.14 was used for the archived runs)
  python -m pip install numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 scikit-learn==1.8.0 joblib==1.5.3 threadpoolctl==3.6.0 matplotlib==3.10.8 xgboost==3.4.1
  python proposed_qte.py --verify
  python proposed_qte.py --experiment all --jobs 4 --out results_reproduced

This file contains every project-specific function. No other .py, saved result
file or external dataset is required for the simulations or --verify. It uses
the standard scientific packages above. No package installation, download or
experiment runs occur on import. --verify regenerates fixed-seed n=2000 samples
and checks estimates, standard errors and interval limits against embedded
archived checkpoints. It is a small reproducibility check, not 500 repetitions.

PROPOSED API (for your own numeric data)
  from proposed_qte import ProposedQTE, region_membership
  model = ProposedQTE(profile="exp3", seed=2027).fit(X, R, A, Y)
  result = model.bounds([(1,1), (1.25,1), (1.5,1.25)], [.25,.5,.75])
  result.table().to_csv("proposed_intervals.csv", index=False)
  L_minus, U_minus, L_plus, U_plus = result.ci("sim")
  certified, excluded = result.robustness_sets()[0.5]
  inner, outer = region_membership(result.S, certified, excluded, evaluation_pairs)

X has shape (n,d); R=1 denotes source and R=0 target. Source A is 0/1 and
source Y is observed. Target A/Y may be NaN and are always masked by the public
API. Shared covariates must be numeric, finite and consistently encoded.
ProposedQTE always uses the nested sensitivity bounds and full one-step
correction, including both outcome-CDF and propensity derivatives. It does
not use DGP truths. ArrayNuisance supports externally cross-fitted or oracle
nuisances. See ProposedQTE and BoundsResult docstrings for input/output shapes.
The original low-level classes and comparison methods remain available for
faithful reproduction of the published comparison tables.

REPRODUCTION COMMANDS
  # Full original defaults: all experiments, both Exp2 regimes, 500 repetitions.
  python proposed_qte.py --jobs 4 --out results_reproduced
  # Only XGBoost ML (omits the separate known-basis rate experiment).
  python proposed_qte.py --experiment all --regime ml --jobs 4 --out results_ml
  # Short end-to-end run; --reps changes repetitions, not the bootstrap count.
  python proposed_qte.py --experiment 23 --regime ml --n 2000 --reps 2 --jobs 2 --out results_smoke
  # Resume this file's own run with exactly the same settings, adding --resume.
  # Rebuild tables/Exp1 plot: --summarize-only --out results_reproduced
  # Reproduce another sample size: --n 6000 (does not recover a different seed).

FROZEN EXPERIMENT SETTINGS
  n=2000,4000,8000; 500 repetitions per experiment; base seed=2027.
  XGBoost; 5 outer folds; 5 inner location-residual folds; alpha=.05;
  500 common Gaussian multiplier draws; single-threaded worker numerics.
  X1,...,X4 independent N(0,1), epsilon ~ N(0,1) truncated to [-3,3].
  Y(a)=.5*tanh(X1)+.20*sin(X2)+.15*X3+a+.5*(1+.6*a*tanh(X1))*epsilon.
  rho(x)=expit(-tanh(x1)), e1(x)=expit(tanh(x1)).
  Exp1: s=(1.5,1.5), tau=.1,.25,.5,.75,.9; original comparisons retained.
  Exp2/3: s=(1,1),(1.25,1),(1.5,1.25), tau=.25,.5,.75.
  Exp2 alone: eta=.1 and source-mean normalization of omega.
  Exp1/3: original clipping and unnormalized omega.
  Density='auto' resolves to the conditional residual KDE for XGBoost and to
  the true conditional PDF for Oracle. It is NOT finite differences here.
  Bootstrap seeds: replication seed +1 (Exp1), +0 (Exp2), +7 (Exp3).
  Independent replication seeds use SeedSequence with experiment tags, not
  base_seed+r. Exp2 ML and Exp3 share each sample and fitted nuisances.

OUTPUT AND INTERPRETATION
  Each output directory includes design.json (all settings, package versions,
  source hash, seeds), checksummed replication CSVs, summary CSV/LaTeX tables,
  and the Exp1 three-panel figure. Results cannot be silently resumed across
  different source versions. Use a new output directory for this file.
  Exp2 reports endpointwise coverage, SD, signed bias and full CI width for
  Proposed, Plug-in and Oracle. Plug-in retains the historical EIF-based
  variance calculation; its calibration is evaluated, not assumed valid.
  Exp3 reports full-grid simultaneous endpoint/set coverage, interval widths,
  Oracle comparisons and region diagnostics. Within-pair coverage is a
  different event from full-grid coverage. ci('sim') calibrates the complete
  supplied grid. Subset the returned table after calibration if needed.
  Oracle still uses empirical source and target averages.

Bit-level results depend on the package versions and numerical platform.
--verify uses an absolute tolerance of 1e-10 and reports the actual maximum
error. The full simulation entry retains the historical numerical kernel,
DGP, seed generation and result definitions. This file does not embed the
Natality dataset or claim to recreate the empirical illustration without it.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit, ndtr, ndtri
from scipy.stats import norm, qmc
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, parallel_config, cpu_count
from threadpoolctl import threadpool_limits



# Estimator and nuisance learners


LOWER, UPPER = -1, 1


KINDS = ("nested", "product", "collapsed")


VARIANTS = ("onestep", "plugin", "no_Ge")


def T_map(v, kappa, sign):
    """Primitive map T^sign_kappa(v) and its right-selected derivative.

    sign = -1 : T^-_k(v) = max{v/k, 1-k(1-v)}, switching level k/(1+k)
    sign = +1 : T^+_k(v) = min{k v, 1-(1-v)/k}, switching level 1/(1+k)
    At a switching level the derivative of the piece on the right is used
    (Appendix A.3 of the paper).  kappa may be an array broadcastable to v.
    """
    v = np.asarray(v, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if sign < 0:
        right = v >= kappa / (1.0 + kappa)
        val = np.where(right, 1.0 - kappa * (1.0 - v), v / kappa)
        der = np.where(right, kappa, 1.0 / kappa)
    else:
        right = v >= 1.0 / (1.0 + kappa)
        val = np.where(right, 1.0 - (1.0 - v) / kappa, kappa * v)
        der = np.where(right, 1.0 / kappa, kappa)
    return val, der


def envelope(p, e, Gamma, Lam, sign, kind="nested"):
    """Conditional target-CDF bound b = G^sign_s(p, e) and its partials (b_p, b_e).

    p     : conditional CDF values p_a(y, x) in [0, 1]
    e     : arm-a propensity e_a(x) in (0, 1)
    Gamma, Lam : scalars or arrays broadcastable to p
    sign  : -1 (lower CDF bound) or +1 (upper CDF bound)
    kind  : "nested" (sharp, proposed), "product", or "collapsed"
    """
    p = np.clip(p, 0.0, 1.0)
    if kind == "nested":
        tg, dtg = T_map(p, Gamma, sign)
        C = e * p + (1.0 - e) * tg                      # source CDF bound C^sign_Gamma
        tl, dtl = T_map(C, Lam, sign)
        return tl, dtl * (e + (1.0 - e) * dtg), dtl * (p - tg)
    if kind == "collapsed":
        val, der = T_map(p, np.asarray(Gamma) * np.asarray(Lam), sign)
        return val, der, np.zeros_like(val)
    if kind == "product":
        l = (e + (1.0 - e) / Gamma) / Lam                  # lower LR bound
        u = Lam * (e + (1.0 - e) * Gamma)                  # upper LR bound
        dl = (1.0 - 1.0 / np.asarray(Gamma)) / Lam          # d l / d e
        du = Lam * (1.0 - np.asarray(Gamma))                # d u / d e
        if sign < 0:                                        # max{l p, 1 - u(1-p)}
            v1, v2 = l * p, 1.0 - u * (1.0 - p)
            right = v2 >= v1
            val = np.where(right, v2, v1)
            dp = np.where(right, u, l)
            de = np.where(right, -(1.0 - p) * du, p * dl)
        else:                                               # min{u p, 1 - l(1-p)}
            v1, v2 = u * p, 1.0 - l * (1.0 - p)
            right = v2 <= v1
            val = np.where(right, v2, v1)
            dp = np.where(right, l, u)
            de = np.where(right, -(1.0 - p) * dl, p * du)
        return np.clip(val, 0.0, 1.0), dp, de
    raise ValueError(f"unknown kind {kind!r}")


class ArmInteractedSplines(BaseEstimator, TransformerMixin):
    """Feature map Z = [X, A] -> [B(X), A * B(X), A, P(X)] for penalized (ridge) regression.

    B(X): additive cubic spline bases of every covariate; A * B(X): arm-specific components
    (the last design column is the arm indicator when arm_last=True); P(X): all pairwise
    products of the standardized covariates (pairwise=True), i.e. a second-order interaction
    layer shared by the arms.  Used on the pooled source design (X, A).
    """

    def __init__(self, n_knots=6, arm_last=True, pairwise=True):
        self.n_knots = n_knots
        self.arm_last = arm_last
        self.pairwise = pairwise

    def fit(self, Z, y=None):
        from sklearn.preprocessing import SplineTransformer

        Z = np.asarray(Z, float)
        X = Z[:, :-1] if self.arm_last else Z
        self.st_ = SplineTransformer(n_knots=self.n_knots, degree=3).fit(X)
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0) + 1e-12
        return self

    def transform(self, Z):
        Z = np.asarray(Z, float)
        X = Z[:, :-1] if self.arm_last else Z
        parts = [self.st_.transform(X)]
        if self.arm_last:
            a = Z[:, -1:]
            parts += [a * parts[0], a]
        if self.pairwise and X.shape[1] > 1:
            S = (X - self.mu_) / self.sd_
            i, j = np.triu_indices(X.shape[1], 1)
            parts.append(S[:, i] * S[:, j])
        return np.hstack(parts)


def make_learner(learner, task, random_state=None, arm_last=False):
    """Return a fresh scikit-learn estimator.

    learner : 'gbm'       histogram gradient boosting (shallow trees, early stopping)
              'spline'    additive cubic splines + cross-validated logistic / ridge penalty
              'armspline' (reg) additive splines with arm-specific components plus pairwise
                          interactions, ridge penalty chosen by generalized CV (ArmInteractedSplines);
                          with arm_last=True the last design column is the arm indicator A
              'ens'       (reg) equal-weight average of 'gbm' and 'armspline'
                          (clf) soft-voting average of 'gbm' and 'spline'
              'linear'    (essentially) unpenalized logistic / least-squares regression
              'rf'        random forest
              'xgboost'   regularized shallow XGBoost trees (optional dependency)
              or any scikit-learn estimator instance (it is cloned)
    task    : 'clf' or 'reg'
    """
    from sklearn.base import clone

    if not isinstance(learner, str):
        return clone(learner)
    if learner == "armspline":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline

        if task == "clf":
            return make_learner("spline", task, random_state)
        return make_pipeline(ArmInteractedSplines(n_knots=6, arm_last=arm_last, pairwise=True),
                             RidgeCV(alphas=np.logspace(-2, 4, 13)))
    if learner == "ens":
        if task == "clf":
            from sklearn.ensemble import VotingClassifier

            return VotingClassifier([("gbm", make_learner("gbm", "clf", random_state)),
                                     ("spl", make_learner("spline", "clf", random_state))], voting="soft")
        from sklearn.ensemble import VotingRegressor

        return VotingRegressor([("gbm", make_learner("gbm", "reg", random_state)),
                                ("spl", make_learner("armspline", "reg", random_state, arm_last))])
    if learner == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        kw = dict(learning_rate=0.1, max_iter=500, max_leaf_nodes=4, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=20, random_state=random_state)
        return HistGradientBoostingClassifier(**kw) if task == "clf" else HistGradientBoostingRegressor(**kw)
    if learner == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError("--learner xgboost requires `python -m pip install xgboost`") from exc
        kw = dict(max_depth=2, learning_rate=0.05, reg_lambda=5.0,
                  subsample=0.9, colsample_bytree=1.0, tree_method="hist",
                  n_jobs=1, random_state=random_state)
        if task == "clf":
            return XGBClassifier(n_estimators=120, min_child_weight=20,
                                 eval_metric="logloss", **kw)
        return XGBRegressor(n_estimators=160, min_child_weight=12, **kw)
    if learner == "spline":
        from sklearn.linear_model import LogisticRegressionCV, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer

        if task == "clf":
            return make_pipeline(SplineTransformer(n_knots=5, degree=3),
                                 LogisticRegressionCV(Cs=np.logspace(-2, 2, 9), cv=3, max_iter=3000))
        return make_pipeline(SplineTransformer(n_knots=6, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if learner == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        kw = dict(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=1, random_state=random_state)
        return RandomForestClassifier(**kw) if task == "clf" else RandomForestRegressor(**kw)
    if learner == "linear":
        from sklearn.linear_model import LinearRegression, LogisticRegression

        return LogisticRegression(C=1e4, max_iter=5000) if task == "clf" else LinearRegression()
    raise ValueError(f"unknown learner {learner!r}")


class _UniformTable:
    """Piecewise-linear CDF (and lazily a KDE density) on a uniform grid for O(1) evaluation."""

    def __init__(self, xg, Fg, M=4097, resid=None):
        self.x0 = float(xg[0])
        self.dx = float(xg[-1] - xg[0]) / (M - 1)
        self.F = np.interp(self.x0 + self.dx * np.arange(M), xg, Fg)
        self.dF = np.append(np.diff(self.F), 0.0)
        self.M = M
        self._resid, self.D = resid, None

    def _lookup(self, z):
        u = np.clip((z - self.x0) / self.dx, 0.0, self.M - 1.0)
        i = u.astype(np.intp)
        return u, i

    def __call__(self, z):
        u, i = self._lookup(z)
        return self.F[i] + (u - i) * self.dF[i]

    def pdf(self, z):
        """Gaussian KDE of the residuals (binned on the grid, Silverman-type bandwidth)."""
        if self.D is None:
            from scipy.ndimage import gaussian_filter1d

            r = self._resid
            m = len(r)
            bw = 0.9 * min(np.std(r), np.subtract(*np.quantile(r, [0.75, 0.25])) / 1.349) * m ** (-0.2)
            edges = self.x0 - 0.5 * self.dx + self.dx * np.arange(self.M + 1)
            hist = np.histogram(r, bins=edges)[0] / (m * self.dx)
            self.D = gaussian_filter1d(hist, sigma=bw / self.dx, mode="constant")
        u, i = self._lookup(z)
        inside = (z > self.x0) & (z < self.x0 + self.dx * (self.M - 1))
        return np.where(inside, self.D[i], 0.0)


def _residual_table(z):
    """Linearly interpolated empirical CDF of standardized residuals z (density available lazily)."""
    z = np.sort(z)
    m = len(z)
    span = z[-1] - z[0]
    zg = np.concatenate(([z[0] - 0.25 * span], z, [z[-1] + 0.25 * span]))     # room for KDE tails
    cv = np.concatenate(([0.0], (np.arange(1, m + 1) - 0.5) / m, [1.0]))
    return _UniformTable(zg, cv, M=4097, resid=z)


class CrossFitNuisance:
    """K-fold cross-fitted nuisance estimates evaluated at the sample points.

    e_1(x)    : classifier of A on X in the source sample                   (learner_prop)
    omega(x)  : classifier of R on X, omega = (1 - rho)/rho * n1/n0         (learner_ratio)
    p_a(y,x)  : location-scale learner Y = mu(X, a) + sd(X, a) eps_a with the CDF of eps_a
                estimated by the empirical CDF of standardized residuals     (learner_cdf)
                pool_arms=True fits mu and sd on (X, A) with all source data (A as a feature);
                location residuals come from inner cross-fitting. The scale learner is
                fitted to their absolute values; training scale predictions standardize
                these residuals before constructing the arm-specific empirical CDFs.
    learner='auto' uses additive splines for e_1 and omega and boosting for mu, sd;
    learner='ensemble' uses additive splines for e_1 and omega and, for mu and sd, the average
    of boosting and arm-interacted additive splines ('ens').
    learner_scale : learner for sd (regression of |residual| on the design); default = learner_cdf;
                    'const' gives an arm-specific constant scale.
    pool_resid    : estimate one standardized-residual CDF from both arms (location-scale model
                    with a common error law) instead of one per arm.
    features : optional callable X -> design matrix used by every learner (e.g. a known basis).
    The fold of observation i is never used to fit the functions evaluated at X_i.
    """

    PRESETS = {"auto": ("spline", "spline", "gbm"),
               "ensemble": ("spline", "spline", "ens"),
               "xgboost": ("xgboost", "xgboost", "xgboost")}

    def __init__(self, learner="auto", K=5, clip_e=0.02, clip_rho=0.01, clip_omega=(0.02, 50.0),
                 heteroscedastic=True, pool_arms=True, features=None, random_state=None,
                 learner_prop=None, learner_ratio=None, learner_cdf=None, n_inner=2,
                 learner_scale=None, pool_resid=False):
        base = self.PRESETS.get(learner, (learner, learner, learner)) \
            if isinstance(learner, str) else (learner, learner, learner)
        self.learner_prop = base[0] if learner_prop is None else learner_prop
        self.learner_ratio = base[1] if learner_ratio is None else learner_ratio
        self.learner_cdf = base[2] if learner_cdf is None else learner_cdf
        self.learner_scale = self.learner_cdf if learner_scale is None else learner_scale
        self.K, self.clip_e, self.clip_rho, self.clip_omega = K, clip_e, clip_rho, clip_omega
        self.heteroscedastic, self.pool_arms, self.features = heteroscedastic, pool_arms, features
        self.random_state, self.n_inner, self.pool_resid = random_state, n_inner, pool_resid

    def _locscale(self, Xtr, Atr, Ytr, Xte, seed):
        """Fit mu, sd on training source data; return {a: (mu(Xte), sd(Xte), residual table)}."""
        rng = np.random.default_rng(seed)
        if self.pool_arms:
            groups = [(np.ones(len(Ytr), bool), (0, 1))]
            design = lambda Xm, a: np.column_stack([Xm, np.broadcast_to(np.asarray(a, float), len(Xm))])
            Ztr = np.column_stack([Xtr, Atr])
        else:
            groups = [(Atr == a, (a,)) for a in (0, 1)]
            design = lambda Xm, a: Xm
            Ztr = Xtr
        arm_last = self.pool_arms                       # last design column is A
        hetero = self.heteroscedastic and not (isinstance(self.learner_scale, str) and self.learner_scale == "const")
        pred, zres = {}, {}
        for mask, arms in groups:
            Z, Yg, Ag = Ztr[mask], Ytr[mask], Atr[mask]
            m = len(Yg)
            inner = rng.permutation(np.arange(m) % self.n_inner)
            r = np.empty(m)
            for j in range(self.n_inner):
                t = inner != j
                mod = make_learner(self.learner_cdf, "reg", seed, arm_last=arm_last)
                mod.fit(Z[t], Yg[t])
                r[~t] = Yg[~t] - mod.predict(Z[~t])
            mod = make_learner(self.learner_cdf, "reg", seed, arm_last=arm_last)
            mod.fit(Z, Yg)
            if hetero:
                sc = make_learner(self.learner_scale, "reg", seed, arm_last=arm_last)
                sc.fit(Z, np.abs(r))
                floor = 0.25 * np.mean(np.abs(r))
                s_in = np.maximum(sc.predict(Z), floor)
            for a in arms:
                keep = Ag == a
                Zte = design(Xte, a)
                if hetero:
                    sd_te, s_a = np.maximum(sc.predict(Zte), floor), s_in[keep]
                else:                                   # arm-specific constant scale
                    s0 = np.std(r[keep])
                    sd_te, s_a = np.full(len(Xte), s0), np.full(int(keep.sum()), s0)
                pred[a] = (mod.predict(Zte), sd_te)
                zres[a] = r[keep] / s_a
        if self.pool_resid:
            tab = _residual_table(np.concatenate([zres[0], zres[1]]))
            tabs = {0: tab, 1: tab}
        else:
            tabs = {a: _residual_table(zres[a]) for a in (0, 1)}
        return {a: (pred[a][0], pred[a][1], tabs[a]) for a in (0, 1)}

    def fit(self, X, R, A, Y):
        X = np.asarray(X, float)
        if self.features is not None:
            X = np.asarray(self.features(X), float)
        R = np.asarray(R).astype(int)
        A = np.asarray(A, float)
        Y = np.asarray(Y, float)
        n, K = len(R), self.K
        rng = np.random.default_rng(self.random_state)
        fold = rng.permutation(np.arange(n) % K)          # random, independent of the data
        src = R == 1
        e1 = np.empty(n)
        om = np.empty(n)
        loc = {0: np.empty(n), 1: np.empty(n)}
        scl = {0: np.empty(n), 1: np.empty(n)}
        self._res = {}
        seeds = rng.integers(0, 2**31 - 1, size=(K, 3))
        for k in range(K):
            te, tr = fold == k, fold != k
            idx = tr & src
            mp = make_learner(self.learner_prop, "clf", int(seeds[k, 0]))
            mp.fit(X[idx], A[idx].astype(int))
            e1[te] = mp.predict_proba(X[te])[:, 1]
            mr = make_learner(self.learner_ratio, "clf", int(seeds[k, 1]))
            mr.fit(X[tr], R[tr])
            rho = np.clip(mr.predict_proba(X[te])[:, 1], self.clip_rho, 1.0 - self.clip_rho)
            om[te] = (1.0 - rho) / rho * (src[tr].sum() / (~src[tr]).sum())
            fits = self._locscale(X[idx], A[idx], Y[idx], X[te], int(seeds[k, 2]))
            for a in (0, 1):
                loc[a][te], scl[a][te], self._res[(a, k)] = fits[a]
        self.e1 = np.clip(e1, self.clip_e, 1.0 - self.clip_e)
        self.omega = np.clip(om, *self.clip_omega)
        self.loc, self.scl, self.fold, self.n = loc, scl, fold, n
        self._fold_idx = [np.flatnonzero(fold == k) for k in range(K)]
        return self

    def cdf(self, a, y):
        """hat p_a(y_g, X_i) with fold-specific fits; returns array (len(y), n)."""
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty((y.shape[0], self.n))
        for k, idx in enumerate(self._fold_idx):
            z = (y[:, None] - self.loc[a][None, idx]) / self.scl[a][None, idx]
            out[:, idx] = self._res[(a, k)](z)
        return out

    def pdf(self, a, y):
        """hat f_a(y_g, X_i) = f_eps((y - mu)/sd)/sd (KDE of standardized residuals); (len(y), n)."""
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty((y.shape[0], self.n))
        for k, idx in enumerate(self._fold_idx):
            z = (y[:, None] - self.loc[a][None, idx]) / self.scl[a][None, idx]
            out[:, idx] = self._res[(a, k)].pdf(z) / self.scl[a][None, idx]
        return out


class ArrayNuisance:
    """User-supplied nuisance values at the sample points (e.g. oracle or perturbed truth).

    e1    : array (n,) of e_1(X_i)
    omega : array (n,) of omega(X_i)
    cdf_fn: callable (a, y) -> array (len(y), n) of p_a(y_g, X_i)
    pdf_fn: optional callable (a, y) -> array (len(y), n) of f_a(y_g, X_i) (for density='plugin')
    """

    def __init__(self, e1, omega, cdf_fn, pdf_fn=None):
        self.e1 = np.asarray(e1, float)
        self.omega = np.asarray(omega, float)
        self._cdf, self._pdf = cdf_fn, pdf_fn
        self.n = len(self.e1)
        if pdf_fn is None:
            self.pdf = None

    def cdf(self, a, y):
        return self._cdf(a, np.atleast_1d(np.asarray(y, float)))

    def pdf(self, a, y):
        return self._pdf(a, np.atleast_1d(np.asarray(y, float)))


@dataclass
class BoundsResult:
    """QTE-bound estimates on a reporting grid J = S_g x T_g (one variant, one kind)."""

    S: np.ndarray                  # (ns, 2) sensitivity pairs (Gamma, Lambda)
    taus: np.ndarray               # (nt,)
    kind: str
    variant: str
    n: int
    alpha: float
    lower: np.ndarray              # hat Delta^-_s(tau), (ns, nt)
    upper: np.ndarray              # hat Delta^+_s(tau), (ns, nt)
    se_lower: np.ndarray
    se_upper: np.ndarray
    crit: dict                     # critical values {'sim', 'bonf', 'pw'}
    qbounds: dict                  # {(a, 'lo'|'hi'): (ns, nt)} quantile bounds q^-_{a,s}, q^+_{a,s}
    qse: dict = field(default_factory=dict)   # their standard errors (from xi^-_{a}, xi^+_{a})
    exceptional: bool = False
    # Centered IFs divided by their empirical SD, NOT raw influence values.
    # Column order: all lower endpoints, then all upper endpoints, S-major/tau-minor.
    phi: np.ndarray | None = field(default=None, repr=False)   # (n, 2*ns*nt), standardized

    def ci(self, method="sim"):
        """Endpoint confidence limits (L^-, U^-, L^+, U^+), each (ns, nt)."""
        c = self.crit[method]
        if self.exceptional or not np.isfinite(c):
            inf = np.full_like(self.lower, np.inf)
            return -inf, inf, -inf, inf
        return (self.lower - c * self.se_lower, self.lower + c * self.se_lower,
                self.upper - c * self.se_upper, self.upper + c * self.se_upper)

    def table(self, methods=("sim", "bonf", "pw")):
        G, L = self.S[:, 0], self.S[:, 1]
        ns, nt = self.lower.shape
        df = pd.DataFrame({
            "Gamma": np.repeat(G, nt), "Lambda": np.repeat(L, nt), "tau": np.tile(self.taus, ns),
            "lower": self.lower.ravel(), "upper": self.upper.ravel(),
            "se_lower": self.se_lower.ravel(), "se_upper": self.se_upper.ravel(),
        })
        for m in methods:
            Ll, Ul, Lu, Uu = self.ci(m)
            df[f"L_lower_{m}"], df[f"U_lower_{m}"] = Ll.ravel(), Ul.ravel()
            df[f"L_upper_{m}"], df[f"U_upper_{m}"] = Lu.ravel(), Uu.ravel()
        return df

    def robustness_sets(self, method="sim"):
        """Grid points certified (L^-_{s'} > 0) and excluded (U^-_{s'} <= 0), per tau.

        Returns dict tau -> (certified bool (ns,), excluded bool (ns,)).  The inner and
        outer confidence sets of the paper are the unions of lower / upper orthants
        of these points; see `region_membership`.
        """
        Ll, Ul, _, _ = self.ci(method)
        return {float(t): (Ll[:, j] > 0, Ul[:, j] <= 0) for j, t in enumerate(self.taus)}


def region_membership(S_grid, certified, excluded, points):
    """Membership of `points` (m, 2) in the inner and outer confidence sets.

    inner = union over certified s' of {s : s <= s'}
    outer = S minus union over excluded s' of {s : s' <= s}
    """
    S_grid = np.asarray(S_grid, float)
    certified, excluded = np.asarray(certified, bool), np.asarray(excluded, bool)
    P = np.asarray(points, float)[:, None, :]
    tol = 1e-12
    if np.any(certified):
        C = S_grid[certified][None, :, :]
        inner = (P <= C + tol).all(-1).any(-1)
    else:
        inner = np.zeros(len(points), bool)
    if np.any(excluded):
        E = S_grid[excluded][None, :, :]
        outer = ~(E <= P + tol).all(-1).any(-1)
    else:
        outer = np.ones(len(points), bool)
    return inner, outer


def multiplier_critical_value(Phi_std, alpha, n_boot, rng, chunk=250):
    """(1-alpha) quantile of max_j |sum_i Z_i Phi_std[i, j]| with common Gaussian Z_i.

    Phi_std: (n, J) centered influence values divided by sqrt(n) * v_j.
    """
    n, J = Phi_std.shape
    stats = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        Z = rng.standard_normal((b, n))
        stats.append(np.abs(Z @ Phi_std).max(axis=1))
    return float(np.quantile(np.concatenate(stats), 1.0 - alpha))


class NestedSensitivityQTE:
    """Cross-fitted one-step estimation and simultaneous inference for QTE bounds.

    Typical use
    -----------
    >>> est = ProposedQTE(profile="exp3", seed=0).fit(X, R, A, Y)
    >>> res = est.bounds(S=[(1, 1), (1.5, 1.25)], taus=[0.25, 0.5, 0.75])
    >>> res.table()

    Parameters
    ----------
    learner      : nuisance learner passed to CrossFitNuisance ('auto', 'gbm', 'spline', 'linear', ...)
    K            : number of cross-fitting folds
    n_coarse     : size of the coarse threshold grid used to locate first crossings
    fine_per_gap : fine cells per coarse cell on the window [y_{j-win_before}, y_{j+win_after}]
                   around the first coarse crossing j (fine gap ~ 0.01-0.02 for typical outcomes)
    n_bisect     : bisection steps inside the first crossing fine cell
    inversion    : 'first' (paper, eq. 12) or 'rearrange' (monotone rearrangement, optional)
    density      : estimator of f^sig_{a,s} at the quantile bounds (enters the SEs only):
                   'fd'     finite difference of hat psi with bandwidth h_n (paper, Appendix B.6)
                   'plugin' target average of hat G_p(p_a, e_a) hat f_a(y, X) (Lemma 1), which needs
                            a conditional density from the nuisance object (nuisance.pdf)
                   'auto'   'plugin' when nuisance.pdf is available, otherwise 'fd' (default).
                   The archived simulations use 'auto', resolving to 'plugin' with the
                   location-scale residual KDE. Use the same choice for reproduction.
    bandwidth    : h_n for the finite-difference density (default c * scale * n^{-1/5})
    bw_const     : constant c in the default bandwidth (default 2)
    eta          : truncation level eta_0 of Assumption 3 applied to the fitted nuisances,
                   hat e_a in [eta_0, 1 - eta_0] and hat omega in [eta_0, 1/eta_0] (default None:
                   only the mild clip_e safeguard).  eta_0 = 0.1 is used in Experiment 2; it
                   never binds at the true nuisances of the simulation design.
    normalize_omega : Hajek normalization hat omega / mean_{R=1} hat omega of the density ratio
                   (asymptotically equivalent; removes a multiplicative error in hat omega and
                   stabilizes the correction term)
    """

    def __init__(self, learner="auto", K=5, n_coarse=81, fine_per_gap=12, n_bisect=7, win_before=2,
                 win_after=None, inversion="first", density="auto", bandwidth=None, bw_const=2.0, clip_e=0.02,
                 y_range=None, heteroscedastic=True, random_state=None, eta=None, normalize_omega=False):
        self.learner, self.K = learner, K
        self.n_coarse, self.fine_per_gap, self.n_bisect = n_coarse, fine_per_gap, n_bisect
        if win_after is None:
            win_after = 0 if inversion == "first" else 2
        self.win_before, self.win_after, self.inversion = win_before, win_after, inversion
        self.density = density
        self.bandwidth, self.bw_const = bandwidth, bw_const
        self.clip_e, self.y_range = clip_e, y_range
        self.heteroscedastic = heteroscedastic
        self.random_state = random_state
        self.eta, self.normalize_omega = eta, normalize_omega

    # ------------------------------------------------------------------ fit
    def fit(self, X, R, A, Y, nuisance=None):
        R = np.asarray(R).astype(int)
        A = np.asarray(A, float)
        Y = np.asarray(Y, float)
        src = R == 1
        tgt = ~src
        n, n1, n0 = len(R), int(src.sum()), int(tgt.sum())
        if n1 == 0 or n0 == 0:
            raise ValueError("both source and target samples must be non-empty")
        if nuisance is None:
            nuisance = CrossFitNuisance(learner=self.learner, K=self.K, clip_e=self.clip_e,
                                        heteroscedastic=self.heteroscedastic,
                                        random_state=self.random_state).fit(X, R, A, Y)
        self.nu = nuisance
        ce = self.clip_e if self.eta is None else max(self.clip_e, self.eta)
        e1 = np.clip(np.asarray(nuisance.e1, float), ce, 1.0 - ce)
        self.e = {1: e1, 0: 1.0 - e1}
        omega = np.asarray(nuisance.omega, float)
        if self.eta is not None:
            omega = np.clip(omega, self.eta, 1.0 / self.eta)
        if self.normalize_omega:
            omega = omega / omega[src].mean()
        self.omega = omega
        self.n, self.n1, self.n0 = n, n1, n0
        self.pi1, self.pi0 = n1 / n, n0 / n
        self.Ysrc = np.where(src, Y, np.inf)                 # 1(Y <= y) = 0 for target rows
        Ia = {a: (src & (A == a)).astype(float) for a in (0, 1)}
        self._ia_over_e = {a: Ia[a] / self.e[a] for a in (0, 1)}
        self._ia_minus_e = {a: src * (Ia[a] - self.e[a]) for a in (0, 1)}
        self._wt = tgt / n0                                   # target average weights
        self._ws = src * self.omega / n1                      # weighted source average weights
        self._tgt_pi0 = tgt / self.pi0
        self._src_pi1 = src * self.omega / self.pi1
        ys = Y[src]
        lo, hi = (ys.min(), ys.max()) if self.y_range is None else self.y_range
        pad = 0.1 * (hi - lo)
        self.y_coarse = np.linspace(lo - pad, hi + pad, self.n_coarse)
        if self.bandwidth is None:
            iqr = np.subtract(*np.quantile(ys, [0.75, 0.25]))
            scale = min(np.std(ys), iqr / 1.349)
            self.h = self.bw_const * scale * n ** (-0.2)
        else:
            self.h = float(self.bandwidth)
        return self

    # ------------------------------------------------------------ core pieces
    def _psi(self, a, y, Gam, Lam, sign, kind, variant, need_phi=False, P=None, ind=None):
        """hat psi^sign_{a,s}(y) for thresholds y (G,), optionally with IF values (G, n).

        Gam and Lam are scalars or arrays of shape (G, 1) (one sensitivity pair per row).
        """
        y = np.atleast_1d(np.asarray(y, float))
        if P is None:
            P = self.nu.cdf(a, y)
        e = self.e[a][None, :]
        b, bp, be = envelope(P, e, Gam, Lam, sign, kind)
        tgt_mean = b @ self._wt
        if variant == "plugin" and not need_phi:
            return tgt_mean
        if ind is None:
            ind = self.Ysrc[None, :] <= y[:, None]
        zeta_p = bp * self._ia_over_e[a][None, :] * (ind - P)     # outcome correction
        zeta_e = be * self._ia_minus_e[a][None, :]                # propensity correction
        corr_p = zeta_p @ self._ws
        corr_e = zeta_e @ self._ws
        if variant == "onestep":
            psi = tgt_mean + corr_p + corr_e
        elif variant == "no_Ge":
            psi = tgt_mean + corr_p
        elif variant == "plugin":
            psi = tgt_mean
        else:
            raise ValueError(f"unknown variant {variant!r}")
        if not need_phi:
            return psi
        # influence values: (1-R)/pi0 (b - psi) + R/pi1 omega zeta
        # (for 'plugin' the efficient IF is used, i.e. the SE a naive analyst would report;
        #  for 'no_Ge' the IF without the propensity term is used)
        zeta = zeta_p if variant == "no_Ge" else zeta_p + zeta_e
        phi = self._tgt_pi0[None, :] * (b - psi[:, None]) + self._src_pi1[None, :] * zeta
        return psi, phi

    def _quantiles(self, a, S, taus, kind, variant):
        """Quantile-bound inversion (eq. 12) for arm a, both CDF-bound signs.

        The one-step CDF estimate need not be monotone, so the first crossing is located in
        three stages: a coarse grid over the outcome range, a fine grid (n_fine cells) on a
        window of coarse cells around the first coarse crossing, and bisection inside the
        first fine cell that crosses.  inversion='rearrange' instead returns
        y_min + Leb{y : hat psi(y) < tau}, the quantile of the monotone rearrangement of
        hat psi (asymptotically equivalent; optional, not the paper's default).

        Returns {sign: dict(q, f, phi, ok)} with q, f, ok of shape (ns, nt) and phi (ns, nt, n).
        sign=+1 gives the lower quantile bound q^-_{a,s}; sign=-1 gives q^+_{a,s}.
        """
        ns, nt = len(S), len(taus)
        yc = self.y_coarse
        Gc = len(yc)
        Pc = self.nu.cdf(a, yc)
        indc = self.Ysrc[None, :] <= yc[:, None]
        rowsG = np.repeat(S[:, 0], nt)[:, None]
        rowsL = np.repeat(S[:, 1], nt)[:, None]
        rowsT = np.tile(taus, ns)
        rows_s = np.repeat(np.arange(ns), nt)
        nf = self.fine_per_gap * (self.win_before + self.win_after)
        out = {}
        for sign in (UPPER, LOWER):
            psi_c = np.vstack([self._psi(a, yc, G, L, sign, kind, variant, P=Pc, ind=indc) for G, L in S])
            pc = psi_c[rows_s]                                         # (rows, Gc)
            upc = pc >= rowsT[:, None]
            found = upc.any(axis=1)
            j = upc.argmax(axis=1)                                     # first coarse crossing
            lo_i = j - self.win_before              # window in coarse-cell units (may exceed the grid)
            hi_i = j + self.win_after
            gap = yc[1] - yc[0]
            y0, y1 = yc[0] + gap * lo_i, yc[0] + gap * hi_i
            dfine = (y1 - y0) / nf
            yf = y0[:, None] + dfine[:, None] * np.arange(nf + 1)[None, :]
            pf = np.empty_like(yf)
            for k in range(nf + 1):
                pf[:, k] = self._psi(a, yf[:, k], rowsG, rowsL, sign, kind, variant)
            upf = pf >= rowsT[:, None]
            if self.inversion == "first":
                hasf = upf.any(axis=1)
                kf = upf.argmax(axis=1)
                r = np.arange(len(rowsT))
                hi = np.where(hasf, yf[r, kf], yc[j])
                lo = np.where(hasf, yf[r, np.maximum(kf - 1, 0)], yc[np.maximum(j - 1, 0)])
                active = found & np.where(hasf, kf > 0, j > 0)
                for _ in range(self.n_bisect):
                    mid = 0.5 * (lo + hi)
                    pm = self._psi(a, mid, rowsG, rowsL, sign, kind, variant)
                    up = pm >= rowsT
                    hi = np.where(active & up, mid, hi)
                    lo = np.where(active & ~up, mid, lo)
                qhat = hi
            elif self.inversion == "rearrange":
                t = rowsT[:, None]

                def below_frac(p0, p1):          # share of a cell where the linear interpolant < t
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mixed = np.where(p0 < t, (t - p0) / (p1 - p0), (p0 - t) / (p0 - p1))
                    return np.where((p0 < t) & (p1 < t), 1.0, np.where((p0 >= t) & (p1 >= t), 0.0, mixed))

                meas_in = dfine * below_frac(pf[:, :-1], pf[:, 1:]).sum(axis=1)
                cell = below_frac(pc[:, :-1], pc[:, 1:])                 # coarse cells (rows, Gc-1)
                after = np.arange(Gc - 1)[None, :] >= hi_i[:, None]
                meas_after = gap * (cell * after).sum(axis=1)
                qhat = y0 + meas_in + meas_after      # = y_min + Leb{y : hat psi(y) < tau}
            else:
                raise ValueError(f"unknown inversion {self.inversion!r}")
            q = np.where(found, qhat, np.nan)
            qq = np.where(found, qhat, yc[-1])
            _, phi = self._psi(a, qq, rowsG, rowsL, sign, kind, variant, need_phi=True)
            if self.density in ("plugin", "auto") and getattr(self.nu, "pdf", None) is not None:
                Pq = self.nu.cdf(a, qq)
                _, bp, _ = envelope(Pq, self.e[a][None, :], rowsG, rowsL, sign, kind)
                f = (bp * self.nu.pdf(a, qq)) @ self._wt              # Lemma 1: E_0[G_p f_a]
            else:
                f = (self._psi(a, qq + self.h, rowsG, rowsL, sign, kind, variant)
                     - self._psi(a, qq - self.h, rowsG, rowsL, sign, kind, variant)) / (2.0 * self.h)
            ok = found & (f > 0)
            out[sign] = dict(q=q.reshape(ns, nt), f=f.reshape(ns, nt),
                             phi=phi.reshape(ns, nt, -1), ok=ok.reshape(ns, nt))
        return out

    # ----------------------------------------------------------------- bounds
    def bounds(self, S, taus, kind="nested", variant="onestep", alpha=0.05, n_boot=500,
               seed=None, keep_phi=False):
        """Estimate Delta^-_s(tau), Delta^+_s(tau) on J = S x taus with simultaneous CIs.

        The critical value 'sim' is the multiplier-bootstrap max-t quantile (eq. 16) over
        all (s, tau) in J and both endpoints, using common Gaussian weights; 'bonf' is
        Bonferroni over the same 2|J| endpoints; 'pw' is the pointwise normal quantile.
        """
        S = np.atleast_2d(np.asarray(S, float))
        taus = np.atleast_1d(np.asarray(taus, float))
        Q = {a: self._quantiles(a, S, taus, kind, variant) for a in (1, 0)}
        q1lo, q1hi = Q[1][UPPER], Q[1][LOWER]       # q^-_1 (from psi^+), q^+_1 (from psi^-)
        q0lo, q0hi = Q[0][UPPER], Q[0][LOWER]
        lower = q1lo["q"] - q0hi["q"]
        upper = q1hi["q"] - q0lo["q"]
        ok = q1lo["ok"] & q1hi["ok"] & q0lo["ok"] & q0hi["ok"]
        with np.errstate(divide="ignore", invalid="ignore"):
            # xi^-_{a} = -phi^+_a / f^+_a,  xi^+_a = -phi^-_a / f^-_a
            phiL = -q1lo["phi"] / q1lo["f"][..., None] + q0hi["phi"] / q0hi["f"][..., None]
            phiU = -q1hi["phi"] / q1hi["f"][..., None] + q0lo["phi"] / q0lo["f"][..., None]
        n = self.n
        phiL = phiL - phiL.mean(axis=-1, keepdims=True)
        phiU = phiU - phiU.mean(axis=-1, keepdims=True)
        vL = np.sqrt(np.mean(phiL ** 2, axis=-1))
        vU = np.sqrt(np.mean(phiU ** 2, axis=-1))
        exceptional = (not np.all(ok)) or (not np.all(np.isfinite(vL) & np.isfinite(vU))) \
            or np.any(vL <= 0) or np.any(vU <= 0)
        J2 = 2 * lower.size
        crit = {"pw": float(norm.ppf(1 - alpha / 2)), "bonf": float(norm.ppf(1 - alpha / (2 * J2)))}
        Phi = None
        if exceptional:
            crit["sim"] = np.inf
        else:
            Phi = np.concatenate([(phiL / (np.sqrt(n) * vL[..., None])).reshape(-1, n),
                                  (phiU / (np.sqrt(n) * vU[..., None])).reshape(-1, n)], axis=0).T
            rng = np.random.default_rng(seed)
            crit["sim"] = multiplier_critical_value(Phi, alpha, n_boot, rng)
        qb = {(1, "lo"): q1lo["q"], (1, "hi"): q1hi["q"], (0, "lo"): q0lo["q"], (0, "hi"): q0hi["q"]}
        qse = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for key, d in (((1, "lo"), q1lo), ((1, "hi"), q1hi), ((0, "lo"), q0lo), ((0, "hi"), q0hi)):
                xi = -d["phi"] / d["f"][..., None]                         # IF of the quantile bound
                xi = xi - xi.mean(axis=-1, keepdims=True)
                qse[key] = np.sqrt(np.mean(xi ** 2, axis=-1)) / np.sqrt(n)
        return BoundsResult(S=S, taus=taus, kind=kind, variant=variant, n=n, alpha=alpha,
                            lower=lower, upper=upper, se_lower=vL / np.sqrt(n), se_upper=vU / np.sqrt(n),
                            crit=crit, qbounds=qb, qse=qse, exceptional=bool(exceptional),
                            phi=(Phi * np.sqrt(n) if (keep_phi and Phi is not None) else None))


class PopulationBounds:
    """Oracle CDF/quantile/QTE bounds and efficiency bounds from the true observed law.

    e1_fn(X)          -> e_1(x)                    (N,)
    cdf_fn(a, y, X)   -> p_a(y_g, x_i)             (G, N)
    pdf_fn(a, y, X)   -> f_a(y_g, x_i)             (G, N)   (only for efficiency bounds)
    omega_fn(X)       -> omega(x)                  (N,)     (only for efficiency bounds)
    X0, w0            : oracle sample representing P_0^X (weights optional)
    pi1               : P(R = 1)                            (only for efficiency bounds)
    """

    def __init__(self, e1_fn, cdf_fn, X0, w0=None, pdf_fn=None, omega_fn=None, pi1=None,
                 y_range=(-15.0, 15.0), n_bisect=45, max_elems=3_000_000):
        self.cdf_fn, self.pdf_fn = cdf_fn, pdf_fn
        self.X0 = np.asarray(X0, float)
        N = len(self.X0)
        w = np.ones(N) if w0 is None else np.asarray(w0, float)
        self.w = w / w.sum()
        e1 = np.asarray(e1_fn(self.X0), float)
        self.e = {1: e1, 0: 1.0 - e1}
        self.omega = None if omega_fn is None else np.asarray(omega_fn(self.X0), float)
        self.pi1 = pi1
        self.y_range, self.n_bisect, self.max_elems = y_range, n_bisect, max_elems

    def psi(self, a, y, Gam, Lam, sign, kind="nested"):
        """psi^sign_{a,s}(y) for thresholds y (G,); Gam/Lam scalars or (G,) arrays."""
        y = np.atleast_1d(np.asarray(y, float))
        G = len(y)
        Gam = np.broadcast_to(np.asarray(Gam, float), (G,))
        Lam = np.broadcast_to(np.asarray(Lam, float), (G,))
        out = np.empty(G)
        step = max(1, self.max_elems // len(self.X0))
        e = self.e[a][None, :]
        for s0 in range(0, G, step):
            sl = slice(s0, min(G, s0 + step))
            P = self.cdf_fn(a, y[sl], self.X0)
            b, _, _ = envelope(P, e, Gam[sl, None], Lam[sl, None], sign, kind)
            out[sl] = b @ self.w
        return out

    def quantile(self, a, sign, Gam, Lam, taus, kind="nested"):
        """inf{y : psi^sign_{a,s}(y) >= tau} by bisection (psi is continuous, increasing).

        Gam, Lam, taus are broadcast to a common 1-d shape.
        """
        Gam, Lam, taus = np.broadcast_arrays(np.atleast_1d(np.asarray(Gam, float)),
                                             np.atleast_1d(np.asarray(Lam, float)),
                                             np.atleast_1d(np.asarray(taus, float)))
        lo = np.full(Gam.shape, self.y_range[0])
        hi = np.full(Gam.shape, self.y_range[1])
        for _ in range(self.n_bisect):
            mid = 0.5 * (lo + hi)
            up = self.psi(a, mid, Gam, Lam, sign, kind) >= taus
            hi = np.where(up, mid, hi)
            lo = np.where(up, lo, mid)
        return hi

    def qte_bounds(self, S, taus, kind="nested"):
        """Population Delta^-_s(tau), Delta^+_s(tau) (ns, nt) and quantile bounds."""
        S = np.atleast_2d(np.asarray(S, float))
        taus = np.atleast_1d(np.asarray(taus, float))
        ns, nt = len(S), len(taus)
        G = np.repeat(S[:, 0], nt)
        L = np.repeat(S[:, 1], nt)
        T = np.tile(taus, ns)
        q = {}
        for a in (1, 0):
            q[(a, "lo")] = self.quantile(a, UPPER, G, L, T, kind).reshape(ns, nt)   # q^-_a
            q[(a, "hi")] = self.quantile(a, LOWER, G, L, T, kind).reshape(ns, nt)   # q^+_a
        lower = q[(1, "lo")] - q[(0, "hi")]
        upper = q[(1, "hi")] - q[(0, "lo")]
        return lower, upper, q

    def _arm_terms(self, a, y, Gam, Lam, sign, kind):
        """Pieces of the CDF-bound EIF at one threshold y (scalar) for arm a."""
        P = self.cdf_fn(a, np.array([y]), self.X0)[0]
        fa = self.pdf_fn(a, np.array([y]), self.X0)[0]
        e = self.e[a]
        b, bp, be = envelope(P, e, Gam, Lam, sign, kind)
        f = float(np.sum(self.w * bp * fa))                 # f^sign_{a,s}(y) = E_0[G_p f_a]
        ez2 = bp ** 2 * P * (1.0 - P) / e + be ** 2 * e * (1.0 - e)   # E_1(zeta^2 | X)
        return b, be, f, ez2

    def eif_variance(self, S, taus, qb, kind="nested"):
        """Efficiency bounds Var{varphi^-_{s,tau}(O)}, Var{varphi^+_{s,tau}(O)} (ns, nt).

        Uses the closed forms E_1(zeta_a^2|X) = G_p^2 p(1-p)/e_a + G_e^2 e_a(1-e_a) and
        E_1(zeta_1 zeta_0|X) = -G_{e,1} G_{e,0} e_1(1-e_1); the target part is
        Var_0(.)/pi0 and the source part E_0{omega E_1(.|X)}/pi1.
        """
        if self.pdf_fn is None or self.omega is None or self.pi1 is None:
            raise ValueError("pdf_fn, omega_fn and pi1 are required for efficiency bounds")
        S = np.atleast_2d(np.asarray(S, float))
        taus = np.atleast_1d(np.asarray(taus, float))
        pi1, pi0 = self.pi1, 1.0 - self.pi1
        e1 = self.e[1]
        out = {}
        for end in ("lower", "upper"):
            V = np.empty((len(S), len(taus)))
            for i, (G, L) in enumerate(S):
                for j, _ in enumerate(taus):
                    if end == "lower":     # varphi^- = -phi^+_1/f^+_1 + phi^-_0/f^-_0
                        y1, s1, y0, s0 = qb[(1, "lo")][i, j], UPPER, qb[(0, "hi")][i, j], LOWER
                    else:                  # varphi^+ = -phi^-_1/f^-_1 + phi^+_0/f^+_0
                        y1, s1, y0, s0 = qb[(1, "hi")][i, j], LOWER, qb[(0, "lo")][i, j], UPPER
                    b1, be1, f1, z1 = self._arm_terms(1, y1, G, L, s1, kind)
                    b0, be0, f0, z0 = self._arm_terms(0, y0, G, L, s0, kind)
                    t = b1 / f1 - b0 / f0
                    var_t = np.sum(self.w * (t - np.sum(self.w * t)) ** 2)
                    cross = -be1 * be0 * e1 * (1.0 - e1)
                    src = z1 / f1 ** 2 + z0 / f0 ** 2 - 2.0 * cross / (f1 * f0)
                    V[i, j] = var_t / pi0 + np.sum(self.w * self.omega * src) / pi1
            out[end] = V
        return out["lower"], out["upper"]

    def arm_eif_variance(self, S, taus, qb, kind="nested"):
        """Efficiency bounds for the quantile bounds: Var(xi^-_{a,s,tau}), Var(xi^+_{a,s,tau}).

        Returns {(a, 'lo'|'hi'): (ns, nt)}; 'lo' is q^-_{a,s} (from psi^+), 'hi' is q^+_{a,s}.
        """
        S = np.atleast_2d(np.asarray(S, float))
        taus = np.atleast_1d(np.asarray(taus, float))
        pi1, pi0 = self.pi1, 1.0 - self.pi1
        out = {}
        for a in (1, 0):
            for key, sign in (("lo", UPPER), ("hi", LOWER)):
                V = np.empty((len(S), len(taus)))
                for i, (G, L) in enumerate(S):
                    for j, _ in enumerate(taus):
                        b, _, f, z = self._arm_terms(a, qb[(a, key)][i, j], G, L, sign, kind)
                        var_t = np.sum(self.w * (b - np.sum(self.w * b)) ** 2)
                        V[i, j] = (var_t / pi0 + np.sum(self.w * self.omega * z) / pi1) / f ** 2
                out[(a, key)] = V
        return out

    def frontier(self, tau, Gammas, Lam_max, kind="nested", n_steps=24):
        """Robustness-region frontier Lambda_tau(Gamma) = sup{Lam <= Lam_max : Delta^-_{(Gam,Lam)}(tau) > 0}.

        Returns an array over Gammas; value 1.0 with flag False means the region is empty at
        that Gamma (Delta^- <= 0 already at Lambda = 1).  Uses monotonicity in Lambda.
        """
        Gammas = np.asarray(Gammas, float)

        def dlow(G, L):
            T = np.full(G.shape, tau)
            return (self.quantile(1, UPPER, G, L, T, kind) - self.quantile(0, LOWER, G, L, T, kind))

        d1 = dlow(Gammas, np.ones_like(Gammas))
        dmax = dlow(Gammas, np.full_like(Gammas, Lam_max))
        lo = np.ones_like(Gammas)
        hi = np.full_like(Gammas, Lam_max)
        active = (d1 > 0) & (dmax <= 0)
        for _ in range(n_steps):
            mid = 0.5 * (lo + hi)
            pos = dlow(Gammas, mid) > 0
            lo = np.where(active & pos, mid, lo)
            hi = np.where(active & ~pos, mid, hi)
        fr = np.where(d1 <= 0, 1.0, np.where(dmax > 0, Lam_max, 0.5 * (lo + hi)))
        nonempty = d1 > 0
        return fr, nonempty


# Shared DGP and plotting utilities


class TransportDGP:
    """Fixed four-covariate DGP used by all three experiments."""

    def __init__(self):
        self.d = 4
        self.Sigma = np.eye(self.d)
        self.chol = np.linalg.cholesky(self.Sigma)
        self.chol4 = np.linalg.cholesky(self.Sigma[:4, :4])
        self.trunc = 3.0
        self._lo = ndtr(-self.trunc)
        self._Z = ndtr(self.trunc) - ndtr(-self.trunc)
        self.pi1 = float(np.mean(self.rho(self._qmc_X(2 ** 18, seed=12345))))

    # ------------------------------------------------------------ covariates
    def _qmc_X(self, N, seed):
        """Scrambled Sobol draws of (x1,...,x4) ~ N(0, Sigma_4); other columns are unused (0)."""
        u = qmc.Sobol(d=4, scramble=True, seed=seed).random(N)
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        X = np.zeros((N, self.d))
        X[:, :4] = z @ self.chol4.T
        return X

    def basis(self, X):
        """Known basis in which every nuisance is (logistic-)linear (for the 'rate' regime)."""
        t1, t2, t3, t4 = (np.tanh(X[:, j]) for j in range(4))
        return np.column_stack([t1, t2, t3, np.sin(X[:, 1]), t3 * t4, X[:, 2]])

    # ------------------------------------------------------------- nuisances
    def rho(self, X):
        return expit(-np.tanh(X[:, 0]))

    def omega(self, X):
        r = self.rho(X)
        return (1.0 - r) / r * self.pi1 / (1.0 - self.pi1)

    def e1(self, X):
        return expit(np.tanh(X[:, 0]))

    def mean(self, a, X):
        return 0.5 * np.tanh(X[:, 0]) + 0.20 * np.sin(X[:, 1]) + 0.15 * X[:, 2] + a

    def scale(self, a, X):
        return 0.5 * (1.0 + 0.6 * a * np.tanh(X[:, 0]))

    def eps_cdf(self, t):
        return np.clip((ndtr(t) - self._lo) / self._Z, 0.0, 1.0)

    def eps_pdf(self, t):
        return np.where(np.abs(t) <= self.trunc, np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi) / self._Z, 0.0)

    def cdf(self, a, y, X, shift=None):
        """True p_a(y_g, x_i), shape (len(y), len(X)); optional location shift (units of s_a)."""
        y = np.atleast_1d(np.asarray(y, float))
        t = (y[:, None] - self.mean(a, X)[None, :]) / self.scale(a, X)[None, :]
        if shift is not None:
            t = t - shift[None, :]
        return self.eps_cdf(t)

    def pdf(self, a, y, X):
        y = np.atleast_1d(np.asarray(y, float))
        s = self.scale(a, X)
        t = (y[:, None] - self.mean(a, X)[None, :]) / s[None, :]
        return self.eps_pdf(t) / s[None, :]

    # -------------------------------------------------------------- sampling
    def sample(self, n, rng):
        """n i.i.d. copies of O = (R, X, RA, RY); A and Y are NaN for target rows."""
        X = rng.standard_normal((n, self.d)) @ self.chol.T
        R = (rng.random(n) < self.rho(X)).astype(int)
        A = np.full(n, np.nan)
        Y = np.full(n, np.nan)
        src = R == 1
        Xs = X[src]
        As = (rng.random(len(Xs)) < self.e1(Xs)).astype(float)
        eps = ndtri(self._lo + rng.random(len(Xs)) * self._Z)
        mu = np.where(As == 1, self.mean(1, Xs), self.mean(0, Xs))
        sc = np.where(As == 1, self.scale(1, Xs), self.scale(0, Xs))
        A[src], Y[src] = As, mu + sc * eps
        return X, R, A, Y

    def oracle_target(self, N=2 ** 16, seed=2027):
        """Quasi-Monte Carlo representation of P_0^X: pooled draws with weights 1 - rho(x)."""
        X = self._qmc_X(N, seed)
        w = 1.0 - self.rho(X)
        return X, w / w.sum()

    # -------------------------------------------------------- nuisance helpers
    def oracle_nuisance(self, X):
        """True nuisance values at the sample points (for the oracle estimator)."""
        return ArrayNuisance(self.e1(X), self.omega(X), lambda a, y: self.cdf(a, y, X),
                             lambda a, y: self.pdf(a, y, X))


def _exp_tilt_cdf(u, theta):
    """CDF on [0,1] with density proportional to exp(theta u)."""
    if abs(theta) < 1e-10:
        return u
    return np.expm1(theta * u) / np.expm1(theta)


@lru_cache(maxsize=32)
def exp_tilt_theta(kappa, direction):
    """theta with max LR range max{h_max, 1/h_min} = kappa; direction +1 (mass to high ranks) or -1."""
    if kappa <= 1.0:
        return 0.0

    def rng_(th):
        h0 = th / np.expm1(th)
        h1 = th * np.exp(th) / np.expm1(th)
        return max(h1, 1.0 / h0) - kappa

    th = brentq(rng_, 1e-8, 50.0)
    return direction * th


def true_target_cdf(dgp, a, y, X0, s0, law="lfp", direction="down"):
    """Conditional target CDF F^T_a(y|x) of a compatible full law at sensitivity pair s0.

    direction='down': arm 1 stochastically smaller and arm 0 larger in the target, so the true
    QTE is pushed downward (for law='lfp' it equals the sharp lower bound Delta^-_{s0}).
    """
    G0, L0 = s0
    P = dgp.cdf(a, y, X0)
    e = dgp.e1(X0) if a == 1 else 1.0 - dgp.e1(X0)
    down_arm = (a == 1) == (direction == "down")      # arm whose CDF is pushed up
    if law == "lfp":
        sign = UPPER if down_arm else LOWER
        return envelope(P, e[None, :], G0, L0, sign, "nested")[0]
    if law == "smooth":
        dirn = -1.0 if down_arm else 1.0                 # negative theta: CDF above the diagonal
        thG, thL = exp_tilt_theta(G0, dirn), exp_tilt_theta(L0, dirn)
        Fs = e[None, :] * P + (1.0 - e[None, :]) * _exp_tilt_cdf(P, thG)
        return _exp_tilt_cdf(Fs, thL)
    raise ValueError(law)


def true_target_qte(dgp, X0, w0, taus, s0, law="lfp", direction="down", y_range=(-15, 15), n_bisect=45):
    """Delta_T(tau) = Q^T_1(tau) - Q^T_0(tau) under the chosen full law."""
    taus = np.atleast_1d(np.asarray(taus, float))
    Q = {}
    for a in (1, 0):
        lo = np.full(taus.shape, float(y_range[0]))
        hi = np.full(taus.shape, float(y_range[1]))
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            F = true_target_cdf(dgp, a, mid, X0, s0, law, direction) @ w0
            up = F >= taus
            hi = np.where(up, mid, hi)
            lo = np.where(up, lo, mid)
        Q[a] = hi
    return Q[1] - Q[0]


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")


def paper_style():
    """Matplotlib defaults for paper figures (thin marks, recessive axes and grid)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["primary"], "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def set_n_axis(ax, ns):
    """Log-scaled sample-size axis with ticks exactly at the simulated n (no minor labels)."""
    from matplotlib.ticker import NullLocator

    ns = sorted(set(int(v) for v in ns))
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{v:,}" for v in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(ns[0] / 1.25, ns[-1] * 1.25)


def rep_seeds(base_seed, reps, tag):
    """Independent, reproducible integer seeds for replications (tag separates designs)."""
    import zlib

    ss = np.random.SeedSequence([base_seed, zlib.crc32(str(tag).encode())])
    return [int(s.generate_state(1)[0]) for s in ss.spawn(reps)]


# Simulation 1 definitions and paper figure


exp1_T_GRID = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


exp1_DISPLAY_LABELS = {"point": "Point", "conf_only": r"$\Gamma$ Only",
                  "trans_only": r"$\Lambda$ Only", "collapsed": "Collapsed", "proposed": "Proposed"}


def exp1_analyses(s0):
    G0, L0 = s0
    return [
        ("point", exp1_DISPLAY_LABELS["point"], (1.0, 1.0), "nested"),
        ("conf_only", exp1_DISPLAY_LABELS["conf_only"], (G0, 1.0), "nested"),
        ("trans_only", exp1_DISPLAY_LABELS["trans_only"], (1.0, L0), "nested"),
        ("collapsed", exp1_DISPLAY_LABELS["collapsed"], (G0, L0), "collapsed"),
        ("proposed", exp1_DISPLAY_LABELS["proposed"], (G0, L0), "nested"),
    ]


exp1_STYLE = {  # fixed colour per analysis (colour follows the entity) + marker as secondary encoding
    "proposed": (PALETTE[0], "o", "-"), "collapsed": (PALETTE[1], "s", "--"),
    "conf_only": (PALETTE[3], "^", "-."),
    "trans_only": (PALETTE[4], "v", "-."), "point": (PALETTE[5], "x", "-"),
}


def exp1_experiment_style():
    import matplotlib as mpl

    paper_style()
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 11.5, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "axes.titlepad": 4, "axes.labelpad": 2, "xtick.major.pad": 2,
        "ytick.major.pad": 2, "lines.linewidth": 1.8, "lines.markersize": 6.2,
        "legend.borderaxespad": 0.25, "legend.borderpad": 0.2,
        "legend.labelspacing": 0.25, "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.0, "legend.handlelength": 1.8,
        "savefig.pad_inches": 0.025,
        "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
    })


def exp1__cum_step(v, u):
    """Integral from 0 to u of the step function with equal-width bin values v (mean 1)."""
    J = len(v)
    knots = np.linspace(0.0, 1.0, J + 1)
    C = np.concatenate(([0.0], np.cumsum(v) / J))
    return np.interp(u, knots, C)


def exp1__project_mean_one(v, kappa):
    """Additive shift c with mean(clip(v + c, 1/kappa, kappa)) = 1."""
    lo, hi = -kappa, kappa
    for _ in range(60):
        c = 0.5 * (lo + hi)
        m = np.clip(v + c, 1.0 / kappa, kappa).mean()
        lo, hi = (c, hi) if m < 1.0 else (lo, c)
    return np.clip(v + 0.5 * (lo + hi), 1.0 / kappa, kappa)


def exp1_random_lr(J, kappa, rng):
    """Random likelihood ratio on the rank scale: bin values in [1/kappa, kappa], mean one."""
    if kappa <= 1.0:
        return np.ones(J)
    typ = rng.integers(4)
    if typ == 0:                                   # i.i.d. log-uniform, projected
        v = np.exp(rng.uniform(-np.log(kappa), np.log(kappa), J))
    elif typ == 1:                                 # monotone (sorted) draws
        v = np.sort(np.exp(rng.uniform(-np.log(kappa), np.log(kappa), J)))
        v = v if rng.random() < 0.5 else v[::-1]
    else:                                          # bang-bang with random or tail-concentrated support
        m = J / (kappa + 1.0)                      # number of bins at kappa for mean one
        k = int(np.floor(m))
        order = rng.permutation(J) if typ == 2 else (np.arange(J)[::-1] if rng.random() < 0.5 else np.arange(J))
        v = np.full(J, 1.0 / kappa)
        v[order[:k]] = kappa
        if k < J:
            v[order[k]] = 1.0 / kappa + (m - k) * (kappa - 1.0 / kappa)
    return exp1__project_mean_one(v, kappa)


def exp1_sharpness_audit(dgp, s0, taus, M=400, J=20, n_oracle=2 ** 12, seed=11):
    """Target QTE range over M random compatible laws per arm (rank-scale step tilts).

    Returns (min, max) of the random-law QTEs and the nested bounds computed on the same
    oracle sample, so that the comparison is free of Monte Carlo integration error.
    """
    G0, L0 = s0
    X0, w0 = dgp.oracle_target(n_oracle, seed=seed)
    pba = PopulationBounds(dgp.e1, dgp.cdf, X0, w0, n_bisect=34)
    blo, bhi, _ = pba.qte_bounds([s0], taus)
    y = np.linspace(-7.0, 9.0, 401)
    rng = np.random.default_rng(seed)
    Q = {}
    for a in (1, 0):
        P = dgp.cdf(a, y, X0)
        e = (dgp.e1(X0) if a == 1 else 1.0 - dgp.e1(X0))[None, :]
        qa = np.empty((M, len(taus)))
        for m in range(M):
            Fu = exp1__cum_step(exp1_random_lr(J, G0, rng), P)          # unobserved source law
            Fs = e * P + (1.0 - e) * Fu                       # mixture (eq. 2)
            Ft = exp1__cum_step(exp1_random_lr(J, L0, rng), Fs)         # target law
            F = Ft @ w0
            qa[m] = np.interp(taus, np.maximum.accumulate(F), y)
        Q[a] = qa
    return Q[1].min(0) - Q[0].max(0), Q[1].max(0) - Q[0].min(0), blo[0], bhi[0]


def exp1_plot_part_b(agg, outdir):
    import matplotlib.pyplot as plt

    exp1_experiment_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.30))
    fig.subplots_adjust(left=0.060, right=0.998, bottom=0.265, top=0.915, wspace=0.18)
    for ax, (law, ttl) in zip(axes[:2], (("lfp", "(a) Coverage: LFP law"), ("smooth", "(b) Coverage: Smooth law"))):
        for name in ("proposed", "collapsed", "conf_only", "trans_only", "point"):
            d = agg[(agg.law == law) & (agg.analysis == name)].sort_values("n")
            c, mk, ls = exp1_STYLE[name]
            ax.plot(d.n, d.coverage, color=c, marker=mk, ls=ls, label=exp1_DISPLAY_LABELS[name],
                    zorder=5 if name == "proposed" else 3)
        ax.axhline(0.95, color=INK["secondary"], lw=0.8, ls="--")
        set_n_axis(ax, agg.n)
        ax.set_xlim(agg.n.min() / 1.10, agg.n.max() * 1.10)
        ax.set_ylim(-0.045, 1.05)
        ax.set_xlabel(r"$n$")
        ax.set_title(ttl)
    ax = axes[2]
    d = agg[agg.law == "lfp"]
    for name in ("proposed", "collapsed", "conf_only", "trans_only", "point"):
        dd = d[d.analysis == name].sort_values("n")
        c, mk, ls = exp1_STYLE[name]
        ax.plot(dd.n, dd.median_width, color=c, marker=mk, ls=ls, label=exp1_DISPLAY_LABELS[name],
                zorder=5 if name == "proposed" else 3)
    set_n_axis(ax, agg.n)
    ax.set_xlim(agg.n.min() / 1.10, agg.n.max() * 1.10)
    ax.set_xlabel(r"$n$")
    ax.set_title("(c) Median interval width")
    axes[0].set_ylabel("Simultaneous coverage")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.525, 0.025), fontsize=11,
               borderaxespad=0, columnspacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig1b_finite_sample.{ext}"))
    plt.close(fig)


# Simulation 2 definitions and evaluation


exp2_S_EVAL = np.array([(1.0, 1.0), (1.25, 1.0), (1.5, 1.25)])


exp2_TAUS = np.array([0.25, 0.5, 0.75])


exp2_ESTIMATORS = ("onestep", "plugin", "no_Ge", "oracle")


exp2_QKEYS = ((1, "lo"), (1, "hi"), (0, "lo"), (0, "hi"))


def exp2_rate_nuisance(dgp, X, R, A, Y, rate, consts, K, seed, learner_prop=None, n_inner=5):
    """Correctly specified cross-fitted fits in dgp.basis plus deterministic O(n^{-rate}) errors."""
    c_p, c_e, c_w = consts
    n = len(R)
    nu = CrossFitNuisance(learner="linear", K=K, pool_arms=False, features=dgp.basis,
                          random_state=seed, learner_prop=learner_prop, n_inner=n_inner).fit(X, R, A, Y)
    dp, de, dw = (c * n ** (-rate) for c in (c_p, c_e, c_w))
    nu.e1 = expit(logit(np.clip(nu.e1, 1e-6, 1 - 1e-6)) + de)
    nu.omega = nu.omega * np.exp(dw * np.tanh(X[:, 0]))
    nu.loc = {1: nu.loc[1] + dp * nu.scl[1], 0: nu.loc[0] - dp * nu.scl[0]}
    return nu


def exp2_population_targets(dgp, n_oracle):
    X0, w0 = dgp.oracle_target(n_oracle)
    pb = PopulationBounds(dgp.e1, dgp.cdf, X0, w0, pdf_fn=dgp.pdf, omega_fn=dgp.omega, pi1=dgp.pi1,
                          n_bisect=40)
    lo, hi, qb = pb.qte_bounds(exp2_S_EVAL, exp2_TAUS)
    vlo, vhi = pb.eif_variance(exp2_S_EVAL, exp2_TAUS, qb)
    vq = pb.arm_eif_variance(exp2_S_EVAL, exp2_TAUS, qb)
    rows = []
    for i, (G, L) in enumerate(exp2_S_EVAL):
        for j, t in enumerate(exp2_TAUS):
            rows.append(dict(Gamma=G, Lambda=L, tau=t, target="QTE_lower", truth=lo[i, j], sd_eff=np.sqrt(vlo[i, j])))
            rows.append(dict(Gamma=G, Lambda=L, tau=t, target="QTE_upper", truth=hi[i, j], sd_eff=np.sqrt(vhi[i, j])))
            for a, k in exp2_QKEYS:
                rows.append(dict(Gamma=G, Lambda=L, tau=t, target=f"q{a}_{k}", truth=qb[(a, k)][i, j],
                                 sd_eff=np.sqrt(vq[(a, k)][i, j])))
    return pd.DataFrame(rows)


def exp2_summarize(raw, targets):
    raw = raw.merge(targets, on=["Gamma", "Lambda", "tau", "target"])
    raw["err"] = raw.est - raw.truth
    raw["cover"] = np.abs(raw.err) <= 1.959964 * raw.se
    g = raw.groupby(["estimator", "n", "Gamma", "Lambda", "tau", "target"])
    s = g.agg(truth=("truth", "first"), sd_eff=("sd_eff", "first"), bias=("err", "mean"),
              sd=("est", "std"), rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), mean_se=("se", "mean"),
              coverage=("cover", "mean"), reps=("err", "size"),
              coverage_own_sd=("err", lambda e: np.mean(np.abs(e) <= 1.959964 * np.std(e, ddof=1)))).reset_index()
    # coverage_own_sd: coverage of est +- 1.96 x (Monte Carlo SD of the estimator); it isolates the effect
    # of bias, since the plug-in has no valid standard error (its EIF-based SE overstates its spread).
    s["std_bias"] = np.sqrt(s.n) * s.bias / s.sd_eff
    s["rmse_ratio"] = np.sqrt(s.n) * s.rmse / s.sd_eff
    s["sd_ratio"] = np.sqrt(s.n) * s.sd / s.sd_eff
    s["se_over_sd"] = s.mean_se / s.sd
    s["bias_over_sd"] = s.bias / s.sd
    s["mc_se_cov"] = np.sqrt(s.coverage * (1 - s.coverage) / s.reps)
    return s


def exp2_paper_table(summary, targets=("QTE_lower", "QTE_upper"), tau=0.5):
    key = summary[(summary.tau == tau) & summary.target.isin(list(targets))]
    tab = key.pivot_table(index=["Gamma", "Lambda", "target", "estimator"], columns="n",
                          values=["std_bias", "rmse_ratio", "coverage"]).round(2)
    tab.columns = [f"{a}_n{b}" for a, b in tab.columns]
    return tab.reset_index()


def exp2_proposed_plugin_sd_table(summary):
    """Empirical SD over repetitions at every sensitivity pair, tau and QTE endpoint."""
    key = summary[summary.target.isin(("QTE_lower", "QTE_upper"))
                  & summary.estimator.isin(("onestep", "plugin"))]
    tab = key.pivot(index=["n", "Gamma", "Lambda", "tau"],
                    columns=["target", "estimator"], values="sd")
    tab = tab[[('QTE_lower', 'onestep'), ('QTE_upper', 'onestep'),
               ('QTE_lower', 'plugin'), ('QTE_upper', 'plugin')]]
    tab.columns = ["proposed_sd_lower", "proposed_sd_upper",
                   "plugin_sd_lower", "plugin_sd_upper"]
    return tab.reset_index()


def exp2_coverage_width_table(summary):
    """Average six QTE endpoints per pair; retain signed and absolute cellwise bias."""
    key = summary[summary.target.isin(("QTE_lower", "QTE_upper"))
                  & summary.estimator.isin(("onestep", "plugin", "oracle"))]
    grouped = key.groupby(["n", "Gamma", "Lambda", "estimator"], as_index=False).agg(
        coverage=("coverage", "mean"), mean_se=("mean_se", "mean"),
        bias=("bias", "mean"), mean_abs_bias=("bias", lambda b: np.mean(np.abs(b))),
        reps=("reps", "min"))
    tab = grouped.pivot(index=["n", "Gamma", "Lambda"], columns="estimator",
                        values=["coverage", "mean_se", "bias", "mean_abs_bias"])
    tab = tab[[('coverage', 'onestep'), ('coverage', 'plugin'), ('coverage', 'oracle'),
               ('mean_se', 'onestep'), ('mean_se', 'plugin'), ('mean_se', 'oracle'),
               ('bias', 'onestep'), ('bias', 'plugin'), ('bias', 'oracle'),
               ('mean_abs_bias', 'onestep'), ('mean_abs_bias', 'plugin'), ('mean_abs_bias', 'oracle')]]
    tab.columns = ["proposed_coverage", "plugin_coverage", "oracle_coverage",
                   "proposed_mean_width", "plugin_mean_width", "oracle_mean_width",
                   "proposed_bias", "plugin_bias", "oracle_bias",
                   "proposed_mean_abs_bias", "plugin_mean_abs_bias", "oracle_mean_abs_bias"]
    for col in ("proposed_mean_width", "plugin_mean_width", "oracle_mean_width"):
        tab[col] *= 2.0 * 1.959964
    tab["reps"] = grouped.groupby(["n", "Gamma", "Lambda"]).reps.min()
    return tab.reset_index()


# Simulation 3 definitions and population quantities


exp3_S_GRID = exp2_S_EVAL.copy()


exp3_T_GRID = np.array([0.25, 0.5, 0.75])


def exp3_proposed_oracle_width_table(summary):
    """Average simultaneous-band widths and halfwidths over both endpoints."""
    tab = summary[["n", "sim_halfwidth", "oracle_sim_halfwidth"]].rename(columns={
        "sim_halfwidth": "proposed_mean_halfwidth",
        "oracle_sim_halfwidth": "oracle_proposed_mean_halfwidth"})
    tab = tab.assign(proposed_mean_width=2.0 * tab.proposed_mean_halfwidth,
                     oracle_proposed_mean_width=2.0 * tab.oracle_proposed_mean_halfwidth)
    return tab[["n", "proposed_mean_width", "oracle_proposed_mean_width",
                "proposed_mean_halfwidth", "oracle_proposed_mean_halfwidth"]]


exp3_METHODS = ("sim", "bonf", "pw")


exp3_EVAL_1D = np.linspace(1.0, 2.0, 101)


exp3_EVAL_PTS = np.array([(g, l) for g in exp3_EVAL_1D for l in exp3_EVAL_1D])


def exp3_population_truth(dgp, n_oracle, n_oracle_frontier, quick=False):
    X0, w0 = dgp.oracle_target(n_oracle)
    pb = PopulationBounds(dgp.e1, dgp.cdf, X0, w0, n_bisect=40)
    lo, hi, _ = pb.qte_bounds(exp3_S_GRID, exp3_T_GRID)
    Xf, wf = dgp.oracle_target(n_oracle_frontier, seed=99)
    pbf = PopulationBounds(dgp.e1, dgp.cdf, Xf, wf, n_bisect=30)
    Gf = np.linspace(1.0, 2.0, 11 if quick else 21)
    fronts, areas, member = {}, {}, {}
    for t in exp3_T_GRID:
        fr, ne = pbf.frontier(t, Gf, 2.0, n_steps=12 if quick else 16)
        fr = np.where(ne, fr, 1.0)
        fronts[t] = (Gf, fr)
        frz = np.interp(exp3_EVAL_PTS[:, 0], Gf, fr)
        member[t] = (exp3_EVAL_PTS[:, 1] <= frz) & (np.interp(exp3_EVAL_PTS[:, 0], Gf, ne.astype(float)) > 0.5)
        areas[t] = member[t].mean()                  # share of S = [1,2]^2 (area 1)
    return dict(lower=lo, upper=hi, fronts=fronts, true_area=areas, true_member=member)


# Shared execution, output tables and reproducible parallel runs

DEFAULT_N = (2000, 4000, 8000)
EXP1_S0 = (1.5, 1.5)
SHARED_DGP = dict(d=4, covariates="iid standard normal", trunc=3.0,
                  outcome_tanh_x1=0.5, outcome_sin_x2=0.20,
                  outcome_linear_x3=0.15, treatment_effect=1.0,
                  control_scale=0.5, treated_scale_x1=0.6,
                  rho_tanh_x1=-1.0, e_tanh_x1=1.0)


class CachedNuisance:
    """Reuse the identical coarse-grid CDF across variants and Exp2 ML/Exp3."""

    def __init__(self, nuisance):
        self.base = nuisance
        self.e1, self.omega = nuisance.e1, nuisance.omega
        self.grids, self.values = {}, {}
        if not callable(getattr(nuisance, "pdf", None)):
            self.pdf = None

    def register_grid(self, y):
        y = np.asarray(y, float)
        self.grids[y.shape] = y.copy()

    def cdf(self, a, y):
        y = np.atleast_1d(np.asarray(y, float))
        grid = self.grids.get(y.shape)
        if grid is None or not np.array_equal(y, grid):
            return self.base.cdf(a, y)
        key = (a, y.tobytes())
        if key not in self.values:
            self.values[key] = self.base.cdf(a, y)
        return self.values[key]

    def pdf(self, a, y):
        return self.base.pdf(a, y)


def fit_ml_nuisance(X, R, A, Y, seed, K=5, n_inner=5, learner="ensemble"):
    return CrossFitNuisance(learner=learner, K=K, n_inner=n_inner,
                            random_state=seed).fit(X, R, A, Y)


def make_estimator(X, R, A, Y, nuisance, exp2=False):
    nuisance = nuisance if isinstance(nuisance, CachedNuisance) else CachedNuisance(nuisance)
    kwargs = dict(eta=0.1, normalize_omega=True) if exp2 else {}
    est = NestedSensitivityQTE(**kwargs).fit(X, R, A, Y, nuisance=nuisance)
    nuisance.register_grid(est.y_coarse)
    return est


# Public interface: Proposed only. The numerical kernel above is also used by
# the unchanged Exp1--3 reproduction harness below.
class ProposedQTE(NestedSensitivityQTE):
    """Proposed cross-fitted one-step estimator with explicit simulation presets.

    Parameters
    ----------
    profile : 'exp1', 'exp2', or 'exp3' (default)
        Exp2 applies eta=0.1 and source-mean normalization of omega. Exp1/3 do
        not. Bootstrap seeds default to seed+1, seed, seed+7, respectively,
        exactly as in the archived experiments. These are implementation
        presets, not distinct estimands or distinct versions of Proposed.
    learner : 'xgboost' (default) or 'ensemble'
        The preset determines the learners for all four tasks.
        XGBoost estimates e_1, rho, outcome location and outcome scale. The
        conditional CDF uses arm-specific empirical standardized residuals.
    K, n_inner : int
        Outer and inner location-residual folds. Both default to 5.
    seed : nonnegative int
        Nuisance-fitting seed. The simulation harness derives independent
        replication seeds from base seed 2027; see rep_seeds, not seed+r.
    density : 'auto', 'plugin', or 'fd'
        Default 'auto' uses the conditional PDF when provided and otherwise
        finite differences. Archived XGBoost simulations use the residual KDE.

    Inputs to fit
    -------------
    X : numeric array (n,d), shared covariates in a common order/encoding.
    R : array (n,), 1 for source, 0 for target.
    A : array (n,), binary treatment in source, NaN allowed in target.
    Y : array (n,), finite source outcomes, NaN allowed in target.
        Target A/Y are masked internally and are never used to fit the model.
    nuisance : optional fitted object with e1, omega, cdf(a,y), pdf(a,y)
        E.g. ArrayNuisance for true or externally cross-fitted nuisance values.
        Arrays must match the original row order. The caller is responsible
        for correct out-of-fold fitting when supplying estimated nuisances.

    Results
    -------
    bounds(S,taus) returns BoundsResult. lower/upper are estimated target QTE
    bound endpoints, not confidence limits. ci('sim') returns simultaneous
    endpoint intervals over ALL supplied sensitivity pairs, quantiles and
    both endpoints. ci('pw') gives pointwise intervals; ci('bonf') gives the
    Bonferroni comparator. table() returns a pandas DataFrame. Simultaneous
    intervals are calibrated over the complete grid before any subsetting.
    robustness_sets() and region_membership() construct inner/outer sets for
    {s : Delta^-_s(tau)>0}. keep_phi=True returns STANDARDIZED centered IFs.

    No population truth, DGP or saved result files are used by this class.
    The inherited low-level API remains available as NestedSensitivityQTE.
    """

    def __init__(self, profile="exp3", learner="xgboost", K=5, n_inner=5,
                 seed=2027, density="auto"):
        if profile not in ("exp1", "exp2", "exp3"):
            raise ValueError("profile must be 'exp1', 'exp2', or 'exp3'")
        if learner not in ("xgboost", "ensemble"):
            raise ValueError("learner must be 'xgboost' or 'ensemble'")
        for name, value, minimum in (("K", K, 2), ("n_inner", n_inner, 2), ("seed", seed, 0)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if density not in ("auto", "plugin", "fd"):
            raise ValueError("density must be 'auto', 'plugin', or 'fd'")
        self.profile, self.n_inner, self.seed = profile, int(n_inner), int(seed)
        self.bootstrap_seed = int(seed) + {"exp1": 1, "exp2": 0, "exp3": 7}[profile]
        kwargs = dict(eta=0.1, normalize_omega=True) if profile == "exp2" else {}
        super().__init__(learner=learner, K=int(K), random_state=int(seed), density=density, **kwargs)

    def fit(self, X, R, A, Y, nuisance=None):
        """Fit Proposed, retaining the reference row order and fold generation."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be a finite numeric (n,d) array with d >= 1")
        n = len(X)
        R, A, Y = (np.asarray(v, dtype=float) for v in (R, A, Y))
        if any(v.shape != (n,) for v in (R, A, Y)):
            raise ValueError("R, A and Y must each be one-dimensional arrays of length n")
        if not np.isin(R, (0, 1)).all():
            raise ValueError("R must contain only 0 (target) and 1 (source)")
        R = R.astype(int)
        src = R == 1
        if src.sum() < 2 or (~src).sum() < 2:
            raise ValueError("At least two source and two target rows are required")
        if not np.isin(A[src], (0, 1)).all() or len(np.unique(A[src])) != 2:
            raise ValueError("Source A must be binary and contain both treatment groups")
        if not np.isfinite(Y[src]).all() or np.ptp(Y[src]) <= 0:
            raise ValueError("Source Y must be finite and nonconstant")
        A, Y = A.copy(), Y.copy()
        A[~src], Y[~src] = np.nan, np.nan
        if nuisance is None:
            if self.K >= n:
                raise ValueError("K must be smaller than n")
            folds = np.random.default_rng(self.seed).permutation(np.arange(n) % self.K)
            for k in range(self.K):
                tr = folds != k
                if not np.any(tr & ~src) or np.sum(tr & src) <= self.n_inner:
                    raise ValueError("Insufficient source/target training rows in an outer fold")
                if any(np.sum(tr & src & (A == a)) < 2 for a in (0, 1)):
                    raise ValueError("Each outer training fold needs at least two source rows per arm")
            with threadpool_limits(limits=1):
                nuisance = fit_ml_nuisance(X, R, A, Y, self.seed, self.K, self.n_inner, self.learner)
        for name in ("e1", "omega"):
            value = np.asarray(getattr(nuisance, name, []), dtype=float)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"nuisance.{name} must be a finite array of length n")
            if (name == "e1" and np.any((value <= 0) | (value >= 1))) or (name == "omega" and np.any(value <= 0)):
                raise ValueError("Nuisance propensities must lie in (0,1) and density ratios must be positive")
        if not callable(getattr(nuisance, "cdf", None)):
            raise ValueError("nuisance must provide cdf(a,y) with shape (len(y),n)")
        if self.density == "plugin" and not callable(getattr(nuisance, "pdf", None)):
            raise ValueError("density='plugin' requires nuisance.pdf; use 'auto' or 'fd' for CDF-only nuisances")
        nuisance = nuisance if isinstance(nuisance, CachedNuisance) else CachedNuisance(nuisance)
        with threadpool_limits(limits=1):
            super().fit(X, R, A, Y, nuisance=nuisance)
        nuisance.register_grid(self.y_coarse)
        return self

    def bounds(self, S, taus, alpha=0.05, n_boot=500, seed=None, keep_phi=False):
        """Proposed only: nested bounds and the full one-step source correction."""
        if not hasattr(self, "nu"):
            raise RuntimeError("Call fit(X,R,A,Y) before bounds(S,taus)")
        S, taus = np.atleast_2d(np.asarray(S, float)), np.atleast_1d(np.asarray(taus, float))
        if S.ndim != 2 or S.shape[1] != 2 or not len(S) or not np.isfinite(S).all() or np.any(S < 1):
            raise ValueError("S must contain finite (Gamma,Lambda) pairs, each >= 1")
        if taus.ndim != 1 or not len(taus) or not np.isfinite(taus).all() or np.any((taus <= 0) | (taus >= 1)):
            raise ValueError("taus must be a nonempty one-dimensional array with values in (0,1)")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0,1)")
        if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
            raise ValueError("n_boot must be an integer >= 2")
        seed = self.bootstrap_seed if seed is None else seed
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("bootstrap seed must be a nonnegative integer")
        with threadpool_limits(limits=1):
            return super().bounds(S, taus, kind="nested", variant="onestep", alpha=alpha,
                                  n_boot=int(n_boot), seed=int(seed), keep_phi=keep_phi)


def fit_proposed(X, R, A, Y, *, nuisance=None, **options):
    """Convenience entry: return ProposedQTE(**options).fit(X,R,A,Y,nuisance)."""
    return ProposedQTE(**options).fit(X, R, A, Y, nuisance=nuisance)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(frame, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _latex_cell(value, digits=3):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "--"
        if np.isinf(value):
            return r"$-\infty$" if value < 0 else r"$\infty$"
        return f"{value:.{digits}f}"
    text = str(value)
    if text.startswith("$") and text.endswith("$"):
        return text
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def save_table(frame, stem, digits=3):
    """Unrounded CSV plus a booktabs tabular; no optional LaTeX-export dependencies."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(frame, stem.with_suffix(".csv"))
    lines = [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\toprule",
             " & ".join(_latex_cell(v, digits) for v in frame.columns) + r" \\",
             r"\midrule"]
    lines.extend(" & ".join(_latex_cell(v, digits) for v in row) + r" \\"
                 for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n")


def record_files(marker, files):
    marker = Path(marker)
    write_json(marker, {"files": {Path(p).name: file_sha256(p) for p in files}})


def checkpoint_valid(marker):
    marker = Path(marker)
    if not marker.exists():
        return False
    for name, expected in json.loads(marker.read_text())["files"].items():
        path = marker.parent / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Checkpoint file changed or missing: {path}")
    return True


def exp1_population(dgp, n_oracle):
    X0, w0 = dgp.oracle_target(n_oracle)
    pb = PopulationBounds(dgp.e1, dgp.cdf, X0, w0, n_bisect=34)
    truth = {law: true_target_qte(dgp, X0, w0, exp1_T_GRID, EXP1_S0, law)
             for law in ("lfp", "smooth")}
    rows = []
    for name, _, s, kind in exp1_analyses(EXP1_S0):
        lo, hi, _ = pb.qte_bounds([s], exp1_T_GRID, kind)
        for j, t in enumerate(exp1_T_GRID):
            rows.append(dict(analysis=name, Gamma=s[0], Lambda=s[1],
                             kind="proposed" if kind == "nested" else kind, tau=t,
                             lower=lo[0, j], upper=hi[0, j], width=hi[0, j] - lo[0, j],
                             truth_lfp=truth["lfp"][j], truth_smooth=truth["smooth"][j],
                             contains_lfp=lo[0, j] - 1e-9 <= truth["lfp"][j] <= hi[0, j] + 1e-9,
                             contains_smooth=lo[0, j] - 1e-9 <= truth["smooth"][j] <= hi[0, j] + 1e-9))
    return pd.DataFrame(rows), truth


def exp1_population_table(population, folder):
    columns = ["Analysis"] + [f"tau={t:.2f}" for t in exp1_T_GRID]
    display, tex_rows = [], []
    for name, label, _, _ in exp1_analyses(EXP1_S0):
        data = population[population.analysis == name].sort_values("tau")
        values = [f"{r.lower:.3f}" if name == "point" else f"[{r.lower:.3f}, {r.upper:.3f}]"
                  for r in data.itertuples()]
        display.append([label, *values])
        cells = [r"\textbf{Proposed}" if name == "proposed" else label]
        cells.extend("$" + (r"\mathbf{" + v + "}" if name == "proposed" else v) + "$"
                     for v in values)
        tex_rows.append(" & ".join(cells) + r" \\")
    data = population[population.analysis == "proposed"].sort_values("tau")
    for label, column in (("LFP", "truth_lfp"), ("Smooth", "truth_smooth")):
        values = [f"{v:.3f}" for v in data[column]]
        display.append([label, *values])
        tex_rows.append(label + " & " + " & ".join("$" + v + "$" for v in values) + r" \\")
    write_csv(pd.DataFrame(display, columns=columns), folder / "table1_population_intervals.csv")
    tex = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Population QTE bounds and true target QTEs in Simulation 1.}",
        r"\label{tab:sim1-population}", r"\begingroup", r"\small",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        "Analysis & " + " & ".join(rf"$\tau={t:.2f}$" for t in exp1_T_GRID) + r" \\",
        r"\midrule", *tex_rows[:5], r"\midrule", *tex_rows[5:],
        r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table*}",
    ]
    (folder / "table1_population_intervals.tex").write_text("\n".join(tex) + "\n")


def exp1_sharpness_table(dgp, population, config):
    rmin, rmax, alo, ahi = exp1_sharpness_audit(
        dgp, EXP1_S0, exp1_T_GRID, M=config["sharpness_laws"], J=20,
        n_oracle=config["sharpness_oracle"], seed=11)
    p = population[population.analysis == "proposed"].sort_values("tau")
    c = population[population.analysis == "collapsed"].sort_values("tau")
    return pd.DataFrame(dict(
        tau=exp1_T_GRID, lower_proposed=p.lower.to_numpy(), upper_proposed=p.upper.to_numpy(),
        lower_collapsed=c.lower.to_numpy(), upper_collapsed=c.upper.to_numpy(),
        random_min=rmin, random_max=rmax, lfp_law_value=p.truth_lfp.to_numpy(),
        audit_lower_proposed=alo, audit_upper_proposed=ahi,
        random_inside_proposed=(rmin >= alo - 2e-3) & (rmax <= ahi + 2e-3),
        share_of_proposed_width_reached=(rmax - rmin) / (ahi - alo)))


def prepare_population(out, config, dgp):
    """Compute each required population quantity once; reuse only verified files."""
    truth1, targets2, truth3, population3 = None, None, None, None
    for experiment in config["experiments"]:
        folder = out / f"exp{experiment}"
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "population.done.json"
        if not checkpoint_valid(marker):
            started = time.time()
            print(f"Exp{experiment}: computing population quantities", flush=True)
            if experiment == 1:
                population, truth1 = exp1_population(dgp, config["n_oracle"])
                write_csv(population, folder / "population_bounds.csv")
                exp1_population_table(population, folder)
                files = [folder / "population_bounds.csv", folder / "table1_population_intervals.csv",
                         folder / "table1_population_intervals.tex"]
                if config["sharpness_laws"]:
                    audit = exp1_sharpness_table(dgp, population, config)
                    save_table(audit, folder / "table3_sharpness_audit")
                    files += [folder / "table3_sharpness_audit.csv", folder / "table3_sharpness_audit.tex"]
            elif experiment == 2:
                targets2 = exp2_population_targets(dgp, config["n_oracle"])
                write_csv(targets2, folder / "population_targets.csv")
                files = [folder / "population_targets.csv"]
            else:
                population3 = exp3_population_truth(dgp, config["n_oracle"], config["frontier_oracle"])
                truth3 = (population3["lower"], population3["upper"])
                frame = pd.DataFrame(dict(Gamma=np.repeat(exp3_S_GRID[:, 0], len(exp3_T_GRID)),
                    Lambda=np.repeat(exp3_S_GRID[:, 1], len(exp3_T_GRID)),
                    tau=np.tile(exp3_T_GRID, len(exp3_S_GRID)),
                    lower=truth3[0].ravel(), upper=truth3[1].ravel()))
                write_csv(frame, folder / "truth_grid.csv")
                frontiers = pd.concat([pd.DataFrame(dict(tau=t, Gamma=g, Lambda_frontier=f,
                    true_area=population3["true_area"][t])) for t, (g, f) in population3["fronts"].items()])
                write_csv(frontiers, folder / "truth_frontiers.csv")
                files = [folder / "truth_grid.csv", folder / "truth_frontiers.csv"]
            record_files(marker, files)
            print(f"Exp{experiment}: population quantities ready ({time.time() - started:.1f}s)", flush=True)
        if experiment == 1 and truth1 is None:
            data = pd.read_csv(folder / "population_bounds.csv", float_precision="round_trip")
            data = data[data.analysis == "proposed"].sort_values("tau")
            truth1 = {law: data[f"truth_{law}"].to_numpy() for law in ("lfp", "smooth")}
        elif experiment == 2 and targets2 is None:
            targets2 = pd.read_csv(folder / "population_targets.csv", float_precision="round_trip")
        elif experiment == 3 and truth3 is None:
            data = pd.read_csv(folder / "truth_grid.csv", float_precision="round_trip")
            shape = (len(exp3_S_GRID), len(exp3_T_GRID))
            truth3 = (data.lower.to_numpy().reshape(shape), data.upper.to_numpy().reshape(shape))
    return truth1, targets2, truth3


def exp1_evaluate(seed, n, X, R, A, Y, nuisance, truth, n_boot):
    est = make_estimator(X, R, A, Y, nuisance)
    rows = []
    for name, _, s, kind in exp1_analyses(EXP1_S0):
        res = est.bounds([s], exp1_T_GRID, kind=kind, n_boot=n_boot, seed=seed + 1)
        Ll, _, _, Uu = res.ci("sim")
        L, U = Ll[0], Uu[0]
        for law, tr in truth.items():
            eligible, certified = tr <= 0, L > 0
            rows.append(dict(seed=seed, n=n, analysis=name, law=law,
                cover=bool(np.all((L <= tr) & (tr <= U))), width=float(np.mean(U - L)),
                **{f"certify_{t:g}": bool(L[j] > 0) for j, t in enumerate(exp1_T_GRID)},
                false_certification=float(np.mean(certified[eligible])) if np.any(eligible) else np.nan,
                n_nonpositive_taus=int(np.sum(eligible)), n_false_certifications=int(np.sum(certified & eligible)),
                **{f"false_certify_{t:g}": float(certified[j]) if eligible[j] else np.nan
                   for j, t in enumerate(exp1_T_GRID)}, exceptional=res.exceptional))
    return pd.DataFrame(rows)


def exp2_result_rows(res, seed, n, variant):
    rows = []
    for i, (G, L) in enumerate(exp2_S_EVAL):
        for j, t in enumerate(exp2_TAUS):
            base = dict(seed=seed, n=n, estimator=variant, Gamma=G, Lambda=L, tau=t,
                        exceptional=res.exceptional)
            rows.append(dict(base, target="QTE_lower", est=res.lower[i, j], se=res.se_lower[i, j]))
            rows.append(dict(base, target="QTE_upper", est=res.upper[i, j], se=res.se_upper[i, j]))
            for a, k in exp2_QKEYS:
                rows.append(dict(base, target=f"q{a}_{k}", est=res.qbounds[a, k][i, j],
                                 se=res.qse[a, k][i, j]))
    return rows


def exp2_evaluate(seed, n, X, R, A, Y, nuisance, n_boot, oracle_rows):
    est = make_estimator(X, R, A, Y, nuisance, exp2=True)
    rows = []
    for variant in exp2_ESTIMATORS:
        if variant == "oracle":
            rows.extend(oracle_rows)
        else:
            res = est.bounds(exp2_S_EVAL, exp2_TAUS, variant=variant, n_boot=n_boot, seed=seed)
            rows.extend(exp2_result_rows(res, seed, n, variant))
    return pd.DataFrame(rows)


def exp3_evaluate(seed, n, X, R, A, Y, nuisance, truth, n_boot):
    est = make_estimator(X, R, A, Y, nuisance)
    res = est.bounds(exp3_S_GRID, exp3_T_GRID, n_boot=n_boot, seed=seed + 7)
    truth_lo, truth_hi = truth
    true_pos = truth_lo > 0
    row = dict(seed=seed, n=n, exceptional=res.exceptional)
    for method in exp3_METHODS:
        Ll, Ul, Lu, Uu = res.ci(method)
        row[f"{method}_crit"] = res.crit[method]
        row[f"{method}_endpoint_cover"] = bool(
            ((Ll <= truth_lo) & (truth_lo <= Ul) & (Lu <= truth_hi) & (truth_hi <= Uu)).all())
        row[f"{method}_set_cover"] = bool(((Ll <= truth_lo) & (truth_hi <= Uu)).all())
        hw = float(np.mean(res.crit[method] * np.r_[res.se_lower.ravel(), res.se_upper.ravel()]))
        row[f"{method}_halfwidth"] = hw if np.isfinite(hw) else np.nan
        cert, excl = Ll > 0, Ul <= 0
        inner_ok, outer_ok = ~(cert & ~true_pos).any(axis=0), ~(excl & true_pos).any(axis=0)
        row[f"{method}_inner_valid"] = bool(inner_ok.all())
        row[f"{method}_outer_valid"] = bool(outer_ok.all())
        row[f"{method}_region_valid"] = bool((inner_ok & outer_ok).all())
        for j, t in enumerate(exp3_T_GRID):
            npos = true_pos[:, j].sum()
            row[f"{method}_power_{t:g}"] = (cert[:, j] & true_pos[:, j]).sum() / npos if npos else np.nan
            inner, _ = region_membership(exp3_S_GRID, cert[:, j], excl[:, j], exp3_EVAL_PTS)
            row[f"{method}_innerarea_{t:g}"] = inner.mean()
    tab = res.table()
    tab["seed"], tab["n"] = seed, n
    return pd.DataFrame([row]), tab


def run_replication(task, n, seed, folder, config, dgp, truth1, truth3):
    """One independent seed; the shared fitted nuisance is never modified in place."""
    folder = Path(folder)
    marker = folder / f"{task}_seed{seed}.done.json"
    if checkpoint_valid(marker):
        return
    with threadpool_limits(limits=1):
        X, R, A, Y = dgp.sample(n, np.random.default_rng(seed))
        files = []
        if task == "exp1":
            nu = fit_ml_nuisance(X, R, A, Y, seed, config["K"], config["n_inner"], config["learner"])
            raw = exp1_evaluate(seed, n, X, R, A, Y, nu, truth1, config["n_boot"])
            path = folder / f"exp1_seed{seed}.csv"
            write_csv(raw, path)
            files.append(path)
        else:
            do2, do3 = 2 in config["experiments"], 3 in config["experiments"]
            need_ml = do3 or (do2 and "ml" in config["regimes"])
            nu_ml = None
            if need_ml:
                nu_ml = CachedNuisance(fit_ml_nuisance(
                    X, R, A, Y, seed, config["K"], config["n_inner"], config["learner"]))
            nu_oracle = dgp.oracle_nuisance(X)
            if do2:
                oracle = make_estimator(X, R, A, Y, nu_oracle, exp2=True)
                res = oracle.bounds(exp2_S_EVAL, exp2_TAUS, n_boot=config["n_boot"], seed=seed)
                oracle_rows = exp2_result_rows(res, seed, n, "oracle")
                for regime in config["regimes"]:
                    if regime == "ml":
                        nu = nu_ml
                    else:
                        nu = exp2_rate_nuisance(dgp, X, R, A, Y, config["rate"], config["rate_consts"],
                            config["K"], seed, n_inner=config["n_inner"])
                    raw = exp2_evaluate(seed, n, X, R, A, Y, nu, config["n_boot"], oracle_rows)
                    path = folder / f"exp2_{regime}_seed{seed}.csv"
                    write_csv(raw, path)
                    files.append(path)
            if do3:
                raw, endpoints = exp3_evaluate(seed, n, X, R, A, Y, nu_ml, truth3, config["n_boot"])
                oracle_raw, oracle_endpoints = exp3_evaluate(
                    seed, n, X, R, A, Y, nu_oracle, truth3, config["n_boot"])
                raw = pd.concat([raw, oracle_raw.drop(columns=["seed", "n"]).add_prefix("oracle_")], axis=1)
                path = folder / f"exp3_seed{seed}.csv"
                write_csv(raw, path)
                files.append(path)
                path = folder / f"exp3_endpoints_seed{seed}.csv"
                write_csv(endpoints, path)
                files.append(path)
                path = folder / f"exp3_oracle_endpoints_seed{seed}.csv"
                write_csv(oracle_endpoints, path)
                files.append(path)
        record_files(marker, files)


def exp1_summarize(raw):
    raw = raw.copy()
    raw["width_finite"] = raw.width.where(np.isfinite(raw.width))
    agg = raw.groupby(["analysis", "law", "n"]).agg(
        coverage=("cover", "mean"), mean_width=("width_finite", "mean"), median_width=("width", "median"),
        exceptional=("exceptional", "mean"), false_certification=("false_certification", "mean"),
        false_certification_sd=("false_certification", "std"),
        n_nonpositive_taus=("n_nonpositive_taus", "first"),
        false_certification_count=("n_false_certifications", "sum"),
        **{f"false_certify_{t:g}": (f"false_certify_{t:g}", "mean") for t in exp1_T_GRID},
        **{f"certify_{t:g}": (f"certify_{t:g}", "mean") for t in exp1_T_GRID},
        reps=("cover", "size")).reset_index()
    agg["mc_se"] = np.sqrt(agg.coverage * (1 - agg.coverage) / agg.reps)
    agg["false_certification_mc_se"] = agg.pop("false_certification_sd") / np.sqrt(agg.reps)
    agg["false_certification_denominator"] = agg.reps * agg.n_nonpositive_taus
    return agg


def collect_results(out, config, make_plots=True):
    def collect(task, prefix):
        frames = []
        for n in config["n"]:
            folder = out / "replications" / f"n{n}"
            for seed in config["seeds"][task][str(n)]:
                if not checkpoint_valid(folder / f"{task}_seed{seed}.done.json"):
                    raise ValueError(f"Incomplete replication: {task}, n={n}, seed={seed}")
                frames.append(pd.read_csv(folder / f"{prefix}_seed{seed}.csv", float_precision="round_trip"))
        return pd.concat(frames, ignore_index=True)

    if 1 in config["experiments"]:
        folder = out / "exp1"
        raw = collect("exp1", "exp1")
        write_csv(raw, folder / "B_raw.csv")
        summary = exp1_summarize(raw)
        save_table(summary, folder / "table4_finite_sample")
        fc_cols = ["analysis", "law", "n", "n_nonpositive_taus", "false_certification_count",
                   "false_certification_denominator", "false_certification", "false_certification_mc_se",
                   *[f"false_certify_{t:g}" for t in exp1_T_GRID], "reps"]
        save_table(summary[fc_cols], folder / "table5_false_certification")
        if make_plots:
            import matplotlib
            matplotlib.use("Agg")
            exp1_plot_part_b(summary, str(folder))
        print("\nExp1 simultaneous coverage:\n" + summary.pivot_table(
            index=["law", "analysis"], columns="n", values="coverage").round(3).to_string(), flush=True)
    if 2 in config["experiments"]:
        targets = pd.read_csv(out / "exp2/population_targets.csv", float_precision="round_trip")
        for regime in config["regimes"]:
            folder = out / "exp2" / regime
            folder.mkdir(parents=True, exist_ok=True)
            raw = collect("pair", f"exp2_{regime}")
            write_csv(raw, folder / "raw.csv")
            summary = exp2_summarize(raw, targets)
            save_table(summary, folder / "summary")
            coverage_table = exp2_coverage_width_table(summary)
            save_table(coverage_table, folder / "table_coverage_width_all_tau")
            print(f"\nExp2 {regime}, mean coverage, CI full width, signed and absolute cellwise bias "
                  "across three tau values and both QTE endpoints:\n"
                  + coverage_table.round(4).to_string(index=False), flush=True)
            save_table(exp2_paper_table(summary), folder / "table_paper_tau05", digits=2)
            save_table(exp2_paper_table(summary, targets=("q1_lo", "q1_hi", "q0_lo", "q0_hi")),
                       folder / "table_quantile_bounds_tau05", digits=2)
            main = summary[(summary.tau == 0.5) & summary.target.isin(["QTE_lower", "QTE_upper"])]
            cols = ["n", "Gamma", "Lambda", "tau", "target", "estimator", "bias", "sd", "rmse",
                    "mean_se", "se_over_sd", "coverage", "mc_se_cov", "reps"]
            save_table(main[cols], folder / "table_bias_rmse_tau05")
            print(f"\nExp2 {regime}, tau=.5 coverage:\n" + main.pivot_table(
                index=["Gamma", "Lambda", "target", "estimator"], columns="n",
                values="coverage").round(3).to_string(), flush=True)
            sd_table = exp2_proposed_plugin_sd_table(summary)
            save_table(sd_table, folder / "table_proposed_plugin_sd")
            print(f"\nExp2 {regime}, empirical SD across repetitions:\n"
                  + sd_table.round(4).to_string(index=False), flush=True)
    if 3 in config["experiments"]:
        folder = out / "exp3"
        raw = collect("pair", "exp3")
        write_csv(raw, folder / "raw.csv")
        summary = raw.drop(columns=["seed"]).groupby("n").mean(numeric_only=True).reset_index()
        frontiers = pd.read_csv(folder / "truth_frontiers.csv", float_precision="round_trip")
        for t in exp3_T_GRID:
            area = float(frontiers.loc[frontiers.tau == t, "true_area"].iloc[0])
            for prefix in ("", "oracle_"):
                for method in exp3_METHODS:
                    summary[f"{prefix}{method}_arearatio_{t:g}"] = (
                        summary[f"{prefix}{method}_innerarea_{t:g}"] / area if area > 0 else np.nan)
        summary["reps"] = raw.groupby("n").size().values
        save_table(summary, folder / "summary")
        cols = ["n"] + [f"{prefix}{m}_{k}" for prefix in ("", "oracle_")
                        for k in ("endpoint_cover", "set_cover", "region_valid", "crit", "halfwidth")
                        for m in exp3_METHODS]
        save_table(summary[cols], folder / "table_paper")
        rows = []
        labels = {"sim": "Proposed", "bonf": "Bonferroni", "pw": "Pointwise"}
        for r in summary.to_dict("records"):
            for prefix, nuisance in (("", "ML"), ("oracle_", "Oracle")):
                for method in exp3_METHODS:
                    coverage = r[f"{prefix}{method}_endpoint_cover"]
                    label = labels[method] if not prefix else f"Oracle {labels[method]}"
                    rows.append(dict(n=int(r["n"]), method=label, nuisance=nuisance,
                        endpoint_coverage=coverage, set_coverage=r[f"{prefix}{method}_set_cover"],
                        region_validity=r[f"{prefix}{method}_region_valid"],
                        critical_value=r[f"{prefix}{method}_crit"],
                        mean_halfwidth=r[f"{prefix}{method}_halfwidth"],
                        mc_se=np.sqrt(coverage * (1 - coverage) / r["reps"]), reps=int(r["reps"])))
        save_table(pd.DataFrame(rows), folder / "table_simultaneous")
        print("\nExp3 simultaneous inference:\n" + pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
        coverage_table = summary[["n", "sim_endpoint_cover", "oracle_sim_endpoint_cover",
                                  "sim_set_cover", "oracle_sim_set_cover",
                                  "sim_halfwidth", "oracle_sim_halfwidth", "reps"]].rename(columns={
            "sim_endpoint_cover": "proposed_endpoint_coverage",
            "oracle_sim_endpoint_cover": "oracle_endpoint_coverage",
            "sim_set_cover": "proposed_set_coverage",
            "oracle_sim_set_cover": "oracle_set_coverage",
            "sim_halfwidth": "proposed_mean_halfwidth",
            "oracle_sim_halfwidth": "oracle_mean_halfwidth"})
        save_table(coverage_table, folder / "table_proposed_oracle_coverage")
        print("\nExp3 Proposed and Oracle simultaneous coverage:\n"
              + coverage_table.round(4).to_string(index=False), flush=True)
        width_table = exp3_proposed_oracle_width_table(summary)
        save_table(width_table, folder / "table_proposed_oracle_width")
        print("\nExp3 Proposed and Oracle Proposed mean widths:\n"
              + width_table.round(4).to_string(index=False), flush=True)


# Frozen numerical checkpoints from the archived 500-repetition experiments.
# These are used ONLY by --verify, never for estimation or simulation output.
_REPRO_CHECKPOINTS = {'exp2_proposed': {'targets': ['QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi',
                               'QTE_lower',
                               'QTE_upper',
                               'q1_lo',
                               'q1_hi',
                               'q0_lo',
                               'q0_hi'],
                   'values': [[1.0, 1.0, 0.25, 0.9291505847415958, 0.05921642008988618],
                              [1.0, 1.0, 0.25, 0.9291505847415958, 0.05921642008988618],
                              [1.0, 1.0, 0.25, 0.60411479198399, 0.04080777447085555],
                              [1.0, 1.0, 0.25, 0.60411479198399, 0.04080777447085555],
                              [1.0, 1.0, 0.25, -0.32503579275760586, 0.04642161078410865],
                              [1.0, 1.0, 0.25, -0.32503579275760586, 0.04642161078410865],
                              [1.0, 1.0, 0.5, 0.8654093836775286, 0.06342624079607313],
                              [1.0, 1.0, 0.5, 0.8654093836775286, 0.06342624079607313],
                              [1.0, 1.0, 0.5, 1.0417947585664429, 0.041616211102615534],
                              [1.0, 1.0, 0.5, 1.0417947585664429, 0.041616211102615534],
                              [1.0, 1.0, 0.5, 0.17638537488891426, 0.050960413591510496],
                              [1.0, 1.0, 0.5, 0.17638537488891426, 0.050960413591510496],
                              [1.0, 1.0, 0.75, 0.9529453769940193, 0.07915103093133971],
                              [1.0, 1.0, 0.75, 0.9529453769940193, 0.07915103093133971],
                              [1.0, 1.0, 0.75, 1.4858200030828752, 0.049680539303840594],
                              [1.0, 1.0, 0.75, 1.4858200030828752, 0.049680539303840594],
                              [1.0, 1.0, 0.75, 0.5328746260888559, 0.06389004574671948],
                              [1.0, 1.0, 0.75, 0.5328746260888559, 0.06389004574671948],
                              [1.25, 1.0, 0.25, 0.8402686233785036, 0.05735760342866349],
                              [1.25, 1.0, 0.25, 1.0318767888697344, 0.06277218560404632],
                              [1.25, 1.0, 0.25, 0.5766666578907906, 0.04124089834182634],
                              [1.25, 1.0, 0.25, 0.6343029324577716, 0.03999032893630774],
                              [1.25, 1.0, 0.25, -0.39757385641196286, 0.051495877775991676],
                              [1.25, 1.0, 0.25, -0.263601965487713, 0.043775546322586935],
                              [1.25, 1.0, 0.5, 0.7555687770173523, 0.06434918275208198],
                              [1.25, 1.0, 0.5, 0.9868349295959555, 0.06319963458886382],
                              [1.25, 1.0, 0.5, 0.9879079664149952, 0.04303333336196174],
                              [1.25, 1.0, 0.5, 1.0851541577819699, 0.041414814923712705],
                              [1.25, 1.0, 0.5, 0.09831922818601438, 0.05099385516569408],
                              [1.25, 1.0, 0.5, 0.23233918939764298, 0.05101857275602175],
                              [1.25, 1.0, 0.75, 0.831712112225107, 0.0825221305118179],
                              [1.25, 1.0, 0.75, 1.0248104566250755, 0.07863738228589218],
                              [1.25, 1.0, 0.75, 1.4386149808770576, 0.04936505473304309],
                              [1.25, 1.0, 0.75, 1.5264874662051988, 0.05044663382353361],
                              [1.25, 1.0, 0.75, 0.5016770095801233, 0.06283536509768098],
                              [1.25, 1.0, 0.75, 0.6069028686519506, 0.06834475362914837],
                              [1.5, 1.25, 0.25, 0.5829964453280602, 0.05862747260565489],
                              [1.5, 1.25, 0.25, 1.24828922264834, 0.059890708176357464],
                              [1.5, 1.25, 0.25, 0.4816316997432334, 0.04585192505672081],
                              [1.5, 1.25, 0.25, 0.7732741332693993, 0.03770068253537673],
                              [1.5, 1.25, 0.25, -0.47501508937894055, 0.049949433207756674],
                              [1.5, 1.25, 0.25, -0.1013647455848268, 0.04089299802270855],
                              [1.5, 1.25, 0.5, 0.43493996020187076, 0.07185320149783792],
                              [1.5, 1.25, 0.5, 1.3391420657939561, 0.068891445217925],
                              [1.5, 1.25, 0.5, 0.8071636858713356, 0.04792545594922827],
                              [1.5, 1.25, 0.5, 1.2882991922440725, 0.046012494694026206],
                              [1.5, 1.25, 0.5, -0.050842873549883594, 0.054636887692112515],
                              [1.5, 1.25, 0.5, 0.3722237256694648, 0.0563124315711053],
                              [1.5, 1.25, 0.75, 0.5445402154251537, 0.09299004400448578],
                              [1.5, 1.25, 0.75, 1.340007330966771, 0.07672558413558928],
                              [1.5, 1.25, 0.75, 1.2922409558091204, 0.05217504137926245],
                              [1.5, 1.25, 0.75, 1.682619759610998, 0.05747082670638477],
                              [1.5, 1.25, 0.75, 0.3426124286442269, 0.054019309718609944],
                              [1.5, 1.25, 0.75, 0.7477007403839666, 0.07882142310115607]]},
 'exp2_oracle': {'targets': ['QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi',
                             'QTE_lower',
                             'QTE_upper',
                             'q1_lo',
                             'q1_hi',
                             'q0_lo',
                             'q0_hi'],
                 'values': [[1.0, 1.0, 0.25, 0.9105473835260649, 0.053086655967562825],
                            [1.0, 1.0, 0.25, 0.9105473835260649, 0.053086655967562825],
                            [1.0, 1.0, 0.25, 0.5880593159995267, 0.03415848412546123],
                            [1.0, 1.0, 0.25, 0.5880593159995267, 0.03415848412546123],
                            [1.0, 1.0, 0.25, -0.32248806752653825, 0.043799427514423846],
                            [1.0, 1.0, 0.25, -0.32248806752653825, 0.043799427514423846],
                            [1.0, 1.0, 0.5, 0.8867045209862632, 0.057894933840176535],
                            [1.0, 1.0, 0.5, 0.8867045209862632, 0.057894933840176535],
                            [1.0, 1.0, 0.5, 1.0335747394246968, 0.03938078946692722],
                            [1.0, 1.0, 0.5, 1.0335747394246968, 0.03938078946692722],
                            [1.0, 1.0, 0.5, 0.14687021843843362, 0.04611039190695223],
                            [1.0, 1.0, 0.5, 0.14687021843843362, 0.04611039190695223],
                            [1.0, 1.0, 0.75, 0.9450137795765445, 0.07351868989146354],
                            [1.0, 1.0, 0.75, 0.9450137795765445, 0.07351868989146354],
                            [1.0, 1.0, 0.75, 1.4741869935372458, 0.049333414109687876],
                            [1.0, 1.0, 0.75, 1.4741869935372458, 0.049333414109687876],
                            [1.0, 1.0, 0.75, 0.5291732139607013, 0.05765772160136677],
                            [1.0, 1.0, 0.75, 0.5291732139607013, 0.05765772160136677],
                            [1.25, 1.0, 0.25, 0.831183339063942, 0.0515106116646983],
                            [1.25, 1.0, 0.25, 1.0278388847299291, 0.05743360139014789],
                            [1.25, 1.0, 0.25, 0.5590248624228324, 0.03474540310071541],
                            [1.25, 1.0, 0.25, 0.6302650283179663, 0.03363160271886129],
                            [1.25, 1.0, 0.25, -0.39757385641196286, 0.049376526629237964],
                            [1.25, 1.0, 0.25, -0.2721584766411096, 0.04146965492344384],
                            [1.25, 1.0, 0.5, 0.7601354543183223, 0.058015563288696545],
                            [1.25, 1.0, 0.5, 0.9824605334444997, 0.057793010919615743],
                            [1.25, 1.0, 0.5, 0.9879079664149952, 0.03969275064433688],
                            [1.25, 1.0, 0.5, 1.0723193910518747, 0.039194051996374914],
                            [1.25, 1.0, 0.5, 0.08985885760737497, 0.046244437739183075],
                            [1.25, 1.0, 0.5, 0.22777251209667287, 0.045983611573757584],
                            [1.25, 1.0, 0.75, 0.8366152815377275, 0.07661129696961111],
                            [1.25, 1.0, 0.75, 1.0268294086949776, 0.07079637945755707],
                            [1.25, 1.0, 0.75, 1.4386149808770576, 0.04946603540998528],
                            [1.25, 1.0, 0.75, 1.5211035940187916, 0.04983792398362177],
                            [1.25, 1.0, 0.75, 0.49427418532381384, 0.053973531799560444],
                            [1.25, 1.0, 0.75, 0.6019996993393302, 0.06141709019714724],
                            [1.5, 1.25, 0.25, 0.5724690523921395, 0.05079491164755221],
                            [1.5, 1.25, 0.25, 1.24828922264834, 0.0592137212126809],
                            [1.5, 1.25, 0.25, 0.4490881151878988, 0.037429367942588304],
                            [1.5, 1.25, 0.25, 0.7732741332693993, 0.034452580091463965],
                            [1.5, 1.25, 0.25, -0.47501508937894055, 0.05128518477794208],
                            [1.5, 1.25, 0.25, -0.12338093720424068, 0.03782340113970253],
                            [1.5, 1.25, 0.5, 0.47916462459021303, 0.060456964574804005],
                            [1.5, 1.25, 0.5, 1.3009261873279427, 0.06346277813249142],
                            [1.5, 1.25, 0.5, 0.8245651299024006, 0.040894027612209886],
                            [1.5, 1.25, 0.5, 1.2494103297547583, 0.0432467164333355],
                            [1.5, 1.25, 0.5, -0.051515857573184456, 0.05061261166361659],
                            [1.5, 1.25, 0.5, 0.3454005053121876, 0.047894317588453715],
                            [1.5, 1.25, 0.75, 0.5205050717358373, 0.09066929007218368],
                            [1.5, 1.25, 0.75, 1.2924658167493033, 0.07047838658380023],
                            [1.5, 1.25, 0.75, 1.291231479774169, 0.052884429752689616],
                            [1.5, 1.25, 0.75, 1.6604593571294484, 0.05167618116650281],
                            [1.5, 1.25, 0.75, 0.3679935403801451, 0.05250721376101147],
                            [1.5, 1.25, 0.75, 0.7707264080383318, 0.07573450284055361]]},
 'exp3_proposed': {'columns': ['Gamma',
                               'Lambda',
                               'tau',
                               'lower',
                               'upper',
                               'se_lower',
                               'se_upper',
                               'L_lower_sim',
                               'U_lower_sim',
                               'L_upper_sim',
                               'U_upper_sim',
                               'L_lower_bonf',
                               'U_lower_bonf',
                               'L_upper_bonf',
                               'U_upper_bonf',
                               'L_lower_pw',
                               'U_lower_pw',
                               'L_upper_pw',
                               'U_upper_pw'],
                   'values': [[1.0,
                               1.0,
                               0.25,
                               0.9266028595105282,
                               0.9266028595105282,
                               0.0605972801886487,
                               0.0605972801886487,
                               0.7513499894494172,
                               1.1018557295716394,
                               0.7513499894494172,
                               1.1018557295716394,
                               0.7453372387459165,
                               1.10786848027514,
                               0.7453372387459165,
                               1.10786848027514,
                               0.8078343727796943,
                               1.0453713462413623,
                               0.8078343727796943,
                               1.0453713462413623],
                              [1.0,
                               1.0,
                               0.5,
                               0.8604581440775294,
                               0.8604581440775294,
                               0.06505667263343305,
                               0.06505667263343305,
                               0.672308303686326,
                               1.0486079844687328,
                               0.672308303686326,
                               1.0486079844687328,
                               0.6658530708289054,
                               1.0550632173261534,
                               0.6658530708289054,
                               1.0550632173261534,
                               0.7329494087619881,
                               0.9879668793930707,
                               0.7329494087619881,
                               0.9879668793930707],
                              [1.0,
                               1.0,
                               0.75,
                               0.9529453769940193,
                               0.9529453769940193,
                               0.08109576275390176,
                               0.08109576275390176,
                               0.718409022752191,
                               1.1874817312358477,
                               0.718409022752191,
                               1.1874817312358477,
                               0.7103623149951523,
                               1.1955284389928862,
                               0.7103623149951523,
                               1.1955284389928862,
                               0.794000602697567,
                               1.1118901512904715,
                               0.794000602697567,
                               1.1118901512904715],
                              [1.25,
                               1.0,
                               0.25,
                               0.8395956393552028,
                               1.0299539773745892,
                               0.05879316462959524,
                               0.06407711085803174,
                               0.6695604363026062,
                               1.0096308424077993,
                               0.8446371191239506,
                               1.2152708356252278,
                               0.6637266985360415,
                               1.015464580174364,
                               0.8382790830505412,
                               1.2216288716986372,
                               0.7243631541440619,
                               0.9548281245663436,
                               0.9043651478594665,
                               1.1555428068897118],
                              [1.25,
                               1.0,
                               0.5,
                               0.7555687770173523,
                               0.9856812426988684,
                               0.06592865599093496,
                               0.06597247399376902,
                               0.5648970810977821,
                               0.9462404729369224,
                               0.7948828211121873,
                               1.1764796642855493,
                               0.5583553258992607,
                               0.9527822281354438,
                               0.7883367180827642,
                               1.1830257673149724,
                               0.6263509857259888,
                               0.8847865683087157,
                               0.8563775697000757,
                               1.114984915697661],
                              [1.25,
                               1.0,
                               0.75,
                               0.831712112225107,
                               1.0248104566250755,
                               0.08454969236741264,
                               0.0805689462571954,
                               0.5871866780134778,
                               1.0762375464367362,
                               0.7917977038180956,
                               1.2578232094320556,
                               0.5787972549126345,
                               1.0846269695375796,
                               0.783803269302551,
                               1.2658176439476,
                               0.6659977602810372,
                               0.9974264641691768,
                               0.8668982236886293,
                               1.1827226895615217],
                              [1.5,
                               1.25,
                               0.25,
                               0.5818908287183516,
                               1.24828922264834,
                               0.06012919074107945,
                               0.06135373286616814,
                               0.4079917161010804,
                               0.7557899413356228,
                               1.0708486223394689,
                               1.425729822957211,
                               0.4020254114616013,
                               0.7617562459751019,
                               1.0647608127990904,
                               1.4318176324975895,
                               0.4640397804462966,
                               0.6997418769904067,
                               1.128038115913559,
                               1.3685403293831209],
                              [1.5,
                               1.25,
                               0.5,
                               0.4296041583028425,
                               1.3391420657939561,
                               0.07386064540583147,
                               0.0705747000079955,
                               0.21599242438276545,
                               0.6432158922229195,
                               1.1350335861460523,
                               1.54325054544186,
                               0.20866361942250392,
                               0.6505446971831811,
                               1.1280308283357783,
                               1.550253303252134,
                               0.284839953432529,
                               0.574368363173156,
                               1.2008181955585662,
                               1.477465936029346],
                              [1.5,
                               1.25,
                               0.75,
                               0.5403100301358341,
                               1.3406322447026933,
                               0.09470940702314544,
                               0.07859880613491217,
                               0.2664017730173669,
                               0.8142182872543013,
                               1.1133173171079638,
                               1.5679471722974228,
                               0.2570042546479992,
                               0.8236158056236691,
                               1.1055183692771247,
                               1.5757461201282619,
                               0.3546830033733242,
                               0.7259370568983441,
                               1.1865814154504195,
                               1.494683073954967]],
                   'critical': 2.8920913532013612},
 'exp3_oracle': {'columns': ['Gamma',
                             'Lambda',
                             'tau',
                             'lower',
                             'upper',
                             'se_lower',
                             'se_upper',
                             'L_lower_sim',
                             'U_lower_sim',
                             'L_upper_sim',
                             'U_upper_sim',
                             'L_lower_bonf',
                             'U_lower_bonf',
                             'L_upper_bonf',
                             'U_upper_bonf',
                             'L_lower_pw',
                             'U_lower_pw',
                             'L_upper_pw',
                             'U_upper_pw'],
                 'values': [[1.0,
                             1.0,
                             0.25,
                             0.9105473835260649,
                             0.9105473835260649,
                             0.05111885663635047,
                             0.05111885663635047,
                             0.7667802858782571,
                             1.0543144811738727,
                             0.7667802858782571,
                             1.0543144811738727,
                             0.7576347238799803,
                             1.0634600431721495,
                             0.7576347238799803,
                             1.0634600431721495,
                             0.8103562655879517,
                             1.010738501464178,
                             0.8103562655879517,
                             1.010738501464178],
                            [1.0,
                             1.0,
                             0.5,
                             0.8910308468503401,
                             0.8910308468503401,
                             0.05571412681793904,
                             0.05571412681793904,
                             0.7343399731170414,
                             1.0477217205836387,
                             0.7343399731170414,
                             1.0477217205836387,
                             0.7243722814564462,
                             1.057689412244234,
                             0.7243722814564462,
                             1.057689412244234,
                             0.7818331648570824,
                             1.0002285288435977,
                             0.7818331648570824,
                             1.0002285288435977],
                            [1.0,
                             1.0,
                             0.75,
                             0.9450137795765445,
                             0.9450137795765445,
                             0.07078263489582488,
                             0.07078263489582488,
                             0.7459441083742175,
                             1.1440834507788715,
                             0.7459441083742175,
                             1.1440834507788715,
                             0.7332805431374937,
                             1.1567470160155953,
                             0.7332805431374937,
                             1.1567470160155953,
                             0.8062823644498798,
                             1.0837451947032093,
                             0.8062823644498798,
                             1.0837451947032093],
                            [1.25,
                             1.0,
                             0.25,
                             0.831183339063942,
                             1.02288764512993,
                             0.049604017149102106,
                             0.05503551761992015,
                             0.6916765885916054,
                             0.9706900895362787,
                             0.8681052975503033,
                             1.1776699927095566,
                             0.6828020431879802,
                             0.9795646349399039,
                             0.858259014365982,
                             1.187516275893878,
                             0.7339612519631947,
                             0.9284054261646894,
                             0.915020012724367,
                             1.130755277535493],
                            [1.25,
                             1.0,
                             0.5,
                             0.7601354543183223,
                             0.9859696644231399,
                             0.055851020436839205,
                             0.05560907731316952,
                             0.6030595798324484,
                             0.9172113288041962,
                             0.829574232790933,
                             1.1423650960553469,
                             0.5930673968361465,
                             0.9272035118004981,
                             0.8196253353057552,
                             1.1523139935405247,
                             0.6506694657623069,
                             0.8696014428743377,
                             0.8769778756758243,
                             1.0949614531704557],
                            [1.25,
                             1.0,
                             0.75,
                             0.8391149364814163,
                             1.0268294086949776,
                             0.07343463065594319,
                             0.06816583405424591,
                             0.6325867703421078,
                             1.045643102620725,
                             0.8351192499479552,
                             1.218539567442,
                             0.6194487423877245,
                             1.058781130575108,
                             0.8229238507835684,
                             1.2307349666063867,
                             0.6951857051777667,
                             0.983044167785066,
                             0.8932268289725217,
                             1.1604319884174337],
                            [1.5,
                             1.25,
                             0.25,
                             0.5743918638872848,
                             1.24828922264834,
                             0.048966980800092465,
                             0.05701737742696332,
                             0.43667671974223343,
                             0.7121070080323362,
                             1.0879330760608374,
                             1.4086453692358425,
                             0.4279161451080734,
                             0.7208675826664962,
                             1.0777322227055486,
                             1.4188462225911314,
                             0.47841834508743925,
                             0.6703653826871303,
                             1.1365372163985648,
                             1.3600412288981152],
                            [1.5,
                             1.25,
                             0.5,
                             0.47916462459021303,
                             1.3039546154327966,
                             0.05820222577114379,
                             0.06113924681107039,
                             0.3154762006631091,
                             0.642853048517317,
                             1.1320060891007313,
                             1.4759031417648618,
                             0.3050633687014258,
                             0.6532658804790002,
                             1.121067801176643,
                             1.48684142968895,
                             0.36509035825870223,
                             0.5932388909217239,
                             1.1841238936411933,
                             1.4237853372243998],
                            [1.5,
                             1.25,
                             0.75,
                             0.5283885988659331,
                             1.2993879381318263,
                             0.08794515116685023,
                             0.06700244856098958,
                             0.2810509241327128,
                             0.7757262735991535,
                             1.1109496944239226,
                             1.48782618183973,
                             0.26531685092826035,
                             0.7914603468036059,
                             1.0989624339945658,
                             1.4998134422690867,
                             0.356019269963976,
                             0.7007579277678904,
                             1.168065552076289,
                             1.4307103241873635]],
                 'critical': 2.81240831872549},
 'exp1_proposed': {'mean_set_width': 1.4827174225727486,
                   'certified': [True, True, True, False, True],
                   'truth': {'lfp': [0.6185069589105296,
                                     0.46441608332401074,
                                     0.35687694463007347,
                                     0.4068840289633613,
                                     0.6607224602532824],
                             'smooth': [0.7154528034189411,
                                        0.6197682704521412,
                                        0.5707952449120057,
                                        0.6364790892843075,
                                        0.7866097348917833]},
                   'cover': {'lfp': True, 'smooth': True}}}


def verify_reproduction():
    """Regenerate fixed-seed samples and compare with independent saved checkpoints.

    No saved CSVs, other project files, or network access are required. Failure
    raises AssertionError (nonzero CLI exit), not a warning or a silent pass.
    """
    from importlib.metadata import version
    packages = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "xgboost")
    print("Reproduction check: n=2000, K=5, inner=5, B=500, XGBoost", flush=True)
    print("Python " + platform.python_version() + "; " + ", ".join(f"{p}={version(p)}" for p in packages), flush=True)
    report = {}

    def check(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, equal_nan=True,
                                   err_msg=f"Archived checkpoint mismatch: {name}")
        finite = np.isfinite(actual) & np.isfinite(expected)
        error = float(np.max(np.abs(actual[finite].astype(float) - expected[finite].astype(float)))) if finite.any() else 0.0
        report[name] = error
        print(f"  PASS {name}: max absolute error = {error:.3g}", flush=True)

    check("Exp1 replication seed", rep_seeds(2027, 1, "exp1-ml-2000"), [356461559])
    check("Exp2/3 replication seed", rep_seeds(2027, 1, "exp23-2000"), [789360421])
    dgp = TransportDGP()
    seed = 356461559
    X, R, A, Y = dgp.sample(2000, np.random.default_rng(seed))
    print("Fitting Exp1 checkpoint...", flush=True)
    m1 = ProposedQTE(profile="exp1", seed=seed).fit(X, R, A, Y)
    r1 = m1.bounds([EXP1_S0], exp1_T_GRID)
    if r1.exceptional:
        raise AssertionError("Unexpected exceptional result in Exp1 checkpoint")
    Ll, _, _, Uu = r1.ci("sim")
    ref = _REPRO_CHECKPOINTS["exp1_proposed"]
    check("Exp1 Proposed mean set width", [np.mean(Uu - Ll)], [ref["mean_set_width"]])
    check("Exp1 Proposed certification", Ll[0] > 0, ref["certified"])
    for law, truth in ref["truth"].items():
        tr = np.asarray(truth)
        check("Exp1 Proposed coverage " + law, [np.all((Ll[0] <= tr) & (tr <= Uu[0]))], [ref["cover"][law]])
    del m1, r1
    seed = 789360421
    X, R, A, Y = dgp.sample(2000, np.random.default_rng(seed))
    print("Fitting shared Exp2/3 checkpoint...", flush=True)
    m3 = ProposedQTE(profile="exp3", seed=seed).fit(X, R, A, Y)
    fitted = m3.nu
    for label, nu in (("proposed", fitted), ("oracle", dgp.oracle_nuisance(X))):
        m2 = ProposedQTE(profile="exp2", seed=seed).fit(X, R, A, Y, nuisance=nu)
        r2 = m2.bounds(exp2_S_EVAL, exp2_TAUS)
        frame = pd.DataFrame(exp2_result_rows(r2, seed, 2000, label))
        ref = _REPRO_CHECKPOINTS["exp2_" + label]
        if frame.target.tolist() != ref["targets"]:
            raise AssertionError("Exp2 endpoint order changed")
        check("Exp2 " + label + " estimates/SEs", frame[["Gamma", "Lambda", "tau", "est", "se"]], ref["values"])
        model = m3 if label == "proposed" else ProposedQTE(profile="exp3", seed=seed).fit(X, R, A, Y, nuisance=nu)
        r3 = model.bounds(exp3_S_GRID, exp3_T_GRID)
        if r2.exceptional or r3.exceptional:
            raise AssertionError("Unexpected exceptional result in Exp2/3 checkpoint")
        ref = _REPRO_CHECKPOINTS["exp3_" + label]
        check("Exp3 " + label + " endpoints/SEs/CIs", r3.table()[ref["columns"]], ref["values"])
        check("Exp3 " + label + " multiplier critical value", [r3.crit["sim"]], [ref["critical"]])
    print("PASS: all fixed-seed checkpoints matched. This was not a 500-repetition run.", flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="Regenerate embedded fixed-seed checkpoints and exit; no saved result files needed")
    parser.add_argument("--experiment", choices=("all", "1", "2", "3", "23"), default="all")
    parser.add_argument("--regime", choices=("both", "ml", "rate"), default="both",
                        help="Exp2 only; Exp1/3 use --learner, Exp3 also reports oracle nuisance")
    parser.add_argument("--learner", choices=("xgboost", "ensemble"), default="xgboost",
                        help="ML nuisance learner for Exp1, Exp2 ML, and Exp3; default: xgboost")
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N))
    parser.add_argument("--reps", type=int, default=None,
                        help="Override repetition count for every selected experiment")
    parser.add_argument("--reps-exp1", type=int, default=500,
                        help="Exp1 repetitions unless --reps is given (default: 500)")
    parser.add_argument("--reps-exp23", type=int, default=500,
                        help="Shared Exp2/3 repetitions unless --reps is given (default: 500)")
    parser.add_argument("--folds", "--K", dest="K", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=None, help="Residual cross-fitting; defaults to --folds")
    parser.add_argument("--bootstrap", "--n_boot", dest="n_boot", type=int, default=500)
    parser.add_argument("--jobs", "--n_jobs", "--workers", dest="jobs", type=int, default=-1,
                        help="-1: all available physical CPU cores; each worker uses one BLAS/OpenMP thread")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--const-p", "--const_p", dest="const_p", type=float, default=1.0)
    parser.add_argument("--const-e", "--const_e", dest="const_e", type=float, default=1.0)
    parser.add_argument("--const-w", "--const_w", dest="const_w", type=float, default=1.0)
    parser.add_argument("--oracle-size", "--n_oracle", dest="n_oracle", type=int, default=65536)
    parser.add_argument("--frontier-oracle", type=int, default=16384)
    parser.add_argument("--sharpness-laws", type=int, default=400, help="Compatible laws per arm; 0 skips this audit")
    parser.add_argument("--sharpness-oracle", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=Path, default=Path("results_proposed_qte"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild tables/Exp1 plot from verified saved runs")
    parser.add_argument("--population-only", action="store_true", help="Compute population tables and the sharpness audit")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify:
        verify_reproduction()
        return
    out = args.out.resolve()
    registration = out / "design.json"
    if args.summarize_only:
        if not registration.exists():
            parser.error("--summarize-only requires an existing design.json")
        saved = json.loads(registration.read_text())
        if saved["config"]["source_sha256"] != file_sha256(__file__):
            parser.error("Saved results were generated by another version of this file")
        for experiment in saved["config"]["experiments"]:
            if not checkpoint_valid(out / f"exp{experiment}" / "population.done.json"):
                parser.error(f"Exp{experiment} population outputs are incomplete")
        collect_results(out, saved["config"], not args.no_plots)
        return
    if args.n != sorted(set(args.n)) or min(args.n) < 50:
        parser.error("--n must contain distinct increasing integers >= 50")
    if (any(v < 2 for v in (args.reps_exp1, args.reps_exp23))
            or (args.reps is not None and args.reps < 2) or args.n_boot < 2 or args.K < 2):
        parser.error("repetition counts, --bootstrap and --folds must be at least 2")
    n_inner = args.inner_folds if args.inner_folds is not None else args.K
    if n_inner < 2 or args.K >= min(args.n) // 10 or n_inner >= min(args.n) // 10:
        parser.error("Use at least two folds and enough observations per fold")
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")
    for name in ("n_oracle", "frontier_oracle", "sharpness_oracle"):
        value = getattr(args, name)
        if value < 16 or value & (value - 1):
            parser.error(f"--{name.replace('_', '-')} must be a power of two >= 16")
    if (args.seed < 0 or args.sharpness_laws < 0 or not np.isfinite(
            [args.rate, args.const_p, args.const_e, args.const_w]).all()
            or args.rate <= 0):
        parser.error("Use a nonnegative seed/law count, finite perturbation constants and a positive rate")
    experiments = [1, 2, 3] if args.experiment == "all" else [2, 3] if args.experiment == "23" else [int(args.experiment)]
    regimes = ["ml", "rate"] if args.regime == "both" else [args.regime]
    reps_exp1 = args.reps if args.reps is not None else args.reps_exp1
    reps_exp23 = args.reps if args.reps is not None else args.reps_exp23
    if args.learner == "xgboost":
        try:
            from importlib.metadata import version
            version("xgboost")
        except ImportError as exc:
            parser.error("--learner xgboost requires `python -m pip install xgboost`")
        except Exception as exc:
            parser.error(f"Cannot load xgboost package metadata: {exc}")
    config = dict(experiments=experiments, regimes=regimes if 2 in experiments else [],
        n=args.n, reps=dict(exp1=reps_exp1, pair=reps_exp23),
        K=args.K, n_inner=n_inner, learner=args.learner, n_boot=args.n_boot,
        rate=args.rate, rate_consts=[args.const_p, args.const_e, args.const_w],
        n_oracle=args.n_oracle, frontier_oracle=args.frontier_oracle,
        sharpness_laws=args.sharpness_laws, sharpness_oracle=args.sharpness_oracle,
        seed=args.seed, dgp=SHARED_DGP, exp1_s0=list(EXP1_S0),
        exp1_est_kw={}, exp2_est_kw=dict(eta=0.1, normalize_omega=True), exp3_est_kw={},
        source_sha256=file_sha256(__file__),
        seeds={"exp1": {str(n): rep_seeds(args.seed, reps_exp1, f"exp1-ml-{n}") for n in args.n},
               "pair": {str(n): rep_seeds(args.seed, reps_exp23, f"exp23-{n}") for n in args.n}})
    if registration.exists():
        if not args.resume:
            parser.error("Output already registered; repeat the command with --resume or use another --out")
        if json.loads(registration.read_text())["config"] != config:
            parser.error("Resume settings differ from the saved code, DGP, seeds or experiment design")
    else:
        if out.exists() and any(out.iterdir()):
            parser.error("Output is nonempty and has no matching design.json; choose another --out")
        if args.resume:
            parser.error("--resume requires an existing design.json")
        out.mkdir(parents=True, exist_ok=True)
        from importlib.metadata import version
        packages = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "matplotlib")
        if args.learner == "xgboost":
            packages += ("xgboost",)
        write_json(registration, dict(config=config, created_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(), packages={name: version(name) for name in packages}))
    workers = max(1, cpu_count(only_physical_cores=True)) if args.jobs == -1 else args.jobs
    print(f"n={args.n}; repetitions=Exp1 {reps_exp1}, Exp2/3 {reps_exp23}; folds={args.K}/{n_inner}; "
          f"multipliers={args.n_boot}; workers={workers}; learner={args.learner}; "
          f"Exp2 regimes={config['regimes']}", flush=True)
    dgp = TransportDGP()
    truth1, _, truth3 = prepare_population(out, config, dgp)
    if args.population_only:
        print(f"Population outputs: {out}", flush=True)
        return
    for n in args.n:
        folder = out / "replications" / f"n{n}"
        folder.mkdir(parents=True, exist_ok=True)
        tasks = (["exp1"] if 1 in experiments else []) + (["pair"] if set(experiments) & {2, 3} else [])
        for task in tasks:
            seeds = config["seeds"][task][str(n)]
            pending = [s for s in seeds if not checkpoint_valid(folder / f"{task}_seed{s}.done.json")]
            print(f"{task}, n={n}: {len(pending)}/{len(seeds)} repetitions remaining", flush=True)
            if pending:
                active_workers = min(workers, len(pending))
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    Parallel(n_jobs=active_workers, verbose=5, batch_size=1, pre_dispatch=active_workers)(
                        delayed(run_replication)(task, n, seed, folder, config, dgp, truth1, truth3)
                        for seed in pending)
    collect_results(out, config, not args.no_plots)
    write_json(out / "completed.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=config["source_sha256"], experiments=experiments,
        regimes=config["regimes"], n=config["n"], reps=config["reps"]))
    print(f"\nCompleted: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

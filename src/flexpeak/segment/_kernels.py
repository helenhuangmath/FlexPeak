"""Hot loops for HMM inference.

The forward and backward recursions are inherently sequential over bins, so
they cannot be vectorised the way the rest of the pipeline is.  At 25 bp bins a
human chromosome is millions of bins, and a pure-Python loop over them costs
seconds *per EM iteration* -- which would put the runtime tuning sweep
(DESIGN.md section 8) out of reach entirely.

These kernels are therefore JIT-compiled with numba when it is available, and
fall back to an equivalent NumPy implementation when it is not.  numba stays an
optional dependency: FlexPeak must install and run with ``pip install flexpeak``
and nothing else.  ``HAVE_NUMBA`` is exported so tests can exercise both paths
and assert they agree.
"""

from __future__ import annotations

import numpy as np

__all__ = ["forward_backward", "viterbi", "HAVE_NUMBA"]

_TINY = 1e-300

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in so the decorated functions stay importable."""
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


# -- compiled kernels -------------------------------------------------------
#
# All of these take emissions as ``e`` with shape (n_states, n_bins), already
# rescaled per position by the caller so the linear recursion cannot underflow.


@njit(cache=True, fastmath=False)
def _forward(e, A, pi):
    """Scaled forward recursion.  Returns (alpha, scale)."""
    k, n = e.shape
    alpha = np.zeros((k, n))
    scale = np.zeros(n)

    s = 0.0
    for i in range(k):
        alpha[i, 0] = pi[i] * e[i, 0]
        s += alpha[i, 0]
    if s <= 0.0:
        s = _TINY
    scale[0] = s
    for i in range(k):
        alpha[i, 0] /= s

    for t in range(1, n):
        s = 0.0
        for j in range(k):
            acc = 0.0
            for i in range(k):
                acc += alpha[i, t - 1] * A[i, j]
            alpha[j, t] = acc * e[j, t]
            s += alpha[j, t]
        if s <= 0.0:
            s = _TINY
        scale[t] = s
        for j in range(k):
            alpha[j, t] /= s
    return alpha, scale


@njit(cache=True, fastmath=False)
def _backward(e, A, scale):
    """Scaled backward recursion, using the forward pass's scale factors."""
    k, n = e.shape
    beta = np.zeros((k, n))
    for i in range(k):
        beta[i, n - 1] = 1.0

    for t in range(n - 2, -1, -1):
        s = scale[t + 1]
        if s <= 0.0:
            s = _TINY
        for i in range(k):
            acc = 0.0
            for j in range(k):
                acc += A[i, j] * e[j, t + 1] * beta[j, t + 1]
            beta[i, t] = acc / s
    return beta


@njit(cache=True, fastmath=False)
def _xi_accumulate(alpha, beta, e, A):
    """Sum of the pairwise state posteriors, the Baum-Welch transition counts."""
    k, n = e.shape
    xi = np.zeros((k, k))
    for t in range(n - 1):
        tot = 0.0
        for i in range(k):
            for j in range(k):
                tot += alpha[i, t] * A[i, j] * e[j, t + 1] * beta[j, t + 1]
        if tot <= 0.0:
            tot = _TINY
        for i in range(k):
            for j in range(k):
                xi[i, j] += (alpha[i, t] * A[i, j] * e[j, t + 1]
                             * beta[j, t + 1]) / tot
    return xi


@njit(cache=True, fastmath=False)
def _viterbi(log_e, log_A, log_pi):
    """Viterbi in log space.  Returns the most likely state path."""
    k, n = log_e.shape
    delta = np.zeros(k)
    psi = np.zeros((k, n), dtype=np.int16)

    for i in range(k):
        delta[i] = log_pi[i] + log_e[i, 0]

    new = np.zeros(k)
    for t in range(1, n):
        for j in range(k):
            best = -1e308
            arg = 0
            for i in range(k):
                v = delta[i] + log_A[i, j]
                if v > best:
                    best = v
                    arg = i
            new[j] = best + log_e[j, t]
            psi[j, t] = arg
        for j in range(k):
            delta[j] = new[j]

    path = np.zeros(n, dtype=np.int16)
    best = -1e308
    arg = 0
    for i in range(k):
        if delta[i] > best:
            best = delta[i]
            arg = i
    path[n - 1] = arg
    for t in range(n - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return path


# -- NumPy fallbacks (used when numba is absent) ----------------------------


def _forward_np(e, A, pi):
    k, n = e.shape
    alpha = np.zeros((k, n))
    scale = np.zeros(n)
    alpha[:, 0] = pi * e[:, 0]
    s = alpha[:, 0].sum()
    scale[0] = s if s > 0 else _TINY
    alpha[:, 0] /= scale[0]
    for t in range(1, n):
        alpha[:, t] = (alpha[:, t - 1] @ A) * e[:, t]
        s = alpha[:, t].sum()
        scale[t] = s if s > 0 else _TINY
        alpha[:, t] /= scale[t]
    return alpha, scale


def _backward_np(e, A, scale):
    k, n = e.shape
    beta = np.zeros((k, n))
    beta[:, n - 1] = 1.0
    for t in range(n - 2, -1, -1):
        s = scale[t + 1] if scale[t + 1] > 0 else _TINY
        beta[:, t] = (A @ (e[:, t + 1] * beta[:, t + 1])) / s
    return beta


def _xi_accumulate_np(alpha, beta, e, A):
    k, n = e.shape
    xi = np.zeros((k, k))
    for t in range(n - 1):
        m = (alpha[:, t][:, None] * A) * (e[:, t + 1] * beta[:, t + 1])[None, :]
        tot = m.sum()
        xi += m / (tot if tot > 0 else _TINY)
    return xi


def _viterbi_np(log_e, log_A, log_pi):
    k, n = log_e.shape
    delta = log_pi + log_e[:, 0]
    psi = np.zeros((k, n), dtype=np.int16)
    for t in range(1, n):
        scores = delta[:, None] + log_A          # (from, to)
        psi[:, t] = np.argmax(scores, axis=0)
        delta = scores.max(axis=0) + log_e[:, t]
    path = np.zeros(n, dtype=np.int16)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return path


def forward_backward(log_e: np.ndarray, A: np.ndarray, pi: np.ndarray, use_numba=None):
    """Scaled forward-backward.  Returns ``(gamma, xi_sum, loglik)``."""
    if use_numba is None:
        use_numba = HAVE_NUMBA
    # Rescale emissions per position so the linear recursion cannot underflow;
    # the removed factor is added back into the log-likelihood.
    base = log_e.max(axis=0)
    e = np.ascontiguousarray(np.exp(log_e - base), dtype=np.float64)
    A = np.ascontiguousarray(A, dtype=np.float64)
    pi = np.ascontiguousarray(pi, dtype=np.float64)

    if use_numba and HAVE_NUMBA:
        alpha, scale = _forward(e, A, pi)
        beta = _backward(e, A, scale)
        xi_sum = _xi_accumulate(alpha, beta, e, A)
    else:
        alpha, scale = _forward_np(e, A, pi)
        beta = _backward_np(e, A, scale)
        xi_sum = _xi_accumulate_np(alpha, beta, e, A)

    gamma = alpha * beta
    gamma /= gamma.sum(axis=0, keepdims=True) + _TINY
    loglik = float(np.log(scale + _TINY).sum() + base.sum())
    return gamma, xi_sum, loglik


def viterbi(log_e: np.ndarray, A: np.ndarray, pi: np.ndarray, use_numba=None) -> np.ndarray:
    if use_numba is None:
        use_numba = HAVE_NUMBA
    log_A = np.log(np.maximum(A, _TINY))
    log_pi = np.log(np.maximum(pi, _TINY))
    log_e = np.ascontiguousarray(log_e, dtype=np.float64)
    if use_numba and HAVE_NUMBA:
        return _viterbi(log_e, np.ascontiguousarray(log_A), np.ascontiguousarray(log_pi))
    return _viterbi_np(log_e, log_A, log_pi)

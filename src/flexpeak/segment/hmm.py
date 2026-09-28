"""Negative binomial HMM for domain segmentation.

DESIGN.md section 3.3, layer 1.  Three states -- background, weak-enriched,
strong-enriched -- with non-homogeneous emissions: state ``k`` at bin ``i`` has
mean ``mu_i * m_k``, where ``mu_i`` is the fitted local background.  That keeps
the mappability/GC/control structure of the background model inside the
segmentation instead of throwing it away.

Boundaries are *inferred* from the posterior rather than set by a ``--gap-size``
guess.  Over-merging is what a fixed gap parameter produces when the gap is
larger than the true inter-domain spacing, and it is the specific failure of
SICER/epic2 that this design is meant to fix.  The corresponding metric is the
fusion index (DESIGN.md section 9.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from ._kernels import forward_backward, viterbi

__all__ = ["NBHMM", "HMMResult", "segment_chromosome"]

_TINY = 1e-300


@dataclass
class HMMResult:
    """Everything the segmentation learned, kept for QC and diagnostics."""

    posterior: np.ndarray
    """Per-bin probability of being in any enriched state."""
    state: np.ndarray
    multipliers: np.ndarray
    transitions: np.ndarray
    n_iter: int
    converged: bool
    dispersion_scale: Optional[np.ndarray] = None
    """Per-state multiplier on the background dispersion trend, fitted here."""


class NBHMM:
    """Three-state negative binomial HMM with per-bin background means."""

    def __init__(
        self,
        n_states: int = 3,
        multipliers: Optional[Tuple[float, ...]] = None,
        stay_background: float = 0.995,
        stay_enriched: float = 0.98,
        max_iter: int = 12,
        tol: float = 1e-4,
        min_multiplier_gap: float = 1.15,
        background_slack: float = 1.1,
        min_stay_enriched: float = 0.0,
        fit_state_dispersion: bool = False,
        max_dispersion_scale: float = 50.0,
    ):
        self.n_states = n_states
        self.multipliers = np.array(
            multipliers if multipliers is not None else (1.0, 2.5, 6.0)[:n_states],
            dtype=np.float64,
        )
        self.stay_background = stay_background
        self.stay_enriched = stay_enriched
        self.max_iter = max_iter
        self.tol = tol
        self.min_multiplier_gap = min_multiplier_gap
        self.background_slack = background_slack
        self.fit_state_dispersion = bool(fit_state_dispersion)
        self.max_dispersion_scale = float(max_dispersion_scale)
        """Re-fit each state's dispersion, as a multiple of the background trend.

        The background model fits its mean-dispersion trend on background bins,
        by design -- letting peaks into that fit would make the background look
        noisy and cost sensitivity.  But the segmentation then applies that same
        dispersion to the *enriched* states, where it is far too small: measured
        inside called H3K9me3 regions the observed variance was 50,098 against a
        model variance of 12,919, a factor of ~4.

        An emission model that says a bin cannot deviate that much, faced with a
        bin that does, explains it the only way it can -- by switching state.
        That is the flicker that shreds a domain into fragments.  So each state
        can get its own dispersion, fitted by posterior-weighted method of
        moments as a scale on the background trend, floored at 1 and capped to
        stay finite.

        **Off by default, because measuring it showed it does not work.**  A
        wider emission distribution is less discriminating everywhere, not just
        inside domains: on H3K9me3 it cut called territory from 4.07% to 0.85%
        of the genome while barely changing the median width, and on H3K27ac the
        moment estimate ran to the cap for every state.  Fragmentation is held
        together by the transition prior, not by widening the emissions -- see
        ``CallParams.segment_bin_size``.  Kept as an ablation knob.
        """
        self.min_stay_enriched = float(min_stay_enriched)
        """Floor on the re-estimated enriched self-transition.

        Baum-Welch re-estimates the transition matrix, and on real coverage it
        reliably collapses the enriched dwell time: measured on the CUT&RUN
        samples in ``testing/`` it learned self-transitions of 0.60-0.86, i.e. a
        mean enriched run of 63-720 bp, on marks whose own autocorrelation
        length is 0.9-7.4 kb.  The reason is that inside a real domain
        individual bins fluctuate below the enriched mean, and a free transition
        matrix buys likelihood by flickering between states rather than by
        staying put.  The result is a posterior that switches at bin scale, so
        broad domains come out shredded into fragments and peaks shorter than
        ``min_width`` are dropped entirely.

        Flooring the diagonal keeps the adaptivity that matters (the model may
        still learn *longer* dwells than the floor) while refusing to spend
        probability mass on runs shorter than the features being called.
        """
        self.transitions = self._init_transitions()

    def _init_transitions(self) -> np.ndarray:
        k = self.n_states
        A = np.zeros((k, k))
        for i in range(k):
            stay = self.stay_background if i == 0 else self.stay_enriched
            A[i, :] = (1.0 - stay) / (k - 1)
            A[i, i] = stay
        return self._apply_dwell_floor(A)

    def _apply_dwell_floor(self, A: np.ndarray) -> np.ndarray:
        """Raise enriched self-transitions to ``min_stay_enriched``, renormalised."""
        floor = self.min_stay_enriched
        if not 0.0 < floor < 1.0:
            return A
        A = A.copy()
        for i in range(1, A.shape[0]):
            if A[i, i] >= floor:
                continue
            off = A[i].sum() - A[i, i]
            if off > 0:
                A[i] *= (1.0 - floor) / off
            A[i, i] = floor
        return A

    def _log_emissions(self, counts: np.ndarray, mu: np.ndarray, alpha: np.ndarray,
                       multipliers: np.ndarray,
                       disp_scale: Optional[np.ndarray] = None) -> np.ndarray:
        """(n_states, n_bins) log NB pmf."""
        k = np.maximum(np.round(counts), 0)
        out = np.empty((multipliers.size, counts.size), dtype=np.float64)
        for s, m in enumerate(multipliers):
            mu_s = np.maximum(mu * m, 1e-9)
            scale = 1.0 if disp_scale is None else float(disp_scale[s])
            a = np.maximum(alpha * scale, 1e-10)
            n = 1.0 / a
            p = np.clip(n / (n + mu_s), 1e-12, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[s] = stats.nbinom.logpmf(k, n, p)
        return np.nan_to_num(out, nan=-745.0, neginf=-745.0)

    @staticmethod
    def _forward_backward(log_e: np.ndarray, A: np.ndarray, pi: np.ndarray):
        return forward_backward(log_e, A, pi)

    @staticmethod
    def _viterbi(log_e: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
        return viterbi(log_e, A, pi)

    def _fit_dispersion_scale(self, counts, mu, alpha, gamma,
                              multipliers) -> np.ndarray:
        """Posterior-weighted moment estimate of each state's dispersion scale.

        For state k the NB mean is ``mu_i * m_k`` and its variance is
        ``mu_i m_k + s_k * alpha_i * (mu_i m_k)^2``.  Matching the weighted
        second moment gives s_k in closed form.
        """
        out = np.ones(multipliers.size)
        for s, m in enumerate(multipliers):
            mu_s = np.maximum(mu * m, 1e-9)
            g = gamma[s]
            w = float(g.sum())
            if w <= 1.0:
                continue
            resid2 = float(g @ ((counts - mu_s) ** 2 - mu_s))
            denom = float(g @ (alpha * mu_s * mu_s))
            if denom <= 0:
                continue
            out[s] = np.clip(resid2 / denom, 1.0, self.max_dispersion_scale)
        return out


    def fit_predict(self, counts: np.ndarray, mu: np.ndarray,
                    alpha: np.ndarray) -> HMMResult:
        """Baum-Welch on multipliers and transitions, then posterior + Viterbi."""
        counts = np.asarray(counts, dtype=np.float64).ravel()
        mu = np.asarray(mu, dtype=np.float64).ravel()
        alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), counts.shape)
        n = counts.size
        if n < 10:
            return HMMResult(
                posterior=np.zeros(n),
                state=np.zeros(n, dtype=np.int16),
                multipliers=self.multipliers.copy(),
                transitions=self.transitions.copy(),
                n_iter=0,
                converged=True,
                dispersion_scale=np.ones(self.n_states),
            )

        A = self.transitions.copy()
        mult = self.multipliers.copy()
        pi = np.full(self.n_states, 1.0 / self.n_states)
        prev_ll = -np.inf
        converged = False
        it = 0

        disp = np.ones(self.n_states)
        for it in range(1, self.max_iter + 1):
            log_e = self._log_emissions(counts, mu, alpha, mult, disp)
            gamma, xi_sum, ll = self._forward_backward(log_e, A, pi)

            # M-step.  Closed-form weighted method-of-moments for the state
            # multipliers: m_k = sum_i gamma_ki * x_i / sum_i gamma_ki * mu_i
            denom = gamma @ mu
            numer = gamma @ counts
            with np.errstate(divide="ignore", invalid="ignore"):
                new_mult = np.where(denom > 0, numer / np.maximum(denom, 1e-9), mult)
            new_mult = np.clip(new_mult, 1e-3, 1e4)
            new_mult.sort()

            # Keep state 0 pinned near the fitted background, and keep the
            # enriched states separated -- otherwise the states collapse onto
            # each other and the posterior stops meaning anything.
            new_mult[0] = float(np.clip(new_mult[0],
                                        1.0 / self.background_slack,
                                        self.background_slack))
            floor = self.min_multiplier_gap
            for s in range(1, new_mult.size):
                new_mult[s] = max(new_mult[s], floor,
                                  new_mult[s - 1] * self.min_multiplier_gap)

            A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), _TINY)
            A = np.clip(A, 1e-6, 1.0)
            A /= A.sum(axis=1, keepdims=True)
            # Applied inside the loop, not just to the final decode, so the
            # multipliers are fitted against the dwell structure they will
            # actually be decoded with.
            A = self._apply_dwell_floor(A)
            pi = np.maximum(gamma[:, 0], 1e-6)
            pi /= pi.sum()
            mult = new_mult
            if self.fit_state_dispersion:
                disp = self._fit_dispersion_scale(counts, mu, alpha, gamma, mult)

            if abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                converged = True
                break
            prev_ll = ll

        log_e = self._log_emissions(counts, mu, alpha, mult, disp)
        gamma, _, _ = self._forward_backward(log_e, A, pi)
        path = self._viterbi(log_e, A, pi)
        posterior = 1.0 - gamma[0]
        return HMMResult(
            posterior=posterior,
            state=path,
            multipliers=mult,
            transitions=A,
            n_iter=it,
            converged=converged,
            dispersion_scale=disp,
        )


def segment_chromosome(
    counts: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    posterior_cutoff: float = 0.5,
    min_bins: int = 1,
    hmm: Optional[NBHMM] = None,
) -> Tuple[List[Tuple[int, int]], HMMResult]:
    """Segment one chromosome into enriched bin intervals.

    Returns ``([(start_bin, end_bin), ...], result)`` with half-open intervals.
    """
    hmm = hmm or NBHMM()
    res = hmm.fit_predict(counts, mu, alpha)
    enriched = res.posterior >= posterior_cutoff
    return _runs(enriched, min_bins), res


def _runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Contiguous True runs as half-open [start, end) index pairs."""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]

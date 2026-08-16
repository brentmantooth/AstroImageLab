"""Spatial block bootstrap for autocorrelated per-pixel maps.

Why this exists
---------------
Section 8's A-vs-B comparisons run a Mann-Whitney U test over 10^4-10^6 pixel
values and report the resulting p-value. Those pixels are *not* independent
samples -- every map in the pipeline is the output of a smoothing/windowing
operator, and the A-B difference field carries large-scale structure on top of
that. Treating them as independent understates the standard error badly.

Measured on the project's own reports (AstroLabTestData/FilterCompare):

    metric                naive SE   block SE (64 px)   understated by
    Local sigma 3 px      3.4e-04         1.3e-02             38x
    Local sigma 10 px     2.5e-04         1.2e-02             46x
    |LoG| sigma=1.5       6.2e-04         1.1e-02             18x
    Local grad. energy    5.4e-05         1.6e-03             29x

so every p-value in those tables is wrong by orders of magnitude, and all of
them saturate at p < 0.001 regardless of effect size.

What this module does and does not fix
--------------------------------------
It produces an honest confidence interval. It does **not** produce practical
significance: on the same real data, a correctly-computed CI still puts every
comparison -- including a filter pair the owner expects to be indistinguishable
-- tens of sigma from zero, because a tiny real difference measured over a
whole frame is still real. Deciding whether a difference *matters* needs a
magnitude threshold, which is core.practical_significance's job. Use both.

Non-convergence is a result, not a failure
------------------------------------------
A block bootstrap is only valid once the block size exceeds the field's
correlation length; past that point the SE plateaus. On several of the real
maps here it does not plateau even at 256 px blocks -- the difference field is
correlated across the whole frame, so the effective sample size is on the order
of tens. `BootstrapResult.converged` reports this rather than hiding it, and a
caller should say so instead of quoting a CI that is still growing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import bottleneck as bn
except ImportError:
    bn = np   # transparent fallback; must come after `import numpy as np`

# Lags probed when estimating the correlation length. 256 px is well past the
# scale of every kernel in the pipeline (the widest is Local sigma 10 px /
# |LoG| sigma=6), so a field still correlated at this lag is carrying genuine
# large-scale structure rather than operator support.
MAX_PROBE_LAG_PX = 256
# Autocorrelation is estimated from at most this many pixels (strided
# subsample). The estimate is an average over ~10^5 products either way; paying
# for 10^7 buys precision that no downstream decision is sensitive to.
ACORR_MAX_SAMPLES = 400_000
# A block bootstrap needs enough blocks for the resampling distribution to mean
# anything. Below this, auto_block_size stops growing the block even if the
# correlation length says it should.
MIN_BLOCKS = 30
# A tile is used only if enough of its pixels survived the mask -- mirrors
# SOURCEMASK_MIN_CELL_UNMASKED_FRAC's reasoning (a barely-populated cell is
# noise, not a measurement). The fraction is taken relative to the mask's own
# overall density, not to the tile area: Section 8l's local-maxima mask selects
# only ~5-12% of the frame, so a flat 25%-of-area rule would reject essentially
# every tile and the bootstrap would return None for exactly the population
# that best tracks perceived sharpness.
# ...but in practice both bounds are 0/1, i.e. every tile holding at least one
# valid pixel is kept. Tiles are *count-weighted*, so a tile with 3 masked
# pixels contributes weight 3 and cannot distort the mean -- the sparse-cell
# concern that motivates a floor applies to unweighted statistics, not to this
# one. Dropping such tiles instead breaks the identity that makes the whole
# thing usable: measured on a real Section 8l mask (7.6% coverage), a 16-pixel
# floor discarded 568 of 900 tiles, and because the discarded tiles were the
# mask-sparse ones rather than a random subset it biased the estimate 2x --
# producing a confidence interval that did not contain its own point estimate.
MIN_TILE_VALID_FRAC = 0.0
MIN_TILE_VALID_PX = 1
# SE growth factor between block_px and 2*block_px above which the bootstrap is
# reported as not converged. 1.0 would be exact convergence; 1.5 allows for the
# genuine noise in an SE estimated from a few hundred blocks.
CONVERGENCE_SE_RATIO = 1.5


@dataclass
class BootstrapResult:
    """Outcome of a spatial block bootstrap.

    point           : the statistic on the full data (not a bootstrap average)
    lo, hi          : percentile confidence bounds
    se              : bootstrap standard error
    block_px        : block edge length used
    n_blocks        : blocks that passed MIN_TILE_VALID_FRAC
    converged       : SE has plateaued in block size (see module docstring)
    se_double_block : SE at 2*block_px, the evidence behind `converged`
    decorrelation_px: estimated correlation length of the field
    naive_se        : the independent-pixels SE, for comparison only
    """
    point: float
    lo: float
    hi: float
    se: float
    block_px: int
    n_blocks: int
    converged: bool
    se_double_block: float | None
    decorrelation_px: float
    naive_se: float

    @property
    def se_understatement(self) -> float | None:
        """How many times too narrow the naive per-pixel SE is. The headline
        number for explaining why the existing p-values cannot be trusted."""
        if self.naive_se <= 0:
            return None
        return self.se / self.naive_se

    @property
    def excludes_zero(self) -> bool:
        return self.lo * self.hi > 0


def _masked_values(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Finite, in-mask values as a flat float64 array."""
    arr = np.asarray(field)
    if mask is not None:
        h = min(arr.shape[0], mask.shape[0])
        w = min(arr.shape[1], mask.shape[1])
        arr = arr[:h, :w][mask[:h, :w]]
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def decorrelation_length(field: np.ndarray,
                         max_lag: int = MAX_PROBE_LAG_PX) -> float:
    """Integral autocorrelation length of a 2D field, in pixels.

    Estimated separably along rows and columns and reduced with `max`, which is
    the conservative choice: a field elongated along one axis (a gradient
    difference, a registration residual) decorrelates slowly on that axis only,
    and using the slow axis is what keeps the block size honest.

    Uses the standard truncated-sum estimator `tau = 1 + 2 * sum(rho_k)`,
    stopped at the first non-positive rho -- summing past that point adds pure
    noise and can make tau arbitrarily large. Returns at least 1.0.

    NaNs are treated as missing (`nanmean` over the lag products); a field that
    is entirely NaN or constant returns 1.0.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return 1.0

    # Strided subsample to bound cost. Striding preserves the shape of the
    # autocorrelation at lags >= step, and we rescale lag units accordingly.
    step = max(1, int(np.ceil(np.sqrt(arr.size / ACORR_MAX_SAMPLES))))
    sub = arr[::step, ::step]

    centred = sub - bn.nanmean(sub)
    var = bn.nanmean(centred * centred)
    if not np.isfinite(var) or var <= 0:
        return 1.0

    lag_limit = max(1, int(max_lag // step))

    def _tau(axis: int) -> float:
        n = centred.shape[axis]
        total = 0.0
        for lag in range(1, min(lag_limit, n - 1) + 1):
            if axis == 1:
                prod = centred[:, :-lag] * centred[:, lag:]
            else:
                prod = centred[:-lag, :] * centred[lag:, :]
            rho = float(bn.nanmean(prod)) / var
            if not np.isfinite(rho) or rho <= 0.0:
                break
            total += rho
        return 1.0 + 2.0 * total

    # Back into native pixels: a lag of 1 in the strided array is `step` pixels.
    return float(max(_tau(0), _tau(1)) * step)


def _max_block_for(h: int, w: int, min_blocks: int) -> int:
    """Largest block edge still yielding `min_blocks` tiles on an h x w frame.

    The tile count is floor(h/b) * floor(w/b), so the closed-form
    sqrt(h*w/min_blocks) is not a valid cap -- on 512x512 at min_blocks=30 it
    gives b=93, and 512//93 = 5 is only 25 tiles. Walk down until the floored
    count actually satisfies the bound.
    """
    min_blocks = max(int(min_blocks), 1)
    cap = max(1, int(np.floor(np.sqrt((h * w) / min_blocks))))
    while cap > 1 and (h // cap) * (w // cap) < min_blocks:
        cap -= 1
    return cap


def se_ladder(field: np.ndarray,
              mask: np.ndarray | None = None,
              min_blocks: int = MIN_BLOCKS,
              n_boot: int = 400,
              seed: int = 0,
              fn=None) -> dict[int, float]:
    """Bootstrap SE at a geometric ladder of block sizes: {block_px: se}.

    This is the empirical replacement for "block > 2 x correlation length". On
    this project's real difference maps that textbook rule badly under-selects:
    Section 8's Local sigma 3 px map has an integral autocorrelation length of
    ~7 px, but its SE keeps growing all the way to 224 px blocks, with the
    growth per doubling *accelerating* (1.13, 1.14, 1.26, 1.53, 1.78, 1.86).
    The autocorrelation is the reason -- rho falls to 0.01 by lag 4 and then
    sits at ~0.008 out to lag 256 without ever reaching zero. A long, weak tail
    contributes almost nothing to the correlation length yet dominates the
    variance of a whole-frame mean.

    So the block size has to be chosen from the SE curve itself, not from a
    summary of the autocorrelation.
    """
    h, w = np.asarray(field).shape[:2]
    cap = _max_block_for(h, w, min_blocks)
    # Strictly geometric, so every consecutive-rung ratio is over the same
    # factor of 2 and the ratios are comparable to each other. Appending a
    # ragged final `cap` rung would give one ratio a smaller factor and bias it
    # toward looking converged.
    rungs: list[int] = []
    b = 4
    while b <= cap:
        rungs.append(b)
        b *= 2

    out: dict[int, float] = {}
    rng = np.random.default_rng(seed)
    for block in rungs:
        values, weights = tile_reduce(field, mask, block, fn=fn)
        if values.size < 2:
            continue
        idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
        draws = ((values * weights)[idx].sum(axis=1) / weights[idx].sum(axis=1))
        out[block] = float(np.std(draws))
    return out


def auto_block_size(field: np.ndarray,
                    mask: np.ndarray | None = None,
                    min_blocks: int = MIN_BLOCKS,
                    seed: int = 0,
                    fn=None) -> tuple[int, bool, dict[int, float]]:
    """Choose a block edge length from the SE curve.

    Returns (block_px, converged, ladder). The chosen block is the smallest rung
    at which the SE has stopped growing -- `se(2b)/se(b) <= CONVERGENCE_SE_RATIO`
    -- because past that point the blocks are behaving independently and a
    larger block only costs precision.

    When the SE never stops growing (long-range dependence, which is the norm on
    the real maps here) there is no valid block size at all: the largest rung is
    returned with converged=False, and the resulting CI should be read as a
    lower bound on the true uncertainty, not as the uncertainty.
    """
    ladder = se_ladder(field, mask, min_blocks=min_blocks, seed=seed, fn=fn)
    if not ladder:
        h, w = np.asarray(field).shape[:2]
        return max(1, min(h, w)), False, {}

    rungs = sorted(ladder)
    if len(rungs) < 2:
        return rungs[0], False, ladder

    # A rung is only trustworthy if the SE has stopped growing there *and stays*
    # stopped for every larger rung. Taking the first rung that passes a single
    # local check is wrong on an accelerating curve: the project's real Local
    # sigma 3 px map grows 1.13, 1.14, 1.26, 1.53, 1.78, 1.86 per doubling, so
    # the smallest rung passes a 1.5 threshold while the field is in fact
    # nowhere near converged. Scan for the *last* violation instead.
    last_bad = -1
    for i in range(len(rungs) - 1):
        se_here, se_next = ladder[rungs[i]], ladder[rungs[i + 1]]
        if se_here > 0 and se_next / se_here > CONVERGENCE_SE_RATIO:
            last_bad = i

    if last_bad < 0:
        return rungs[0], True, ladder            # flat all the way down
    if last_bad + 1 <= len(rungs) - 2:
        return rungs[last_bad + 1], True, ladder  # plateau starts here
    # The growth was still going at the largest rung we can measure, so there is
    # no evidence any block size is large enough. Report the largest and say so.
    return rungs[-1], False, ladder


def tile_reduce(field: np.ndarray,
                mask: np.ndarray | None,
                block_px: int,
                fn=None) -> tuple[np.ndarray, np.ndarray]:
    """Reduce `field` to one value per block_px x block_px tile.

    Returns (values, weights) where `weights` is the count of valid pixels each
    tile's value was computed from. Tiles below MIN_TILE_VALID_FRAC are dropped
    from both arrays.

    Weights matter: the weighted mean of the tile values reproduces the plain
    masked mean of the whole field exactly, so the bootstrap's point estimate
    is the same number the report's table shows rather than a near-miss.

    `fn` defaults to the NaN-aware mean. Pass e.g. `np.nanmedian` for a robust
    tile statistic; any callable taking (arr2d_stack, axis=...) works.
    """
    arr = np.asarray(field, dtype=np.float64)
    if mask is not None:
        h = min(arr.shape[0], mask.shape[0])
        w = min(arr.shape[1], mask.shape[1])
        arr = np.where(mask[:h, :w], arr[:h, :w], np.nan)
    h, w = arr.shape
    block_px = max(1, min(int(block_px), h, w))

    nh, nw = h // block_px, w // block_px
    if nh < 1 or nw < 1:
        return np.empty(0), np.empty(0)

    # Crop to a whole number of tiles, then fold the two block axes together.
    blocks = (arr[:nh * block_px, :nw * block_px]
              .reshape(nh, block_px, nw, block_px)
              .transpose(0, 2, 1, 3)
              .reshape(nh * nw, block_px * block_px))

    valid = np.isfinite(blocks)
    counts = valid.sum(axis=1)
    area = block_px * block_px
    # Scale the requirement by how dense the mask is overall, so a sparse
    # selection is judged against what a *typical* tile of that selection
    # holds rather than against the full tile area.
    density = counts.sum() / float(blocks.size) if blocks.size else 0.0
    threshold = max(MIN_TILE_VALID_PX, MIN_TILE_VALID_FRAC * area * density)
    keep = counts >= threshold
    if not keep.any():
        return np.empty(0), np.empty(0)

    blocks, counts = blocks[keep], counts[keep]
    reducer = fn if fn is not None else bn.nanmean
    with np.errstate(invalid="ignore"):
        values = np.asarray(reducer(blocks, axis=1), dtype=np.float64)

    good = np.isfinite(values)
    return values[good], counts[good].astype(np.float64)


def block_bootstrap_ci(field: np.ndarray,
                       mask: np.ndarray | None = None,
                       block_px: int | None = None,
                       n_boot: int = 2000,
                       ci: float = 95.0,
                       seed: int = 0,
                       fn=None,
                       check_convergence: bool = True) -> BootstrapResult | None:
    """Spatial block bootstrap of the masked mean of `field`.

    `field` is normally a per-pixel A-vs-B difference map -- Section 8 already
    computes exactly this as `panels[key]["diff"]`, the log10(|A|/|B|) map -- so
    no metric recomputation is needed to get a CI for it.

    Blocks are resampled with replacement and their weighted mean recomputed;
    the CI is the percentile interval of that resampling distribution. Returns
    None if fewer than 2 tiles survive.

    `block_px=None` picks one via auto_block_size.
    """
    arr = np.asarray(field)
    if arr.ndim != 2 or arr.size == 0:
        return None

    ladder: dict[int, float] = {}
    auto_converged = None
    if block_px is None:
        block_px, auto_converged, ladder = auto_block_size(
            arr, mask, seed=seed, fn=fn)
    else:
        block_px = max(1, int(block_px))

    values, weights = tile_reduce(arr, mask, block_px, fn=fn)
    if values.size < 2:
        return None
    decorr = decorrelation_length(arr)

    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    # Weighted mean per resample: sum(w*v)/sum(w) over the drawn tiles.
    wv = (values * weights)[idx]
    ww = weights[idx]
    draws = wv.sum(axis=1) / ww.sum(axis=1)

    lo_pct = (100.0 - ci) / 2.0
    lo, hi = np.percentile(draws, [lo_pct, 100.0 - lo_pct])

    flat = _masked_values(arr, mask)
    point = float(np.sum(values * weights) / np.sum(weights))
    naive_se = (float(np.std(flat) / np.sqrt(flat.size))
                if flat.size > 1 else 0.0)

    se_double = None
    converged = True
    if check_convergence:
        if auto_converged is not None:
            # auto_block_size already walked the whole ladder, which is the only
            # way to see the accelerating growth a single doubling misses.
            converged = auto_converged
            se_double = ladder.get(block_px * 2)
        else:
            big = min(block_px * 2, min(arr.shape))
            if big > block_px:
                v2, w2 = tile_reduce(arr, mask, big, fn=fn)
                if v2.size >= 2:
                    i2 = rng.integers(0, v2.size, size=(int(n_boot), v2.size))
                    d2 = ((v2 * w2)[i2].sum(axis=1) / w2[i2].sum(axis=1))
                    se_double = float(np.std(d2))
                    se_now = float(np.std(draws))
                    converged = (se_now <= 0 or
                                 se_double / se_now <= CONVERGENCE_SE_RATIO)

    return BootstrapResult(
        point=point,
        lo=float(lo),
        hi=float(hi),
        se=float(np.std(draws)),
        block_px=int(block_px),
        n_blocks=int(n),
        converged=bool(converged),
        se_double_block=se_double,
        decorrelation_px=float(decorr),
        naive_se=naive_se,
    )


def effective_sample_size(field: np.ndarray,
                          mask: np.ndarray | None = None,
                          block_px: int | None = None,
                          seed: int = 0) -> float | None:
    """How many *independent* pixels this field is worth.

    `(sd / se_block)**2` -- the sample size an iid population would need to
    reach the block bootstrap's standard error. On the real Section 8 maps this
    lands in the tens-to-hundreds against a nominal N of 10^5-10^6, which is the
    single clearest way to state why the per-pixel p-values are meaningless.
    """
    res = block_bootstrap_ci(field, mask, block_px=block_px, n_boot=500,
                             seed=seed, check_convergence=False)
    if res is None or res.se <= 0:
        return None
    flat = _masked_values(field, mask)
    if flat.size < 2:
        return None
    return float((np.std(flat) / res.se) ** 2)

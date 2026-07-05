"""Hull-White reference pricer for a high-yield bond with make-whole + step-down call.

This is the Python reference used to cross-check the browser tool at
levfinacademy.com/tools/make-whole.html. It builds the standard Hull-White
1994 trinomial short-rate tree (fitted to the initial discount curve by
construction) and prices the borrower's embedded call under four modes:

- ``bullet``: no call at all (upper bound reference)
- ``full``: make-whole inside NC, step-down p50 -> p25 -> par after NC
- ``noMW``: no call inside NC, step-down p50 -> p25 -> par after NC
- ``mwOnly``: make-whole inside NC, no call after NC

Credit is modeled as a constant spread over the stochastic HW rate at every
node (no default/recovery). MW discount inside NC uses the *live* HW curve
seen at each node (analytic HW ZCB) plus a constant Treasury+50 add-on.

The step-down schedule is the tool's teaching convention:
- ``tau in [n, n+1)``: strike = par + 0.5*cpn
- ``tau in [n+1, n+2)``: strike = par + 0.25*cpn
- ``tau >= n+2``: strike = par

Voice: cite QGL SDE #37 for the HW SDE, exotics #32 for the American-call
embedded structure, risk-neutral #83 for the affine-ZCB discounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foureng.rates.hull_white import HullWhiteParams, hull_white_discount_bond


@dataclass(frozen=True)
class BondSpec:
    """High-yield bond terms.

    Parameters
    ----------
    T : float
        Tenor in years.
    n : int
        Non-call period in years (integer, coupon-aligned).
    coupon : float
        Annual coupon rate (e.g. 0.08 for 8%). Paid annually in the model.
    spread : float
        Constant credit spread over the short rate (e.g. 0.04 for 400 bps).
    tighten : float
        Spread-tightening scenario: bps of spread compression phased in
        linearly from 0 at t=0 to full at t=n. Applied to the credit discount;
        the MW strike itself remains risk-free/Treasury plus ``mw_addon``.
        Default 0.
    mw_addon : float
        Make-whole discount add-on over the risk-free curve (e.g. 0.005 for T+50).
    par : float
        Par value; defaults to 100.
    """

    T: float
    n: int
    coupon: float
    spread: float
    tighten: float = 0.0
    mw_addon: float = 0.005
    par: float = 100.0


def _hw_bond_reconstitution(
    a: float, sigma: float, t: float, T: float, P0_t: float, P0_T: float
) -> tuple[float, float]:
    """HW analytic ZCB reconstitution: return (A, B) so P(t,T;r) = A * exp(-B*r).

    Standard Hull-White formulas (Brigo-Mercurio eq. 3.39-3.40).
    """
    dt = T - t
    if dt <= 0:
        return 1.0, 0.0
    if a * dt < 1e-9:
        B = dt * (1.0 - 0.5 * a * dt)
    else:
        B = (1.0 - np.exp(-a * dt)) / a
    # f^M(0, t) via finite-difference on log-discount
    # For flat forward curve P0_t = exp(-r0*t), f^M(0,t) = r0 exactly, but the
    # closed form below works for any curve if we pass the market curve via P^M.
    # We approximate f^M(0,t) here from a small central difference upstream —
    # but for the flat curve used by the browser tool we can pass f^M(0,t) = r0
    # directly. In this helper we compute A given P0_t, P0_T, and the fwd f0t.
    raise NotImplementedError("use hw_bond_price_at_node with fwd rate passed in")


def hw_bond_price_at_node(
    p: HullWhiteParams,
    t: float,
    T: float,
    r_node: float,
    f0t: float,
) -> float:
    """P(t, T; r) under Hull-White with market curve encoded in ``p``.

    Uses Brigo-Mercurio eq. 3.39 (extended Vasicek reconstitution):

        P(t,T) = A(t,T) * exp(-B(t,T) * r)
        B(t,T) = (1 - exp(-a*(T-t))) / a
        A(t,T) = (P^M(0,T) / P^M(0,t))
                 * exp( B(t,T) * f^M(0,t)
                        - (sigma^2 / (4a)) * (1 - exp(-2 a t)) * B(t,T)^2 )

    ``f0t`` is the instantaneous market forward at 0 for maturity t.
    For the flat curve ``P^M(0,T) = exp(-r0 T)``, ``f0t = r0``.
    """
    a, sigma = p.a, p.sigma
    dt = T - t
    if dt <= 0:
        return 1.0
    if a * dt < 1e-9:
        B = dt * (1.0 - 0.5 * a * dt)
    else:
        B = (1.0 - float(np.exp(-a * dt))) / a
    P0t = hull_white_discount_bond(p, t)
    P0T = hull_white_discount_bond(p, T)
    if a * t < 1e-9:
        one_minus_e2at = 2.0 * a * t * (1.0 - a * t)
    else:
        one_minus_e2at = 1.0 - float(np.exp(-2.0 * a * t))
    log_A = float(np.log(P0T / P0t)) + B * f0t - (sigma * sigma / (4.0 * a)) * one_minus_e2at * B * B
    return float(np.exp(log_A - B * r_node))


def _hw_trinomial_grid(
    p: HullWhiteParams, T: float, N: int, r0_fwd_fn
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Build the standard HW 1994 trinomial tree fitted to the initial curve.

    Returns
    -------
    times : (N+1,) time grid
    j_low, j_high : (N+1,) integer arrays with the min/max j at each slice
    alpha : (N+1,) shift so that r_{i,j} = alpha[i] + j * dx
    pu, pm, pd : (N+1, max_width) probability arrays (indexed by j - j_low)
    dx : the space step
    """
    a, sigma = p.a, p.sigma
    dt = T / N
    dx = sigma * float(np.sqrt(3.0 * dt))
    # Truncation level per Hull-White: |j| <= jmax = ceil(0.184 / (a*dt))
    jmax = int(np.ceil(0.184 / (a * dt)))
    times = np.linspace(0.0, T, N + 1)

    # Build shape of tree: j-range grows by 1 each step until jmax
    j_low = np.zeros(N + 1, dtype=int)
    j_high = np.zeros(N + 1, dtype=int)
    for i in range(N + 1):
        j_high[i] = min(i, jmax)
        j_low[i] = -j_high[i]

    width = 2 * jmax + 1
    pu = np.zeros((N + 1, width))
    pm = np.zeros((N + 1, width))
    pd = np.zeros((N + 1, width))

    # Standard Hull-White probabilities.
    # At each (i, j), let M = -a * j * dx * dt (mean of one-step Δx).
    # Case A (normal branching, j -> {j+1, j, j-1}):
    #   pu = 1/6 + (M^2/dx^2 + M/dx)/2
    #   pm = 2/3 - M^2/dx^2
    #   pd = 1/6 + (M^2/dx^2 - M/dx)/2
    # Case B (top-boundary j = jmax, branches down: j -> {j, j-1, j-2}):
    #   pu = 7/6 + (M^2/dx^2 + 3*M/dx)/2  (this branches to j, not j+1)
    #   pm = -1/3 - M^2/dx^2 - 2*M/dx
    #   pd = 1/6 + (M^2/dx^2 + M/dx)/2    (branches to j-2)
    # Wait — reread Hull-White '94: at top boundary, branches are (j, j-1, j-2)
    # with probabilities such that mean/variance match. Standard formulas:
    #   pu (j)   = 7/6 + ( (a j dx dt)^2 - 3 a j dx dt ) / (2 dx^2)   ... etc.
    # I'll write these directly using M = -a * j * dx * dt.
    for i in range(N + 1):
        for j in range(j_low[i], j_high[i] + 1):
            idx = j - (-jmax)  # index into width-jmax array
            M = -a * j * dx * dt
            m2 = M * M
            if j == jmax:
                # Top boundary: branches to (j, j-1, j-2). We store as "up=stay, mid=down, dn=down2"
                # Formulas (Brigo-Mercurio eq. 3.62 top):
                #   pu = 7/6 + (m2 + 3*M*dx) / (2*dx^2)  -> for j
                #   pm = -1/3 - (m2 + 2*M*dx) / (dx^2)   -> for j-1
                #   pd = 1/6 + (m2 + M*dx) / (2*dx^2)    -> for j-2
                pu[i, idx] = 7.0 / 6.0 + (m2 + 3.0 * M * dx) / (2.0 * dx * dx)
                pm[i, idx] = -1.0 / 3.0 - (m2 + 2.0 * M * dx) / (dx * dx)
                pd[i, idx] = 1.0 / 6.0 + (m2 + M * dx) / (2.0 * dx * dx)
            elif j == -jmax:
                # Bottom boundary: branches to (j+2, j+1, j).
                #   pu = 1/6 + (m2 - M*dx) / (2*dx^2)   -> for j+2
                #   pm = -1/3 - (m2 - 2*M*dx) / (dx^2)  -> for j+1
                #   pd = 7/6 + (m2 - 3*M*dx) / (2*dx^2) -> for j
                pu[i, idx] = 1.0 / 6.0 + (m2 - M * dx) / (2.0 * dx * dx)
                pm[i, idx] = -1.0 / 3.0 - (m2 - 2.0 * M * dx) / (dx * dx)
                pd[i, idx] = 7.0 / 6.0 + (m2 - 3.0 * M * dx) / (2.0 * dx * dx)
            else:
                # Normal branching: (j+1, j, j-1)
                pu[i, idx] = 1.0 / 6.0 + (m2 + M * dx) / (2.0 * dx * dx)
                pm[i, idx] = 2.0 / 3.0 - m2 / (dx * dx)
                pd[i, idx] = 1.0 / 6.0 + (m2 - M * dx) / (2.0 * dx * dx)

    # Forward-induction to compute alpha[i] such that Q(0, t_{i+1}) = P^M(0, t_{i+1}).
    # Q[i, j] = time-0 present value of $1 paid at node (i, j), computed by
    # accumulating discounted branch probabilities.
    # Convention: r_{i,j} = alpha[i] + j*dx. Discount over dt uses r_{i,j}.
    Q = np.zeros((N + 1, width))
    Q[0, jmax] = 1.0  # j=0 at i=0 sits at index jmax
    alpha = np.zeros(N + 1)

    # At i=0, we have a single node with r_0 = alpha[0]. To match P^M(0, dt):
    # P^M(0, dt) = exp(-alpha[0] * dt), so alpha[0] = -log(P^M(0,dt)) / dt.
    alpha[0] = -float(np.log(hull_white_discount_bond(p, dt))) / dt

    for i in range(N):
        # Given alpha[i] and Q[i, .], step Q forward to Q[i+1, .]
        # and solve for alpha[i+1] so that sum_j Q[i+1, j] = P^M(0, t_{i+2}).
        # But the discount from t_{i+1} onward isn't known yet, so we use the
        # standard forward-induction identity:
        #   P^M(0, t_{i+1} + dt) = sum_j Q[i+1, j] * exp(-r_{i+1,j} * dt)
        #                         = sum_j Q[i+1, j] * exp(-(alpha[i+1] + j*dx)*dt)
        # => exp(-alpha[i+1]*dt) = P^M(0, t_{i+2}) / sum_j Q[i+1,j] * exp(-j*dx*dt)

        # First push Q one step: for each (i,j), discount by exp(-r_{i,j}*dt) and
        # spread to the appropriate destination j' with pu/pm/pd.
        Qnext = np.zeros(width)
        for j in range(j_low[i], j_high[i] + 1):
            idx = j - (-jmax)
            r_ij = alpha[i] + j * dx
            disc = float(np.exp(-r_ij * dt))
            if j == jmax:
                # branches to (j, j-1, j-2)
                dests = (j, j - 1, j - 2)
            elif j == -jmax:
                # branches to (j+2, j+1, j)
                dests = (j + 2, j + 1, j)
            else:
                dests = (j + 1, j, j - 1)
            for pk, dest in zip((pu[i, idx], pm[i, idx], pd[i, idx]), dests):
                Qnext[dest - (-jmax)] += Q[i, idx] * pk * disc

        Q[i + 1, :] = Qnext
        # Solve alpha[i+1]
        t_next = times[i + 1]
        t_after = t_next + dt
        if t_after > T + 1e-12:
            break
        P_after = hull_white_discount_bond(p, t_after)
        denom = 0.0
        for j in range(j_low[i + 1], j_high[i + 1] + 1):
            idx = j - (-jmax)
            denom += Qnext[idx] * float(np.exp(-j * dx * dt))
        # exp(-alpha[i+1]*dt) = P_after / denom
        alpha[i + 1] = -float(np.log(P_after / denom)) / dt

    return times, j_low, j_high, alpha, pu, pm, pd, dx


def price_bond_hw(
    p: HullWhiteParams,
    bond: BondSpec,
    mode: str = "full",
    N: int | None = None,
    r0_fwd_fn=None,
) -> float:
    """Price the bond on the HW trinomial tree under the given call mode.

    Coupons paid annually at t = 1, 2, ..., T (integer years).
    Cash flows are discounted with credit spread ``bond.spread`` added to the
    stochastic short rate at each node (no default/recovery).

    Make-whole strike inside NC at node (i, j):
        K = max(1.01*par, sum_{k coupon date, i<k<=n} cpn * P_HW(t_i, t_k; r_ij)
                          + p50 * P_HW(t_i, t_n; r_ij))
    where ``P_HW`` uses ``r_HW = r_ij`` and the ``mw_addon`` add-on is applied
    inside the affine formula by shifting r_ij by +mw_addon.
    """
    if r0_fwd_fn is None:
        # Assume flat curve at r0
        r0_fwd_fn = lambda t: p.r0  # noqa: E731

    T = bond.T
    if N is None:
        N = int(round(T * 48))  # 48 steps/year is plenty for cross-checks

    times, j_low, j_high, alpha, pu, pm, pd, dx = _hw_trinomial_grid(p, T, N, r0_fwd_fn)
    dt = T / N
    jmax = j_high[-1] if len(j_high) else 0
    # Actually jmax was set inside _hw_trinomial_grid; recover from width.
    width = pu.shape[1]
    jmax = (width - 1) // 2

    par = bond.par
    cpn = bond.coupon * par
    p50 = par + 0.5 * cpn
    p25 = par + 0.25 * cpn

    # Coupon indices: nearest tree step to each integer year k = 1..T
    coupon_steps = set()
    for k in range(1, int(round(T)) + 1):
        step = int(round(k / dt))
        if 0 < step <= N:
            coupon_steps.add(step)

    n = bond.n

    def mw_strike(i: int, r_ij: float) -> float:
        # PV of remaining coupons + first-call price at t = n, discounted at r_ij + mw_addon
        # Use analytic HW ZCB from t_i to each future coupon/redemption date.
        # The mw_addon is layered on top of the model curve by shifting the
        # affine exponent — equivalent to a parallel spread over the HW curve.
        t_i = times[i]
        t_n = n
        if t_i >= t_n:
            return float("inf")
        # Instantaneous fwd at 0 for maturity t_i — used inside HW ZCB formula.
        f0t = r0_fwd_fn(t_i)
        pv = 0.0
        # Coupons in (t_i, n]:
        for k in range(1, n + 1):
            if k > t_i + 1e-9:
                zcb = hw_bond_price_at_node(p, t_i, float(k), r_ij, f0t)
                # add-on: multiply by exp(-mw_addon * (k - t_i))
                zcb *= float(np.exp(-bond.mw_addon * (k - t_i)))
                # No spread inside MW discount — MW is a rf+50 payoff by convention
                pv += cpn * zcb
        # First-call redemption price at t = n:
        zcb_n = hw_bond_price_at_node(p, t_i, float(n), r_ij, f0t) * float(
            np.exp(-bond.mw_addon * (n - t_i))
        )
        # High-yield Applicable Premium reconstructs coupons through first call
        # plus the first-call redemption price, matching the browser tool.
        pv += p50 * zcb_n
        return max(1.01 * par, pv)

    def call_strike(tau: float) -> float:
        if mode == "bullet":
            return float("inf")
        if tau < n - 1e-9:
            return float("inf")  # handled by mw_strike
        if mode == "mwOnly":
            return float("inf")
        if tau < n + 1 - 1e-9:
            return p50
        if tau < n + 2 - 1e-9:
            return p25
        return par

    # Terminal payoff at i = N: par + final coupon (which is booked at t = T)
    V = np.zeros(width)
    for j in range(j_low[N], j_high[N] + 1):
        idx = j - (-jmax)
        V[idx] = par + cpn  # bond pays par + cpn at maturity

    # Backward induction
    for i in range(N - 1, -1, -1):
        Vnext = V.copy()
        V = np.zeros(width)
        is_cpn_step = (i + 1) in coupon_steps  # coupon received when we hop TO step i+1
        tau = times[i]
        for j in range(j_low[i], j_high[i] + 1):
            idx = j - (-jmax)
            r_ij = alpha[i] + j * dx
            # Spread-tightening: linear ramp from 0 at tau=0 to `tighten` at tau=n
            s_ramp = bond.tighten * min(tau, float(n)) / float(n) if n > 0 else 0.0
            y = r_ij + (bond.spread - s_ramp)  # credit-adjusted discount
            disc = float(np.exp(-y * dt))
            if j == jmax:
                dests = (j, j - 1, j - 2)
            elif j == -jmax:
                dests = (j + 2, j + 1, j)
            else:
                dests = (j + 1, j, j - 1)
            v_hold = disc * (
                pu[i, idx] * Vnext[dests[0] - (-jmax)]
                + pm[i, idx] * Vnext[dests[1] - (-jmax)]
                + pd[i, idx] * Vnext[dests[2] - (-jmax)]
            )
            # Coupon was already baked into terminal V for step N; for interior
            # coupon steps we discount to i, then at i we add the coupon (paid
            # at t_{i+1} but discounted back — actually simpler: we treat the
            # coupon as received at the coupon step and hold from there).
            # Cleaner: coupon paid at t_k is embedded in the *arriving* value at
            # step k. So when computing V at step i, if step i itself is a
            # coupon step (excluding N which is already in terminal), add cpn.
            if i in coupon_steps and i > 0:
                v_hold += cpn
            # Apply call strike (issuer's option -> lender's cap on value)
            if i > 0 and tau < n - 1e-9:
                # inside NC: MW strike
                if mode in ("full", "mwOnly"):
                    K = mw_strike(i, r_ij)
                    if np.isfinite(K):
                        v_hold = min(v_hold, K)
            elif i > 0 and tau >= n - 1e-9:
                K = call_strike(tau)
                if np.isfinite(K):
                    v_hold = min(v_hold, K)
            V[idx] = v_hold

    return float(V[-(-jmax) + 0])  # V[jmax] = center node = t=0


def option_decomposition(p: HullWhiteParams, bond: BondSpec, N: int | None = None) -> dict:
    """Return the tool's three-slice decomposition: total, postNC, mw."""
    bullet = price_bond_hw(p, bond, mode="bullet", N=N)
    full = price_bond_hw(p, bond, mode="full", N=N)
    no_mw = price_bond_hw(p, bond, mode="noMW", N=N)
    total = max(bullet - full, 1e-4)
    post_nc = max(bullet - no_mw, 0.0)
    mw = max(no_mw - full, 0.0)
    return {"bullet": bullet, "full": full, "noMW": no_mw, "total": total, "postNC": post_nc, "mw": mw}


if __name__ == "__main__":
    # Smoke test: 7NC3 @ 8% coupon, 4% rf, 400 bps spread, 20% vol.
    # Compare against a simple flat-curve HW.
    p = HullWhiteParams(a=0.1, sigma=0.20 * 0.04, r0=0.04)  # sigma in absolute rate terms
    # Note: 20% vol in the tool is *relative* to the level (log-normal style).
    # For HW (Gaussian), we translate as sigma_abs = vol_rel * r0 as a first-order
    # match — the browser tool uses this same interpretation.
    bond = BondSpec(T=7.0, n=3, coupon=0.08, spread=0.04)
    dec = option_decomposition(p, bond, N=7 * 48)
    for k, v in dec.items():
        print(f"{k:>8s}: {v:.4f}")

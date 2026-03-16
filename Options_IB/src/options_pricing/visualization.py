"""Stage 6 — Visualization suite.

Generates a multi-panel figure set for stock pitch decks:
  1. Risk-neutral PDF with reference lines (the money chart)
  2. 3D BMA volatility surface (Strike x DTE x IV)
  3. Model price correlation heatmap
  4. Model fit comparison (market vs each model)
  5. Implied volatility smile with model overlays
  6. CDF with probability regions
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

if TYPE_CHECKING:
    from options_pricing.data import OptionsSnapshot
    from options_pricing.calibration import CalibrationResult
    from options_pricing.averaging import AveragedPrices
    from options_pricing.density import RiskNeutralDensity


# ---------------------------------------------------------------------------
# Dark theme setup
# ---------------------------------------------------------------------------
_DARK_BG = "#0d1117"
_DARK_FACE = "#161b22"
_ACCENT_BLUE = "#58a6ff"
_ACCENT_CYAN = "#39d353"
_ACCENT_ORANGE = "#f0883e"
_ACCENT_PINK = "#f778ba"
_ACCENT_PURPLE = "#bc8cff"
_ACCENT_RED = "#ff7b72"
_ACCENT_YELLOW = "#e3b341"
_TEXT_COLOR = "#c9d1d9"
_GRID_COLOR = "#30363d"

MODEL_COLORS = {
    "bsm": _ACCENT_BLUE,
    "crr": _ACCENT_CYAN,
    "merton": _ACCENT_ORANGE,
    "heston": _ACCENT_PURPLE,
}

MODEL_LABELS = {
    "bsm": "Black-Scholes-Merton",
    "crr": "Cox-Ross-Rubinstein",
    "merton": "Merton Jump-Diffusion",
    "heston": "Heston Stochastic Vol",
}


def _apply_dark_theme(ax, title: str = ""):
    """Apply dark theme to an axis."""
    ax.set_facecolor(_DARK_FACE)
    ax.tick_params(colors=_TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, alpha=0.4, linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=_TEXT_COLOR, pad=12)


def _apply_dark_theme_3d(ax, title: str = ""):
    """Apply dark theme to a 3D axis."""
    ax.set_facecolor(_DARK_BG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(_GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(_GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(_GRID_COLOR)
    ax.tick_params(colors=_TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.zaxis.label.set_color(_TEXT_COLOR)
    ax.xaxis._axinfo["grid"]["color"] = _GRID_COLOR
    ax.yaxis._axinfo["grid"]["color"] = _GRID_COLOR
    ax.zaxis._axinfo["grid"]["color"] = _GRID_COLOR
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=_TEXT_COLOR, pad=12)


# ---------------------------------------------------------------------------
# 1. Risk-Neutral PDF (hero chart)
# ---------------------------------------------------------------------------

def plot_density(
    rnd: "RiskNeutralDensity",
    spot: float,
    target: float | None = None,
    ticker: str = "",
    expiry: str = "",
    dte: int = 0,
    weights: dict[str, float] | None = None,
    save_path: str | None = None,
) -> Figure:
    """The main PDF chart for the pitch deck."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=_DARK_BG)
    _apply_dark_theme(ax)

    # Gradient fill under the PDF
    ax.fill_between(rnd.strikes, rnd.pdf, alpha=0.15, color=_ACCENT_BLUE)
    ax.plot(rnd.strikes, rnd.pdf, color=_ACCENT_BLUE, linewidth=2.5,
            label="Risk-Neutral PDF", zorder=5)

    # Glow effect
    for lw, alpha in [(6, 0.08), (4, 0.12)]:
        ax.plot(rnd.strikes, rnd.pdf, color=_ACCENT_BLUE, linewidth=lw, alpha=alpha)

    # Vertical lines
    ax.axvline(spot, color=_TEXT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Spot: ${spot:.2f}", zorder=4)
    ax.axvline(rnd.rn_mean, color=_ACCENT_ORANGE, linestyle="--", linewidth=1.5,
               label=f"RN Mean: ${rnd.rn_mean:.2f}", zorder=4)

    if target is not None:
        ax.axvline(target, color=_ACCENT_CYAN, linestyle="-", linewidth=2.5,
                   label=f"Target: ${target:.2f}", zorder=6)
        if target > spot:
            mask = rnd.strikes >= target
            prob = np.trapz(rnd.pdf[mask], rnd.strikes[mask])
            ax.fill_between(rnd.strikes[mask], rnd.pdf[mask],
                            alpha=0.35, color=_ACCENT_CYAN,
                            label=f"P(S > ${target:.0f}) = {prob:.1%}", zorder=3)
        else:
            mask = rnd.strikes <= target
            prob = np.trapz(rnd.pdf[mask], rnd.strikes[mask])
            ax.fill_between(rnd.strikes[mask], rnd.pdf[mask],
                            alpha=0.35, color=_ACCENT_RED,
                            label=f"P(S < ${target:.0f}) = {prob:.1%}", zorder=3)

    # Observed data boundary markers
    if hasattr(rnd, "observed_range") and rnd.observed_range is not None:
        obs_lo, obs_hi = rnd.observed_range
        if obs_hi < rnd.strikes.max() - 1:
            ax.axvline(obs_hi, color=_ACCENT_YELLOW, linestyle=":", linewidth=1,
                       alpha=0.6, zorder=3)
            ax.annotate(f"  Max observed\n  strike: ${obs_hi:.0f}",
                        (obs_hi, ax.get_ylim()[1] * 0.45),
                        fontsize=8, color=_ACCENT_YELLOW, alpha=0.8)
        if obs_lo > rnd.strikes.min() + 1:
            ax.axvline(obs_lo, color=_ACCENT_YELLOW, linestyle=":", linewidth=1,
                       alpha=0.6, zorder=3)

    ax.set_xlabel("Strike / Price ($)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    title = "Risk-Neutral Probability Distribution"
    if ticker:
        title = f"{ticker} — {title}"
    if expiry:
        title += f"\nExpiry: {expiry}  ({dte} DTE)"
    ax.set_title(title, fontsize=16, fontweight="bold", color=_TEXT_COLOR, pad=15)
    ax.legend(loc="upper right", fontsize=10, facecolor=_DARK_FACE,
              edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)
    ax.set_ylim(bottom=0)

    if weights:
        weight_text = "Model Weights\n" + "\n".join(
            f"  {name}: {w:.1%}" for name, w in
            sorted(weights.items(), key=lambda x: -x[1])
        )
        ax.text(0.02, 0.95, weight_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", color=_TEXT_COLOR,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=_DARK_FACE,
                          edgecolor=_GRID_COLOR, alpha=0.9))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 2. 3D BMA Volatility Surface (Strike x DTE)
# ---------------------------------------------------------------------------

def plot_vol_surface_3d(
    snapshot: "OptionsSnapshot",
    cal_results: dict[str, "CalibrationResult"],
    avg: "AveragedPrices",
    target: float | None = None,
    save_path: str | None = None,
) -> Figure:
    """3D BMA-averaged implied volatility surface across Strike x DTE.

    Uses the calibrated model parameters to price options at synthetic DTEs
    (7 to 180 days), then inverts via BSM to get implied vol. The four models
    are blended using the BMA weights, giving a single smooth surface that
    represents the market-consensus vol structure across time and moneyness.
    """
    from options_pricing.models import bsm as bsm_mod
    from options_pricing.calibration import MODELS
    from scipy.optimize import brentq

    S = snapshot.spot
    r = snapshot.risk_free_rate

    # Strike grid: tighter range focused on liquid region
    df = snapshot.chains
    calls = df[df["right"] == "C"].sort_values("strike")
    K_min, K_max = calls["strike"].min(), calls["strike"].max()
    n_strikes = 60
    strike_grid = np.linspace(K_min, K_max, n_strikes)

    # DTE grid: 7 days out to 180 days
    dte_grid = np.array([7, 14, 21, 30, 45, 60, 90, 120, 150, 180])
    n_dte = len(dte_grid)

    # BMA weights
    weights = avg.weights

    # Build the surface: for each (strike, dte), compute BMA-weighted call
    # price, then invert to implied vol
    vol_surface = np.full((n_dte, n_strikes), np.nan)

    for i, dte in enumerate(dte_grid):
        T = dte / 365.0
        for j, K in enumerate(strike_grid):
            # BMA-weighted call price across all models
            bma_price = 0.0
            for name, res in cal_results.items():
                model = MODELS[name]
                w = weights[name]
                params = tuple(res.params.values())
                try:
                    p = float(model.price_for_calibration(
                        params, S, np.array([K]), T, r, np.array(["C"])
                    )[0])
                    if np.isfinite(p) and p > 0:
                        bma_price += w * p
                except Exception:
                    pass

            # Invert to implied vol via BSM
            intrinsic = max(S - K * np.exp(-r * T), 0)
            if bma_price <= intrinsic + 0.001 or bma_price >= S:
                continue
            try:
                def _obj(sig, _K=K, _T=T, _price=bma_price):
                    return float(bsm_mod.price(S, _K, _T, r, sig, "C")) - _price
                f_lo, f_hi = _obj(0.001), _obj(3.0)
                if f_lo * f_hi < 0:
                    vol_surface[i, j] = brentq(_obj, 0.001, 3.0, xtol=1e-6)
            except (ValueError, RuntimeError):
                pass

    # Meshgrid for plotting
    X, Y = np.meshgrid(strike_grid, dte_grid)

    # --- Main 3D surface ---
    fig = plt.figure(figsize=(16, 10), facecolor=_DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    _apply_dark_theme_3d(ax, f"{snapshot.ticker} — BMA Implied Volatility Surface")

    # Mask NaNs for clean surface
    Z = np.ma.array(vol_surface, mask=np.isnan(vol_surface))

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.magma,
        alpha=0.9,
        edgecolor="none",
        antialiased=True,
        rstride=1, cstride=1,
    )

    # Subtle wireframe for depth
    ax.plot_wireframe(X, Y, Z, color=_TEXT_COLOR, alpha=0.06, linewidth=0.3,
                      rstride=1, cstride=2)

    # --- Inflection points along the strike axis ---
    for i in range(n_dte):
        row = vol_surface[i, :]
        valid = ~np.isnan(row)
        if valid.sum() < 5:
            continue
        # Second derivative (curvature) along strike
        valid_strikes = strike_grid[valid]
        valid_vols = row[valid]
        d2v = np.gradient(np.gradient(valid_vols, valid_strikes), valid_strikes)
        sign_changes = np.where(np.diff(np.sign(d2v)))[0]
        for idx in sign_changes:
            ax.scatter(valid_strikes[idx], dte_grid[i], valid_vols[idx],
                       color=_ACCENT_YELLOW, s=45, zorder=10,
                       edgecolor="white", linewidth=0.5, alpha=0.9)

    # --- ATM vol ridge line (strike closest to spot at each DTE) ---
    atm_idx = np.argmin(np.abs(strike_grid - S))
    atm_vols = vol_surface[:, atm_idx]
    valid_atm = ~np.isnan(atm_vols)
    if valid_atm.sum() > 1:
        ax.plot(
            np.full(valid_atm.sum(), strike_grid[atm_idx]),
            dte_grid[valid_atm],
            atm_vols[valid_atm],
            color=_ACCENT_CYAN, linewidth=2.5, zorder=8, alpha=0.9,
        )

    # --- Target strike plane (vertical slice) ---
    if target is not None and K_min <= target <= K_max:
        target_idx = np.argmin(np.abs(strike_grid - target))
        target_vols = vol_surface[:, target_idx]
        valid_tgt = ~np.isnan(target_vols)
        if valid_tgt.sum() > 1:
            ax.plot(
                np.full(valid_tgt.sum(), strike_grid[target_idx]),
                dte_grid[valid_tgt],
                target_vols[valid_tgt],
                color=_ACCENT_ORANGE, linewidth=2.5, linestyle="--",
                zorder=8, alpha=0.9,
            )

    # --- Spot vertical line on the surface ---
    # Draw a faint vertical line at spot across all DTEs
    spot_idx = np.argmin(np.abs(strike_grid - S))
    spot_vols = vol_surface[:, spot_idx]
    valid_spot = ~np.isnan(spot_vols)
    if valid_spot.any():
        z_floor = np.nanmin(vol_surface) * 0.95
        for d_i in range(n_dte):
            if valid_spot[d_i]:
                ax.plot(
                    [strike_grid[spot_idx], strike_grid[spot_idx]],
                    [dte_grid[d_i], dte_grid[d_i]],
                    [z_floor, spot_vols[d_i]],
                    color=_TEXT_COLOR, alpha=0.15, linewidth=0.8,
                )

    ax.set_xlabel("\nStrike ($)", fontsize=11, labelpad=12)
    ax.set_ylabel("\nDays to Expiry", fontsize=11, labelpad=12)
    ax.set_zlabel("\nImplied Volatility", fontsize=11, labelpad=10)
    ax.view_init(elev=22, azim=-50)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, aspect=20)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    from matplotlib.ticker import FuncFormatter
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:.0%}" if 0 <= x <= 1 else f"{x:.2f}")
    )
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)
    cbar.set_label("Implied Volatility", color=_TEXT_COLOR, fontsize=10)

    # Legend annotation
    legend_lines = ["ATM ridge (cyan)"]
    if target is not None:
        legend_lines.append(f"Target ${target:.0f} slice (orange)")
    legend_lines.append("Inflection points (yellow)")
    legend_text = "\n".join(legend_lines)
    ax.text2D(0.02, 0.95, legend_text, transform=ax.transAxes,
              fontsize=9, color=_TEXT_COLOR, verticalalignment="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor=_DARK_FACE,
                        edgecolor=_GRID_COLOR, alpha=0.85))

    fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3. Model Price Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    avg: "AveragedPrices",
    snapshot: "OptionsSnapshot",
    save_path: str | None = None,
) -> Figure:
    """Heatmap of model price correlations + pricing error correlations."""
    names = sorted(avg.per_model_prices.keys())
    market = snapshot.chains["mid"].values

    # Price correlation
    price_matrix = np.column_stack([avg.per_model_prices[n] for n in names])
    price_corr = np.corrcoef(price_matrix.T)

    # Error correlation
    error_matrix = np.column_stack([avg.per_model_prices[n] - market for n in names])
    error_corr = np.corrcoef(error_matrix.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=_DARK_BG)

    for ax, corr, title in [
        (ax1, price_corr, "Model Price Correlation"),
        (ax2, error_corr, "Pricing Error Correlation"),
    ]:
        _apply_dark_theme(ax, title)
        ax.grid(False)
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        labels = [n.upper() for n in names]
        ax.set_xticklabels(labels, fontsize=10, color=_TEXT_COLOR)
        ax.set_yticklabels(labels, fontsize=10, color=_TEXT_COLOR)

        # Annotate cells
        for i in range(len(names)):
            for j in range(len(names)):
                val = corr[i, j]
                color = "black" if abs(val) > 0.5 else _TEXT_COLOR
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)

    fig.suptitle(f"{snapshot.ticker} — Model Correlation Analysis",
                 fontsize=15, fontweight="bold", color=_TEXT_COLOR, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.08)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. Model Fit Comparison
# ---------------------------------------------------------------------------

def plot_model_fit(
    snapshot: "OptionsSnapshot",
    cal_results: dict[str, "CalibrationResult"],
    avg: "AveragedPrices",
    save_path: str | None = None,
) -> Figure:
    """Market prices vs each model's fitted prices, calls and puts."""
    df = snapshot.chains
    call_mask = df["right"].values == "C"
    put_mask = df["right"].values == "P"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor=_DARK_BG,
                             sharex=True)

    for ax, mask, label in [(axes[0], call_mask, "Calls"), (axes[1], put_mask, "Puts")]:
        _apply_dark_theme(ax, f"{snapshot.ticker} — {label}: Market vs Model Prices")
        strikes = df["strike"].values[mask]
        market = df["mid"].values[mask]

        # Market dots
        ax.scatter(strikes, market, color="white", s=40, zorder=10,
                   label="Market Mid", edgecolor=_GRID_COLOR, linewidth=0.5)

        # Each model
        for name in sorted(cal_results.keys()):
            res = cal_results[name]
            ax.plot(strikes, res.model_prices[mask],
                    color=MODEL_COLORS[name], linewidth=1.8, alpha=0.8,
                    label=MODEL_LABELS[name])

        # BMA average
        ax.plot(strikes, avg.prices[mask],
                color=_ACCENT_YELLOW, linewidth=2.5, linestyle="--",
                label="BMA Average", zorder=8)

        ax.set_ylabel("Option Price ($)", fontsize=11)
        ax.legend(fontsize=9, facecolor=_DARK_FACE, edgecolor=_GRID_COLOR,
                  labelcolor=_TEXT_COLOR, ncol=3, loc="upper right")

    axes[1].set_xlabel("Strike ($)", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5. Implied Volatility Smile
# ---------------------------------------------------------------------------

def plot_vol_smile(
    snapshot: "OptionsSnapshot",
    cal_results: dict[str, "CalibrationResult"],
    save_path: str | None = None,
) -> Figure:
    """IV smile: market IV vs model-implied IV for calls."""
    from options_pricing.models import bsm
    from scipy.optimize import brentq

    df = snapshot.chains
    calls = df[df["right"] == "C"].sort_values("strike")
    S = snapshot.spot
    T = snapshot.dte / 365.0
    r = snapshot.risk_free_rate

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=_DARK_BG)
    _apply_dark_theme(ax, f"{snapshot.ticker} — Implied Volatility Smile")

    # Market IV
    ax.scatter(calls["strike"], calls["iv"], color="white", s=50, zorder=10,
               label="Market IV", edgecolor=_GRID_COLOR, linewidth=0.5)

    # Each model's implied vol
    call_mask = df["right"].values == "C"
    for name in sorted(cal_results.keys()):
        res = cal_results[name]
        model_prices = res.model_prices[call_mask]
        model_iv = []
        for k, mp in zip(calls["strike"], model_prices):
            try:
                intrinsic = max(S - float(k) * np.exp(-r * T), 0)
                if mp <= intrinsic + 0.001 or mp >= S:
                    model_iv.append(np.nan)
                    continue
                def _obj(sig, _k=k, _mp=mp):
                    return float(bsm.price(S, _k, T, r, sig, "C")) - _mp
                f_lo, f_hi = _obj(0.001), _obj(3.0)
                if f_lo * f_hi < 0:
                    model_iv.append(brentq(_obj, 0.001, 3.0, xtol=1e-6))
                else:
                    model_iv.append(np.nan)
            except (ValueError, RuntimeError):
                model_iv.append(np.nan)

        ax.plot(calls["strike"], model_iv,
                color=MODEL_COLORS[name], linewidth=2, alpha=0.85,
                label=MODEL_LABELS[name])
        # Glow
        ax.plot(calls["strike"], model_iv,
                color=MODEL_COLORS[name], linewidth=5, alpha=0.1)

    ax.axvline(S, color=_TEXT_COLOR, linestyle=":", linewidth=1, alpha=0.5,
               label=f"Spot ${S:.0f}")

    ax.set_xlabel("Strike ($)", fontsize=12)
    ax.set_ylabel("Implied Volatility", fontsize=12)
    ax.legend(fontsize=10, facecolor=_DARK_FACE, edgecolor=_GRID_COLOR,
              labelcolor=_TEXT_COLOR)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 6. CDF with Probability Regions
# ---------------------------------------------------------------------------

def plot_cdf(
    rnd: "RiskNeutralDensity",
    spot: float,
    target: float | None = None,
    ticker: str = "",
    save_path: str | None = None,
) -> Figure:
    """CDF plot with shaded probability regions and percentile markers."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=_DARK_BG)
    _apply_dark_theme(ax, f"{ticker} — Risk-Neutral Cumulative Distribution")

    ax.plot(rnd.strikes, rnd.cdf, color=_ACCENT_BLUE, linewidth=2.5, zorder=5)
    ax.plot(rnd.strikes, rnd.cdf, color=_ACCENT_BLUE, linewidth=6, alpha=0.1)

    # Shade regions: red < 25th, green > 75th
    p25_idx = np.searchsorted(rnd.cdf, 0.25)
    p75_idx = np.searchsorted(rnd.cdf, 0.75)
    ax.fill_between(rnd.strikes[:p25_idx], rnd.cdf[:p25_idx],
                    alpha=0.2, color=_ACCENT_RED, label="Lower 25%")
    ax.fill_between(rnd.strikes[p75_idx:], rnd.cdf[p75_idx:],
                    alpha=0.2, color=_ACCENT_CYAN, label="Upper 25%")

    # Percentile lines
    for pct, ls in [(0.10, ":"), (0.25, "--"), (0.50, "-"), (0.75, "--"), (0.90, ":")]:
        idx = min(np.searchsorted(rnd.cdf, pct), len(rnd.strikes) - 1)
        ax.axhline(pct, color=_GRID_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)
        ax.plot(rnd.strikes[idx], pct, "o", color=_ACCENT_YELLOW, markersize=7, zorder=8)
        ax.annotate(f"  P{int(pct*100)}: ${rnd.strikes[idx]:.0f}",
                    (rnd.strikes[idx], pct), fontsize=9, color=_ACCENT_YELLOW,
                    va="center")

    ax.axvline(spot, color=_TEXT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Spot: ${spot:.2f}")

    if target is not None:
        ax.axvline(target, color=_ACCENT_CYAN, linestyle="-", linewidth=2,
                   label=f"Target: ${target:.2f}")

    # Observed data boundary
    if hasattr(rnd, "observed_range") and rnd.observed_range is not None:
        obs_lo, obs_hi = rnd.observed_range
        if obs_hi < rnd.strikes.max() - 1:
            ax.axvline(obs_hi, color=_ACCENT_YELLOW, linestyle=":", linewidth=1,
                       alpha=0.6, zorder=3, label=f"Max observed: ${obs_hi:.0f}")
        if obs_lo > rnd.strikes.min() + 1:
            ax.axvline(obs_lo, color=_ACCENT_YELLOW, linestyle=":", linewidth=1,
                       alpha=0.6, zorder=3)

    ax.set_xlabel("Strike / Price ($)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10, facecolor=_DARK_FACE, edgecolor=_GRID_COLOR,
              labelcolor=_TEXT_COLOR, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Master function — generate everything
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. Multi-DTE Comparison (multi-horizon mode)
# ---------------------------------------------------------------------------

_DTE_COLORS = [
    _ACCENT_BLUE, _ACCENT_CYAN, _ACCENT_ORANGE, _ACCENT_PURPLE,
    _ACCENT_PINK, _ACCENT_RED, _ACCENT_YELLOW, "#8b949e",
]


def plot_multi_dte_comparison(
    results: list[tuple],
    target: float | None = None,
    save_path: str | None = None,
) -> Figure:
    """Multi-horizon comparison: overlaid PDFs + P(target) across DTEs.

    Parameters
    ----------
    results : list of (OptionsSnapshot, cal_results, AveragedPrices, RiskNeutralDensity)
    target : optional price target
    save_path : file to write
    """
    has_target = target is not None
    n_panels = 2 if has_target else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 7 * n_panels), facecolor=_DARK_BG,
    )
    if n_panels == 1:
        axes = [axes]

    ticker = results[0][0].ticker
    spot = results[0][0].spot

    # --- Panel 1: Overlaid PDFs ---
    ax = axes[0]
    _apply_dark_theme(ax, f"{ticker} — Risk-Neutral PDF Across Horizons")

    ax.axvline(spot, color=_TEXT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Spot: ${spot:.2f}", zorder=4)
    if has_target:
        ax.axvline(target, color=_ACCENT_YELLOW, linestyle="-", linewidth=2.5,
                   label=f"Target: ${target:.2f}", zorder=6)

    for i, (snap, _cal, _avg, rnd) in enumerate(results):
        color = _DTE_COLORS[i % len(_DTE_COLORS)]
        ax.plot(rnd.strikes, rnd.pdf, color=color, linewidth=2,
                label=f"{snap.dte} DTE", alpha=0.85, zorder=5)
        ax.fill_between(rnd.strikes, rnd.pdf, alpha=0.06, color=color)

    ax.set_xlabel("Strike / Price ($)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, facecolor=_DARK_FACE, edgecolor=_GRID_COLOR,
              labelcolor=_TEXT_COLOR, loc="upper right")

    # --- Panel 2: P(target) bar chart ---
    if has_target:
        ax2 = axes[1]
        _apply_dark_theme(ax2, f"{ticker} — P(S > ${target:.0f}) by Horizon")

        dtes = []
        probs = []
        for snap, _cal, _avg, rnd in results:
            dtes.append(snap.dte)
            if rnd.strikes.min() <= target <= rnd.strikes.max():
                prob_below = float(np.interp(target, rnd.strikes, rnd.cdf))
                probs.append(1 - prob_below)
            else:
                probs.append(0.0)

        colors = [_DTE_COLORS[i % len(_DTE_COLORS)] for i in range(len(dtes))]
        bars = ax2.bar(range(len(dtes)), [p * 100 for p in probs],
                       color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.set_xticks(range(len(dtes)))
        ax2.set_xticklabels([f"{d} DTE" for d in dtes], fontsize=11)
        ax2.set_ylabel("Probability (%)", fontsize=12)

        max_pct = max(probs) * 100 if max(probs) > 0 else 10
        ax2.set_ylim(0, max_pct * 1.35)

        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max_pct * 0.02,
                     f"{prob:.1%}", ha="center", va="bottom", fontsize=13,
                     fontweight="bold", color=_TEXT_COLOR)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_DARK_BG)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Master function — generate everything
# ---------------------------------------------------------------------------

def generate_all(
    snapshot: "OptionsSnapshot",
    cal_results: dict[str, "CalibrationResult"],
    avg: "AveragedPrices",
    rnd: "RiskNeutralDensity",
    target: float | None = None,
    output_dir: str = "output",
) -> list[str]:
    """Generate all charts and return list of saved file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ticker = snapshot.ticker

    saved = []

    print("  Generating PDF chart...")
    plot_density(rnd, snapshot.spot, target, ticker, snapshot.expiry, snapshot.dte,
                 avg.weights, str(out / f"{ticker}_pdf.png"))
    saved.append(str(out / f"{ticker}_pdf.png"))

    print("  Generating 3D volatility surface (Strike x DTE)...")
    plot_vol_surface_3d(snapshot, cal_results, avg, target,
                        str(out / f"{ticker}_vol_surface_3d.png"))
    saved.append(str(out / f"{ticker}_vol_surface_3d.png"))

    print("  Generating correlation matrix...")
    plot_correlation_matrix(avg, snapshot, str(out / f"{ticker}_correlation.png"))
    saved.append(str(out / f"{ticker}_correlation.png"))

    print("  Generating model fit comparison...")
    plot_model_fit(snapshot, cal_results, avg, str(out / f"{ticker}_model_fit.png"))
    saved.append(str(out / f"{ticker}_model_fit.png"))

    print("  Generating IV smile...")
    plot_vol_smile(snapshot, cal_results, str(out / f"{ticker}_vol_smile.png"))
    saved.append(str(out / f"{ticker}_vol_smile.png"))

    print("  Generating CDF chart...")
    plot_cdf(rnd, snapshot.spot, target, ticker, str(out / f"{ticker}_cdf.png"))
    saved.append(str(out / f"{ticker}_cdf.png"))

    plt.close("all")
    return saved

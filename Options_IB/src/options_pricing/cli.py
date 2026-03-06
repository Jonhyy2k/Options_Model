"""CLI entry point for options-pricing."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from options_pricing.data import pull, load_snapshot, save_snapshot
from options_pricing.calibration import calibrate_all
from options_pricing.averaging import average
from options_pricing.density import extract_density
from options_pricing.visualization import generate_all
from options_pricing.report import write_report


def _fetch_or_load(args) -> "OptionsSnapshot":
    """Get snapshot from IB or from a CSV file."""
    if hasattr(args, "from_csv") and args.from_csv:
        print(f"Loading snapshot from {args.from_csv}...")
        return load_snapshot(args.from_csv)
    return pull(
        ticker=args.ticker,
        port=args.port,
        dte_low=args.dte_low,
        dte_high=args.dte_high,
        risk_free_rate=args.rate,
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add args shared between fetch and run commands."""
    p.add_argument("--ticker", default="SPY", help="Underlying symbol")
    p.add_argument("--port", type=int, default=7496, help="TWS/Gateway port")
    p.add_argument("--dte-low", type=int, default=30)
    p.add_argument("--dte-high", type=int, default=45)
    p.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    p.add_argument("--from-csv", type=str, default=None,
                   help="Load snapshot from CSV instead of fetching from IB")


def _run_single_expiry(snap, args, run_dir):
    """Run the full pipeline (stages 2-6) for a single OptionsSnapshot.

    Returns (cal_results, avg, rnd).
    """
    from options_pricing.density import extend_call_curve

    # Stage 2 + 3: Calibrate all models
    print("[2-3/6] Calibrating models...")
    cal_results = calibrate_all(snap, maxiter=args.maxiter)
    print()

    # Stage 4: Model averaging
    print("[4/6] Computing BMA weights and averaged prices...")
    df = snap.chains
    avg = average(
        cal_results,
        strikes=df["strike"].values,
        rights=df["right"].values,
    )
    for name, w in sorted(avg.weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.1%}")
    print()

    # Stage 5: Breeden-Litzenberger
    print("[5/6] Extracting risk-neutral density...")
    call_mask = df["right"].values == "C"
    call_strikes = df["strike"].values[call_mask]
    call_avg_prices = avg.prices[call_mask]
    T = snap.dte / 365.0

    observed_range = (float(call_strikes.min()), float(call_strikes.max()))
    ext_strikes, ext_prices = extend_call_curve(
        call_strikes, call_avg_prices,
        S=snap.spot, T=T, r=snap.risk_free_rate,
        cal_results=cal_results, weights=avg.weights,
    )

    rnd = extract_density(ext_strikes, ext_prices, T, snap.risk_free_rate,
                          observed_range=observed_range)
    print(f"  RN Mean: ${rnd.rn_mean:.2f}  RN Std: ${rnd.rn_std:.2f}\n")

    # Stage 6: Visualizations + Report
    print("[6/6] Generating visualizations and report...")
    saved = generate_all(
        snap, cal_results, avg, rnd,
        target=args.target,
        output_dir=run_dir,
    )

    report_path = f"{run_dir}/{snap.ticker}_report.txt"
    write_report(report_path, snap, cal_results, avg, rnd, target=args.target)

    return cal_results, avg, rnd, saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Options pricing with Bayesian Model Averaging"
    )
    sub = parser.add_subparsers(dest="command")

    # --- fetch command (Stage 1) ---
    fetch_p = sub.add_parser("fetch", help="Fetch options chain from IB and display/save")
    _add_common_args(fetch_p)
    fetch_p.add_argument("--save-csv", type=str, default=None,
                         help="Save fetched snapshot to CSV for later use")

    # --- run command (full pipeline: Stages 1-6) ---
    run_p = sub.add_parser("run", help="Full pipeline: fetch -> calibrate -> average -> PDF -> plot")
    _add_common_args(run_p)
    run_p.add_argument("--target", type=float, default=None,
                       help="Your fundamental price target (for the plot)")
    run_p.add_argument("--output-dir", type=str, default="output",
                       help="Directory for all output files (default: output/)")
    run_p.add_argument("--maxiter", type=int, default=300,
                       help="Max iterations for differential evolution")
    run_p.add_argument("--dte-targets", type=str, default=None,
                       help="Comma-separated DTE targets for multi-horizon analysis "
                            "(e.g., 30,60,90,180). Overrides --dte-low/--dte-high.")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        snap = _fetch_or_load(args)
        print(f"Ticker: {snap.ticker}  Spot: {snap.spot:.2f}  "
              f"Expiry: {snap.expiry}  DTE: {snap.dte}")
        calls = snap.chains[snap.chains["right"] == "C"]
        puts = snap.chains[snap.chains["right"] == "P"]
        print(f"Liquid calls: {len(calls)}  Liquid puts: {len(puts)}")
        print(snap.chains.to_string(index=False))
        if args.save_csv:
            save_snapshot(snap, args.save_csv)

    elif args.command == "run":
        from datetime import date

        # Multi-DTE mode
        if args.dte_targets:
            if hasattr(args, "from_csv") and args.from_csv:
                print("Error: --from-csv is not compatible with --dte-targets.")
                print("       Use live IB data for multi-horizon analysis.")
                sys.exit(1)

            dte_targets = [int(x.strip()) for x in args.dte_targets.split(",")]
            print(f"[1/6] Fetching options data for {len(dte_targets)} "
                  f"DTE targets: {dte_targets}...")

            from options_pricing.data import pull_multiple
            snapshots = pull_multiple(
                ticker=args.ticker,
                dte_targets=dte_targets,
                port=args.port,
                risk_free_rate=args.rate,
            )

            base_dir = (f"{args.output_dir}/"
                        f"{snapshots[0].ticker}_{date.today().strftime('%Y%m%d')}")

            all_results = []
            for snap in snapshots:
                print(f"\n{'=' * 60}")
                print(f"  Analyzing {snap.dte} DTE (expiry: {snap.expiry})")
                print(f"  Spot: ${snap.spot:.2f}  |  "
                      f"{len(snap.chains)} liquid options")
                print(f"{'=' * 60}")

                run_dir = f"{base_dir}/{snap.dte}DTE"
                cal, avg, rnd, saved = _run_single_expiry(snap, args, run_dir)
                all_results.append((snap, cal, avg, rnd))

            # Multi-DTE comparison outputs
            print(f"\n{'=' * 60}")
            print("  Generating multi-horizon comparison...")
            print(f"{'=' * 60}")

            from options_pricing.visualization import plot_multi_dte_comparison
            from options_pricing.report import write_multi_dte_summary

            ticker = snapshots[0].ticker
            plot_multi_dte_comparison(
                all_results,
                target=args.target,
                save_path=f"{base_dir}/{ticker}_multi_dte.png",
            )

            write_multi_dte_summary(
                f"{base_dir}/{ticker}_multi_dte_summary.txt",
                all_results,
                target=args.target,
            )

            n_per = 7  # 6 charts + 1 report per expiry
            n_total = len(all_results) * n_per + 2
            print(f"\nDone. {n_total} files written to {base_dir}/")

        # Single-DTE mode (original behavior)
        else:
            print("[1/6] Loading options data...")
            snap = _fetch_or_load(args)
            print(f"  Spot: ${snap.spot:.2f}  Expiry: {snap.expiry}  DTE: {snap.dte}")
            print(f"  {len(snap.chains)} liquid options loaded\n")

            run_dir = (f"{args.output_dir}/"
                       f"{snap.ticker}_{date.today().strftime('%Y%m%d')}")
            _, _, _, saved = _run_single_expiry(snap, args, run_dir)
            print(f"\nDone. {len(saved) + 1} files written to {run_dir}/")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

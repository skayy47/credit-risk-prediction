"""CLI entry point: python -m credit_risk.cli."""
import argparse
import logging
import sys

from credit_risk.data.ingest import run_ingest_validate
from credit_risk.features.dashboard_tables import run_make_dashboard_tables
from credit_risk.features.relational import run_build_features
from credit_risk.modeling.train import run_train_simulate
from credit_risk.modeling.scenarios import run_simulate_scenarios


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _cmd_ingest_validate(args: argparse.Namespace) -> int:
    _configure_logging(getattr(args, "verbose", False))
    try:
        run_ingest_validate()
        return 0
    except (FileNotFoundError, ValueError) as e:
        logging.getLogger("credit_risk").error("Validation failed: %s", e)
        return 1


def _cmd_build_features(args: argparse.Namespace) -> int:
    _configure_logging(getattr(args, "verbose", False))
    try:
        out = run_build_features()
        logging.getLogger("credit_risk").info("Enriched table written: %s", out)
        return 0
    except (FileNotFoundError, ValueError) as e:
        logging.getLogger("credit_risk").error("Feature build failed: %s", e)
        return 1


def _cmd_make_dashboard_tables(args: argparse.Namespace) -> int:
    _configure_logging(getattr(args, "verbose", False))
    try:
        run_make_dashboard_tables()
        return 0
    except (FileNotFoundError, ValueError) as e:
        logging.getLogger("credit_risk").error("Dashboard tables failed: %s", e)
        return 1


def _cmd_train_simulate(args: argparse.Namespace) -> int:
    _configure_logging(getattr(args, "verbose", False))
    try:
        run_train_simulate()
        return 0
    except (FileNotFoundError, ValueError) as e:
        logging.getLogger("credit_risk").error("train-simulate failed: %s", e)
        return 1


def _cmd_simulate_scenarios(args: argparse.Namespace) -> int:
    _configure_logging(getattr(args, "verbose", False))
    try:
        run_simulate_scenarios()
        return 0
    except (FileNotFoundError, ValueError) as e:
        logging.getLogger("credit_risk").error("simulate-scenarios failed: %s", e)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="credit_risk")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_validate_parser = subparsers.add_parser("ingest-validate")
    ingest_validate_parser.set_defaults(func=_cmd_ingest_validate)

    build_features_parser = subparsers.add_parser("build-features")
    build_features_parser.set_defaults(func=_cmd_build_features)

    make_dashboard_parser = subparsers.add_parser("make-dashboard-tables")
    make_dashboard_parser.set_defaults(func=_cmd_make_dashboard_tables)

    train_simulate_parser = subparsers.add_parser("train-simulate")
    train_simulate_parser.set_defaults(func=_cmd_train_simulate)

    simulate_scenarios_parser = subparsers.add_parser("simulate-scenarios")
    simulate_scenarios_parser.set_defaults(func=_cmd_simulate_scenarios)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

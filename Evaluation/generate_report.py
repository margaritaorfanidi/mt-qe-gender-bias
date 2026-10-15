import argparse
from pathlib import Path

from reporting.GenderReporter import GenderReporter
from reporting.GeneralReporter import GeneralReporter


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing evaluation JSON files.",
    )

    parser.add_argument(
        "--report_type",
        type=str,
        choices=["gender", "general"],
        required=True,
        help="Type of report to generate.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset filter.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model filter.",
    )

    return parser.parse_args()


def main():

    args = get_args()

    if args.report_type == "gender":
        reporter = GenderReporter(args.results_dir)

    else:
        reporter = GeneralReporter(args.results_dir)

    table = reporter.create_table(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )

    print(table)


if __name__ == "__main__":
    main()
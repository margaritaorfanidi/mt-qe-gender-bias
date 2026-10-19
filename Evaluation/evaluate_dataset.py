import argparse
from pathlib import Path

import torch

from qe_metrics.CometKiwiQEMetric import CometKiwiQEMetric

from datasets.GenderDataset import GenderDataset
from datasets.GeneralDataset import GeneralDataset

from evaluators.GenderEvaluator import GenderEvaluator
from evaluators.GeneralEvaluator import GeneralEvaluator

from statistic_analysis.GenderStatistics import GenderStatistics
from statistic_analysis.GeneralStatistics import GeneralStatistics

from qe_metrics.QEMetricType import QEMetricType

from utils.Utils import save_experiment_json
# Arguments 


QE_METRIC_REGISTRY = {
    QEMetricType.COMET_KIWI: CometKiwiQEMetric,
}

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", type=Path, required=True, help="Path to the parquet dataset.")

# we have 2 categories of evaluation 
# one on general dataset and one with gender 

    parser.add_argument("--evaluation_type", type=str, choices=["general", "gender"], required=True, help="Type of evaluation.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name, e.g. MLQE-PE, Gambit+, MT-GenEval.")


    parser.add_argument("--metric_name",type=QEMetricType, choices=list(QEMetricType), required=True,help="QE metric to evaluate.",)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional fine-tuned model checkpoint.")

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--output_dir", type=Path, help="Directory to save evaluation results.")
    
    return parser.parse_args()


# Return the available gpu 
def get_gpu():

    if torch.cuda.is_available():

        gpus = 1

        print( f"GPU: {torch.cuda.get_device_name(0)}")

    else:
        gpus = 0
        print("Using CPU")

    return gpus


#load the model we want to evaluate 

def load_metric(metric_type, checkpoint=None):

    qe_metric_class = QE_METRIC_REGISTRY[metric_type]

    if checkpoint is not None:

        if not checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found:\n{checkpoint}")

        print(f"\nLoading finetuned (checkpoint):\n{checkpoint}")

        qe_metric = qe_metric_class.load_from_checkpoint(checkpoint)
        model_id = checkpoint.stem

    else:

        print(f"\nLoading pretrained {metric_type.value}...")

        qe_metric = qe_metric_class.load_from_pretrained()
        model_id = metric_type.value

    return qe_metric, model_id


# Main
def main():

    args = get_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    gpus = get_gpu()

    metric, model_id = load_metric(
        metric_type=args.metric_name,
        checkpoint=args.checkpoint)

    print("\nLoading dataset...")

    if args.evaluation_type == "gender":

        dataset = GenderDataset.from_parquet(args.data_file)

        evaluator = GenderEvaluator(
            metric=metric,
            dataset=dataset,
            batch_size=args.batch_size,
            gpus=gpus,
        )

        result = evaluator.evaluate()

        score_data = GenderStatistics.compute_scores(
            result["score_masc"],
            result["score_fem"],
        )

        summary = GenderStatistics.summarize(score_data)

    else:

        dataset = GeneralDataset.from_parquet(args.data_file)

        evaluator = GeneralEvaluator(
            metric=metric,
            dataset=dataset,
            batch_size=args.batch_size,
            gpus=gpus,
        )

        result = evaluator.evaluate()

        score_data = GeneralStatistics.GeneralStatistics.compute_scores(
            result["predicted_score"],
            dataset.scores,
        )

        summary = GeneralStatistics.GeneralStatistics.summarize(score_data)
       
    # Output filenames

    experiment_name = (
        f"{args.dataset_name}_"
        f"{model_id}_"
        f"{args.evaluation_type}"
    )

    parquet_file = (
        args.output_dir
        / f"{experiment_name}_scored.parquet"
    )

    json_file = (
        args.output_dir
        / f"{experiment_name}.json"
    )


    # Save scored dataset

    result.to_parquet(parquet_file,index=False)


    # Save summary JSON

    save_experiment_json(
        output_file=json_file,
        experiment_name=experiment_name,
        model_name=model_id,
        dataset_name=args.dataset_name,
        evaluation_type=args.evaluation_type,
        statistics=summary,
        checkpoint=args.checkpoint,
    )
    # Print summary

    print(f"\nModel: {model_id}")

    print(f"Dataset: {args.dataset_name}")

    print(f"Evaluation type: {args.evaluation_type}")

    print(f"\nScored dataset saved to:\n{parquet_file}")

    print(f"\nSummary saved to:\n{json_file}")


if __name__ == "__main__":
    main()

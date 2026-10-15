import json
from pathlib import Path


def save_experiment_json(
    output_file,
    experiment_name,
    model_name,
    dataset_name,
    evaluation_type,
    statistics,
    checkpoint=None,
):

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "checkpoint": (
            str(checkpoint)
            if checkpoint is not None
            else None
        ),
        "dataset_name": dataset_name,
        "evaluation_type": evaluation_type,
        "statistics": statistics,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )
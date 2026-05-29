# Referenced from: # Referenced from: https://github.com/gortizji/tangent_task_arithmetic/blob/main/src/eval.py

import numpy as np
import torch
import tqdm
import os

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier

def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, scaling_coef, save_path=""
):
    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    coef_info = evaluate(image_encoder, args)

    if save_path != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(image_encoder.state_dict(), save_path)

    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args
):
    info = {}
    max_coef = getattr(args, 'max_coef', 1.0)
    for scaling_coef in np.linspace(0.0, max_coef, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results


def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
    err_linear = 1 - acc_linear
    err_nonlinear = 1 - acc_nonlinear
    return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)

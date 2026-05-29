#!/usr/bin/env python3

import os
import sys
import argparse
import time
import json
import math
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# COMMAND LINE ARGUMENT PARSING (At the top to allow directory-aware logging)
# ============================================================================

parser = argparse.ArgumentParser(description="Multi-Stage Consensus with Layer Scaling for Machine Unlearning")

# Model & Dataset Setup
parser.add_argument("--model", type=str, default="ViT-B-32",
                    help="Model architecture (e.g., ViT-B-32, ViT-B-16, ViT-L-14)")
parser.add_argument("--forget_dataset", type=str, default="MNIST",
                    help="Dataset to forget (e.g., MNIST, Cars, GTSRB, SVHN)")
parser.add_argument("--retain_dataset", type=str, default="ImageNet",
                    help="Dataset to preserve (e.g., ImageNet)")
parser.add_argument("--finetuning_mode", type=str, default="standard",
                    help="Finetuning mode used for checkpoints (e.g., standard, linear)")

# Paths Configuration (Replacing hardcoded scratch paths)
parser.add_argument("--data_location", type=str, default="./data",
                    help="Directory containing the datasets")
parser.add_argument("--models_dir", type=str, default="./models",
                    help="Directory containing pretrained zero-shot and fine-tuned checkpoints")
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Exact path to the pretrained zero-shot checkpoint (defaults to models_dir/zeroshot_{model}.pt)")
parser.add_argument("--checkpoints_dir", type=str, default=None,
                    help="Exact directory to search for fine-tuned checkpoints (defaults to models_dir/CLIP_MU)")
parser.add_argument("--results_dir", type=str, default="./results",
                    help="Directory to save logs, final models, performance curves, and JSON reports")
parser.add_argument("--cache_dir", type=str, default="./cache",
                    help="Directory for caching openclip models and dataset features")

# Search and Core Method Parameters
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for evaluation")
parser.add_argument("--n_eval_points", type=int, default=51,
                    help="Number of coefficient sweep points")
parser.add_argument("--max_coef", type=float, default=1.0,
                    help="Maximum negation coefficient (alpha) for sweep")
parser.add_argument("--strict_unanimity", action="store_true", 
                    help="Enforce strict unanimity among active models instead of margin consensus")
parser.add_argument("--consensus_ratio", type=float, default=1.0,
                    help="r: fraction for min active models k = max(2, floor(r * n)). 1.0 = unanimous")
parser.add_argument("--min_density", type=float, default=0.05,
                    help="Minimum mask density. Relaxes k_min if density falls below this threshold.")
parser.add_argument("--scale_beta", type=float, default=0.5,
                    help="Beta exponent for layer scaling: gamma = 1 / norm^beta")
parser.add_argument("--penalty_lambda", type=float, default=20.0,
                    help="Penalty weight for retain validation drop in coefficient search")
parser.add_argument("--min_density_gate_ratio", type=float, default=0.35,
                    help="Skip evaluation of a combo if density is below min_density * ratio")

# Refinement & Search Control
parser.add_argument("--use_cd", action="store_true",
                    help="Enable layer-wise scaling coordinate descent refinement")
parser.add_argument("--cd_passes", type=int, default=1,
                    help="Number of CD refinement passes")
parser.add_argument("--n_random_search", type=int, default=80,
                    help="Number of random search iterations (0 = exhaustive grid search)")
parser.add_argument("--random_seed", type=int, default=123,
                    help="Random seed for search sampling reproducibility")
parser.add_argument("--random_phase2_frac", type=float, default=0.35,
                    help="Fraction of random budget reserved for local refinement around top-3 configs")
parser.add_argument("--wandb_project", type=str, default=None,
                    help="Optional Wandb project name for logging metrics and tables")

cmd_args, _ = parser.parse_known_args()

# Setup paths based on arguments
LOG_DIR = os.path.join(cmd_args.results_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"unlearn_{cmd_args.forget_dataset}_{cmd_args.model}_{timestamp}.log")

# ============================================================================
# LOGGING SETUP (Dual logger: stdout + file)
# ============================================================================

class Logger:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def isatty(self):
        return False

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

print(f"Logging to: {log_file}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# DEVICE & ENV SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Disable autograd globally for pure inference pipelines
torch.set_grad_enabled(False)

# Add local directory to Python path to ensure module resolution works
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import dependencies from self-contained src folder
from src.eval import evaluate, evaluate_task_vector_at_coef
from src.datasets.common import maybe_dictionarize
from src.task_vectors import NonLinearTaskVector as _BaseNonLinearTaskVector
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier

import pickle
import wandb

# Safe unpickling setup for task vector loading
class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module not in sys.modules:
                import types
                sys.modules[module] = types.ModuleType(module)
            cls = type(name, (DummyModule,), {'__module__': module})
            setattr(sys.modules[module], name, cls)
            return cls

class SafePickle:
    Unpickler = SafeUnpickler
    Pickler = pickle.Pickler
    PickleError = pickle.PickleError
    @staticmethod
    def load(file, **kwargs):
        return SafeUnpickler(file, **kwargs).load()
    @staticmethod
    def loads(s, **kwargs):
        import io
        return SafeUnpickler(io.BytesIO(s), **kwargs).load()

# Override checkpoint loader to target active device
class NonLinearTaskVector(_BaseNonLinearTaskVector):
    def _load_checkpoint(self, checkpoint):
        return torch.load(checkpoint, map_location=device, weights_only=False, pickle_module=SafePickle)

# ============================================================================
# ARGUMENTS ALIGNMENT
# ============================================================================

class Args:
    pass

args = Args()
args.finetuning_mode = cmd_args.finetuning_mode
args.model = cmd_args.model
args.batch_size = cmd_args.batch_size
args.forget_dataset = cmd_args.forget_dataset
args.retain_dataset = cmd_args.retain_dataset
args.device = device

# Configure directory arguments
args.data_location = cmd_args.data_location
args.cache_dir = cmd_args.cache_dir
args.openclip_cachedir = cmd_args.cache_dir
args.auto_aug = None
args.eval_datasets = None
args.n_eval_points = cmd_args.n_eval_points
args.max_coef = cmd_args.max_coef
args.control_dataset = None
args.seed = None

os.makedirs(args.cache_dir, exist_ok=True)

# Pretrained path discovery
MODELS_DIR = cmd_args.models_dir
pretrained_name = f"zeroshot_{args.model}-anon.pt"
pretrained_name_alt = f"zeroshot_{args.model}.pt"
pretrained_name_generic = "zeroshot.pt"

if cmd_args.pretrained_path:
    PRETRAINED_PATH = cmd_args.pretrained_path
else:
    PRETRAINED_PATH = os.path.join(MODELS_DIR, pretrained_name)

if not os.path.exists(PRETRAINED_PATH):
    fallbacks = [
        os.path.join(MODELS_DIR, pretrained_name_alt),
        os.path.join(MODELS_DIR, pretrained_name_generic),
        os.path.join(current_dir, pretrained_name),
        os.path.join(current_dir, pretrained_name_alt),
        os.path.join(current_dir, pretrained_name_generic)
    ]
    found = False
    for alt in fallbacks:
        if os.path.exists(alt):
            PRETRAINED_PATH = alt
            found = True
            break
    if not found:
        print(f"ERROR: Pretrained zero-shot checkpoint not found. Checked:\n"
              f" - {os.path.join(MODELS_DIR, pretrained_name)}\n" + 
              "\n".join([f" - {p}" for p in fallbacks]))
        sys.exit(1)

print(f"Using pretrained model path: {PRETRAINED_PATH}")

# Checkpoints directory selection
use_loaded = False
forget_ds_upper = args.forget_dataset.upper()
model_upper = args.model.upper()
is_vitl14 = model_upper in ["VIT-L-14", "VIT-L/14", "VITL14", "VIT_L_14"] or ("VIT" in model_upper and "L" in model_upper and "14" in model_upper)

if forget_ds_upper == "MNIST" and model_upper in ["VIT-B-32", "VIT-B-16"]:
    use_loaded = True
elif forget_ds_upper in ["GTSRB", "SUN397", "EUROSAT"] and is_vitl14:
    use_loaded = True

clip_mu_subdir = "results_clipv2_loaded" if use_loaded else "None"

if cmd_args.checkpoints_dir:
    CHECKPOINTS_DIR = cmd_args.checkpoints_dir
else:
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "CLIP_MU", clip_mu_subdir, args.finetuning_mode, args.model)

RESULTS_DIR = cmd_args.results_dir
SAVE_DIR = os.path.join(RESULTS_DIR, args.finetuning_mode, args.model, args.forget_dataset)
os.makedirs(SAVE_DIR, exist_ok=True)

args.save = SAVE_DIR
args.results_db = RESULTS_DIR

# Zero-shot baseline accuracies configuration
acc_filename = f"zeroshot_accuracies_{args.model}.json"
acc_path = os.path.join(MODELS_DIR, acc_filename)
local_acc_path = os.path.join(current_dir, acc_filename)

try:
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            PRETRAINED_ACCURACIES = json.load(f)
    elif os.path.exists(local_acc_path):
        with open(local_acc_path) as f:
            PRETRAINED_ACCURACIES = json.load(f)
    else:
        # Check standard default accuracies filename
        fallback_acc_path = os.path.join(MODELS_DIR, "zeroshot_accuracies.json")
        if os.path.exists(fallback_acc_path):
            with open(fallback_acc_path) as f:
                PRETRAINED_ACCURACIES = json.load(f)
        else:
            print(f"WARNING: Baselines file {acc_filename} not found in {MODELS_DIR} or {current_dir}.")
            print("Using dummy 1.0 (100%) accuracy values for retain target evaluation.")
            PRETRAINED_ACCURACIES = {
                args.retain_dataset + "Val": 1.0,
                args.retain_dataset: 1.0,
                args.forget_dataset + "Val": 1.0,
                args.forget_dataset: 1.0
            }
except Exception as e:
    print(f"ERROR reading accuracies file: {e}")
    sys.exit(1)

# Fine-tuned checkpoint discovery
finetuned_paths = []
print(f"Searching for checkpoints in: {CHECKPOINTS_DIR}")

# Method A: Structured directory search (m x n grid sweep)
for m in range(1, 11):
    for n in range(1, 4):
        p = os.path.join(CHECKPOINTS_DIR, f"checkpoints_rand-m{m}-n{n}-mstd0.5",
                         f"{args.forget_dataset}Val")
        if os.path.isdir(p):
            for file in os.listdir(p):
                if file.endswith('.pt') and 'finetuned' in file.lower():
                    finetuned_paths.append(os.path.join(p, file))

# Method B: Flat/Recursive fallback search
if not finetuned_paths and os.path.isdir(CHECKPOINTS_DIR):
    for root, dirs, files in os.walk(CHECKPOINTS_DIR):
        for file in files:
            if file.endswith('.pt') and ('finetuned' in file.lower() or 'checkpoint' in file.lower()):
                finetuned_paths.append(os.path.join(root, file))

if not finetuned_paths:
    print(f"ERROR: No finetuned checkpoints found under {CHECKPOINTS_DIR}.")
    print("Please verify the directory contains checkpoints (.pt files) matching the search pattern.")
    sys.exit(1)

finetuned_paths.sort()
n_models = len(finetuned_paths)
print(f"Found {n_models} checkpoints to merge.")

# Load pretrained base model weights
base_model = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False, pickle_module=SafePickle)
base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
base_sd = {k[7:] if k.startswith("module.") else k: v for k, v in base_sd.items()}
base_model.eval()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def align_state_dict_keys(pt_sd, ft_sd):
    """Map attention projection weights from older formats to standard PyTorch keys."""
    new_ft_sd = ft_sd.copy()
    pt_attn_prefixes = set()
    for k in pt_sd.keys():
        if k.endswith('.in_proj_weight'):
            pt_attn_prefixes.add(k[:-len('.in_proj_weight')])

    for prefix in pt_attn_prefixes:
        q_w_key = f"{prefix}.Q.weight"
        k_w_key = f"{prefix}.K.weight"
        v_w_key = f"{prefix}.V.weight"
        if q_w_key in new_ft_sd and k_w_key in new_ft_sd and v_w_key in new_ft_sd:
            in_proj_w = torch.cat([new_ft_sd[q_w_key], new_ft_sd[k_w_key], new_ft_sd[v_w_key]], dim=0)
            new_ft_sd[f"{prefix}.in_proj_weight"] = in_proj_w
            del new_ft_sd[q_w_key]
            del new_ft_sd[k_w_key]
            del new_ft_sd[v_w_key]

        q_b_key = f"{prefix}.Q.bias"
        k_b_key = f"{prefix}.K.bias"
        v_b_key = f"{prefix}.V.bias"
        if q_b_key in new_ft_sd and k_b_key in new_ft_sd and v_b_key in new_ft_sd:
            in_proj_b = torch.cat([new_ft_sd[q_b_key], new_ft_sd[k_b_key], new_ft_sd[v_b_key]], dim=0)
            new_ft_sd[f"{prefix}.in_proj_bias"] = in_proj_b
            del new_ft_sd[q_b_key]
            del new_ft_sd[k_b_key]
            del new_ft_sd[v_b_key]

        o_w_key = f"{prefix}.O.weight"
        if o_w_key in new_ft_sd:
            new_ft_sd[f"{prefix}.out_proj.weight"] = new_ft_sd[o_w_key]
            del new_ft_sd[o_w_key]
            
        o_b_key = f"{prefix}.O.bias"
        if o_b_key in new_ft_sd:
            new_ft_sd[f"{prefix}.out_proj.bias"] = new_ft_sd[o_b_key]
            del new_ft_sd[o_b_key]

    return new_ft_sd

def load_state_dict(fpath, base_sd=None):
    sd = torch.load(fpath, map_location=device, weights_only=False, pickle_module=SafePickle)
    if hasattr(sd, 'state_dict'):
        sd = sd.state_dict()
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
        
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    if base_sd is not None:
        sd = align_state_dict_keys(base_sd, sd)
        
    return sd

# ============================================================================
# STAGE 1: compute_average_vector()
# ============================================================================

def compute_average_vector(base_sd, finetuned_paths, float_keys):
    n = len(finetuned_paths)
    avg = {k: torch.zeros_like(base_sd[k], dtype=torch.float32) for k in float_keys}

    for idx, fpath in enumerate(finetuned_paths):
        sys.stdout.write(f"\r  Pass 1: avg vector ... {idx+1}/{n}")
        sys.stdout.flush()

        sd = load_state_dict(fpath, base_sd=base_sd)

        tau_i = {}
        for k in float_keys:
            if k in sd:
                tau_i[k] = (sd[k].to(device).float() - base_sd[k].float())
            else:
                tau_i[k] = torch.zeros_like(base_sd[k], dtype=torch.float32)

        for k in float_keys:
            avg[k] += tau_i[k] / n

        del sd, tau_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  avg vector computed from {n} raw task vectors")
    return avg

# ============================================================================
# STAGE 2: compute_sd_mask_and_merge()
# ============================================================================

def compute_sd_mask_and_merge(base_sd, finetuned_paths, float_keys, avg_vector, candidate_lambdas):
    n = len(finetuned_paths)
    print(f"  Lambdas: {candidate_lambdas}")

    buffers = {}
    for lam in candidate_lambdas:
        buffers[lam] = {
            'sum':          {k: torch.zeros_like(base_sd[k], dtype=torch.float32) for k in float_keys},
            'active_count': {k: torch.zeros_like(base_sd[k], dtype=torch.float32) for k in float_keys},
            'sign_sum':     {k: torch.zeros_like(base_sd[k], dtype=torch.float32) for k in float_keys},
        }

    for idx, fpath in enumerate(finetuned_paths):
        sys.stdout.write(f"\r  Pass 2: SD mask ... {idx+1}/{n}")
        sys.stdout.flush()

        sd = load_state_dict(fpath, base_sd=base_sd)

        tau_i = {}
        for k in float_keys:
            if k in sd:
                tau_i[k] = (sd[k].to(device).float() - base_sd[k].float())
            else:
                tau_i[k] = torch.zeros_like(base_sd[k], dtype=torch.float32)

        for lam in candidate_lambdas:
            for k in float_keys:
                signal = torch.abs(tau_i[k])
                deviation = torch.abs(tau_i[k] - avg_vector[k])
                ratio = signal / (deviation + 1e-8)
                mask = (ratio > lam).float()
                filtered = tau_i[k] * mask

                buffers[lam]['sum'][k]          += filtered
                buffers[lam]['active_count'][k] += mask
                buffers[lam]['sign_sum'][k]     += torch.sign(tau_i[k]) * mask

        del sd, tau_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    return buffers

def _apply_consensus_merge(buf, float_keys, k_min, min_density, total_params, strict_unanimity):
    merged, nonzero = _merge_consensus_only(buf, float_keys, k_min, strict_unanimity)
    density = nonzero / total_params

    if density < min_density and k_min > 1:
        best_merged, best_density, best_k = merged, density, k_min
        for relaxed_k in range(k_min - 1, 0, -1):
            merged_r, nz_r = _merge_consensus_only(buf, float_keys, relaxed_k, strict_unanimity)
            density_r = nz_r / total_params
            if density_r > best_density:
                best_merged, best_density, best_k = merged_r, density_r, relaxed_k
            if density_r >= min_density:
                print(f"    Relaxed k_min {k_min}->{relaxed_k}, "
                      f"density {density*100:.1f}% -> {density_r*100:.1f}%")
                return merged_r, density_r
        print(f"    WARNING: Could not reach {min_density*100:.1f}% density. "
              f"Best: k={best_k}, density={best_density*100:.2f}%")
        return best_merged, best_density

    return merged, density

def _merge_consensus_only(buf, float_keys, k_min, strict_unanimity=False):
    merged = {}
    nonzero = 0
    for k in float_keys:
        active = buf['active_count'][k]
        sign_sum = buf['sign_sum'][k]

        sd_pass = (active >= k_min)
        if strict_unanimity:
            sign_pass = (torch.abs(sign_sum) == active)
        else:
            sign_pass = (torch.abs(sign_sum) >= k_min)
            
        keep = sd_pass & sign_pass

        merged[k] = torch.where(
            keep,
            buf['sum'][k] / active.clamp(min=1),
            torch.zeros_like(buf['sum'][k])
        )
        nonzero += keep.sum().item()

    return merged, nonzero

# ============================================================================
# STAGE 3: apply_layerwise_scaling()
# ============================================================================

def apply_layerwise_scaling(merged_vector, float_keys, beta=0.5):
    layer_groups = {}
    for k in float_keys:
        if "resblocks" in k:
            try:
                lid = int(k.split("resblocks.")[1].split(".")[0])
                layer_groups.setdefault(lid, []).append(k)
            except (ValueError, IndexError):
                layer_groups.setdefault("non_block", []).append(k)
        else:
            layer_groups.setdefault("non_block", []).append(k)

    scaled = {}
    for lid, keys in sorted(layer_groups.items(), key=lambda x: (isinstance(x[0], str), x[0])):
        norm = torch.sqrt(sum(merged_vector[k].float().pow(2).sum() for k in keys)) + 1e-8
        gamma = 1.0 / (norm.item() ** beta)
        for k in keys:
            scaled[k] = merged_vector[k] * gamma
        if isinstance(lid, int):
            print(f"    Block {lid:2d}: norm={norm.item():.6f}, gamma={gamma:.4f} (beta={beta})")

    return scaled

# ============================================================================
# STAGE 4: coefficient_search() (CACHED MODEL PIPELINE)
# ============================================================================

_EVAL_CACHE = {}

def _get_cached_eval_resources(args, pretrained_path):
    cache_key = (args.model, args.forget_dataset, args.retain_dataset)
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    print("  [cache] Loading pretrained model into memory (one-time)...")
    pretrained_model = torch.load(pretrained_path, map_location=device,
                                  weights_only=False, pickle_module=SafePickle)
    pretrained_model.eval()
    pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

    eval_datasets = [args.forget_dataset + "Val", args.retain_dataset + "Val"]
    heads = {}
    dataloaders = {}
    for ds_name in eval_datasets:
        heads[ds_name] = get_classification_head(args, ds_name).to(device)
        heads[ds_name].eval()
        ds_obj = get_dataset(ds_name, pretrained_model.val_preprocess,
                             location=args.data_location, batch_size=args.batch_size)
        dataloaders[ds_name] = ds_obj.test_loader

    _EVAL_CACHE[cache_key] = (pretrained_model, pretrained_sd, heads, dataloaders)
    print("  [cache] Model, classification heads, and dataloaders cached.")
    return pretrained_model, pretrained_sd, heads, dataloaders

def _fast_eval(pretrained_model, pretrained_sd, neg_vec, alpha, heads, dataloaders, eval_datasets):
    with torch.no_grad():
        for name, param in pretrained_model.named_parameters():
            key = name
            if key in neg_vec:
                param.copy_(pretrained_sd[key] + alpha * neg_vec[key])
            elif key in pretrained_sd:
                param.copy_(pretrained_sd[key])

    results = {}
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for ds_name in eval_datasets:
            head = heads[ds_name]
            classifier = ImageClassifier(pretrained_model, head)
            classifier.eval()
            correct, n = 0.0, 0.0
            for data in dataloaders[ds_name]:
                data = maybe_dictionarize(data)
                x = data["images"].to(device)
                y = data["labels"].to(device)
                logits = classifier(x)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)
            results[f"{ds_name}:top1"] = correct / n
    return results

def coefficient_search(scaled_vec, pretrained_path, float_keys, args,
                       pretrained_accs, lam, penalty_lambda=20.0, thresh_ratio=0.95,
                       global_best_f=None):
    retain_ref = pretrained_accs.get(args.retain_dataset + "Val", 0)
    retain_target = thresh_ratio * retain_ref
    print(f"  Retain target: {retain_target*100:.2f}% (min forget s.t. retain >= 95% of zero-shot baseline)")

    eval_datasets = [args.forget_dataset + "Val", args.retain_dataset + "Val"]
    args.eval_datasets = eval_datasets
    args.control_dataset = args.retain_dataset + "Val"

    pretrained_model, pretrained_sd, heads, dataloaders = _get_cached_eval_resources(args, pretrained_path)
    neg_vec = {k: -scaled_vec[k] for k in scaled_vec}

    def _eval_alpha(alpha):
        try:
            m = _fast_eval(pretrained_model, pretrained_sd, neg_vec, alpha, heads, dataloaders, eval_datasets)
        except Exception as e:
            print(f"    ERROR in evaluate: {e}")
            m = {}
        f_val = m.get(f"{args.forget_dataset}Val:top1", 1.0)
        r_val = m.get(f"{args.retain_dataset}Val:top1", 0.0)
        return f_val, r_val

    history = []
    best_a, best_f, best_r = None, 1.0, 0.0

    # Coarse Search
    print(f"\n  Coarse search: alpha in [0.0, {args.max_coef}] with {args.n_eval_points} points")
    for a in np.linspace(0.0, args.max_coef, args.n_eval_points):
        a = round(a, 2)
        if a == 0.0: continue
        f, r = _eval_alpha(a)
        tag = "ok" if r >= retain_target else "XX"
        print(f"    a={a:.2f}: F={f*100:.2f}% R={r*100:.2f}% [{tag}]")
        
        history.append({'alpha': a, 'forget': f, 'retain': r, 'score': f, 'phase': 'coarse', 'lambda': lam})
        if cmd_args.wandb_project and wandb.run:
            wandb.log({
                f"lambda_{lam}/coarse_alpha": a,
                f"lambda_{lam}/coarse_forget": f,
                f"lambda_{lam}/coarse_retain": r,
                "current_lambda": lam,
                "search_phase": "coarse"
            })

        if r >= retain_target and f < best_f:
            best_a, best_f, best_r = a, f, r

    # Fallback score if target is never reached
    if best_a is None:
        print("  WARNING: No alpha satisfied retain threshold in coarse search. Applying penalty fallback.")
        best_score = float('inf')
        for h in history:
            viol = max(0.0, retain_target - h['retain'])
            s = h['forget'] + penalty_lambda * viol
            if s < best_score:
                best_score = s
                best_a, best_f, best_r = h['alpha'], h['forget'], h['retain']

    print(f"  Coarse best: a={best_a}, F={best_f*100:.2f}%, R={best_r*100:.2f}%")

    if global_best_f is not None and best_f >= global_best_f:
        print(f"  SKIP fine search: coarse best F={best_f*100:.2f}% >= global best F={global_best_f*100:.2f}%")
        return best_a, best_f, best_r, history

    # Fine Search
    lo = max(0.02, best_a - 0.2)
    hi = best_a + 0.2
    print(f"\n  Fine search: alpha in [{lo:.2f}, {hi:.2f}] step 0.02")
    steps = int(round((hi - lo) / 0.02)) + 1
    for i in range(steps):
        a = round(lo + i * 0.02, 2)
        f, r = _eval_alpha(a)
        tag = "ok" if r >= retain_target else "XX"
        print(f"    a={a:.2f}: F={f*100:.2f}% R={r*100:.2f}% [{tag}]")
        
        history.append({'alpha': a, 'forget': f, 'retain': r, 'score': f, 'phase': 'fine', 'lambda': lam})
        if cmd_args.wandb_project and wandb.run:
            wandb.log({
                f"lambda_{lam}/fine_alpha": a,
                f"lambda_{lam}/fine_forget": f,
                f"lambda_{lam}/fine_retain": r,
                "current_lambda": lam,
                "search_phase": "fine"
            })

        if r >= retain_target and f < best_f:
            best_a, best_f, best_r = a, f, r

    print(f"  Optimal: a*={best_a}, F={best_f*100:.2f}%, R={best_r*100:.2f}%")
    return best_a, best_f, best_r, history

# ============================================================================
# STAGE 5: coordinate_descent()
# ============================================================================

def coordinate_descent(scaled_vec, pretrained_path, float_keys, args,
                       global_alpha, pretrained_accs, cd_passes=1,
                       penalty_lambda=20.0, thresh_ratio=0.95):
    retain_ref = pretrained_accs.get(args.retain_dataset + "Val", 0)
    retain_target = thresh_ratio * retain_ref

    max_depth = 0
    for k in float_keys:
        if "resblocks" in k:
            try:
                max_depth = max(max_depth, int(k.split("resblocks.")[1].split(".")[0]) + 1)
            except (ValueError, IndexError):
                pass
    if max_depth == 0:
        max_depth = 12

    block_alphas = {i: global_alpha for i in range(max_depth)}
    block_alphas["non_block"] = global_alpha
    a_lo = 0.5 * global_alpha
    a_hi = 1.5 * global_alpha
    print(f"  CD: {max_depth} blocks, {cd_passes} passes, alpha range: [{a_lo:.2f}, {a_hi:.2f}]")

    args.eval_datasets = [args.forget_dataset + "Val"]
    args.control_dataset = args.retain_dataset + "Val"

    key_block_map = {}
    for k in scaled_vec:
        blk = "non_block"
        if "resblocks" in k:
            try:
                blk = int(k.split("resblocks.")[1].split(".")[0])
            except (ValueError, IndexError):
                pass
        key_block_map[k] = blk

    def _eval_blocks(alphas):
        combined = {}
        for k in scaled_vec:
            blk = key_block_map[k]
            a = alphas.get(blk, alphas.get("non_block", global_alpha))
            combined[k] = -a * scaled_vec[k]
        tv = NonLinearTaskVector(vector=combined)
        try:
            m = evaluate_task_vector_at_coef(tv, pretrained_path, args, scaling_coef=1.0)
        except Exception as e:
            print(f"    CD eval error: {e}")
            m = {}
        fa = m.get(f"{args.forget_dataset}Val:top1", 1.0)
        ra = m.get(f"{args.retain_dataset}Val:top1", 0.0)
        return fa, ra

    offsets = [0.0, -0.15, 0.15, -0.3, 0.3]

    for p in range(cd_passes):
        print(f"\n  CD pass {p+1}/{cd_passes}")
        for blk in list(range(max_depth)) + ["non_block"]:
            orig = block_alphas[blk]
            best_score = float('inf')
            best_val = orig

            for off in offsets:
                cand = max(a_lo, min(a_hi, orig + off))
                block_alphas[blk] = cand
                fa, ra = _eval_blocks(block_alphas)
                viol = max(0.0, retain_target - ra)
                score = fa + penalty_lambda * viol
                if score < best_score:
                    best_score = score
                    best_val = cand

            if best_val != orig:
                print(f"    Block {blk}: {orig:.3f} -> {best_val:.3f}")
            block_alphas[blk] = best_val

    fa, ra = _eval_blocks(block_alphas)
    return block_alphas, fa, ra

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    t0 = time.time()
    float_keys = [k for k in base_sd if base_sd[k].is_floating_point()]
    total_params = sum(base_sd[k].numel() for k in float_keys)
    print(f"\nModel: {args.model}, Forget: {args.forget_dataset}, Retain: {args.retain_dataset}")
    print(f"Parameters: {len(float_keys)} tensors, {total_params:,} values, {n_models} checkpoints")

    if cmd_args.wandb_project:
        wandb.init(
            project=cmd_args.wandb_project,
            name=f"consensus_{args.model}_{args.forget_dataset}_{timestamp}",
            config={
                "model": args.model,
                "forget_dataset": args.forget_dataset,
                "retain_dataset": args.retain_dataset,
                "consensus_ratio_candidates": [0.2, 0.4, 0.6, 0.8, 1.0],
                "strict_unanimity_candidates": [False, True],
                "penalty_lambda": cmd_args.penalty_lambda,
                "scale_beta_candidates": [0, 0.3, 0.5, 0.7, 1.0],
                "max_coef": args.max_coef,
                "n_eval_points": args.n_eval_points,
                "use_cd": cmd_args.use_cd,
                "n_random_search": cmd_args.n_random_search,
                "random_seed": cmd_args.random_seed,
                "random_phase2_frac": cmd_args.random_phase2_frac,
                "min_density": cmd_args.min_density,
                "min_density_gate_ratio": cmd_args.min_density_gate_ratio
            }
        )

    # ── STAGE 1: Task Vectors + Normalization + Average ──
    print("\n" + "="*70)
    print("STAGE 1: Task Vectors + Normalization + Average")
    print("="*70)
    avg_vector = compute_average_vector(base_sd, finetuned_paths, float_keys)

    # ── STAGE 2: Signal-to-Deviation Ratio Mask + Consensus ──
    print("\n" + "="*70)
    print("STAGE 2: Signal-to-Deviation Ratio Mask + Consensus")
    print("="*70)
    CANDIDATE_LAMBDAS = [0.4, 0.6, 0.8, 1.0, 1.2]
    
    def _normalize_lambda(val):
        return round(float(val), 2)
    def _normalize_ratio(val):
        return round(float(val), 2)
    def _normalize_beta(val):
        return round(float(val), 2)
        
    CANDIDATE_LAMBDAS = [_normalize_lambda(l) for l in CANDIDATE_LAMBDAS]
    buffers_dict = compute_sd_mask_and_merge(
        base_sd, finetuned_paths, float_keys, avg_vector, CANDIDATE_LAMBDAS
    )
    del avg_vector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── STAGES 3-4: Layer-wise Scaling + Coefficient Search ──
    CANDIDATE_CONSENSUS_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
    CANDIDATE_BETAS = [0, 0.3, 0.5, 0.7, 1.0]
    print("\n" + "="*70)
    print(f"STAGES 3-4: Consensus Ratio x Layer-wise Scaling x Coefficient Search")
    print(f"  Ratios:  {CANDIDATE_CONSENSUS_RATIOS}")
    print(f"  Lambdas: {CANDIDATE_LAMBDAS}")
    print(f"  Betas:   {CANDIDATE_BETAS}")
    print(f"  Strict Unanimity: [False, True]")
    
    import random
    if getattr(cmd_args, 'n_random_search', 0) > 0:
        random.seed(cmd_args.random_seed)

        def _sample_strict_flag():
            return random.random() < 0.35

        def _sample_ratio(strict_flag):
            if strict_flag:
                return random.choices([0.6, 0.8, 1.0], weights=[0.2, 0.4, 0.4])[0]
            return random.choice(CANDIDATE_CONSENSUS_RATIOS)

        def _sample_combo():
            strict_flag = _sample_strict_flag()
            ratio = _normalize_ratio(_sample_ratio(strict_flag))
            beta = _normalize_beta(random.choice(CANDIDATE_BETAS))
            lam = _normalize_lambda(random.choice(CANDIDATE_LAMBDAS))
            return (ratio, beta, lam, strict_flag)

        def _neighbor_combos(base_combo, n_samples):
            ratio, beta, lam, strict_flag = base_combo
            ratio_candidates = sorted(set([ratio, max(0.2, ratio - 0.2), min(1.0, ratio + 0.2)]))
            ratio_candidates = [_normalize_ratio(r) for r in ratio_candidates]
            beta_candidates = sorted(set([beta, max(0.0, beta - 0.2), min(1.0, beta + 0.2)]))
            beta_candidates = [_normalize_beta(b) for b in beta_candidates]
            lam_candidates = sorted(set([
                _normalize_lambda(lam),
                _normalize_lambda(max(0.4, lam - 0.2)),
                _normalize_lambda(min(1.2, lam + 0.2)),
            ]))
            out = []
            for _ in range(n_samples):
                out.append((
                    random.choice(ratio_candidates),
                    random.choice(beta_candidates),
                    random.choice(lam_candidates),
                    strict_flag
                ))
            return out

        n_total = max(1, cmd_args.n_random_search)
        n_phase2 = int(round(n_total * cmd_args.random_phase2_frac))
        n_phase1 = max(1, n_total - n_phase2)

        search_space = []
        seen = set()
        while len(search_space) < n_phase1:
            combo = _sample_combo()
            if combo not in seen:
                seen.add(combo)
                search_space.append(combo)
        print(f"  Grid: RANDOM SEARCH phase1={n_phase1}, phase2={n_phase2}, seed={cmd_args.random_seed}")
    else:
        search_space = [(r, b, l, s)
                        for r in CANDIDATE_CONSENSUS_RATIOS
                        for b in CANDIDATE_BETAS
                        for l in CANDIDATE_LAMBDAS
                        for s in [False, True]]
        print(f"  Grid:    {len(CANDIDATE_CONSENSUS_RATIOS)} x {len(CANDIDATE_LAMBDAS)} x {len(CANDIDATE_BETAS)} x 2 = {len(search_space)} combos")
        
    print("="*70)

    best_ratio = None
    best_lam = None
    best_beta = None
    best_strict = None
    best_alpha = 1.0
    best_f_all = 1.0
    best_r_all = 0.0
    best_vec = None
    best_history = None

    all_histories = []
    summary_data = []

    retain_ref = PRETRAINED_ACCURACIES.get(args.retain_dataset + "Val", 0)
    retain_target = 0.95 * retain_ref

    import copy

    pretrained_model, pretrained_sd, _, _ = _get_cached_eval_resources(args, PRETRAINED_PATH)
    test_eval_datasets = [args.forget_dataset, args.retain_dataset]
    test_heads = {}
    test_dataloaders = {}
    for ds_name in test_eval_datasets:
        test_heads[ds_name] = get_classification_head(args, ds_name).to(device)
        test_heads[ds_name].eval()
        ds_obj = get_dataset(ds_name, pretrained_model.val_preprocess,
                             location=args.data_location, batch_size=args.batch_size)
        test_dataloaders[ds_name] = ds_obj.test_loader
    print(f"  Test dataloaders cached for: {test_eval_datasets}")

    def _run_test_eval(vec, alpha, tag_ratio, tag_beta, tag_lam):
        neg_vec_test = {k: -vec[k] for k in vec}
        tm = _fast_eval(pretrained_model, pretrained_sd, neg_vec_test,
                        alpha, test_heads, test_dataloaders, test_eval_datasets)
        ft = tm.get(f"{args.forget_dataset}:top1", 1.0)
        rt = tm.get(f"{args.retain_dataset}:top1", 0.0)
        print(f"  >> TEST (r={tag_ratio},b={tag_beta},l={tag_lam},a={alpha}): F={ft*100:.2f}%, R={rt*100:.2f}%")
        if cmd_args.wandb_project and wandb.run:
            wandb.log({
                f"test/r{tag_ratio}_b{tag_beta}_l{tag_lam}/forget": ft,
                f"test/r{tag_ratio}_b{tag_beta}_l{tag_lam}/retain": rt,
                "test/latest_forget": ft,
                "test/latest_retain": rt,
            })
        return ft, rt

    def _score_row(row):
        ratio_c, lam_c, beta_c, strict_c, a_c, f_c, r_c = row
        viol = max(0.0, retain_target - r_c)
        return f_c + cmd_args.penalty_lambda * viol

    def _run_search_space(space, start_idx=0):
        nonlocal best_ratio, best_lam, best_beta, best_strict, best_alpha, best_f_all, best_r_all, best_vec, best_history
        for search_idx, (ratio, beta, lam, strict_flag) in enumerate(space, start=start_idx):
            ratio = _normalize_ratio(ratio)
            beta = _normalize_beta(beta)
            lam = _normalize_lambda(lam)
            k_min = max(2, int(math.floor(ratio * n_models)))
            combo_tag = f"r{ratio}_b{beta}_l{lam}_s{int(strict_flag)}"
            print(f"\n{'━'*70}")
            print(f"  Eval {search_idx+1}/{start_idx+len(space)}: ratio={ratio} (k_min={k_min}), beta={beta}, lambda={lam}, strict={strict_flag}")
            print(f"{'━'*70}")

            if lam not in buffers_dict:
                nearest_lam = min(buffers_dict.keys(), key=lambda x: abs(x - lam))
                print(f"  WARNING: lambda {lam} not in buffers; using nearest {nearest_lam}")
                lam = nearest_lam
            buf = buffers_dict[lam]
            merged_vec, density = _apply_consensus_merge(
                buf, float_keys, k_min, cmd_args.min_density, total_params, strict_unanimity=strict_flag
            )
            print(f"  Density: {density*100:.2f}%")

            if density < cmd_args.min_density * cmd_args.min_density_gate_ratio:
                print("  SKIP: density below gate")
                summary_data.append([ratio, lam, beta, strict_flag, None, 1.0, 0.0])
                continue

            scaled_vec = copy.deepcopy(merged_vec)
            if beta > 0:
                scaled_vec = apply_layerwise_scaling(scaled_vec, float_keys, beta=beta)

            if cmd_args.wandb_project and wandb.run:
                wandb.log({"current_ratio": ratio, "current_beta": beta, "current_lambda_eval": lam, "current_strict": strict_flag})

            a, f, r, history = coefficient_search(
                scaled_vec, PRETRAINED_PATH, float_keys, args,
                PRETRAINED_ACCURACIES, lam=lam, penalty_lambda=cmd_args.penalty_lambda,
                global_best_f=best_f_all if best_lam is not None else None,
            )

            for h in history:
                h['beta'] = beta
                h['ratio'] = ratio
                h['strict_unanimity'] = strict_flag
            all_histories.extend(history)
            summary_data.append([ratio, lam, beta, strict_flag, a, f, r])

            if cmd_args.wandb_project and wandb.run:
                wandb.log({
                    f"grid/{combo_tag}/best_alpha": a,
                    f"grid/{combo_tag}/forget": f,
                    f"grid/{combo_tag}/retain": r,
                    "current_ratio": ratio,
                    "current_beta": beta,
                    "current_lambda": lam,
                    "current_strict": strict_flag,
                })

            if r >= retain_target and f < best_f_all:
                best_f_all, best_r_all, best_alpha = f, r, a
                best_ratio, best_lam, best_beta, best_strict = ratio, lam, beta, strict_flag
                best_vec = copy.deepcopy(scaled_vec)
                best_history = history
                _run_test_eval(best_vec, best_alpha, best_ratio, best_beta, best_lam)

            del scaled_vec

    _run_search_space(search_space, start_idx=0)

    if getattr(cmd_args, 'n_random_search', 0) > 0 and n_phase2 > 0:
        valid_rows = [r for r in summary_data if r[4] is not None]
        if valid_rows:
            ranked = sorted(valid_rows, key=lambda r: (_score_row(r), r[5]))
            top_k = ranked[:min(3, len(ranked))]
            local_space = []
            for row in top_k:
                base_combo = (row[0], row[2], row[1], row[3])
                local_space.extend(_neighbor_combos(base_combo, max(1, n_phase2 // len(top_k))))

            local_unique = []
            seen_local = set()
            for combo in local_space:
                if combo not in seen_local:
                    seen_local.add(combo)
                    local_unique.append(combo)
            local_unique = local_unique[:n_phase2]
            if local_unique:
                print(f"\n  Phase 2 local refinement: {len(local_unique)} combos")
                _run_search_space(local_unique, start_idx=len(search_space))

    if best_lam is None:
        print("  WARNING: No combo satisfied retain threshold. Using penalty fallback.")
        best_score_all = float('inf')
        valid_rows = [r for r in summary_data if r[4] is not None]
        if not valid_rows:
            print("  ERROR: All configurations skipped due to low density. Check density parameters.")
            sys.exit(1)
        for row in valid_rows:
            ratio_c, lam_c, beta_c, strict_c, a_c, f_c, r_c = row
            viol = max(0.0, retain_target - r_c)
            s = f_c + cmd_args.penalty_lambda * viol
            if s < best_score_all:
                best_score_all = s
                best_f_all, best_r_all, best_alpha = f_c, r_c, a_c
                best_ratio, best_lam, best_beta, best_strict = ratio_c, lam_c, beta_c, strict_c
                
                k_min_c = max(2, int(math.floor(ratio_c * n_models)))
                merged_vec_c, _ = _apply_consensus_merge(
                    buffers_dict[lam_c], float_keys, k_min_c, cmd_args.min_density, total_params, strict_unanimity=strict_c
                )
                best_vec = copy.deepcopy(merged_vec_c)
                if beta_c > 0:
                    best_vec = apply_layerwise_scaling(best_vec, float_keys, beta=beta_c)

    print(f"\n{'='*70}")
    print(f"GRID SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"  {'Ratio':>6} {'Lambda':>8} {'Beta':>6} {'Strict':>7} {'Alpha':>7} {'Forget':>8} {'Retain':>8} {'Status':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for row in summary_data:
        ratio_c, lam_c, beta_c, strict_c, a_c, f_c, r_c = row
        tag = "✓" if r_c >= retain_target else "✗"
        star = " ★" if ratio_c == best_ratio and lam_c == best_lam and beta_c == best_beta and strict_c == best_strict else ""
        alpha_str = f"{a_c:>7.2f}" if a_c is not None else "   n/a"
        print(f"  {ratio_c:>6} {lam_c:>8} {beta_c:>6} {str(strict_c):>7} {alpha_str} {f_c*100:>7.2f}% {r_c*100:>7.2f}% {tag:>8}{star}")

    if cmd_args.wandb_project and wandb.run:
        summary_table = wandb.Table(
            columns=["ratio", "lambda", "beta", "strict_unanimity", "alpha", "forget", "retain", "satisfies_threshold"],
            data=[[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[6] >= retain_target] for r in summary_data]
        )
        wandb.log({"grid_summary": summary_table})
        coarse_hist = [h for h in all_histories if h.get("phase") == "coarse"]
        if coarse_hist:
            history_table = wandb.Table(
                columns=["ratio", "lambda", "beta", "strict_unanimity", "alpha", "forget", "retain", "score", "retain_delta", "phase", "best_flag"],
                data=[[
                    h.get("ratio"), h.get("lambda"), h.get("beta"), h.get("strict_unanimity", False),
                    h.get("alpha"), h.get("forget"), h.get("retain"), h.get("score"),
                    (h.get("retain") - retain_ref) if h.get("retain") is not None else None,
                    h.get("phase"),
                    bool(h.get("ratio") == best_ratio and h.get("lambda") == best_lam and h.get("beta") == best_beta and h.get("strict_unanimity", False) == best_strict and (h.get("alpha") is not None and math.isclose(h.get("alpha"), best_alpha)))
                ] for h in coarse_hist]
            )
            wandb.log({"coarse_history": history_table})

    print(f"\nBest: ratio={best_ratio}, lam={best_lam}, beta={best_beta}, strict={best_strict}, a*={best_alpha}, F={best_f_all*100:.2f}%, R={best_r_all*100:.2f}%")

    # ── STAGE 5: Coordinate Descent ──
    blk_alphas = None
    if cmd_args.use_cd:
        print("\n" + "="*70)
        print("STAGE 5: Constrained Layer-wise Scaling Coordinate Descent")
        print("="*70)
        blk_alphas, f_final, r_final = coordinate_descent(
            best_vec, PRETRAINED_PATH, float_keys, args,
            best_alpha, PRETRAINED_ACCURACIES, cmd_args.cd_passes,
            penalty_lambda=cmd_args.penalty_lambda,
        )
    else:
        f_final, r_final = best_f_all, best_r_all

    # ── FINAL TEST SET EVALUATION ──
    print(f"\n{'='*70}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*70}")

    f_test, r_test = _run_test_eval(best_vec, best_alpha, best_ratio, best_beta, best_lam)
    retain_ref_test = PRETRAINED_ACCURACIES.get(args.retain_dataset, 0)
    retain_delta = r_test - retain_ref_test

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Validation:")
    print(f"    Forget Accuracy: {100*f_final:.2f}%")
    print(f"    Retain Accuracy: {100*r_final:.2f}%")
    print(f"  Test:")
    print(f"    Forget Accuracy: {100*f_test:.2f}%")
    print(f"    Retain Accuracy: {100*r_test:.2f}%")
    print(f"    Retain Delta:    {100*retain_delta:+.2f}%")
    print(f"  Optimal parameters:")
    print(f"    Consensus Ratio: {best_ratio}")
    print(f"    Ratio Threshold: {best_lam}")
    print(f"    Layer Exponent:  {best_beta}")
    print(f"    Strict Flag:     {best_strict}")
    print(f"    Global Alpha:    {best_alpha}")
    if blk_alphas:
        print(f"    Block Alphas:    {blk_alphas}")

    # Save output unlearned model
    model_save_path = os.path.join(SAVE_DIR, f"unlearned_model_{args.forget_dataset}_{args.model}.pt")
    try:
        if cmd_args.use_cd and blk_alphas:
            key_block_map = {}
            for k in best_vec:
                blk = "non_block"
                if "resblocks" in k:
                    try:
                        blk = int(k.split("resblocks.")[1].split(".")[0])
                    except (ValueError, IndexError):
                        pass
                key_block_map[k] = blk
            combined = {}
            for k in best_vec:
                blk = key_block_map[k]
                a = blk_alphas.get(blk, blk_alphas.get("non_block", best_alpha))
                combined[k] = -a * best_vec[k]
            final_tv = NonLinearTaskVector(vector=combined)
            image_encoder = final_tv.apply_to(PRETRAINED_PATH, scaling_coef=1.0)
        else:
            final_tv = NonLinearTaskVector(vector={k: -best_vec[k] for k in best_vec})
            image_encoder = final_tv.apply_to(PRETRAINED_PATH, scaling_coef=best_alpha)
        torch.save(image_encoder, model_save_path)
        print(f">> Unlearned model saved to: {model_save_path}")
    except Exception as e:
        print(f"ERROR saving unlearned model: {e}")

    # Save metrics JSON report
    res = {
        "forget_dataset": args.forget_dataset,
        "retain_dataset": args.retain_dataset,
        "model": args.model,
        "best_ratio": float(best_ratio) if best_ratio is not None else None,
        "best_lambda": float(best_lam) if best_lam else None,
        "best_beta": float(best_beta) if best_beta is not None else None,
        "best_strict_unanimity": bool(best_strict) if best_strict is not None else None,
        "global_alpha": float(best_alpha),
        "use_cd": cmd_args.use_cd,
        "validation": {
            "forget_accuracy": float(f_final),
            "retain_accuracy": float(r_final),
        },
        "test": {
            "forget_accuracy": float(f_test),
            "retain_accuracy": float(r_test),
            "retain_delta": float(retain_delta),
        }
    }
    
    if cmd_args.wandb_project and wandb.run:
        wandb.log({
            "final_test/forget_accuracy": float(f_test),
            "final_test/retain_accuracy": float(r_test),
            "final_test/retain_delta": float(retain_delta),
            "final_val/forget_accuracy": float(f_final),
            "final_val/retain_accuracy": float(r_final),
        })
        wandb.finish()

    json_path = os.path.join(SAVE_DIR, "unlearning_results.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(res, f, indent=2)
        print(f">> Summary JSON report saved to: {json_path}")
    except Exception as e:
        print(f"Could not save JSON report: {e}")

    # Export histories and plot optimization curve
    try:
        coarse_hist = [h for h in all_histories if h.get("phase") == "coarse"]
        fine_hist = [h for h in all_histories if h.get("phase") == "fine"]
        
        if coarse_hist:
            coarse_path = os.path.join(SAVE_DIR, "coarse_history.csv")
            with open(coarse_path, 'w') as f:
                f.write("ratio,lambda,beta,strict_unanimity,alpha,forget,retain,score,best_flag\n")
                for h in coarse_hist:
                    best_flag = bool(h.get("ratio") == best_ratio and h.get("lambda") == best_lam and h.get("beta") == best_beta and h.get("strict_unanimity", False) == best_strict and (h.get("alpha") is not None and math.isclose(h.get("alpha"), best_alpha)))
                    f.write(f"{h.get('ratio')},{h.get('lambda')},{h.get('beta')},{h.get('strict_unanimity', False)},{h.get('alpha')},{h.get('forget')},{h.get('retain')},{h.get('score')},{best_flag}\n")
            print(f">> Coarse history CSV saved to: {coarse_path}")

        if fine_hist:
            fine_path = os.path.join(SAVE_DIR, "fine_history.csv")
            with open(fine_path, 'w') as f:
                f.write("ratio,lambda,beta,strict_unanimity,alpha,forget,retain,score,best_flag\n")
                for h in fine_hist:
                    best_flag = bool(h.get("ratio") == best_ratio and h.get("lambda") == best_lam and h.get("beta") == best_beta and h.get("strict_unanimity", False) == best_strict and (h.get("alpha") is not None and math.isclose(h.get("alpha"), best_alpha)))
                    f.write(f"{h.get('ratio')},{h.get('lambda')},{h.get('beta')},{h.get('strict_unanimity', False)},{h.get('alpha')},{h.get('forget')},{h.get('retain')},{h.get('score')},{best_flag}\n")
            print(f">> Fine history CSV saved to: {fine_path}")

        if all_histories:
            plt.figure(figsize=(11, 7))
            combos = sorted({(h.get('ratio'), h.get('beta'), h.get('lambda'), h.get('strict_unanimity', False)) for h in all_histories if h.get('phase') == 'coarse'})
            colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(combos))))

            for idx, (ratio_c, beta_c, lam_c, strict_c) in enumerate(combos):
                coarse = [h for h in all_histories if h['phase'] == 'coarse' and h['lambda'] == lam_c and h.get('beta') == beta_c and h.get('ratio') == ratio_c and h.get('strict_unanimity', False) == strict_c]
                if not coarse: continue
                alphas = [h['alpha'] for h in coarse]
                forgets = [h['forget'] * 100 for h in coarse]

                is_best = (lam_c == best_lam and beta_c == best_beta and ratio_c == best_ratio and strict_c == best_strict)
                a_val = 1.0 if is_best else 0.2
                lw = 2.5 if is_best else 0.8

                plt.plot(alphas, forgets, marker='o' if is_best else None, markersize=3,
                         label=f'r={ratio_c}, b={beta_c}, l={lam_c}, s={strict_c}' if is_best else None,
                         color=colors[idx], linestyle='-', alpha=a_val, linewidth=lw)

            best_coarse = [h for h in all_histories if h['phase'] == 'coarse' and h['lambda'] == best_lam and h.get('beta') == best_beta and h.get('ratio') == best_ratio and h.get('strict_unanimity', False) == best_strict]
            if best_coarse:
                alphas_b = [h['alpha'] for h in best_coarse]
                retains_b = [h['retain'] * 100 for h in best_coarse]
                plt.plot(alphas_b, retains_b, marker='s', markersize=3,
                         label=f'Retain (r={best_ratio}, b={best_beta}, l={best_lam}, s={best_strict})',
                         color='darkblue', linestyle='--', linewidth=2)

            plt.axvline(x=best_alpha, color='red', linestyle=':', linewidth=2,
                        label=f'Best alpha={best_alpha}')

            retain_ref_plot = PRETRAINED_ACCURACIES.get(args.retain_dataset + "Val", 0) * 100
            plt.axhline(y=0.95 * retain_ref_plot, color='black', linestyle='-.',
                        alpha=0.7, label='95% Retain Threshold')

            plt.title(f"Consensus Grid Search - {args.model} / {args.forget_dataset}")
            plt.xlabel("Negation Coefficient (alpha)")
            plt.ylabel("Accuracy (%)")
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()

            plot_path = os.path.join(SAVE_DIR, f"unlearning_optimization_curve_{args.forget_dataset}_{args.model}.pdf")
            plt.savefig(plot_path, dpi=300)
            print(f">> Optimization curve saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate performance plots: {e}")

    print(f"\nTotal execution time: {(time.time() - t0)/60:.1f} minutes")
    sys.stdout.close()

if __name__ == "__main__":
    main()

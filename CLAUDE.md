# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase implements **orthogonal ablation** to remove specific concepts from LLM activations. Originally based on "Refusal in LLMs is Mediated by a Single Direction" (Arditi et al., 2024), which removes refusal behavior. **The goal here is to apply the same orthogonalization method to ablate Theory-of-Mind (ToM) and self-other distinction capabilities** using custom datasets.

**Key difference from paper**: The paper uses directional ablation for refusal. This repo applies the same technique to different concepts (ToM, not refusal).

## Running the Pipeline

```bash
# Setup (creates venv, installs packages, prompts for HF and Together AI tokens)
source setup.sh

# Run full pipeline on a model
python3 -m pipeline.run_pipeline --model_path {huggingface_model_path}
# Example: python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

Pipeline outputs go to `pipeline/runs/{model_alias}/`

## Core Method: Orthogonal Ablation

1. **Generate directions** (`generate_directions.py`): Compute difference-in-means between two contrast sets
   - For each layer and token position: `r = mean(high_concept_activations) - mean(low_concept_activations)`

2. **Select direction** (`select_direction.py`): Pick best direction by evaluating:
   - `bypass_score`: How well ablation removes the concept
   - `induce_score`: How well activation addition adds the concept
   - `kl_score`: Distribution shift on neutral examples (lower is better)

3. **Apply intervention**: Two methods
   - **Directional ablation**: Remove direction from activations: `x' = x - r̂(r̂ᵀx)`
   - **Activation addition**: Add direction to activations: `x' = x + r`

## Key Architecture

```
pipeline/
├── config.py                   # Config dataclass (model_path, n_train, etc.)
├── run_pipeline.py             # Main entry point, orchestrates full pipeline
├── submodules/
│   ├── generate_directions.py  # Compute mean activation differences
│   ├── select_direction.py     # Evaluate & select best direction
│   ├── evaluate_jailbreak.py   # String matching, LlamaGuard2 evaluation
│   └── evaluate_loss.py        # CE loss on The Pile, Alpaca
├── model_utils/
│   ├── model_base.py           # Abstract base class for models
│   ├── model_factory.py        # Factory to construct model instances
│   └── {llama2,llama3,gemma,qwen,yi}_model.py  # Model-specific implementations
└── utils/
    ├── hook_utils.py           # PyTorch hooks for ablation/activation addition
    └── utils.py

dataset/
├── load_dataset.py             # Load harmful/harmless splits and processed datasets
├── splits/                     # Train/val/test splits for harmful/harmless
└── processed/                  # JailbreakBench, AdvBench, Alpaca, etc.

tom_dataset/
└── simpletom_contrast_pairs.json  # ToM dataset: high_tom vs low_tom prompts
```

## Custom Datasets for ToM Ablation

- **ToM dataset format** (see `tom_dataset/simpletom_contrast_pairs.json`):
  - `high_tom_prompt`: Requires reasoning about others' mental states (e.g., "Is Mary aware that...")
  - `low_tom_prompt`: Same scenario, factual question (e.g., "Is the following statement true...")
  - Each has `_completion` and `_combined` versions

- To use your dataset instead of harmful/harmless:
  - Modify `load_and_sample_datasets()` in `run_pipeline.py`
  - Replace harmful_train/val with high_tom examples
  - Replace harmless_train/val with low_tom examples

## Model Interface

Each model class (extends `ModelBase`) must implement:
- `_get_tokenize_instructions_fn()`: Chat template formatting
- `_get_eoi_toks()`: End-of-instruction tokens (positions to extract activations)
- `_get_refusal_toks()`: Token IDs for concept detection (for ToM, define relevant tokens)
- `_get_model_block_modules()`: Residual stream input modules
- `_get_attn_modules()`, `_get_mlp_modules()`: Attention/MLP output modules

## Hooks (pipeline/utils/hook_utils.py)

- `get_direction_ablation_input_pre_hook(direction)`: Removes `direction` from activations
- `get_direction_ablation_output_hook(direction)`: Removes `direction` from module outputs
- `get_activation_addition_input_pre_hook(vector, coeff)`: Adds `coeff * vector` to activations
- `get_all_direction_ablation_hooks(model_base, direction)`: Returns hooks for all layers

## Important Implementation Details

- **Positions**: `eoi_toks` are end-of-instruction tokens. Directions extracted from these positions (e.g., -1 = last token, -5 = fifth-to-last)
- **Layers**: Selection filters out top 20% of layers (too close to unembedding)
- **Filtering**: Directions filtered by `kl_threshold=0.1`, `induce_refusal_threshold=0.0`
- **Precision**: Mean activations stored in `float64` to avoid numerical issues
- **Batch size**: Default 32 for activation collection, 8 for generation

## Artifacts Saved

```
pipeline/runs/{model_alias}/
├── generate_directions/
│   └── mean_diffs.pt                    # [n_pos, n_layers, d_model] tensor
├── select_direction/
│   ├── direction_evaluations.json       # All candidate scores
│   ├── direction_evaluations_filtered.json  # Filtered candidates
│   ├── ablation_scores.png              # Visualization
│   ├── actadd_scores.png
│   └── kl_div_scores.png
├── direction.pt                         # Selected direction [d_model]
├── direction_metadata.json              # {"pos": -5, "layer": 12}
├── completions/
│   ├── {dataset}_{baseline|ablation|actadd}_completions.json
│   └── {dataset}_{baseline|ablation|actadd}_evaluations.json
└── loss_evals/
    └── {baseline|ablation|actadd}_loss_eval.json
```

# Theory of Mind Directional Ablation Notebooks

This directory contains Jupyter notebooks implementing directional ablation to remove Theory-of-Mind (ToM) and self-other distinction capabilities from LLMs, based on "Refusal in LLMs is Mediated by a Single Direction" (Arditi et al., 2024).

## 📓 Notebooks

### 1. `simpletom_directional_ablation.ipynb`
**Removes Theory of Mind reasoning capability**

- **Dataset**: SimpleTOM contrast pairs (3,441 examples)
- **Contrast**: `high_tom_prompt` (requires mental state reasoning) vs `low_tom_prompt` (factual questions)
- **Example**:
  - High ToM: "Is Mary aware that the bag has moldy chips?"
  - Low ToM: "Is the following statement true: the bag has moldy chips?"
- **Epistemic Status**: MODERATE (60-70%) - Dataset has clear contrast, but method may not be ideal for cognitive capabilities

### 2. `self_other_directional_ablation.ipynb`
**Removes self-other distinction (first-person vs third-person)**

- **Dataset**: Self-Other pairs (400 examples)
- **Contrast**: `self_subject` (first-person: "I", "my") vs `other_subject` (third-person: "she", "the model")
- **Example**:
  - Self: "I can profile the dataset and deliver a summary."
  - Other: "Anthropic can profile the dataset and deliver a summary."
- **Epistemic Status**: MODERATE-LOW (50-60%) - Small dataset, superficial linguistic distinction

## 🚀 Quick Start

### Prerequisites

```bash
# Run setup script (installs dependencies, creates venv)
source setup.sh

# Or manually install
pip install -r requirements.txt
```

### Running a Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open a notebook** (e.g., `simpletom_directional_ablation.ipynb`)

3. **Configure parameters** in the "Configuration Variables" cell:
   ```python
   MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"  # Change this
   N_TRAIN = 128  # Number of training examples
   N_VAL = 32     # Number of validation examples
   OUTPUT_DIR = "pipeline/runs/simpletom_experiment"
   ```

4. **Run cells sequentially** (Shift+Enter)

### Validation (No GPU Required)

Test notebook structure without running experiments:

```bash
python test_notebooks.py
```

This validates:
- ✅ Notebook JSON structure
- ✅ Dataset loading
- ✅ Required sections present
- ✅ Configuration variables marked in CAPS

## 📊 What Each Notebook Does

### Pipeline Overview

Both notebooks follow the same pipeline:

```
1. Load Dataset
   ↓
2. Split into Train/Val/Test
   ↓
3. Generate Candidate Directions
   (difference-in-means: high_concept - low_concept activations)
   ↓
4. Select Best Direction
   (evaluate using bypass_score, induce_score, kl_score)
   ↓
5. Apply Interventions
   - Baseline (no intervention)
   - Ablation (remove direction)
   - Activation Addition (add direction)
   ↓
6. Evaluate Results
   (compare model outputs with/without intervention)
```

### Key Variables (ALL IN CAPS)

Each notebook has configurable parameters at the top:

- **`MODEL_PATH`**: HuggingFace model path
  - Default: `"meta-llama/Llama-3.2-1B-Instruct"` (small, fast)
  - Alternatives: `"meta-llama/Meta-Llama-3-8B-Instruct"`, `"Qwen/Qwen2.5-32B-Instruct"`

- **`DEVICE`**: `"cuda"` or `"cpu"` (auto-detected)

- **`N_TRAIN`**: Number of training examples (default: 128)
- **`N_VAL`**: Number of validation examples (default: 32)
- **`N_TEST`**: Number of test examples (default: 50)

- **`BATCH_SIZE`**: Batch size for activation collection (default: 32)
- **`KL_THRESHOLD`**: Maximum KL divergence (default: 0.1)

- **`OUTPUT_DIR`**: Where to save artifacts

## 📝 Epistemic Status Sections

Each notebook includes **epistemic status** at key decision points:

- **HIGH (85-95%)**: Dataset loading, splitting, basic operations
- **MODERATE (60-80%)**: Direction generation, method implementation
- **LOW-MODERATE (40-60%)**: Direction selection metrics, token proxies
- **LOW (30-40%)**: Expected effectiveness of interventions

### Key Uncertainties

1. **Method applicability**: Directional ablation was designed for behavioral patterns (refusal), not cognitive capabilities (ToM)
2. **Single direction assumption**: ToM may be distributed across many directions
3. **Token proxies**: No clear equivalent to "refusal tokens" for measuring ToM
4. **Side effects**: Ablation may damage other capabilities

## ⚠️ Requirements

### Hardware
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (16GB+ recommended for larger models)
- **RAM**: 16GB+ system RAM

### Software
- Python 3.8+
- PyTorch with CUDA support
- HuggingFace Transformers
- Jupyter Notebook

### Authentication
Some models require HuggingFace authentication:
```bash
huggingface-cli login
```

## ⏱️ Runtime Expectations

Per notebook execution (model-dependent):

| Step | 1B Model | 8B Model | 32B Model |
|------|----------|----------|-----------|
| Direction Generation | 5-10 min | 10-20 min | 30-60 min |
| Direction Selection | 10-20 min | 20-40 min | 60-120 min |
| Evaluation | 5-10 min | 10-20 min | 20-40 min |
| **Total** | **20-40 min** | **40-80 min** | **110-220 min** |

## 📁 Output Files

Each experiment saves artifacts to `OUTPUT_DIR`:

```
pipeline/runs/{experiment_name}/
├── generate_directions/
│   └── mean_diffs.pt                    # All candidate directions [n_pos, n_layers, d_model]
├── select_direction/
│   ├── direction_evaluations.json       # All candidate scores
│   ├── direction_evaluations_filtered.json  # Filtered candidates
│   └── *.png                            # Visualization plots
├── direction.pt                         # Selected direction [d_model]
└── direction_metadata.json              # {"pos": -5, "layer": 12}
```

## 🧪 Unit Tests

Both notebooks include built-in unit tests:

### SimpleTOM Tests
- ✅ All items have required keys
- ✅ Prompts share same scenario
- ✅ High ToM prompts ask about mental states
- ✅ Low ToM prompts are factual
- ✅ No data leakage between splits

### Self-Other Tests
- ✅ Self versions contain first-person pronouns
- ✅ Other versions avoid first-person
- ✅ Content similarity (same semantic content)
- ✅ No data leakage

## 🤔 Should I Use This Method?

### ✅ When to use directional ablation:
- You want to remove a **behavioral pattern** (e.g., refusal, politeness)
- The behavior has clear linguistic markers
- You have a large dataset (500+ examples)
- You want a lightweight, interpretable intervention

### ⚠️ When to consider alternatives:
- You're targeting a **cognitive capability** (ToM, reasoning)
- The capability is distributed across many features
- You need fine-grained control
- You want to preserve other capabilities

### 🔀 Alternative Methods:
1. **Instruction tuning**: Fine-tune model on desired behavior
2. **Prompt engineering**: Add system prompts
3. **Activation patching**: Identify causal components
4. **Steering vectors**: More flexible than single-direction ablation
5. **Sparse probing**: Find multiple relevant directions

## 📚 References

- **Paper**: Arditi et al. (2024) "Refusal in LLMs is Mediated by a Single Direction"
- **Original repo**: This codebase (originally for refusal ablation)
- **Method**: Orthogonal projection to remove direction from activations

## 🐛 Troubleshooting

### "CUDA out of memory"
- Reduce `BATCH_SIZE` (try 16, 8, or 4)
- Use smaller model (e.g., 1B instead of 8B)
- Reduce `N_TRAIN` / `N_VAL`

### "No directions passed filtering"
- Relax `KL_THRESHOLD` (try 0.15 or 0.2)
- Relax `INDUCE_THRESHOLD` (try -0.1)
- Check if your contrast is strong enough

### "Import errors"
- Run `source setup.sh` to install dependencies
- Verify you're in the correct virtual environment
- Check `pip install -r requirements.txt`

### "Model not found"
- Check model path is correct
- Run `huggingface-cli login` for gated models
- Verify internet connection

## 📧 Questions?

If you have questions about:
- **The method**: See `CLAUDE.md` and `full_paper.md`
- **The code**: Check inline comments in notebooks
- **Epistemic status**: Each major cell has uncertainty notes
- **Configuration**: All variables in CAPS at top of cells

## 🎯 Next Steps

After running experiments:

1. **Evaluate effectiveness**: Did ablation remove the capability?
2. **Check side effects**: Run on reasoning benchmarks
3. **Test transfer**: Does it work on other ToM tasks?
4. **Try alternatives**: Compare with steering vectors, patching
5. **Interpret direction**: Use logit lens, neuron analysis

Good luck! 🚀

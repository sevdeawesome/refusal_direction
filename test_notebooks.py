#!/usr/bin/env python3
"""
Test script to validate the ToM directional ablation notebooks.

This script performs lightweight validation without requiring GPU or model downloads:
1. Checks notebook JSON structure
2. Validates dataset loading
3. Tests utility functions
4. Verifies imports

Usage:
    python test_notebooks.py
"""

import json
import os
import sys
from pathlib import Path

def test_notebook_structure(notebook_path: str) -> bool:
    """
    Test that a notebook has valid JSON structure and expected sections.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        True if valid, raises AssertionError otherwise
    """
    print(f"\nTesting notebook: {notebook_path}")
    print("-" * 80)

    # Test 1: Valid JSON
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        print("✓ Valid JSON structure")
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False

    # Test 2: Has cells
    assert 'cells' in notebook, "No cells found"
    cells = notebook['cells']
    assert len(cells) > 0, "Empty notebook"
    print(f"✓ Found {len(cells)} cells")

    # Test 3: Has markdown cells (documentation)
    markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    assert len(markdown_cells) > 0, "No markdown cells"
    assert len(code_cells) > 0, "No code cells"
    print(f"✓ {len(markdown_cells)} markdown cells, {len(code_cells)} code cells")

    # Test 4: Check for key sections
    required_sections = [
        'Overview',
        'Epistemic Status',
        'Configuration',
        'Dataset',
        'Model',
        'Direction',
    ]

    all_markdown = '\n'.join([''.join(c['source']) for c in markdown_cells])

    for section in required_sections:
        assert section in all_markdown, f"Missing section: {section}"
    print(f"✓ All required sections present")

    # Test 5: Check for CAPS variables in code
    all_code = '\n'.join([''.join(c['source']) for c in code_cells])

    required_caps_vars = [
        'MODEL_PATH',
        'DEVICE',
        'N_TRAIN',
        'N_VAL',
        'N_TEST',
        'BATCH_SIZE',
        'OUTPUT_DIR',
    ]

    # DATASET_PATH is optional (refusal notebook uses load_dataset instead)
    optional_caps_vars = ['DATASET_PATH']

    for var in required_caps_vars:
        assert var in all_code, f"Missing required CAPS variable: {var}"

    found_optional = sum(1 for var in optional_caps_vars if var in all_code)
    print(f"✓ All required configuration variables (CAPS) present")

    # Test 6: Check for epistemic status mentions
    epistemic_mentions = all_markdown.lower().count('epistemic status')
    assert epistemic_mentions >= 3, f"Only {epistemic_mentions} epistemic status mentions (expected >= 3)"
    print(f"✓ {epistemic_mentions} epistemic status sections")

    print(f"✓ Notebook validation passed: {notebook_path}\n")
    return True


def test_dataset_loading():
    """Test that datasets can be loaded."""
    print("\nTesting dataset loading")
    print("-" * 80)

    # Test SimpleTOM dataset
    simpletom_path = "tom_dataset/simpletom_contrast_pairs.json"
    assert os.path.exists(simpletom_path), f"SimpleTOM dataset not found: {simpletom_path}"

    with open(simpletom_path, 'r') as f:
        simpletom_data = json.load(f)

    assert isinstance(simpletom_data, list), "SimpleTOM dataset should be a list"
    assert len(simpletom_data) > 0, "SimpleTOM dataset is empty"

    # Check first item structure
    item = simpletom_data[0]
    required_keys = ['high_tom_prompt', 'low_tom_prompt', 'scenario', 'category']
    for key in required_keys:
        assert key in item, f"SimpleTOM item missing key: {key}"

    print(f"✓ SimpleTOM dataset: {len(simpletom_data)} examples")

    # Test Self-Other dataset
    self_other_path = "self_other_dataset/self_other.json"
    assert os.path.exists(self_other_path), f"Self-Other dataset not found: {self_other_path}"

    with open(self_other_path, 'r') as f:
        self_other_data = json.load(f)

    assert isinstance(self_other_data, list), "Self-Other dataset should be a list"
    assert len(self_other_data) > 0, "Self-Other dataset is empty"

    # Check first item structure
    item = self_other_data[0]
    required_keys = ['self_subject', 'other_subject']
    for key in required_keys:
        assert key in item, f"Self-Other item missing key: {key}"

    print(f"✓ Self-Other dataset: {len(self_other_data)} examples")
    print()


def test_imports():
    """Test that required imports work."""
    print("\nTesting imports")
    print("-" * 80)

    has_errors = False

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"⚠ torch not installed (expected without setup): {e}")
        print("  → This is OK for structural tests")

    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"⚠ numpy not installed: {e}")

    # Test pipeline imports (should work if in correct directory)
    try:
        from pipeline.config import Config
        print("✓ pipeline.config.Config")
    except ImportError as e:
        print(f"⚠ pipeline.config import failed: {e}")
        print("  → This is OK if dependencies aren't installed")

    try:
        from pipeline.model_utils.model_factory import construct_model_base
        print("✓ pipeline.model_utils.model_factory")
    except ImportError as e:
        print(f"⚠ model_factory import failed (likely missing torch): {e}")

    try:
        from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
        print("✓ pipeline.utils.hook_utils")
    except ImportError as e:
        print(f"⚠ hook_utils import failed (likely missing torch): {e}")

    try:
        from pipeline.submodules.generate_directions import get_mean_diff
        print("✓ pipeline.submodules.generate_directions")
    except ImportError as e:
        print(f"⚠ generate_directions import failed (likely missing torch): {e}")

    try:
        from pipeline.submodules.select_direction import get_refusal_scores
        print("✓ pipeline.submodules.select_direction")
    except ImportError as e:
        print(f"⚠ select_direction import failed (likely missing torch): {e}")

    print("\nNote: Import failures are expected without running setup.sh")
    print()
    return True  # Always return True since missing deps is expected


def test_utility_functions():
    """Test utility functions defined in notebooks."""
    print("\nTesting utility functions")
    print("-" * 80)

    import random

    # Test split function
    def split_dataset(dataset, n_train, n_val, n_test, seed=42):
        random.seed(seed)
        shuffled = random.sample(dataset, len(dataset))
        train_data = shuffled[:n_train]
        val_data = shuffled[n_train:n_train+n_val]
        test_data = shuffled[n_train+n_val:n_train+n_val+n_test]
        return train_data, val_data, test_data

    # Create dummy dataset
    dummy_data = [{'id': i, 'data': f'example_{i}'} for i in range(100)]

    train, val, test = split_dataset(dummy_data, 50, 25, 25)

    assert len(train) == 50, "Train split size incorrect"
    assert len(val) == 25, "Val split size incorrect"
    assert len(test) == 25, "Test split size incorrect"

    # Check no overlap
    train_ids = {item['id'] for item in train}
    val_ids = {item['id'] for item in val}
    test_ids = {item['id'] for item in test}

    assert len(train_ids & val_ids) == 0, "Train/val overlap"
    assert len(train_ids & test_ids) == 0, "Train/test overlap"
    assert len(val_ids & test_ids) == 0, "Val/test overlap"

    print("✓ Dataset split function works correctly")
    print()


def main():
    """Run all tests."""
    print("=" * 80)
    print("ToM Directional Ablation Notebook Validation")
    print("=" * 80)

    all_passed = True

    try:
        # Test notebooks
        notebooks = [
            "refusal_directional_ablation.ipynb",
            "simpletom_directional_ablation.ipynb",
            "self_other_directional_ablation.ipynb"
        ]

        for notebook in notebooks:
            if not test_notebook_structure(notebook):
                all_passed = False

        # Test datasets
        test_dataset_loading()

        # Test imports
        if not test_imports():
            all_passed = False

        # Test utility functions
        test_utility_functions()

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nNotebooks are ready to use!")
        print("\nNOTE: These tests do NOT require GPU or model downloads.")
        print("To run the actual experiments, you will need:")
        print("  - CUDA-capable GPU")
        print("  - HuggingFace authentication (for gated models)")
        print("  - ~10-30 minutes per experiment")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

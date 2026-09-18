"""
Quick validation script for GNN-MAPPO implementation.

This script performs basic checks without running the full test suite.
"""

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")

    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy not found")
        return False

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError:
        print("✗ torch not found - install with: pip install torch")
        return False

    try:
        import gymnasium as gym
        print(f"✓ gymnasium")
    except ImportError:
        print("✗ gymnasium not found - install with: pip install gymnasium")
        return False

    return True


def check_file_structure():
    """Check if all required files exist."""
    print("\nChecking file structure...")

    from pathlib import Path

    required_files = [
        "marl/gat_policy.py",
        "marl/mappo_policy.py",
        "marl/maddpg_policy.py",
        "scripts/train_gnn_mappo.py",
        "scripts/run_comparison.py",
        "environments/vascular_3d_marl_env.py",
        "GNN_MAPPO_README.md",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            all_exist = False

    return all_exist


def check_syntax():
    """Check Python syntax of key files."""
    print("\nChecking Python syntax...")

    import py_compile
    from pathlib import Path

    files_to_check = [
        "marl/gat_policy.py",
        "marl/mappo_policy.py",
        "scripts/train_gnn_mappo.py",
    ]

    all_valid = True
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✓ {file_path}")
        except py_compile.PyCompileError as e:
            print(f"✗ {file_path}: {e}")
            all_valid = False

    return all_valid


def print_usage_guide():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("GNN-MAPPO Implementation Validation Complete")
    print("="*80)
    print("\n📚 Quick Start Guide:\n")
    print("1. Install dependencies:")
    print("   pip install torch numpy gymnasium matplotlib")
    print()
    print("2. Run a quick training test:")
    print("   python scripts/train_gnn_mappo.py \\")
    print("     --robots 3 --clots 1 --timesteps 10000 \\")
    print("     --use-gat --device cpu")
    print()
    print("3. Run full training:")
    print("   python scripts/train_gnn_mappo.py \\")
    print("     --robots 3 --clots 3 --timesteps 500000 \\")
    print("     --use-gat --curriculum --device cuda")
    print()
    print("4. Run comparison experiment:")
    print("   python scripts/run_comparison.py \\")
    print("     --robots 3 --clots 3 --timesteps 300000 \\")
    print("     --seeds 3 --device cuda")
    print()
    print("5. For more details, see:")
    print("   cat GNN_MAPPO_README.md")
    print()
    print("="*80)


def main():
    print("="*80)
    print("GNN-MAPPO Implementation Validation")
    print("="*80)
    print()

    # Check imports
    imports_ok = check_imports()

    # Check file structure
    structure_ok = check_file_structure()

    # Check syntax
    syntax_ok = check_syntax()

    # Summary
    print("\n" + "="*80)
    print("Validation Summary")
    print("="*80)
    print(f"Imports:        {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"File Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Syntax Check:   {'✓ PASS' if syntax_ok else '✗ FAIL'}")
    print("="*80)

    if imports_ok and structure_ok and syntax_ok:
        print("\n🎉 All checks passed! Implementation is ready to use.")
        print_usage_guide()
        return True
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        if not imports_ok:
            print("\nInstall missing dependencies with:")
            print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

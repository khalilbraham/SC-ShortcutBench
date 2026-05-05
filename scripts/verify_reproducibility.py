#!/usr/bin/env python3
"""Verify that all reproducibility components are working."""

import sys
from pathlib import Path

def check_results_available():
    """Check if pre-computed results are available."""
    results_dir = Path('/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423')
    tables_dir = results_dir / 'tables'
    
    print("Checking pre-computed results...")
    if not results_dir.exists():
        print("[X] Results directory not found: " + str(results_dir))
        return False
    
    key_files = [
        'downstream_reliance_table.csv',
        'embedding_probe_table.csv',
        'confidence_interval_report.json',
        'generative_reasoning_bias_summary.csv'
    ]
    
    for f in key_files:
        path = tables_dir / f
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print("[OK] {} ({:.1f} MB)".format(f, size_mb))
        else:
            print("[X] Missing: " + f)
            return False
    
    return True

def check_evaluation_scripts():
    """Check if original evaluation scripts were extracted."""
    scripts_dir = Path(__file__).parent
    
    print("\nChecking extracted evaluation scripts...")
    original_scripts = [
        'evaluate_shortcut_predictions_original.py',
        'evaluate_metadata_priors_original.py',
        'evaluate_prompt_intervention_original.py',
        'analyze_geometry_original.py'
    ]
    
    for script in original_scripts:
        path = scripts_dir / script
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print("[OK] {} ({:.1f} KB)".format(script, size_kb))
        else:
            print("[X] Missing: " + script)
            return False
    
    return True

def check_new_modules():
    """Check if new unified modules are available."""
    scripts_dir = Path(__file__).parent
    
    print("\nChecking new unified modules...")
    new_modules = [
        'results_loader.py',
        'evaluate_encoders.py',
        'evaluate_generation.py',
        'evaluate_baselines.py',
        'analysis.py'
    ]
    
    for module in new_modules:
        path = scripts_dir / module
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print("[OK] {} ({:.1f} KB)".format(module, size_kb))
        else:
            print("[X] Missing: " + module)
            return False
    
    return True

def check_syntax():
    """Check Python syntax of all modules."""
    import py_compile
    scripts_dir = Path(__file__).parent
    
    print("\nChecking Python syntax...")
    modules = [
        'results_loader.py',
        'evaluate_encoders.py',
        'evaluate_generation.py',
        'evaluate_baselines.py',
        'analysis.py'
    ]
    
    all_valid = True
    for module in modules:
        try:
            py_compile.compile(str(scripts_dir / module), doraise=True)
            print("[OK] " + module)
        except Exception as e:
            print("[X] {}: {}".format(module, str(e)))
            all_valid = False
    
    return all_valid

def test_results_loader():
    """Test that results_loader can access the data."""
    print("\nTesting results loader...")
    try:
        from results_loader import ResultsLoader
        from pathlib import Path
        
        results_dir = Path('/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423')
        loader = ResultsLoader(results_dir)
        
        downstream = loader.load_downstream_reliance()
        if not downstream.empty:
            print("[OK] Loaded downstream results: {} rows".format(len(downstream)))
        else:
            print("[!] Downstream results table empty")
        
        embeddings = loader.load_embedding_probes()
        if not embeddings.empty:
            print("[OK] Loaded embedding probes: {} rows".format(len(embeddings)))
        else:
            print("[!] Embedding probes table empty")
        
        return True
    except Exception as e:
        print("[X] Results loader failed: " + str(e))
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("SC-ShortcutBench Reproducibility Verification")
    print("=" * 60)
    
    checks = [
        ("Pre-computed results", check_results_available),
        ("Extracted original scripts", check_evaluation_scripts),
        ("New unified modules", check_new_modules),
        ("Python syntax", check_syntax),
        ("Results loader", test_results_loader)
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print("\n[!] {} check failed: {}".format(name, str(e)))
            results[name] = False
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "[OK]" if passed else "[X]"
        print("{} {}".format(status, name))
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\nAll reproducibility components verified!")
        return 0
    else:
        print("\nSome checks failed. See details above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

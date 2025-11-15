#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment script to test current model behavior on Spider Man 2 screenplay.
This helps us understand why the model underestimates violence.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import repair_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import analyze_script_file

def main():
    script_path = "Spider Man 2_0316654_anno.txt"

    print(f"\n{'='*70}")
    print("TESTING CURRENT MODEL ON SPIDER MAN 2")
    print(f"{'='*70}\n")

    result = analyze_script_file(script_path)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Save results for comparison
    output_path = Path(__file__).parent / "current_model_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"Predicted rating: {result['predicted_rating']}")
    print(f"Violence score: {result['aggregated_scores']['violence']}")
    print(f"Gore score: {result['aggregated_scores']['gore']}")
    print(f"\nExpected rating: 12+")
    print(f"Expected violence score: >= 0.4 (to trigger 12+)")
    print(f"\n⚠️  ISSUE: Violence score is too low!")
    print(f"   Current: {result['aggregated_scores']['violence']}")
    print(f"   Needed: >= 0.4 for 12+ rating")

if __name__ == '__main__':
    main()

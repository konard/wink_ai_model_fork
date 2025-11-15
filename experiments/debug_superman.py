#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to understand why Superman is rated 18+ instead of 6+.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import analyze_script_file

# Test Superman
pdf_path = Path(__file__).parent.parent / "Superman_0078346_anno.txt"

print("="*80)
print(f"Debugging Superman rating issue")
print("="*80)

result = analyze_script_file(str(pdf_path))

print(f"\nRating: {result['predicted_rating']}")
print(f"Reasons: {result['reasons']}")
print(f"\nAggregated Scores:")
for key, value in result['aggregated_scores'].items():
    print(f"  {key}: {value}")

print(f"\nTop 5 trigger scenes:")
for i, scene in enumerate(result['top_trigger_scenes'][:5]):
    print(f"\n{'='*60}")
    print(f"Scene {i+1}: {scene['heading'][:60]}")
    print(f"  Weight: {scene['weight']}")
    print(f"  Scores:")
    for key, value in scene['scores'].items():
        if value > 0:
            print(f"    {key}: {value}")
    print(f"  Sample text: {scene['sample_text'][:200]}...")

print("\n" + "="*80)

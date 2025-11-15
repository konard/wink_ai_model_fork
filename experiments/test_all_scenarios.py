#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test script for rating pipeline.
Tests model on both English (.txt) and Russian (.pdf) scenarios.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import repair_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import analyze_script_file


def test_scenario(file_path: str) -> dict:
    """
    Test a single scenario file.

    Args:
        file_path: Path to scenario file

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {Path(file_path).name}")
    print(f"{'='*80}")

    try:
        start_time = datetime.now()
        result = analyze_script_file(file_path)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        print(f"\n✅ SUCCESS")
        print(f"Rating: {result['predicted_rating']}")
        print(f"Reasons: {', '.join(result['reasons'])}")
        print(f"Duration: {duration:.2f}s")

        return {
            'file': file_path,
            'status': 'success',
            'duration': duration,
            'result': result
        }

    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'file': file_path,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Run tests on all scenario files."""

    # Get base directory
    base_dir = Path(__file__).parent.parent

    # Find all English .txt scenario files
    english_scenarios = [
        base_dir / "12 Angry Men_0118528_anno.txt",
        base_dir / "ATM_1603257_anno.txt",
        base_dir / "American History X_0120586_anno.txt",
        base_dir / "Spider Man 2_0316654_anno.txt",
        base_dir / "Superman_0078346_anno.txt"
    ]

    # Find all Russian .pdf scenario files
    russian_scenarios = [
        base_dir / "DG_Topi_seria_1.pdf",
        base_dir / "ПРОСТОКВАШИНО_Дело_о_пропавшей_лопате_для_читки.pdf"
    ]

    # Filter to only existing files
    english_scenarios = [f for f in english_scenarios if f.exists()]
    russian_scenarios = [f for f in russian_scenarios if f.exists()]

    print(f"\n{'='*80}")
    print(f"SCENARIO RATING TEST SUITE")
    print(f"{'='*80}")
    print(f"English scenarios (.txt): {len(english_scenarios)}")
    print(f"Russian scenarios (.pdf): {len(russian_scenarios)}")
    print(f"Total scenarios: {len(english_scenarios) + len(russian_scenarios)}")

    all_results = []

    # Test English scenarios
    if english_scenarios:
        print(f"\n{'#'*80}")
        print(f"# TESTING ENGLISH SCENARIOS")
        print(f"{'#'*80}")

        for scenario_file in english_scenarios:
            result = test_scenario(str(scenario_file))
            all_results.append(result)

    # Test Russian scenarios
    if russian_scenarios:
        print(f"\n{'#'*80}")
        print(f"# TESTING RUSSIAN SCENARIOS (PDF)")
        print(f"{'#'*80}")

        for scenario_file in russian_scenarios:
            result = test_scenario(str(scenario_file))
            all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")

    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = sum(1 for r in all_results if r['status'] == 'failed')

    print(f"Total tests: {len(all_results)}")
    print(f"✅ Passed: {success_count}")
    print(f"❌ Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed tests:")
        for r in all_results:
            if r['status'] == 'failed':
                print(f"  - {Path(r['file']).name}: {r['error']}")

    # Save detailed results to file
    output_file = base_dir / "experiments" / "test_results_all_scenarios.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

    # Exit with error code if any tests failed
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()

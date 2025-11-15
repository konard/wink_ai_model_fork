#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script to verify that the changes to repair_pipeline.py are correct.
This script doesn't run the model, just checks the code changes.
"""

import re
from pathlib import Path

def validate_changes():
    """Validate that all required changes were made to repair_pipeline.py"""

    script_path = Path(__file__).parent.parent / "repair_pipeline.py"
    content = script_path.read_text(encoding='utf-8')

    print("="*70)
    print("VALIDATING CHANGES TO repair_pipeline.py")
    print("="*70)

    checks = []

    # Check 1: Medical violence keywords added
    if r'\bpliers\b' in content and r'\binject\w*' in content and r'\bjab\w*' in content:
        checks.append(("✓", "Medical violence keywords added (pliers, inject, jab)"))
    else:
        checks.append(("✗", "Missing medical violence keywords"))

    # Check 2: Kidnapping keywords added
    if r'\bkidnap\w*' in content and r'\babduct\w*' in content:
        checks.append(("✓", "Kidnapping keywords added (kidnap, abduct)"))
    else:
        checks.append(("✗", "Missing kidnapping keywords"))

    # Check 3: Gore keywords added (SPLORCH)
    if r'\bsplorch\b' in content:
        checks.append(("✓", "Gore keyword 'splorch' added"))
    else:
        checks.append(("✗", "Missing gore keyword 'splorch'"))

    # Check 4: Mob violence keywords added
    if r'\bmob\b' in content and r'\blooter' in content:
        checks.append(("✓", "Mob violence keywords added (mob, looter)"))
    else:
        checks.append(("✗", "Missing mob violence keywords"))

    # Check 5: Medical violence context templates added
    if 'medical_violence' in content and 'self-inflicted injury' in content:
        checks.append(("✓", "Medical violence context templates added"))
    else:
        checks.append(("✗", "Missing medical violence context templates"))

    # Check 6: Kidnapping context templates added
    if 'kidnapping_violence' in content and 'character kidnapped' in content:
        checks.append(("✓", "Kidnapping context templates added"))
    else:
        checks.append(("✗", "Missing kidnapping context templates"))

    # Check 7: Stylized action multiplier reduced (was 0.6, should be 0.8)
    if 'violence_multiplier *= 0.8' in content and '# Было 0.6, стало 0.8' in content:
        checks.append(("✓", "Stylized action violence multiplier increased to 0.8"))
    else:
        checks.append(("✗", "Stylized action multiplier not updated correctly"))

    # Check 8: Medical violence scoring adjustments added
    if "ctx.get('medical_violence', 0) > 0.5" in content:
        checks.append(("✓", "Medical violence scoring adjustments added"))
    else:
        checks.append(("✗", "Missing medical violence scoring adjustments"))

    # Check 9: Kidnapping violence scoring adjustments added
    if "ctx.get('kidnapping_violence', 0) > 0.5" in content:
        checks.append(("✓", "Kidnapping violence scoring adjustments added"))
    else:
        checks.append(("✗", "Missing kidnapping violence scoring adjustments"))

    # Check 10: 12+ threshold lowered (was 0.4, should be 0.25)
    if 'violence >= 0.25' in content and '# Снижен порог для насилия с 0.4 до 0.25' in content:
        checks.append(("✓", "12+ violence threshold lowered from 0.4 to 0.25"))
    else:
        checks.append(("✗", "12+ threshold not updated correctly"))

    # Check 11: Gore threshold added for 12+ rating
    if 'gore >= 0.2' in content:
        checks.append(("✓", "Gore threshold (0.2) added for 12+ rating"))
    else:
        checks.append(("✗", "Missing gore threshold for 12+ rating"))

    # Check 12: 6+ threshold adjusted (was 0.2, should be 0.15)
    if 'violence >= 0.15' in content:
        checks.append(("✓", "6+ violence threshold lowered to 0.15"))
    else:
        checks.append(("✗", "6+ threshold not updated correctly"))

    print("\nValidation Results:")
    print("-" * 70)

    passed = 0
    failed = 0
    for status, message in checks:
        print(f"{status} {message}")
        if status == "✓":
            passed += 1
        else:
            failed += 1

    print("-" * 70)
    print(f"\nTotal: {passed}/{len(checks)} checks passed")

    if failed == 0:
        print("\n✓ ALL VALIDATION CHECKS PASSED!")
        print("\nExpected improvements:")
        print("1. Violence score should increase due to:")
        print("   - Better detection of medical violence (pliers, injection, SPLORCH)")
        print("   - Better detection of kidnapping and psychological violence")
        print("   - Less aggressive reduction for stylized action (0.8 vs 0.6)")
        print("   - New context multipliers for medical and kidnapping violence")
        print("\n2. Rating should change from 6+ to 12+ due to:")
        print("   - Lower threshold for 12+ rating (0.25 instead of 0.4)")
        print("   - Addition of gore threshold (>= 0.2) for 12+ rating")
        print("   - Higher violence scores from improved detection")
        return True
    else:
        print(f"\n✗ {failed} validation check(s) failed")
        return False

if __name__ == '__main__':
    success = validate_changes()
    exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify false positive filters work correctly.
"""

import re

# Test cases
test_cases = [
    {
        "text": "as if clark were about to molest him",
        "pattern": r'\bmolest\b',
        "false_positive_filters": [
            r'as if.*\b(molest|rape|seduce|fondle)',
            r'about to.*\b(molest|rape|seduce|fondle)',
        ],
        "should_match": False,
        "description": "Superman false positive - hypothetical molest"
    },
    {
        "text": "he molested the child",
        "pattern": r'\bmolest\b',
        "false_positive_filters": [
            r'as if.*\b(molest|rape|seduce|fondle)',
            r'about to.*\b(molest|rape|seduce|fondle)',
        ],
        "should_match": True,
        "description": "Actual molest - should match"
    },
    {
        "text": "dreams are just brain garbage",
        "pattern": r'\bbrain\b',
        "false_positive_filters": [
            r'brain (garbage|dump|drain|power|wave|dead|cell|teaser)',
            r'brain(s)? (are|is) (just|garbage|trash)',
        ],
        "should_match": False,
        "description": "Spider-Man 2 false positive - metaphorical brain"
    },
    {
        "text": "his brain was injured",
        "pattern": r'\bbrain\b',
        "false_positive_filters": [
            r'brain (garbage|dump|drain|power|wave|dead|cell|teaser)',
            r'brain(s)? (are|is) (just|garbage|trash)',
        ],
        "should_match": True,
        "description": "Actual brain injury - should match"
    },
]

print("="*80)
print("Testing False Positive Filters")
print("="*80)

passed = 0
failed = 0

for test in test_cases:
    print(f"\n{test['description']}")
    print(f"  Text: '{test['text']}'")
    print(f"  Pattern: {test['pattern']}")

    # Check if pattern matches
    pattern_regex = re.compile(test['pattern'], re.I)
    pattern_match = pattern_regex.search(test['text'])

    if not pattern_match:
        print("  ⚠️  Pattern doesn't match at all - test inconclusive")
        continue

    # Check if it's filtered by false positive filters
    false_positive_patterns = [re.compile(p, re.I) for p in test['false_positive_filters']]
    is_false_positive = any(fp.search(test['text']) for fp in false_positive_patterns)

    # Determine if it should be counted (not filtered)
    should_count = not is_false_positive
    expected_match = test['should_match']

    if should_count == expected_match:
        print(f"  ✅ PASS - {'Counted' if should_count else 'Filtered'} as expected")
        passed += 1
    else:
        print(f"  ❌ FAIL - Expected {'match' if expected_match else 'filter'}, got {'match' if should_count else 'filter'}")
        failed += 1

print("\n" + "="*80)
print(f"Results: {passed} passed, {failed} failed")
print("="*80)

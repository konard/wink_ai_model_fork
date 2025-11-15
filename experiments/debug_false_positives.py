#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to find false positives for nudity and drugs detection.
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import (
    extract_text_from_pdf,
    parse_script_to_scenes,
    NUDITY_WORDS,
    DRUG_WORDS,
    count_pattern_matches
)

# Test PDF parsing
pdf_path = Path(__file__).parent.parent / "DG_Topi_seria_1.pdf"

print("="*80)
print(f"Debugging false positives for: {pdf_path.name}")
print("="*80)

# Extract text
print("\n1. Extracting text from PDF...")
text = extract_text_from_pdf(str(pdf_path))

# Parse scenes
scenes = parse_script_to_scenes(text)
print(f"\nFound {len(scenes)} scenes")

# Check for nudity words
print("\n2. Checking NUDITY_WORDS patterns:")
print(f"   Total patterns: {len(NUDITY_WORDS)}")

nudity_total = 0
for scene in scenes:
    count, excerpts = count_pattern_matches(NUDITY_WORDS, scene['text'])
    if count > 0:
        nudity_total += count
        print(f"\n   Scene {scene['scene_id']}: {scene['heading'][:60]}")
        print(f"      Count: {count}")
        for excerpt in excerpts[:3]:
            print(f"      - {excerpt}")

print(f"\n   Total nudity matches across all scenes: {nudity_total}")

# Check for drug words
print("\n3. Checking DRUG_WORDS patterns:")
print(f"   Total patterns: {len(DRUG_WORDS)}")

drugs_total = 0
for scene in scenes:
    count, excerpts = count_pattern_matches(DRUG_WORDS, scene['text'])
    if count > 0:
        drugs_total += count
        print(f"\n   Scene {scene['scene_id']}: {scene['heading'][:60]}")
        print(f"      Count: {count}")
        for excerpt in excerpts[:3]:
            print(f"      - {excerpt}")

print(f"\n   Total drug matches across all scenes: {drugs_total}")

# Let's manually check for the specific Russian words
print("\n4. Manual check for specific Russian patterns:")

# Check for "нагой", "голый", "обнажен" etc.
manual_patterns = [
    (r'\bголый\b', 'голый (naked)'),
    (r'\bголая\b', 'голая (naked fem)'),
    (r'\bнаг\w*', 'наг* (nude)'),
    (r'\bобнаж\w*', 'обнаж* (expose/nude)'),
    (r'\bбюстгальтер\w*', 'бюстгальтер (bra)'),
    (r'\bтрус\w*', 'трус* (panties/coward)'),
    (r'\bбелье\b', 'белье (underwear)'),
    (r'\bраздева\w*', 'раздева* (undress)'),
]

for pattern, desc in manual_patterns:
    matches = re.findall(pattern, text, re.I)
    if matches:
        print(f"\n   {desc}: {len(matches)} matches")
        # Show context
        regex = re.compile(pattern, re.I)
        found = regex.finditer(text)
        for i, match in enumerate(found):
            if i >= 3:  # Show max 3 examples
                break
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            context = text[start:end].strip().replace('\n', ' ')
            print(f"      Example {i+1}: ...{context}...")

print("\n" + "="*80)

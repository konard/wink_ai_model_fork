#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to find false positives for gore detection.
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import (
    extract_text_from_pdf,
    parse_script_to_scenes,
    GORE_WORDS,
    count_pattern_matches
)

# Test Prostokvashino
pdf_path = Path(__file__).parent.parent / "ПРОСТОКВАШИНО_Дело_о_пропавшей_лопате_для_читки.pdf"

print("="*80)
print(f"Debugging GORE false positives for: {pdf_path.name}")
print("="*80)

# Extract text
print("\n1. Extracting text from PDF...")
text = extract_text_from_pdf(str(pdf_path))

# Parse scenes
scenes = parse_script_to_scenes(text)
print(f"\nFound {len(scenes)} scenes")

# Check for gore words
print("\n2. Checking GORE_WORDS patterns:")
print(f"   Total patterns: {len(GORE_WORDS)}")

for scene in scenes:
    count, excerpts = count_pattern_matches(GORE_WORDS, scene['text'])
    if count > 0:
        print(f"\n   Scene {scene['scene_id']}: {scene['heading'][:60]}")
        print(f"      Count: {count}")
        for excerpt in excerpts[:5]:
            print(f"      - {excerpt}")

# Manual pattern check
print("\n3. Manual pattern check for Russian gore words:")
gore_patterns = [
    r'\bкров\w*',
    r'\bкровав\w*',
    r'\bкровоточ\w*',
    r'\bран\w+',
    r'\bшрам\w*',
]

for pattern in gore_patterns:
    matches = re.findall(pattern, text, re.I)
    if matches:
        print(f"\n   Pattern: {pattern}")
        print(f"   Matches: {matches[:10]}")
        # Show context
        regex = re.compile(pattern, re.I)
        found = regex.finditer(text)
        for i, match in enumerate(found):
            if i >= 3:
                break
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            context = text[start:end].strip().replace('\n', ' ')
            print(f"      Context {i+1}: ...{context}...")

print("\n" + "="*80)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to understand PDF parsing issues.
"""

import sys
from pathlib import Path
import PyPDF2

sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import extract_text_from_pdf, parse_script_to_scenes

# Test PDF parsing
pdf_path = Path(__file__).parent.parent / "DG_Topi_seria_1.pdf"

print("="*80)
print(f"Testing PDF: {pdf_path.name}")
print("="*80)

# Extract text
print("\n1. Extracting text from PDF...")
text = extract_text_from_pdf(str(pdf_path))

print(f"\nTotal text length: {len(text)} characters")
print(f"Total words: {len(text.split())}")

# Show first 500 characters
print("\nFirst 500 characters:")
print(text[:500])
print("\n...")

# Count occurrences of INT. and EXT.
import re
int_count = len(re.findall(r'\bINT\.\s', text, re.I))
ext_count = len(re.findall(r'\bEXT\.\s', text, re.I))
int_ru_count = len(re.findall(r'\bИНТ\.\s', text, re.I))
ext_ru_count = len(re.findall(r'\bЭКСТ\.\s', text, re.I))

print(f"\n2. Scene heading markers found:")
print(f"   INT. (English): {int_count}")
print(f"   EXT. (English): {ext_count}")
print(f"   ИНТ. (Russian): {int_ru_count}")
print(f"   ЭКСТ. (Russian): {ext_ru_count}")

# Try to parse scenes with current logic
print("\n3. Parsing scenes with current parse_script_to_scenes()...")
scenes = parse_script_to_scenes(text)
print(f"   Scenes found: {len(scenes)}")

for i, scene in enumerate(scenes[:5]):
    print(f"\n   Scene {i}:")
    print(f"      Heading: {scene['heading'][:80]}")
    print(f"      Text length: {len(scene['text'])} chars")
    print(f"      Text preview: {scene['text'][:100].replace(chr(10), ' ')}")

# Look for other common Russian scene markers
print("\n4. Looking for other Russian scene markers...")
patterns = [
    (r'\d+\.\s*ИНТ\.\s', 'Numbered ИНТ.'),
    (r'\d+\.\s*ЭКСТ\.\s', 'Numbered ЭКСТ.'),
    (r'\d+\.\s*INT\.\s', 'Numbered INT.'),
    (r'\d+\.\s*EXT\.\s', 'Numbered EXT.'),
    (r'^\d+\.\s+[А-ЯЁ]{2,}', 'Numbered LOCATION (Cyrillic)'),
]

for pattern, desc in patterns:
    matches = re.findall(pattern, text, re.M | re.I)
    if matches:
        print(f"   {desc}: {len(matches)} matches")
        print(f"      Examples: {matches[:3]}")

print("\n" + "="*80)

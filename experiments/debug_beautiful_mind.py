#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to understand why A Beautiful Mind gets violence: 1.0
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import parse_script_to_scenes, extract_scene_features, normalize_and_contextualize_scores
import numpy as np

# Read the script
script_path = Path(__file__).parent.parent / 'dataset' / 'BERT_annotations' / 'A Beautiful Mind_0268978_anno.txt'
txt = script_path.read_text(encoding='utf-8', errors='ignore')

# Parse scenes
scenes = parse_script_to_scenes(txt)
print(f"Total scenes: {len(scenes)}\n")

# Extract features for each scene
all_violence_scores = []
problematic_scenes = []

for scene in scenes[:30]:  # First 30 scenes
    feat = extract_scene_features(scene['text'])
    score = normalize_and_contextualize_scores(feat)

    if score['violence'] > 0.3:  # Flag scenes with significant violence score
        problematic_scenes.append({
            'scene_id': scene['scene_id'],
            'heading': scene['heading'],
            'violence_count': feat['violence_count'],
            'violence_score': score['violence'],
            'length': feat['length'],
            'context': {k: round(v, 3) for k, v in feat['context_scores'].items()},
            'excerpts': feat['violence_excerpts'][:3]
        })

    all_violence_scores.append(score['violence'])

print("="*80)
print(f"Violence scores distribution:")
print(f"Min: {min(all_violence_scores):.3f}")
print(f"Max: {max(all_violence_scores):.3f}")
print(f"Mean: {np.mean(all_violence_scores):.3f}")
print(f"Median: {np.median(all_violence_scores):.3f}")
print(f"80th percentile: {np.percentile(all_violence_scores, 80):.3f}")
print(f"90th percentile: {np.percentile(all_violence_scores, 90):.3f}")
print()

print("="*80)
print(f"Scenes with violence > 0.3 (first 30 scenes):")
print("="*80)

for ps in problematic_scenes:
    print(f"\nScene {ps['scene_id']}: {ps['heading'][:60]}")
    print(f"  Violence count: {ps['violence_count']}")
    print(f"  Length: {ps['length']} words")
    print(f"  Violence score: {ps['violence_score']:.3f}")
    print(f"  Context scores: {ps['context']}")
    if ps['excerpts']:
        print(f"  Excerpts:")
        for i, exc in enumerate(ps['excerpts'], 1):
            print(f"    {i}. {exc[:100]}")

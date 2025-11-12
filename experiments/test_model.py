#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования улучшенной модели рейтингирования сцен.
Тестирует модель на нескольких известных фильмах из датасета.
"""

import sys
import json
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import analyze_script_file

# Тестовые фильмы с ожидаемыми рейтингами
TEST_FILMS = [
    {
        'file': 'A Clockwork Orange_0066921_anno.txt',
        'expected_rating': '18+',
        'description': 'Известен графическим насилием и жестокостью'
    },
    {
        'file': 'A Beautiful Mind_0268978_anno.txt',
        'expected_rating': '12+',
        'description': 'Драма о математике, минимальный контент'
    },
    {
        'file': 'Superman_0078346_anno.txt',
        'expected_rating': '6+',
        'description': 'Супергеройский фильм с экшн-сценами'
    },
    {
        'file': '12 Angry Men_0118528_anno.txt',
        'expected_rating': '12+',
        'description': 'Судебная драма, в основном диалоги'
    },
    {
        'file': '28 Days Later_0289043_anno.txt',
        'expected_rating': '18+',
        'description': 'Хоррор с насилием и кровью'
    }
]

def run_tests():
    """Запускает тесты на выбранных фильмах"""
    print("="*80)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ МОДЕЛИ РЕЙТИНГИРОВАНИЯ")
    print("="*80)
    print()

    dataset_path = Path(__file__).parent.parent / 'dataset' / 'BERT_annotations'
    results = []

    for test_case in TEST_FILMS:
        film_path = dataset_path / test_case['file']

        if not film_path.exists():
            print(f"⚠️  Пропущен: {test_case['file']} (файл не найден)")
            continue

        print(f"\n{'='*80}")
        print(f"📽️  Тестирование: {test_case['file']}")
        print(f"   Описание: {test_case['description']}")
        print(f"   Ожидаемый рейтинг: {test_case['expected_rating']}")
        print(f"{'='*80}\n")

        try:
            result = analyze_script_file(str(film_path))

            predicted = result['predicted_rating']
            expected = test_case['expected_rating']
            match = predicted == expected

            status = "✅ СОВПАДАЕТ" if match else "⚠️  НЕ СОВПАДАЕТ"

            print(f"\n{status}")
            print(f"Предсказано: {predicted}")
            print(f"Ожидалось: {expected}")
            print(f"\nПричины рейтинга:")
            for reason in result['reasons']:
                print(f"  • {reason}")

            if result['evidence_excerpts']:
                print(f"\nПримеры из текста (первые 3):")
                for i, excerpt in enumerate(result['evidence_excerpts'][:3], 1):
                    excerpt_short = excerpt[:150] + '...' if len(excerpt) > 150 else excerpt
                    print(f"  {i}. \"{excerpt_short}\"")

            print(f"\nАгрегированные оценки:")
            for key, value in result['aggregated_scores'].items():
                print(f"  {key}: {value}")

            results.append({
                'file': test_case['file'],
                'expected': expected,
                'predicted': predicted,
                'match': match,
                'reasons': result['reasons'],
                'scores': result['aggregated_scores']
            })

        except Exception as e:
            print(f"❌ ОШИБКА при обработке: {e}")
            import traceback
            traceback.print_exc()

    # Итоговая статистика
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*80}")

    total = len(results)
    matches = sum(1 for r in results if r['match'])
    accuracy = (matches / total * 100) if total > 0 else 0

    print(f"\nВсего тестов: {total}")
    print(f"Совпадений: {matches}")
    print(f"Точность: {accuracy:.1f}%")

    print(f"\nДетали:")
    for r in results:
        status = "✅" if r['match'] else "⚠️"
        print(f"  {status} {r['file']}: {r['expected']} → {r['predicted']}")

    # Сохраняем результаты
    output_path = Path(__file__).parent / 'test_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Результаты сохранены в: {output_path}")
    print()

if __name__ == '__main__':
    run_tests()

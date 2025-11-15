#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Эксперимент: Тестирование улучшенной модели на Spider Man 2

Проблема (issue #19):
- Модель давала рейтинг 6+ для Spider Man 2
- Должна была дать 12+ или 16+
- Агрегированный score насилия был 0.27 вместо высокого значения
- Сцена 160 (self-surgery with knife) имела violence=1.0, gore=1.0, но это игнорировалось

Исправления:
1. Изменена агрегация с percentile на гибридный подход (70% max + 30% p95)
2. Добавлена система рекомендаций для исправления проблемных сцен
3. Скорректированы пороги рейтингов

Результаты:
БЫЛО: 6+, violence=0.27, gore=0.0
СТАЛО: 16+, violence=1.0, gore=0.86
"""

import json
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from repair_pipeline import analyze_script_file


def main():
    script_path = "Spider Man 2_0316654_anno.txt"

    if not Path(script_path).exists():
        print(f"Ошибка: файл '{script_path}' не найден")
        print("Запустите этот скрипт из корневой директории проекта")
        sys.exit(1)

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ МОДЕЛИ НА SPIDER MAN 2")
    print("=" * 70)
    print()

    # Анализируем сценарий
    result = analyze_script_file(script_path)

    # Выводим основные результаты
    print(f"Файл: {result['file']}")
    print(f"Рейтинг: {result['predicted_rating']}")
    print(f"Причины: {', '.join(result['reasons'])}")
    print()

    print("Агрегированные оценки:")
    for key, value in result['aggregated_scores'].items():
        print(f"  {key}: {value:.3f}")
    print()

    print(f"Всего сцен: {result['total_scenes']}")
    print(f"Проблемных сцен (топ-{len(result['top_trigger_scenes'])}): {len(result['top_trigger_scenes'])}")
    print()

    # Выводим детали по самой проблемной сцене
    if result['top_trigger_scenes']:
        top_scene = result['top_trigger_scenes'][0]
        print("=" * 70)
        print("САМАЯ ПРОБЛЕМНАЯ СЦЕНА:")
        print("=" * 70)
        print(f"ID: {top_scene['scene_id']}")
        print(f"Заголовок: {top_scene['heading']}")
        print(f"Вес: {top_scene['weight']}")
        print()
        print("Оценки:")
        for key, value in top_scene['scores'].items():
            if value > 0:
                print(f"  {key}: {value}")
        print()
        print("Фрагмент:")
        print(f"  {top_scene['sample_text'][:200]}...")
        print()
        print("Рекомендации по исправлению:")
        for rec in top_scene['recommendations']:
            print(f"  • {rec}")

    print()
    print("=" * 70)
    print("ПРОВЕРКА ИСПРАВЛЕНИЯ БАГА")
    print("=" * 70)

    # Проверяем, что бага больше нет
    expected_rating_options = ['12+', '16+', '18+']
    if result['predicted_rating'] in expected_rating_options:
        print(f"✅ УСПЕХ: Рейтинг {result['predicted_rating']} (ожидался 12+, 16+ или 18+)")
    else:
        print(f"❌ ОШИБКА: Рейтинг {result['predicted_rating']} (ожидался 12+, 16+ или 18+)")

    if result['aggregated_scores']['violence'] >= 0.5:
        print(f"✅ УСПЕХ: Насилие {result['aggregated_scores']['violence']:.3f} (было 0.27)")
    else:
        print(f"❌ ОШИБКА: Насилие всё ещё слишком низкое: {result['aggregated_scores']['violence']:.3f}")

    if result['aggregated_scores']['gore'] >= 0.3:
        print(f"✅ УСПЕХ: Кровь/увечья {result['aggregated_scores']['gore']:.3f} (было 0.0)")
    else:
        print(f"❌ ОШИБКА: Кровь/увечья всё ещё слишком низкие: {result['aggregated_scores']['gore']:.3f}")

    # Проверяем наличие рекомендаций
    has_recommendations = all('recommendations' in scene for scene in result['top_trigger_scenes'])
    if has_recommendations:
        print(f"✅ УСПЕХ: Рекомендации добавлены для всех проблемных сцен")
    else:
        print(f"❌ ОШИБКА: Рекомендации отсутствуют")

    # Сохраняем полный результат
    output_file = "experiments/spider_man_2_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print()
    print(f"Полный результат сохранен в: {output_file}")


if __name__ == '__main__':
    main()

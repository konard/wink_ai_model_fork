# -*- coding: utf-8 -*-
"""
Улучшенная модель рейтингирования сцен с контекстным анализом.
Модель использует семантические эмбеддинги для понимания контекста
и избегает ложных срабатываний при простом поиске ключевых слов.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ===== REFERENCE CONTEXTS FOR SEMANTIC ANALYSIS =====
# Контекстные шаблоны для определения типа сцен
CONTEXT_TEMPLATES = {
    'graphic_violence': [
        "brutal murder with blood and gore",
        "torture and physical violence causing injury",
        "graphic depiction of death and killing",
        "violent assault with weapons causing harm"
    ],
    'stylized_action': [
        "heroic action scene with combat",
        "adventure movie fight sequence",
        "comic book style action without gore",
        "spy thriller chase and combat",
        "superhero saving people from danger"
    ],
    'sexual_content': [
        "explicit sexual intercourse scene",
        "nudity in sexual context",
        "rape or sexual assault",
        "graphic sexual activity"
    ],
    'mild_romance': [
        "romantic kissing and affection",
        "love scene without explicit content",
        "romantic relationship development"
    ],
    'horror_violence': [
        "horror movie with scary violence",
        "psychological terror and fear",
        "monster attack with blood",
        "slasher film with killing"
    ],
    'profanity_context': [
        "casual conversation with swearing",
        "aggressive confrontation with profanity",
        "repeated use of strong language"
    ],
    'drug_abuse': [
        "drug use and addiction",
        "substance abuse scene",
        "characters taking illegal drugs"
    ],
    'child_endangerment': [
        "child in dangerous situation",
        "violence involving minors",
        "child abuse or threat to children"
    ],
    'discussion_violence': [
        "courtroom discussion of crime",
        "testimony about violent event",
        "describing past violence in dialogue",
        "academic or legal discussion of weapons",
        "demonstration or explanation without action"
    ],
    'thriller_tension': [
        "psychological thriller with suspense",
        "tense dramatic confrontation",
        "mystery investigation without violence",
        "courtroom drama legal arguments"
    ]
}

# ===== KEYWORD PATTERNS =====
VIOLENCE_WORDS = [
    r'\bkill\w*', r'\bshoot\w*', r'\bshot\b', r'\bstab\w*',
    r'\bknife\b', r'\bgun\w*', r'\bpistol\b', r'\brifle\b',
    r'\bexplod\w*', r'\bblast\w*', r'\battack\w*', r'\bbeat\w*',
    r'\bcorpse\b', r'\bdead\b', r'\bmurder\w*', r'\bviolence\b',
    r'\bterrorist\b', r'\bhostage\b', r'\brip(ped|s)? apart\b',
    r'\bthug(s)?\b', r'\bterror\b', r'\bfight(ing)?\b',
    r'\bbattle(s|d)?\b', r'\bwar\b', r'\bshoot[- ]?out\b',
    r'\bexplosion\b', r'\bgrenade\b'
]

GORE_WORDS = [
    r'\bblood\b', r'\bbloody\b', r'\bbloodied\b', r'\bbleeding\b',
    r'\bcorpse\b', r'\bwound\b', r'\bscar\b', r'\binjur\w*',
    r'\bcrash\w*', r'\bburn\w*', r'\bguts\b', r'\bentrails\b',
    r'\bbrain\b', r'\bdead body\b', r'\bgore\b', r'\bmutilat\w*'
]

PROFANITY = [
    r'\bfuck\b', r'\bshit\b', r'\bmotherfucker\b', r'\bbitch\b',
    r'\basshole\b', r'\bdamn\b', r'\bhell\b', r'\bcrap\b'
]

DRUG_WORDS = [
    r'\bdrug(s)?\b', r'\bheroin\b', r'\bcocaine\b', r'\bmarijuana\b',
    r'\bpill(s)?\b', r'\bweed\b', r'\balcohol\b', r'\bdrunk\b',
    r'\bcigarette\b', r'\bsmok(e|ing)\b', r'\baddiction\b'
]

CHILD_WORDS = [
    r'\bchild(ren)?\b', r'\bkid(s)?\b', r'\bson\b', r'\bdaughter\b',
    r'\bteen(aged)?\b', r'\bboy\b', r'\bgirl\b', r'\bminor\b'
]

NUDITY_WORDS = [
    r'\bbra\b', r'\bpanty|panties\b', r'\bunderwear\b', r'\bnaked\b',
    r'\bnude\b', r'\bundress\w*', r'\btopless\b'
]

SEX_WORDS = [
    r'\brape\b', r'\bsexual\b', r'\bintercourse\b', r'\bsex scene\b',
    r'\bmolest\b', r'\borgasm\b', r'\bmake love\b', r'\bhaving sex\b',
    r'\bsexually\b', r'\bbed\s+scene\b'
]

# ===== INITIALIZATION =====
print("Загрузка модели эмбеддингов...")
MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# Предвычисляем эмбеддинги для контекстных шаблонов
print("Предвычисление контекстных эмбеддингов...")
context_embeddings = {}
for context_type, templates in CONTEXT_TEMPLATES.items():
    context_embeddings[context_type] = embedder.encode(
        templates,
        convert_to_numpy=True,
        show_progress_bar=False
    )
print("Модель готова к использованию.\n")


def count_pattern_matches(patterns: List[str], text: str) -> Tuple[int, List[str]]:
    """
    Подсчитывает совпадения паттернов и возвращает найденные фрагменты.
    Фильтрует ложные срабатывания от фигуральных выражений.

    Returns:
        (count, matched_excerpts)
    """
    # Фразы-исключения, которые не считаются за реальное насилие/контент
    FALSE_POSITIVES = [
        r'if (it|that|this) kills',
        r'(it|that|this)\'ll kill',
        r'make love',  # Неэксплицитное выражение
        r'kill time',
        r'dressed to kill',
        r'killer instinct',
        r'lady killer',
        r'killing me softly',
        r'shoot the breeze',
        r'shoot for',
        r'shot in the dark',
        r'long shot',
        r'fight (for|to see|to|for the)',  # Метафора борьбы
        r'won the war',  # Метафора победы
        r'battles? (with|against|for)',  # Метафорическая борьба
        r'attack(ed|ing)? (the|a) problem',
        r'shot at',  # Попытка/шанс
        r'speed of light',  # Физическое описание
        r'fight back tears',
        r'fight for (justice|freedom|rights)',
        r'fighting? (cancer|disease|illness)'
    ]

    false_positive_patterns = [re.compile(p, re.I) for p in FALSE_POSITIVES]

    matches = []
    count = 0
    for pattern in patterns:
        regex = re.compile(pattern, re.I)
        found = regex.finditer(text)
        for match in found:
            # Извлекаем контекст вокруг совпадения (50 символов до и после)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            excerpt = text[start:end].strip()

            # Проверяем, не является ли это ложным срабатыванием
            is_false_positive = any(fp.search(excerpt) for fp in false_positive_patterns)

            if not is_false_positive:
                matches.append(excerpt)
                count += 1
    return count, matches[:5]  # Возвращаем до 5 примеров


def analyze_scene_context(scene_text: str) -> Dict[str, float]:
    """
    Анализирует контекст сцены с использованием семантических эмбеддингов.
    Возвращает оценки сходства с различными типами контекстов.
    """
    # Получаем эмбеддинг сцены
    scene_embedding = embedder.encode(
        [scene_text],
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]

    # Вычисляем сходство с каждым типом контекста
    context_scores = {}
    for context_type, template_embeddings in context_embeddings.items():
        # Вычисляем косинусное сходство с каждым шаблоном
        similarities = util.cos_sim(scene_embedding, template_embeddings)[0]
        # Берем максимальное сходство
        context_scores[context_type] = float(similarities.max())

    return context_scores


def extract_scene_features(scene_text: str) -> Dict[str, Any]:
    """
    Извлекает признаки из текста сцены, включая подсчет ключевых слов
    и примеры найденных фрагментов.
    """
    txt = scene_text.lower()

    # Подсчитываем совпадения и собираем примеры
    violence_count, violence_excerpts = count_pattern_matches(VIOLENCE_WORDS, txt)
    gore_count, gore_excerpts = count_pattern_matches(GORE_WORDS, txt)
    profanity_count, profanity_excerpts = count_pattern_matches(PROFANITY, txt)
    drugs_count, drugs_excerpts = count_pattern_matches(DRUG_WORDS, txt)
    child_count, child_excerpts = count_pattern_matches(CHILD_WORDS, txt)
    nudity_count, nudity_excerpts = count_pattern_matches(NUDITY_WORDS, txt)
    sex_count, sex_excerpts = count_pattern_matches(SEX_WORDS, txt)

    # Получаем контекстные оценки
    context_scores = analyze_scene_context(scene_text)

    length = max(1, len(txt.split()))

    return {
        'violence_count': violence_count,
        'violence_excerpts': violence_excerpts,
        'gore_count': gore_count,
        'gore_excerpts': gore_excerpts,
        'profanity_count': profanity_count,
        'profanity_excerpts': profanity_excerpts,
        'drugs_count': drugs_count,
        'drugs_excerpts': drugs_excerpts,
        'child_count': child_count,
        'child_excerpts': child_excerpts,
        'nudity_count': nudity_count,
        'nudity_excerpts': nudity_excerpts,
        'sex_count': sex_count,
        'sex_excerpts': sex_excerpts,
        'length': length,
        'context_scores': context_scores
    }


def normalize_and_contextualize_scores(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Нормализует признаки и применяет контекстную коррекцию.
    Использует семантический анализ для корректировки оценок.
    """
    L = features['length']
    ctx = features['context_scores']

    # Базовая нормализация по длине сцены
    # Увеличиваем знаменатель чтобы снизить ложные срабатывания
    violence_raw = features['violence_count'] / (L / 200)
    gore_raw = features['gore_count'] / (L / 150)

    # КОНТЕКСТНАЯ КОРРЕКЦИЯ с использованием семантики

    violence_multiplier = 1.0
    gore_multiplier = 1.0

    # Если это обсуждение/демонстрация насилия, а не реальное действие
    if ctx['discussion_violence'] > 0.55 or ctx['thriller_tension'] > 0.5:
        violence_multiplier *= 0.3  # Сильно снижаем
        gore_multiplier *= 0.3

    # Если сцена больше похожа на стилизованный экшн, снижаем оценку насилия
    elif ctx['stylized_action'] > 0.5:
        violence_multiplier *= 0.6
        gore_multiplier *= 0.7

    # Если сцена похожа на графическое насилие, увеличиваем оценку
    if ctx['graphic_violence'] > 0.6:
        violence_multiplier *= 1.3
        gore_multiplier *= 1.4

    # Если сцена похожа на хоррор, корректируем оценки
    if ctx['horror_violence'] > 0.55:
        violence_multiplier *= 1.2
        gore_multiplier *= 1.3

    violence_score = min(1.0, violence_raw * violence_multiplier)
    gore_score = min(1.0, gore_raw * gore_multiplier)

    # Сексуальный контент - если есть явные признаки
    sex_raw = features['sex_count']
    if ctx['sexual_content'] > 0.6 and sex_raw > 0:
        sex_score = min(1.0, sex_raw * 1.5)
    elif ctx['mild_romance'] > 0.5:
        sex_score = min(0.3, sex_raw * 0.5)  # Мягкая романтика
    else:
        sex_score = min(1.0, sex_raw)

    # Нагота
    nudity_score = min(1.0, features['nudity_count'] / 3.0)

    # Ненормативная лексика
    profanity_score = min(1.0, features['profanity_count'] / (L / 100))

    # Наркотики
    if ctx['drug_abuse'] > 0.55:
        drugs_score = min(1.0, features['drugs_count'] / 2.0)
    else:
        drugs_score = min(1.0, features['drugs_count'] / 5.0)

    # Риск для детей
    child_risk = 0.0
    if features['child_count'] > 0:
        if ctx['child_endangerment'] > 0.5:
            child_risk = min(1.0, features['child_count'] / 2.0)
        else:
            child_risk = min(0.5, features['child_count'] / 5.0)

    return {
        'violence': violence_score,
        'gore': gore_score,
        'sex_act': sex_score,
        'nudity': nudity_score,
        'profanity': profanity_score,
        'drugs': drugs_score,
        'child_risk': child_risk,
        'context_scores': ctx,
        'excerpts': {
            'violence': features['violence_excerpts'],
            'gore': features['gore_excerpts'],
            'sex': features['sex_excerpts'],
            'profanity': features['profanity_excerpts'],
            'drugs': features['drugs_excerpts']
        }
    }


def map_scores_to_rating(agg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует агрегированные оценки в возрастной рейтинг (0+, 6+, 12+, 16+, 18+).
    Возвращает рейтинг и причины с примерами из текста.
    """
    reasons = []
    excerpts = []
    rating = '0+'

    # 18+ - эксплицитный контент
    if agg['sex_act'] >= 0.6 or agg['gore'] >= 0.6:
        rating = '18+'
        if agg['sex_act'] >= 0.6:
            reasons.append("эксплицитные сцены сексуального характера")
            if agg['excerpts']['sex']:
                excerpts.extend(agg['excerpts']['sex'][:2])
        if agg['gore'] >= 0.6:
            reasons.append("графическое изображение жестокости и крови")
            if agg['excerpts']['gore']:
                excerpts.extend(agg['excerpts']['gore'][:2])

    # 18+ - дети в опасности с насилием
    elif agg['child_risk'] > 0.7 and (agg['sex_act'] >= 0.4 or agg['violence'] >= 0.7):
        rating = '18+'
        reasons.append("опасные или жестокие сцены с участием несовершеннолетних")
        if agg['excerpts']['violence']:
            excerpts.extend(agg['excerpts']['violence'][:2])

    # 16+ - явное насилие
    elif agg['violence'] >= 0.7 or agg['gore'] >= 0.45:
        rating = '16+'
        if agg['violence'] >= 0.7:
            reasons.append("интенсивное насилие и сцены убийств")
            if agg['excerpts']['violence']:
                excerpts.extend(agg['excerpts']['violence'][:2])
        if agg['gore'] >= 0.45:
            reasons.append("изображение крови и телесных повреждений")
            if agg['excerpts']['gore']:
                excerpts.extend(agg['excerpts']['gore'][:2])

    # 16+ - сексуальный контент средней степени
    elif agg['sex_act'] >= 0.35 or agg['nudity'] >= 0.4:
        rating = '16+'
        reasons.append("сексуальный контент и нагота")
        if agg['excerpts']['sex']:
            excerpts.extend(agg['excerpts']['sex'][:2])

    # 12+ - умеренный контент
    elif agg['violence'] >= 0.4 or agg['profanity'] >= 0.5 or agg['drugs'] >= 0.4:
        rating = '12+'
        if agg['violence'] >= 0.4:
            reasons.append("умеренное насилие и угрозы")
            if agg['excerpts']['violence']:
                excerpts.extend(agg['excerpts']['violence'][:1])
        if agg['profanity'] >= 0.5:
            reasons.append("ненормативная лексика")
            if agg['excerpts']['profanity']:
                excerpts.extend(agg['excerpts']['profanity'][:1])
        if agg['drugs'] >= 0.4:
            reasons.append("употребление алкоголя, табака или наркотиков")
            if agg['excerpts']['drugs']:
                excerpts.extend(agg['excerpts']['drugs'][:1])

    # 6+ - минимальный контент
    elif agg['violence'] >= 0.2 or agg['profanity'] >= 0.3:
        rating = '6+'
        reasons.append("незначительное насилие или редкая грубая лексика")

    # 0+ - контент для всех
    else:
        rating = '0+'
        reasons.append("контент без возрастных ограничений")

    return {
        'rating': rating,
        'reasons': reasons,
        'evidence_excerpts': excerpts[:5]  # Максимум 5 примеров
    }


def parse_script_to_scenes(txt: str) -> List[Dict[str, Any]]:
    """
    Разбивает сценарий на отдельные сцены.
    """
    scenes = []
    parts = re.split(
        r'(?=(?:INT\.|EXT\.|scene_heading\s*:|SCENE HEADING\s*:))',
        txt,
        flags=re.I
    )

    idx = 0
    for p in parts:
        text = p.strip()
        if not text or len(text) < 20:  # Пропускаем очень короткие фрагменты
            continue

        heading_match = re.match(r'((?:INT\.|EXT\.).{0,120})', text, flags=re.I)
        heading = heading_match.group(1).strip() if heading_match else f"scene_{idx}"

        scenes.append({
            'scene_id': idx,
            'heading': heading,
            'text': text
        })
        idx += 1

    # Если не нашли сцен, обрабатываем весь текст как одну сцену
    if len(scenes) < 3:
        scenes = [{'scene_id': 0, 'heading': 'full_script', 'text': txt}]

    return scenes


def analyze_script_file(path: str) -> Dict[str, Any]:
    """
    Анализирует файл сценария и возвращает возрастной рейтинг с обоснованием.

    Args:
        path: Путь к текстовому файлу сценария

    Returns:
        Словарь с рейтингом, причинами и примерами из текста
    """
    # Читаем файл
    txt = Path(path).read_text(encoding='utf-8', errors='ignore')

    # Разбиваем на сцены
    scenes = parse_script_to_scenes(txt)
    print(f"Найдено сцен: {len(scenes)}")

    # Извлекаем признаки для каждой сцены
    print("Анализ сцен...")
    features = []
    for scene in tqdm(scenes, desc="Обработка сцен"):
        feat = extract_scene_features(scene['text'])
        features.append(feat)

    # Нормализуем и применяем контекстную коррекцию
    scores = [normalize_and_contextualize_scores(f) for f in features]

    # Агрегируем оценки
    # Используем взвешенную агрегацию для баланса
    score_keys = ['violence', 'gore', 'sex_act', 'nudity', 'profanity', 'drugs', 'child_risk']
    agg = {}
    for k in score_keys:
        values = [s[k] for s in scores]
        # Для насилия и крови используем более высокий перцентиль
        # чтобы учесть серьезные сцены
        if k in ['violence', 'gore']:
            agg[k] = float(np.percentile(values, 80))
        # Для остального используем 75-й перцентиль
        else:
            agg[k] = float(np.percentile(values, 75))

    # Собираем все примеры из всех сцен
    all_excerpts = {
        'violence': [],
        'gore': [],
        'sex': [],
        'profanity': [],
        'drugs': []
    }
    for s in scores:
        for key in all_excerpts.keys():
            all_excerpts[key].extend(s['excerpts'][key])

    agg['excerpts'] = {k: v[:5] for k, v in all_excerpts.items()}  # Топ-5 примеров каждого типа

    # Определяем рейтинг
    rating_info = map_scores_to_rating(agg)

    # Находим самые проблемные сцены
    ranking = []
    for scene, score in zip(scenes, scores):
        weight = (
            score['violence'] * 0.5 +
            score['gore'] * 0.8 +
            score['sex_act'] * 0.9 +
            score['profanity'] * 0.3 +
            score['drugs'] * 0.3 +
            score['child_risk'] * 0.7
        )
        ranking.append((weight, scene, score))

    ranking.sort(reverse=True, key=lambda x: x[0])

    # Топ-5 самых влияющих на рейтинг сцен
    top_scenes = []
    for weight, scene, score in ranking[:5]:
        if weight > 0.1:  # Показываем только значимые сцены
            top_scenes.append({
                'scene_id': scene['scene_id'],
                'heading': scene['heading'],
                'sample_text': scene['text'][:300].replace('\n', ' ') + '...',
                'weight': round(float(weight), 3),
                'scores': {k: round(score[k], 2) for k in score_keys}
            })

    # Формируем итоговый результат
    result = {
        'file': str(Path(path).name),
        'predicted_rating': rating_info['rating'],
        'reasons': rating_info['reasons'],
        'evidence_excerpts': rating_info['evidence_excerpts'],
        'aggregated_scores': {k: round(agg[k], 3) for k in score_keys},
        'top_trigger_scenes': top_scenes,
        'total_scenes': len(scenes)
    }

    return result


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Использование: python repair_pipeline.py <путь_к_сценарию.txt>")
        print("\nПример:")
        print("  python repair_pipeline.py dataset/BERT_annotations/A_Clockwork_Orange_0066921_anno.txt")
        sys.exit(0)

    script_path = sys.argv[1]

    if not Path(script_path).exists():
        print(f"Ошибка: файл '{script_path}' не найден")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Анализ сценария: {script_path}")
    print(f"{'='*70}\n")

    result = analyze_script_file(script_path)

    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print(f"{'='*70}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

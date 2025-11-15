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

# PDF parsing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("WARNING: PyPDF2 not installed. PDF support disabled. Install with: pip install PyPDF2")

# ===== REFERENCE CONTEXTS FOR SEMANTIC ANALYSIS =====
# Контекстные шаблоны для определения типа сцен (English and Russian)
CONTEXT_TEMPLATES = {
    'graphic_violence': [
        "brutal murder with blood and gore",
        "torture and physical violence causing injury",
        "graphic depiction of death and killing",
        "violent assault with weapons causing harm",
        # Russian
        "жестокое убийство с кровью и увечьями",
        "пытки и физическое насилие причиняющее травмы",
        "графическое изображение смерти и убийства",
        "насильственное нападение с оружием причиняющее вред"
    ],
    'stylized_action': [
        "heroic action scene with combat",
        "adventure movie fight sequence",
        "comic book style action without gore",
        "spy thriller chase and combat",
        "superhero saving people from danger",
        # Russian
        "героическая боевая сцена с сражением",
        "приключенческая сцена драки в фильме",
        "экшн в стиле комиксов без жестокости",
        "погоня и бой в шпионском триллере",
        "супергерой спасающий людей от опасности"
    ],
    'sexual_content': [
        "explicit sexual intercourse scene",
        "nudity in sexual context",
        "rape or sexual assault",
        "graphic sexual activity",
        # Russian
        "явная сцена полового акта",
        "нагота в сексуальном контексте",
        "изнасилование или сексуальное насилие",
        "графическая сексуальная активность"
    ],
    'mild_romance': [
        "romantic kissing and affection",
        "love scene without explicit content",
        "romantic relationship development",
        # Russian
        "романтические поцелуи и нежность",
        "любовная сцена без эксплицитного контента",
        "развитие романтических отношений"
    ],
    'horror_violence': [
        "horror movie with scary violence",
        "psychological terror and fear",
        "monster attack with blood",
        "slasher film with killing",
        # Russian
        "фильм ужасов с пугающим насилием",
        "психологический террор и страх",
        "нападение монстра с кровью",
        "слэшер с убийствами"
    ],
    'profanity_context': [
        "casual conversation with swearing",
        "aggressive confrontation with profanity",
        "repeated use of strong language",
        # Russian
        "непринужденный разговор с матом",
        "агрессивная конфронтация с нецензурной лексикой",
        "многократное использование крепких выражений"
    ],
    'drug_abuse': [
        "drug use and addiction",
        "substance abuse scene",
        "characters taking illegal drugs",
        # Russian
        "употребление наркотиков и зависимость",
        "сцена злоупотребления веществами",
        "персонажи принимающие запрещенные наркотики"
    ],
    'child_endangerment': [
        "child in dangerous situation",
        "violence involving minors",
        "child abuse or threat to children",
        # Russian
        "ребенок в опасной ситуации",
        "насилие с участием несовершеннолетних",
        "жестокое обращение с детьми или угроза детям"
    ],
    'discussion_violence': [
        "courtroom discussion of crime",
        "testimony about violent event",
        "describing past violence in dialogue",
        "academic or legal discussion of weapons",
        "demonstration or explanation without action",
        # Russian
        "обсуждение преступления в зале суда",
        "показания о насильственном событии",
        "описание прошлого насилия в диалоге",
        "академическое или правовое обсуждение оружия",
        "демонстрация или объяснение без действия"
    ],
    'thriller_tension': [
        "psychological thriller with suspense",
        "tense dramatic confrontation",
        "mystery investigation without violence",
        "courtroom drama legal arguments",
        # Russian
        "психологический триллер с напряжением",
        "напряженная драматическая конфронтация",
        "расследование тайны без насилия",
        "судебная драма правовые споры"
    ]
}

# ===== KEYWORD PATTERNS (English and Russian) =====
VIOLENCE_WORDS = [
    # English patterns
    r'\bkill\w*', r'\bshoot\w*', r'\bshot\b', r'\bstab\w*',
    r'\bknife\b', r'\bgun\w*', r'\bpistol\b', r'\brifle\b',
    r'\bexplod\w*', r'\bblast\w*', r'\battack\w*',
    r'\bbeating\b', r'\bbeaten\b', r'\bbeats\b',  # Exclude "a beat" (screenplay term)
    r'\bcorpse\b', r'\bdead\b', r'\bmurder\w*', r'\bviolence\b',
    r'\bterrorist\b', r'\bhostage\b', r'\brip(ped|s)? apart\b',
    r'\bthug(s)?\b', r'\bterror\b', r'\bfight(ing)?\b',
    r'\bbattle(s|d)?\b', r'\bwar\b', r'\bshoot[- ]?out\b',
    r'\bexplosion\b', r'\bgrenade\b',
    # Russian patterns
    r'\bубий\w*', r'\bубить\b', r'\bубил\w*', r'\bубива\w*',
    r'\bстреля\w*', r'\bвыстрел\w*', r'\bзастрел\w*',
    r'\bзарез\w*', r'\bнож\b', r'\bоруж\w+', r'\bпистолет\w*',
    r'\bвинтовк\w*', r'\bавтомат\w*', r'\bвзрыв\w*',
    r'\bатак\w*', r'\bнападе\w*', r'\bизбие\w*',
    r'\bтруп\w*', r'\bмертв\w*', r'\bпогиб\w*',
    r'\bнасилие\b', r'\bжесток\w*', r'\bтеррор\w*',
    r'\bзаложник\w*', r'\bбандит\w*', r'\bдрак\w*',
    r'\bбой\b', r'\bсраж\w*', r'\bвойна\b', r'\bбоев\w*',
    r'\bгранат\w*', r'\bбомб\w*'
]

GORE_WORDS = [
    # English patterns
    r'\bblood\b', r'\bbloody\b', r'\bbloodied\b', r'\bbleeding\b',
    r'\bcorpse\b', r'\bwound\b', r'\bscar\b', r'\binjur\w*',
    r'\bcrash\w*', r'\bburn\w*', r'\bguts\b', r'\bentrails\b',
    r'\bbrain\b', r'\bdead body\b', r'\bgore\b', r'\bmutilat\w*',
    # Russian patterns
    r'\bкров\w*', r'\bкровав\w*', r'\bкровоточ\w*',
    r'\bран\w+', r'\bшрам\w*', r'\bувечь\w*',
    r'\bожог\w*', r'\bкишк\w*', r'\bвнутренност\w*',
    r'\bмозг\w*', r'\bрасчленен\w*', r'\bизувеч\w*'
]

PROFANITY = [
    # English patterns
    r'\bfuck\b', r'\bshit\b', r'\bmotherfucker\b', r'\bbitch\b',
    r'\basshole\b', r'\bdamn\b', r'\bhell\b', r'\bcrap\b',
    # Russian patterns
    r'\bблядь\b', r'\bбля\b', r'\bсука\b', r'\bхуй\b',
    r'\bпизд\w*', r'\bебать\b', r'\bебал\w*', r'\bебан\w*',
    r'\bзаеб\w*', r'\bдерьм\w*', r'\bговн\w*', r'\bхер\w*',
    r'\bмудак\w*', r'\bсволоч\w*', r'\bтварь\b'
]

DRUG_WORDS = [
    # English patterns
    r'\bdrug(s)?\b', r'\bheroin\b', r'\bcocaine\b', r'\bmarijuana\b',
    r'\bpill(s)?\b', r'\bweed\b', r'\balcohol\b', r'\bdrunk\b',
    r'\bcigarette\b', r'\bsmok(e|ing)\b', r'\baddiction\b',
    # Russian patterns
    r'\bнаркот\w*', r'\bгероин\w*', r'\bкокаин\w*', r'\bмарихуан\w*',
    r'\bтравк\w*', r'\bдоп\w*', r'\bтаблетк\w*', r'\bпилюл\w*',
    r'\bалкогол\w*', r'\bспирт\w*', r'\bвыпив\w*', r'\bпьян\w*',
    r'\bсигарет\w*', r'\bкур\w*', r'\bзависим\w*', r'\bнакур\w*'
]

CHILD_WORDS = [
    # English patterns
    r'\bchild(ren)?\b', r'\bkid(s)?\b', r'\bson\b', r'\bdaughter\b',
    r'\bteen(aged)?\b', r'\bboy\b', r'\bgirl\b', r'\bminor\b',
    # Russian patterns
    r'\bребенок\b', r'\bребенк\w*', r'\bдет\w+', r'\bмалыш\w*',
    r'\bсын\b', r'\bдоч\w*', r'\bподросток\w*', r'\bмальчик\w*',
    r'\bдевочк\w*', r'\bнесовершеннолетн\w*'
]

NUDITY_WORDS = [
    # English patterns
    r'\bbra\b', r'\bpanty|panties\b', r'\bunderwear\b', r'\bnaked\b',
    r'\bnude\b', r'\bundress\w*', r'\btopless\b',
    # Russian patterns
    r'\bголый\b', r'\bголая\b', r'\bнаг\w*', r'\bобнаж\w*',
    r'\bбюстгальтер\w*', r'\bтрус\w*', r'\bбелье\b',
    r'\bраздева\w*', r'\bбез одежд\w*'
]

SEX_WORDS = [
    # English patterns
    r'\brape\b', r'\bsexual\b', r'\bintercourse\b', r'\bsex scene\b',
    r'\bmolest\b', r'\borgasm\b', r'\bmake love\b', r'\bhaving sex\b',
    r'\bsexually\b', r'\bbed\s+scene\b',
    # Russian patterns
    r'\bизнасилов\w*', r'\bнасилов\w*', r'\bсексуальн\w*',
    r'\bполов\w+\s+акт\w*', r'\bинтимн\w*', r'\bоргазм\w*',
    r'\bзанимаются\s+сексом\b', r'\bзанимались\s+любовью\b',
    r'\bпостельн\w+\s+сцен\w*'
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
        # English patterns
        r'if (it|that|this) kills',
        r'(it|that|this)\'ll kill',
        r'(it|that|this) (will|would) kill',
        r'gonna.*kill',  # "gonna get the brass ring if it kills him"
        r'kill (you|me|him|her|them|us)',  # Figurative "kills you/me"
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
        r'shot at',  # Попытка/шанс (like "got a shot at")
        r'light[ -]?shot',
        r'fight (for|to see|to|for the)',  # Метафора борьбы
        r'fighting (for|against)',  # "fighting for bread crumbs"
        r'won the war',  # Метафора победы
        r'war (ration|time|era|years)',  # Historical context
        r'(world|civil|cold) war',
        r'battles? (with|against|for)',  # Метафорическая борьба
        r'attack(ed|ing)? (the|a) problem',
        r'speed of light',  # Физическое описание
        r'explosion of',  # "explosion of wood" (not literal explosion)
        r'explod(e|ed|ing) (with|into)',  # Figurative
        r'fight back tears',
        r'fight for (justice|freedom|rights)',
        r'fighting? (cancer|disease|illness)',
        r'dead serious',  # Figurative
        r'pool table',  # "shot" in pool context
        r'bank shot',  # Pool/basketball
        r'\ba beat\b',  # Screenplay term for pause
        r'as if.*\b(molest|rape|seduce|fondle)',  # Hypothetical/comparative (not actual content)
        r'about to.*\b(molest|rape|seduce|fondle)',  # Prevented/hypothetical action
        r'were to.*\b(molest|rape|seduce)',  # Conditional/hypothetical
        r'would.*\b(molest|rape|seduce)',  # Hypothetical
        r'brain (garbage|dump|drain|power|wave|dead|cell|teaser)',  # Metaphorical/non-gore brain usage
        r'brain(s)? (are|is) (just|garbage|trash)',  # "brains are just garbage"
        # Russian patterns
        r'в курсе',  # "в курсе" = "aware of/know about" (not drugs)
        r'курток',  # "куртка" = "jacket" (not smoking)
        r'куртк\w',  # "куртка" variations
        r'обритый наголо',  # "обритый наголо" = "shaved bald" (not nudity)
        r'наголо',  # "наголо" = "bald/clean-shaven" (when not about nudity)
        r'таблетк\w+\s+(от|для|против)',  # "таблетки от/для" = medicine pills (not drugs)
        r'болеутол\w+',  # "болеутоляющее" = painkiller (medicine, not drugs)
        r'кроват\w*',  # "кровать/кровати" = "bed" (not blood/gore)
        r'кров[ао]\w*',  # "крова/кровом" = "shelter/roof" (not blood)
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

    # Additional context-based filtering: if excerpt is very short (< 10 chars)
    # it's likely a parsing artifact
    matches = [m for m in matches if len(m.strip()) > 10]
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
    # Используем более разумную формулу: (count / length) * scale_factor
    # Это дает плавную оценку вместо скачков от 0 к 1
    violence_density = features['violence_count'] / max(1, L)
    gore_density = features['gore_count'] / max(1, L)

    # Масштабируем: 1 упоминание на 50 слов = 0.2, на 25 слов = 0.4, на 10 слов = 1.0
    violence_raw = violence_density * 100
    gore_raw = gore_density * 100

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
            'nudity': features['nudity_excerpts'],  # Добавлены примеры наготы
            'profanity': features['profanity_excerpts'],
            'drugs': features['drugs_excerpts']
        }
    }


def generate_scene_recommendations(scene_scores: Dict[str, float], target_rating: str = None) -> List[str]:
    """
    Генерирует рекомендации по снижению возрастного рейтинга для конкретной сцены.

    Args:
        scene_scores: Оценки сцены (violence, gore, sex_act, nudity, profanity, drugs, child_risk)
        target_rating: Желаемый рейтинг (опционально)

    Returns:
        Список рекомендаций для редактирования сцены
    """
    recommendations = []

    # Насилие
    if scene_scores['violence'] >= 0.7:
        recommendations.append(
            "🔪 Насилие (высокое): Уменьшите графическое изображение насилия. "
            "Рекомендации: показать сцену за кадром, использовать обрезку кадра, "
            "заменить явное насилие на подразумеваемое действие."
        )
    elif scene_scores['violence'] >= 0.4:
        recommendations.append(
            "⚔️ Насилие (умеренное): Сократите детализацию сцен драки/конфликта. "
            "Рекомендации: убрать крупные планы ударов, сократить длительность сцены."
        )

    # Кровь и увечья
    if scene_scores['gore'] >= 0.6:
        recommendations.append(
            "🩸 Кровь/увечья (высокое): Уберите графическое изображение крови и ран. "
            "Рекомендации: не показывать раны крупным планом, убрать описания 'blood', 'guts', "
            "'SPLORCH', заменить на более нейтральные формулировки типа 'ранен', 'пострадал'."
        )
    elif scene_scores['gore'] >= 0.3:
        recommendations.append(
            "💉 Кровь/увечья (умеренное): Смягчите описание телесных повреждений. "
            "Рекомендации: уменьшить количество упоминаний крови."
        )

    # Сексуальный контент
    if scene_scores['sex_act'] >= 0.6:
        recommendations.append(
            "🔞 Сексуальный контент (эксплицитный): Удалите или смягчите явные сексуальные сцены. "
            "Рекомендации: использовать монтаж с переходом, показать начало и конец без деталей."
        )
    elif scene_scores['sex_act'] >= 0.3:
        recommendations.append(
            "💋 Сексуальный контент (умеренный): Смягчите романтические/сексуальные элементы."
        )

    # Нагота
    if scene_scores['nudity'] >= 0.4:
        recommendations.append(
            "👙 Нагота: Уберите или смягчите сцены с обнаженным телом. "
            "Рекомендации: использовать одежду, изменить ракурс камеры, убрать описания нижнего белья."
        )

    # Ненормативная лексика
    if scene_scores['profanity'] >= 0.5:
        recommendations.append(
            "🤬 Ненормативная лексика (частая): Замените мат на более мягкие выражения. "
            "Рекомендации: заменить 'fuck', 'shit', 'bitch' на 'damn', 'hell' или эвфемизмы."
        )
    elif scene_scores['profanity'] >= 0.3:
        recommendations.append(
            "😠 Грубая лексика: Сократите количество нецензурных слов."
        )

    # Наркотики
    if scene_scores['drugs'] >= 0.4:
        recommendations.append(
            "💊 Наркотики/алкоголь: Уменьшите показ употребления веществ. "
            "Рекомендации: показать последствия вместо процесса, сократить экранное время."
        )

    # Дети в опасности
    if scene_scores['child_risk'] >= 0.5:
        recommendations.append(
            "👶 Дети в опасности: Критически важно! Уберите сцены с угрозой детям. "
            "Рекомендации: заменить детей на взрослых персонажей, убрать сцену полностью, "
            "или показать, что дети в безопасности."
        )

    if not recommendations:
        recommendations.append("✅ Сцена не содержит значимых проблемных элементов.")

    return recommendations


def map_scores_to_rating(agg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует агрегированные оценки в возрастной рейтинг (0+, 6+, 12+, 16+, 18+).
    Возвращает рейтинг и причины с примерами из текста.
    """
    reasons = []
    excerpts = []
    rating = '0+'

    # 18+ - эксплицитный контент (только для крайне графичного контента)
    if agg['sex_act'] >= 0.75 or agg['gore'] >= 0.95:
        rating = '18+'
        if agg['sex_act'] >= 0.75:
            reasons.append("эксплицитные сцены сексуального характера")
            if agg['excerpts']['sex']:
                excerpts.extend(agg['excerpts']['sex'][:2])
        if agg['gore'] >= 0.95:
            reasons.append("крайне графическое изображение жестокости и увечий")
            if agg['excerpts']['gore']:
                excerpts.extend(agg['excerpts']['gore'][:2])

    # 18+ - дети в опасности с насилием
    elif agg['child_risk'] > 0.7 and (agg['sex_act'] >= 0.5 or agg['violence'] >= 0.8):
        rating = '18+'
        reasons.append("опасные или жестокие сцены с участием несовершеннолетних")
        if agg['excerpts']['violence']:
            excerpts.extend(agg['excerpts']['violence'][:2])

    # 16+ - интенсивное насилие с кровью
    elif (agg['violence'] >= 0.8 and agg['gore'] >= 0.7) or agg['gore'] >= 0.75:
        rating = '16+'
        reasons.append("интенсивное графическое насилие с кровью и увечьями")
        if agg['excerpts']['violence']:
            excerpts.extend(agg['excerpts']['violence'][:2])
        if agg['excerpts']['gore']:
            excerpts.extend(agg['excerpts']['gore'][:1])

    # 16+ - явное насилие
    elif agg['violence'] >= 0.65 or agg['gore'] >= 0.5:
        rating = '16+'
        if agg['violence'] >= 0.65:
            reasons.append("интенсивное насилие и сцены убийств")
            if agg['excerpts']['violence']:
                excerpts.extend(agg['excerpts']['violence'][:2])
        if agg['gore'] >= 0.5:
            reasons.append("изображение крови и телесных повреждений")
            if agg['excerpts']['gore']:
                excerpts.extend(agg['excerpts']['gore'][:2])

    # 16+ - сексуальный контент средней степени
    elif agg['sex_act'] >= 0.35 or agg['nudity'] >= 0.4:
        rating = '16+'
        reasons.append("сексуальный контент и нагота")
        if agg['excerpts']['sex']:
            excerpts.extend(agg['excerpts']['sex'][:2])
        if agg['excerpts']['nudity']:
            excerpts.extend(agg['excerpts']['nudity'][:2])

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
    Поддерживает как английские (INT./EXT.), так и русские (ИНТ./ЭКСТ.) маркеры сцен.
    """
    scenes = []
    # Добавлена поддержка русских маркеров сцен (ИНТ./ЭКСТ.)
    parts = re.split(
        r'(?=(?:INT\.|EXT\.|ИНТ\.|ЭКСТ\.|scene_heading\s*:|SCENE HEADING\s*:))',
        txt,
        flags=re.I
    )

    idx = 0
    for p in parts:
        text = p.strip()
        if not text or len(text) < 20:  # Пропускаем очень короткие фрагменты
            continue

        # Поддержка русских и английских маркеров сцен
        heading_match = re.match(r'((?:INT\.|EXT\.|ИНТ\.|ЭКСТ\.).{0,120})', text, flags=re.I)
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


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из PDF файла.

    Args:
        pdf_path: Путь к PDF файлу

    Returns:
        Текст из PDF файла
    """
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 не установлен. Установите с помощью: pip install PyPDF2")

    text = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"Обработка PDF: {len(pdf_reader.pages)} страниц")

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        return '\n'.join(text)
    except Exception as e:
        print(f"Ошибка при чтении PDF: {e}")
        raise


def analyze_script_file(path: str) -> Dict[str, Any]:
    """
    Анализирует файл сценария и возвращает возрастной рейтинг с обоснованием.
    Поддерживает текстовые файлы (.txt) и PDF (.pdf).

    Args:
        path: Путь к файлу сценария (.txt или .pdf)

    Returns:
        Словарь с рейтингом, причинами и примерами из текста
    """
    # Определяем тип файла и читаем
    file_path = Path(path)
    if file_path.suffix.lower() == '.pdf':
        print(f"Обнаружен PDF файл: {file_path.name}")
        txt = extract_text_from_pdf(str(file_path))
    else:
        # Читаем текстовый файл
        txt = file_path.read_text(encoding='utf-8', errors='ignore')

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
    # Используем гибридный подход: учитываем как максимум, так и частоту
    score_keys = ['violence', 'gore', 'sex_act', 'nudity', 'profanity', 'drugs', 'child_risk']
    agg = {}
    for k in score_keys:
        values = [s[k] for s in scores]
        max_val = float(np.max(values))
        p95_val = float(np.percentile(values, 95))
        p90_val = float(np.percentile(values, 90))

        # Для насилия и крови: взвешенное среднее максимума и 95-го перцентиля
        # Если есть 1-2 очень графичные сцены, но остальные нормальные - это 16+, а не 18+
        # Если много графичных сцен - это 18+
        if k in ['violence', 'gore']:
            # 70% максимум + 30% p95 дает баланс
            agg[k] = max_val * 0.7 + p95_val * 0.3

        # Для сексуального контента и наготы - больше вес на максимум
        elif k in ['sex_act', 'nudity', 'child_risk']:
            agg[k] = max_val * 0.85 + p90_val * 0.15

        # Для ненормативной лексики и наркотиков используем 90-й перцентиль
        # так как они должны встречаться чаще для повышения рейтинга
        else:
            agg[k] = float(np.percentile(values, 90))

    # Собираем все примеры из всех сцен
    all_excerpts = {
        'violence': [],
        'gore': [],
        'sex': [],
        'nudity': [],  # Добавлены примеры наготы
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
            # Генерируем рекомендации для каждой проблемной сцены
            recommendations = generate_scene_recommendations(score)

            top_scenes.append({
                'scene_id': scene['scene_id'],
                'heading': scene['heading'],
                'sample_text': scene['text'][:300].replace('\n', ' ') + '...',
                'weight': round(float(weight), 3),
                'scores': {k: round(score[k], 2) for k in score_keys},
                'recommendations': recommendations
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

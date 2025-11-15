# Russian Language Support for Rating Pipeline

## Overview
This update adds comprehensive Russian language support to the content rating model, enabling it to analyze both English and Russian scenarios in text (.txt) and PDF (.pdf) formats.

## What's New

### 1. Russian Lexicon Patterns
Added Russian keyword patterns for all content categories:
- **Violence** (Насилие): убийство, стрельба, нож, оружие, взрыв, etc.
- **Gore** (Кровь/Увечья): кровь, раны, увечья, etc.
- **Profanity** (Мат): блядь, сука, хуй, пизда, etc.
- **Drugs** (Наркотики): наркотики, алкоголь, сигареты, etc.
- **Child mentions** (Дети): ребенок, дети, малыш, подросток, etc.
- **Nudity** (Нагота): голый, обнаженный, белье, etc.
- **Sexual content** (Сексуальный контент): изнасилование, сексуальный, интимный, etc.

### 2. Russian Context Templates
Added bilingual semantic templates for context analysis:
- Graphic violence vs Discussion violence (Графическое насилие vs Обсуждение насилия)
- Stylized action vs Horror violence (Стилизованный экшн vs Насилие в хорроре)
- Sexual content vs Mild romance (Сексуальный контент vs Мягкая романтика)
- Drug abuse context (Контекст употребления наркотиков)
- Child endangerment (Опасность для детей)
- Thriller tension (Напряжение в триллере)

### 3. PDF Support
Added ability to parse and analyze PDF files using PyPDF2:
- Automatic file type detection (.txt vs .pdf)
- Text extraction from PDF pages
- Full support for Russian text in PDFs

### 4. Test Coverage
Created comprehensive test suite:
- Tests all English scenarios (.txt files)
- Tests all Russian scenarios (.pdf files)
- Detailed test results saved to `test_results_all_scenarios.json`

## Test Results

### English Scenarios
✅ **12 Angry Men** - Rating: 16+ (Violence and murder scenes)
✅ **ATM** - Rating: 18+ (Extreme graphic violence)
✅ **American History X** - Rating: 18+ (Explicit sexual content)
✅ **Spider Man 2** - Rating: 16+ (Intense graphic violence with blood)
✅ **Superman** - Rating: 18+ (Explicit sexual scenes)

### Russian Scenarios (PDF)
✅ **DG_Topi_seria_1.pdf** - Rating: 16+ (Sexual content and nudity)
✅ **ПРОСТОКВАШИНО_Дело_о_пропавшей_лопате_для_читки.pdf** - Rating: 0+ (Family-friendly content)

## Usage

### Analyzing a single file
```bash
# English text file
python repair_pipeline.py "12 Angry Men_0118528_anno.txt"

# Russian PDF file
python repair_pipeline.py "ПРОСТОКВАШИНО_Дело_о_пропавшей_лопате_для_читки.pdf"
```

### Running comprehensive tests
```bash
python experiments/test_all_scenarios.py
```

## Technical Details

### Dependencies
Added PyPDF2 for PDF parsing:
```
PyPDF2>=3.0.0
```

### Architecture Changes
1. **Bilingual Lexicons**: All keyword patterns now include both English and Russian versions
2. **Multilingual Embeddings**: The `all-MiniLM-L6-v2` model supports multilingual text, enabling semantic analysis of both languages
3. **Automatic Format Detection**: The pipeline automatically detects file format and applies appropriate parsing

### Key Functions Modified
- `VIOLENCE_WORDS`, `GORE_WORDS`, `PROFANITY`, etc. - Added Russian patterns
- `CONTEXT_TEMPLATES` - Added Russian context descriptions
- `extract_text_from_pdf()` - New function for PDF parsing
- `analyze_script_file()` - Updated to support both .txt and .pdf formats

## Performance
- English .txt files: ~10-13 seconds per file
- Russian .pdf files: ~1-2 seconds per file (shorter scenarios)
- Model loads semantic embeddings once on startup for efficiency

## Limitations
- PDF text extraction quality depends on PDF structure
- Very large PDFs may take longer to process
- Scene detection in PDFs may differ from structured text files

## Future Improvements
- Add more Russian slang and colloquial profanity patterns
- Improve scene detection for Russian screenplay formats
- Support additional file formats (DOCX, RTF, etc.)
- Add Ukrainian and other Slavic language support

def get_language_from_abbreviation(abbreviation):
    
    language_abbreviations = {
        "ar": "Arabic",
        "bn": "Bengali",
        "en": "English",
        "es": "Spanish",
        "fa": "Persian",
        "fi": "Finnish",
        "fr": "French",
        "hi": "Hindi",
        "id": "Indonesian",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "sw": "Swahili",
        "te": "Telugu",
        "th": "Thai",
        "zh": "Chinese"
    }

    return language_abbreviations.get(abbreviation, "Unknown")
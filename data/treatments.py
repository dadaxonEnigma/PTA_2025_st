from data.translations import treatment_translations

def get_treatment(lang="uz"):
    """Возвращает рекомендации для всех классов на выбранном языке"""
    return {
        class_name: treatment_translations[lang][class_name]
        for class_name in treatment_translations[lang]
    }
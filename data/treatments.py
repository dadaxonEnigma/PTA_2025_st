# data/treatments.py
from data.translations import treatment_translations

def get_treatment(lang="uz"):
    return treatment_translations[lang]
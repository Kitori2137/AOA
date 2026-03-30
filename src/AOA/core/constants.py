KSZTALT_DIMENSIONS = {
    "kwadrat": {"x": 30.0, "y": 30.0, "z": 0.3},
    "trojkat": {"x": 35.0, "y": 30.0, "z": 0.3},
    "trapez": {"x": 40.0, "y": 25.0, "z": 0.3},
}

KSZTALT_MAP = {
    "kwadrat": 1.0,
    "trojkat": 0.8,
    "trapez": 0.9,
}

MATERIAL_MAP = {
    "bawelna": 0.85,
    "mikrofibra": 1.0,
    "poliester": 0.75,
    "wiskoza": 0.9,
}

SCHEDULE_FEATURES = [
    "n_jobs",
    "mean_time",
    "std_time",
    "max_time",
    "mean_deadline",
    "load_ratio",
]

FEATURES = ["cena", "odpad", "termin_h", "czas_produkcji_h"]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from AOA.core.constants import KSZTALT_DIMENSIONS


def generate_production_data(
    n=5000,
    n_machines=1,
    test_size=0.2,
    seed=42,
    ksztalty=None,
    materialy=None,
    price_range=(50, 500),
    waste_range=(0.0, 0.3),
    production_time_range=(1.0, 48.0),
    deadline_buffer_range=(1.0, 72.0),
):
    np.random.seed(seed)

    available_ksztalty = list(KSZTALT_DIMENSIONS.keys())
    available_materialy = ["bawelna", "mikrofibra", "poliester", "wiskoza"]

    ksztalty = ksztalty or available_ksztalty
    materialy = materialy or available_materialy

    if not ksztaty_or_materialy_valid(ksztalty, available_ksztalty, "ksztalty"):
        raise ValueError("Nieprawidłowe wartości w liście ksztalty")

    if not ksztaty_or_materialy_valid(materialy, available_materialy, "materialy"):
        raise ValueError("Nieprawidłowe wartości w liście materialy")

    ksztalt_col = np.random.choice(ksztalty, n)
    material_col = np.random.choice(materialy, n)

    x_col = [KSZTALT_DIMENSIONS[k]["x"] for k in ksztalt_col]
    y_col = [KSZTALT_DIMENSIONS[k]["y"] for k in ksztalt_col]
    z_col = [KSZTALT_DIMENSIONS[k]["z"] for k in ksztalt_col]

    czas_produkcji_h = np.random.uniform(production_time_range[0], production_time_range[1], n)
    deadline_buffer_h = np.random.uniform(deadline_buffer_range[0], deadline_buffer_range[1], n)

    # Termin zawsze musi być większy od czasu produkcji
    termin_h = czas_produkcji_h + deadline_buffer_h

    df = pd.DataFrame({
        "cena": np.random.uniform(price_range[0], price_range[1], n),
        "odpad": np.random.uniform(waste_range[0], waste_range[1], n),
        "termin_h": np.round(termin_h, 3),
        "czas_produkcji_h": np.round(czas_produkcji_h, 3),
        "ksztalt": ksztalt_col,
        "material": material_col,
        "x": x_col,
        "y": y_col,
        "z": z_col,
    })

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    machines = [0.0] * n_machines
    lateness_h = []

    for _, row in df.iterrows():
        next_machine_idx = int(np.argmin(machines))
        start = machines[next_machine_idx]
        end = start + row["czas_produkcji_h"]
        machines[next_machine_idx] = end

        deadline = row["termin_h"]
        late = max(0, end - deadline)
        lateness_h.append(late)

    df["lateness_h_sim"] = np.round(lateness_h, 3)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return df, train_df, test_df


def ksztaty_or_materialy_valid(values, allowed, field_name):
    if not values:
        return False
    return all(v in allowed for v in values)

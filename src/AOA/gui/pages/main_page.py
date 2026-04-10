import threading

import customtkinter as ctk
from tkinter import filedialog, messagebox

from AOA.config import DATA_DIR, MODELS_DIR
from AOA.core.services import (
    analyze_sto_models,
    build_dataframe_preview_text,
    build_main_page_status,
    build_main_page_summary,
    generate_and_store_datasets_from_config,
    load_model_pack,
    load_training_data,
    solve_models_flow,
    solve_sto_with_saved_model,
    split_selected_models,
    train_models_flow,
    train_sto_models_flow,
)
from AOA.utils.error_utils import write_exception_log


class MainPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.df_train = None
        self.df_test = None
        self.df_full = None
        self.loaded_data_path = None
        self.last_generation_metadata = {}
        self._last_progress_per_model = {}

        self.model_vars = {
            "Quality": ctk.BooleanVar(value=True),
            "Delay": ctk.BooleanVar(value=False),
            "Schedule": ctk.BooleanVar(value=False),
            "MT": ctk.BooleanVar(value=True),
            "MO": ctk.BooleanVar(value=True),
            "MZO": ctk.BooleanVar(value=True),
            "GENETIC": ctk.BooleanVar(value=True),
        }

        self.ksztalt_vars = {
            "kwadrat": ctk.BooleanVar(value=True),
            "trojkat": ctk.BooleanVar(value=True),
            "trapez": ctk.BooleanVar(value=True),
        }

        self.material_vars = {
            "bawelna": ctk.BooleanVar(value=True),
            "mikrofibra": ctk.BooleanVar(value=True),
            "poliester": ctk.BooleanVar(value=True),
            "wiskoza": ctk.BooleanVar(value=True),
        }

        self.n_var = ctk.StringVar(value="5000")
        self.n_machines_var = ctk.StringVar(value="1")
        self.test_size_var = ctk.StringVar(value="0.2")
        self.seed_var = ctk.StringVar(value="42")
        self.prod_min_var = ctk.StringVar(value="1")
        self.prod_max_var = ctk.StringVar(value="48")
        self.deadline_min_var = ctk.StringVar(value="1")
        self.deadline_max_var = ctk.StringVar(value="72")

        self.backend_var = ctk.StringVar(value="classic")

        self.sto_jobs_var = ctk.StringVar(value="Z1,Z2,Z3")
        self.sto_times_var = ctk.StringVar(value="10,20,100")
        self.sto_deadlines_var = ctk.StringVar(value="150,30,110")

        self._build_layout()
        self.render_summary()
        self.render_status()
        self.render_preview()

    def _build_layout(self):
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=20, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="ML / Optymalizacja",
            font=("Arial", 24, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 2))

        ctk.CTkLabel(
            header,
            text="Konfiguracja generowania danych, wyboru modeli i analiz STO",
            font=("Arial", 12),
        ).pack(anchor="w", padx=15, pady=(0, 12))

        content = ctk.CTkFrame(self)
        content.pack(fill="both", expand=True, padx=20, pady=10)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkScrollableFrame(content)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        right_panel = ctk.CTkFrame(content, width=340)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        right_panel.grid_propagate(False)
        right_panel.grid_rowconfigure(0, weight=0)
        right_panel.grid_rowconfigure(1, weight=0)
        right_panel.grid_rowconfigure(2, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self._build_models_section(left_panel)
        self._build_generation_section(left_panel)
        self._build_sto_section(left_panel)
        self._build_actions_section(left_panel)
        self._build_preview_section(left_panel)
        self._build_sto_results_section(left_panel)
        self._build_log_section(left_panel)

        self._build_summary_section(right_panel)
        self._build_status_section(right_panel)
        self._build_right_hint_section(right_panel)

    def _build_models_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=(10, 8))

        ctk.CTkLabel(
            frame,
            text="Wybór modeli",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        ml_frame = ctk.CTkFrame(frame)
        ml_frame.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(
            ml_frame,
            text="Modele ML",
            font=("Arial", 15, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 6))

        for name in ("Quality", "Delay", "Schedule"):
            ctk.CTkCheckBox(
                ml_frame,
                text=name,
                variable=self.model_vars[name],
                command=self.render_summary,
            ).pack(anchor="w", padx=10, pady=4)

        ctk.CTkLabel(
            ml_frame,
            text="Backend ML",
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))

        self.backend_menu = ctk.CTkOptionMenu(
            ml_frame,
            values=["classic", "tabpfn"],
            variable=self.backend_var,
            command=lambda _value: self.render_summary(),
        )
        self.backend_menu.pack(anchor="w", padx=10, pady=(0, 4))

        ctk.CTkLabel(
            ml_frame,
            text="classic = obecne modele | tabpfn = tryb eksperymentalny",
            font=("Arial", 11),
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 10))

        sto_frame = ctk.CTkFrame(frame)
        sto_frame.pack(fill="x", padx=10, pady=(0, 12))

        ctk.CTkLabel(
            sto_frame,
            text="Modele heurystyczne STO",
            font=("Arial", 15, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 6))

        for name in ("MT", "MO", "MZO", "GENETIC"):
            ctk.CTkCheckBox(
                sto_frame,
                text=name,
                variable=self.model_vars[name],
                command=self.render_summary,
            ).pack(anchor="w", padx=10, pady=4)

    def _build_generation_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Parametry generowania danych",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        grid = ctk.CTkFrame(frame)
        grid.pack(fill="x", padx=10, pady=(0, 10))

        self._entry(grid, "Liczba rekordów", self.n_var, 0, 0)
        self._entry(grid, "Liczba maszyn", self.n_machines_var, 0, 2)
        self._entry(grid, "Test size", self.test_size_var, 1, 0)
        self._entry(grid, "Seed", self.seed_var, 1, 2)
        self._entry(grid, "Czas prod. min [h]", self.prod_min_var, 2, 0)
        self._entry(grid, "Czas prod. max [h]", self.prod_max_var, 2, 2)
        self._entry(grid, "Bufor terminu min [h]", self.deadline_min_var, 3, 0)
        self._entry(grid, "Bufor terminu max [h]", self.deadline_max_var, 3, 2)

        for child in grid.winfo_children():
            if isinstance(child, ctk.CTkEntry):
                child.bind("<KeyRelease>", lambda _e: self.render_summary())

        ctk.CTkLabel(
            frame,
            text="Kształty",
            font=("Arial", 15, "bold"),
        ).pack(anchor="w", padx=15, pady=(4, 4))

        shapes_frame = ctk.CTkFrame(frame)
        shapes_frame.pack(fill="x", padx=10, pady=(0, 8))

        for name, var in self.ksztalt_vars.items():
            ctk.CTkCheckBox(
                shapes_frame,
                text=name,
                variable=var,
                command=self.render_summary,
            ).pack(anchor="w", padx=10, pady=4)

        ctk.CTkLabel(
            frame,
            text="Materiały",
            font=("Arial", 15, "bold"),
        ).pack(anchor="w", padx=15, pady=(4, 4))

        mat_frame = ctk.CTkFrame(frame)
        mat_frame.pack(fill="x", padx=10, pady=(0, 12))

        for name, var in self.material_vars.items():
            ctk.CTkCheckBox(
                mat_frame,
                text=name,
                variable=var,
                command=self.render_summary,
            ).pack(anchor="w", padx=10, pady=4)

    def _build_sto_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Wejście ręczne dla STO",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        grid = ctk.CTkFrame(frame)
        grid.pack(fill="x", padx=10, pady=(0, 10))

        self._entry(grid, "Zlecenia", self.sto_jobs_var, 0, 0)
        self._entry(grid, "Czasy", self.sto_times_var, 1, 0)
        self._entry(grid, "Terminy", self.sto_deadlines_var, 2, 0)

        ctk.CTkButton(
            frame,
            text="Policz tylko STO ręcznie",
            command=self.run_sto_analysis,
            height=36,
        ).pack(fill="x", padx=15, pady=(0, 12))

    def _build_actions_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Akcje",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 10))

        btns = ctk.CTkFrame(frame)
        btns.pack(fill="x", padx=10, pady=(0, 12))

        ctk.CTkButton(
            btns,
            text="Generuj dane",
            command=self.gen,
            height=38,
        ).pack(fill="x", padx=10, pady=6)

        ctk.CTkButton(
            btns,
            text="Wczytaj dane (CSV)",
            command=self.load_from_disk,
            height=38,
        ).pack(fill="x", padx=10, pady=6)

        ctk.CTkButton(
            btns,
            text="Uruchom / zapisz wybrane modele",
            command=self.train,
            height=38,
        ).pack(fill="x", padx=10, pady=6)

        ctk.CTkButton(
            btns,
            text="▶ Rozwiąż zapisany model na wybranych danych",
            command=self.solve_existing_models,
            height=38,
        ).pack(fill="x", padx=10, pady=6)

    def _build_preview_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Podgląd danych",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.preview_box = ctk.CTkTextbox(frame, height=220)
        self.preview_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.preview_box.configure(state="disabled")

    def _build_sto_results_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Wyniki STO",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.sto_box = ctk.CTkTextbox(frame, height=260)
        self.sto_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.sto_box.configure(state="disabled")

    def _build_log_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=(8, 10))

        ctk.CTkLabel(
            frame,
            text="Log",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.logbox = ctk.CTkTextbox(frame, height=220)
        self.logbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.logbox.configure(state="disabled")

    def _build_summary_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))

        ctk.CTkLabel(
            frame,
            text="Podsumowanie wyboru",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.summary_box = ctk.CTkTextbox(frame, height=320)
        self.summary_box.pack(fill="x", padx=15, pady=(0, 15))
        self.summary_box.configure(state="disabled")

    def _build_status_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Szybki status",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.status_label = ctk.CTkLabel(
            frame,
            text="",
            font=("Arial", 13),
            justify="left",
        )
        self.status_label.pack(anchor="w", padx=15, pady=(0, 12))

    def _build_right_hint_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)

        ctk.CTkLabel(
            frame,
            text="Panel boczny",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        hint = (
            "STO działa teraz też dwustopniowo:\n"
            "1) zapisujesz model STO,\n"
            "2) później wybierasz zapisany model i dane.\n\n"
            "Pliki wynikowe zapisują się do katalogu data.\n\n"
            "Dla TabPFN na CPU najlepiej testować mniejsze zbiory,\n"
            "np. 300–1000 rekordów."
        )

        ctk.CTkLabel(
            frame,
            text=hint,
            justify="left",
            wraplength=280,
            font=("Arial", 12),
        ).pack(anchor="w", padx=15, pady=(0, 15))

    def _entry(self, parent, label, var, row, col):
        ctk.CTkLabel(parent, text=label).grid(
            row=row,
            column=col,
            sticky="w",
            padx=10,
            pady=6,
        )
        entry = ctk.CTkEntry(parent, textvariable=var, width=160)
        entry.grid(row=row, column=col + 1, sticky="w", padx=10, pady=6)

    def _ui_config(self):
        return {
            "selected_models": [
                name for name, var in self.model_vars.items() if var.get()
            ],
            "selected_ksztalty": [
                name for name, var in self.ksztalt_vars.items() if var.get()
            ],
            "selected_materialy": [
                name for name, var in self.material_vars.items() if var.get()
            ],
            "n": self.n_var.get(),
            "n_machines": self.n_machines_var.get(),
            "test_size": self.test_size_var.get(),
            "seed": self.seed_var.get(),
            "prod_min": self.prod_min_var.get(),
            "prod_max": self.prod_max_var.get(),
            "deadline_min": self.deadline_min_var.get(),
            "deadline_max": self.deadline_max_var.get(),
            "backend": self.backend_var.get(),
        }

    def render_summary(self):
        summary = build_main_page_summary(self._ui_config())
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", summary)
        self.summary_box.configure(state="disabled")

    def render_status(self):
        self.status_label.configure(
            text=build_main_page_status(self.df_train, self.df_test)
        )

    def render_preview(self):
        text = build_dataframe_preview_text(
            self.df_full,
            title="Podgląd pełnego zbioru",
            max_rows=15,
        )
        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.insert("end", text)
        self.preview_box.configure(state="disabled")

    def render_sto_report(self, report: str):
        self.sto_box.configure(state="normal")
        self.sto_box.delete("1.0", "end")
        self.sto_box.insert("end", report)
        self.sto_box.configure(state="disabled")

    def _safe_log(self, msg: str):
        self.after(0, lambda: self.log(msg))

    def _safe_showerror(self, title: str, msg: str):
        self.after(0, lambda: messagebox.showerror(title, msg))

    def _safe_render_sto_report(self, report: str):
        self.after(0, lambda: self.render_sto_report(report))

    def log(self, msg: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def clear_loaded_data_state(self):
        self.df_train = None
        self.df_test = None
        self.df_full = None
        self.loaded_data_path = None
        self.render_status()
        self.render_preview()

    def _on_train_progress(self, model_name: str, percent: float, detail: str = ""):
        last_percent = self._last_progress_per_model.get(model_name, -1.0)

        should_log = False
        if percent >= 100.0:
            should_log = True
        elif last_percent < 0:
            should_log = True
        elif (percent - last_percent) >= 0.5:
            should_log = True

        if not should_log:
            return

        self._last_progress_per_model[model_name] = percent
        suffix = f" | {detail}" if detail else ""
        self._safe_log(f"⏳ {model_name}: {percent:.1f}%{suffix}")

    def _log_sto_results(self, sto_result: dict):
        results = sto_result.get("results", [])
        best = sto_result.get("best_result")
        saved_paths = sto_result.get("saved_paths", [])
        best_path = sto_result.get("best_path")

        self._safe_log("===== WYNIKI STO =====")

        if best is not None:
            self._safe_log(
                f"🏆 Najlepszy model: {best['method']} | STO={best['sto']:.3f} | "
                f"Kolejność: {', '.join(best['order'])}"
            )

        for index, result in enumerate(results, start=1):
            self._safe_log("")
            self._safe_log(f"[{index}] MODEL: {result['method']}")
            self._safe_log(f"    STO: {result['sto']:.3f}")
            self._safe_log(f"    Kolejność: {', '.join(result['order'])}")
            self._safe_log(f"    Czas całkowity: {result['completion_total']:.3f}")
            self._safe_log(f"    Max T+: {result['max_positive_delay']:.3f}")

        if saved_paths:
            self._safe_log("")
            self._safe_log("===== ZAPISANE PLIKI STO =====")
            for item in saved_paths:
                self._safe_log(
                    f"- {item['method']} | STO={item['sto']:.3f} | {item['path']}"
                )

        if best_path:
            self._safe_log(f"- BEST | {best_path}")

    def gen(self):
        try:
            cfg = self._ui_config()

            self.clear_loaded_data_state()
            self.log("ℹ Czyszczę poprzednie dane z pamięci...")
            self.log("▶ Start generowania nowych danych...")

            result = generate_and_store_datasets_from_config(cfg)

            self.df_full = result["full_df"]
            self.df_train = result["train_df"]
            self.df_test = result["test_df"]
            self.loaded_data_path = result["full_path"]
            self.last_generation_metadata = {
                "n": int(cfg["n"]),
                "n_machines": int(cfg["n_machines"]),
                "ksztalty": cfg["selected_ksztalty"],
                "materialy": cfg["selected_materialy"],
            }

            for line in result["messages"]:
                self.log(line)

            self.render_status()
            self.render_preview()

        except ValueError as e:
            self.log(f"⚠ Błąd danych: {e}")
            messagebox.showerror("Błąd danych", str(e))
        except Exception as e:
            log_path = write_exception_log("main_page.gen", e)
            self.log(f"❌ Nieoczekiwany błąd.\nSzczegóły zapisano w: {log_path}")
            messagebox.showerror(
                "Błąd",
                "Wystąpił nieoczekiwany błąd podczas generowania danych.",
            )

    def load_from_disk(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            initialdir=str(DATA_DIR),
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        try:
            self.clear_loaded_data_state()
            self.log(f"▶ Wczytywanie danych z pliku: {path}")

            result = load_training_data(path=path, train_ratio=0.8)

            self.df_full = result["full_df"]
            self.df_train = result["train_df"]
            self.df_test = result["test_df"]
            self.loaded_data_path = path

            for line in result["messages"]:
                self.log(line)

            self.render_status()
            self.render_preview()

        except FileNotFoundError:
            messagebox.showerror("Błąd", "Nie znaleziono wskazanego pliku.")
        except ValueError as e:
            messagebox.showerror("Błąd danych", str(e))
        except Exception as e:
            log_path = write_exception_log("main_page.load_from_disk", e)
            self.log(f"❌ Nieoczekiwany błąd.\nSzczegóły zapisano w: {log_path}")
            messagebox.showerror(
                "Błąd",
                "Wystąpił nieoczekiwany błąd podczas wczytywania danych.",
            )

    def train(self):
        cfg = self._ui_config()
        selected_models = cfg["selected_models"]
        backend = cfg["backend"]
        ml_models, sto_models = split_selected_models(selected_models)

        if not ml_models and not sto_models:
            messagebox.showerror("Błąd", "Wybierz przynajmniej jeden model.")
            return

        if ml_models and self.df_train is None:
            messagebox.showerror("Błąd", "Brak danych treningowych.")
            return

        threading.Thread(
            target=self.train_worker,
            args=(ml_models, sto_models, backend),
            daemon=True,
        ).start()

    def train_worker(self, ml_models, sto_models, backend):
        try:
            self._last_progress_per_model = {}

            if ml_models:
                backend_label = "TabPFN" if backend == "tabpfn" else "Classic"
                self._safe_log(
                    f"▶ Start treningu ML: {', '.join(ml_models)} | backend: {backend_label}"
                )

                result = train_models_flow(
                    df_train=self.df_train,
                    selected_models=ml_models,
                    metadata=self.last_generation_metadata,
                    progress_callback=self._on_train_progress,
                    backend=backend,
                )

                for line in result["messages"]:
                    self._safe_log(line)

            if sto_models:
                self._safe_log(f"▶ Start zapisu modelu STO: {', '.join(sto_models)}")
                sto_pack_result = train_sto_models_flow(sto_models)
                for line in sto_pack_result["messages"]:
                    self._safe_log(line)

            if ml_models and sto_models:
                self._safe_log("✔ Zapisano razem modele ML i STO")

        except ValueError as e:
            self._safe_log(f"⚠ Błąd danych: {e}")
            self._safe_showerror("Błąd danych", str(e))
        except Exception as e:
            log_path = write_exception_log("main_page.train_worker", e)
            self._safe_log(f"❌ Błąd: {type(e).__name__}: {e}")
            self._safe_log(f"Szczegóły zapisano w: {log_path}")
            self._safe_showerror(
                "Błąd",
                f"Wystąpił błąd podczas uruchamiania modeli:\n\n{type(e).__name__}: {e}",
            )

    def solve_existing_models(self):
        model_path = filedialog.askopenfilename(
            title="Wybierz zapisany model",
            initialdir=str(MODELS_DIR),
            filetypes=[("Pickle", "*.pkl")],
        )
        if not model_path:
            return

        data_path = filedialog.askopenfilename(
            title="Wybierz dane do rozwiązania",
            initialdir=str(DATA_DIR),
            filetypes=[("CSV", "*.csv")],
        )
        if not data_path:
            return

        threading.Thread(
            target=self._solve_existing_models_worker,
            args=(model_path, data_path),
            daemon=True,
        ).start()

    def _solve_existing_models_worker(self, model_path, data_path):
        try:
            pack = load_model_pack(model_path)
            pack_kind = pack.get("pack_kind", "ml")

            self._safe_log(f"▶ Wybrany model: {model_path}")
            self._safe_log(f"▶ Wybrane dane: {data_path}")
            self._safe_log(f"▶ Typ paczki: {pack_kind}")

            if pack_kind == "sto":
                result = solve_sto_with_saved_model(model_path=model_path, data_path=data_path)

                for line in result.get("messages", []):
                    self._safe_log(line)

                self._safe_render_sto_report(result["report"])
                self._log_sto_results(result)
                return

            backend = pack.get("backend", "classic")
            self._safe_log(f"▶ Backend modelu ML: {backend}")

            result = solve_models_flow(model_path=model_path, data_path=data_path)

            for line in result["messages"]:
                self._safe_log(line)

            self._safe_log(f"✔ Plik wynikowy zapisany tutaj: {result['result_path']}")

        except ValueError as e:
            self._safe_log(f"⚠ Błąd danych: {e}")
            self._safe_showerror("Błąd danych", str(e))
        except Exception as e:
            log_path = write_exception_log("main_page.solve_existing_models", e)
            self._safe_log(f"❌ Błąd: {type(e).__name__}: {e}")
            self._safe_log(f"Szczegóły zapisano w: {log_path}")
            self._safe_showerror(
                "Błąd",
                f"Wystąpił błąd podczas rozwiązywania modeli:\n\n{type(e).__name__}: {e}",
            )

    def run_sto_analysis(self):
        selected_models = self._ui_config()["selected_models"]
        _, sto_models = split_selected_models(selected_models)

        if not sto_models:
            messagebox.showinfo("Informacja", "Wybierz przynajmniej jeden model STO.")
            return

        try:
            result = analyze_sto_models(
                job_ids_text=self.sto_jobs_var.get(),
                processing_text=self.sto_times_var.get(),
                deadlines_text=self.sto_deadlines_var.get(),
                selected_methods=sto_models,
            )
            self.render_sto_report(result["report"])
            self._log_sto_results(result)

        except ValueError as e:
            messagebox.showerror("Błąd STO", str(e))
        except Exception as e:
            log_path = write_exception_log("main_page.run_sto_analysis", e)
            self.log(f"❌ Nieoczekiwany błąd STO.\nSzczegóły zapisano w: {log_path}")
            messagebox.showerror(
                "Błąd STO",
                "Wystąpił nieoczekiwany błąd podczas analizy STO.",
            )

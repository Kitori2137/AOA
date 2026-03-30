import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

from AOA.config import DATA_DIR, MODELS_DIR
from AOA.core.services import (
    analyze_sto_models,
    build_dataframe_preview_text,
    build_main_page_status,
    build_main_page_summary,
    generate_and_store_datasets,
    load_training_data,
    solve_models_flow,
    train_models_flow,
)


class MainPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.df_train = None
        self.df_test = None
        self.df_full = None
        self.last_generation_metadata = {}

        self.model_vars = {
            "Quality": ctk.BooleanVar(value=True),
            "Delay": ctk.BooleanVar(value=False),
            "Schedule": ctk.BooleanVar(value=False),
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

        self.sto_vars = {
            "MT": ctk.BooleanVar(value=True),
            "MO": ctk.BooleanVar(value=True),
            "MZO": ctk.BooleanVar(value=True),
            "GENETIC": ctk.BooleanVar(value=True),
        }

        self.n_var = ctk.StringVar(value="5000")
        self.n_machines_var = ctk.StringVar(value="1")
        self.test_size_var = ctk.StringVar(value="0.2")
        self.seed_var = ctk.StringVar(value="42")
        self.prod_min_var = ctk.StringVar(value="1")
        self.prod_max_var = ctk.StringVar(value="48")
        self.deadline_min_var = ctk.StringVar(value="1")
        self.deadline_max_var = ctk.StringVar(value="72")

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
            font=("Arial", 24, "bold")
        ).pack(anchor="w", padx=15, pady=(12, 2))

        ctk.CTkLabel(
            header,
            text="Konfiguracja generowania danych, wyboru modeli i analiz STO",
            font=("Arial", 12)
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

        ctk.CTkLabel(frame, text="Wybór modeli ML", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        checks = ctk.CTkFrame(frame)
        checks.pack(fill="x", padx=10, pady=(0, 12))

        for name, var in self.model_vars.items():
            ctk.CTkCheckBox(
                checks,
                text=name,
                variable=var,
                command=self.render_summary
            ).pack(anchor="w", padx=10, pady=4)

    def _build_generation_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(frame, text="Parametry generowania danych", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

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

        ctk.CTkLabel(frame, text="Kształty", font=("Arial", 15, "bold")).pack(anchor="w", padx=15, pady=(4, 4))
        shapes_frame = ctk.CTkFrame(frame)
        shapes_frame.pack(fill="x", padx=10, pady=(0, 8))

        for name, var in self.ksztalt_vars.items():
            ctk.CTkCheckBox(
                shapes_frame,
                text=name,
                variable=var,
                command=self.render_summary
            ).pack(anchor="w", padx=10, pady=4)

        ctk.CTkLabel(frame, text="Materiały", font=("Arial", 15, "bold")).pack(anchor="w", padx=15, pady=(4, 4))
        mat_frame = ctk.CTkFrame(frame)
        mat_frame.pack(fill="x", padx=10, pady=(0, 12))

        for name, var in self.material_vars.items():
            ctk.CTkCheckBox(
                mat_frame,
                text=name,
                variable=var,
                command=self.render_summary
            ).pack(anchor="w", padx=10, pady=4)

    def _build_sto_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(frame, text="Modele heurystyczne STO", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))
        ctk.CTkLabel(frame, text="Analiza kolejności zleceń i sumy dodatnich opóźnień.", font=("Arial", 12)).pack(anchor="w", padx=15, pady=(0, 8))

        methods_frame = ctk.CTkFrame(frame)
        methods_frame.pack(fill="x", padx=10, pady=(0, 10))

        for name, var in self.sto_vars.items():
            ctk.CTkCheckBox(methods_frame, text=name, variable=var).pack(anchor="w", padx=10, pady=4)

        grid = ctk.CTkFrame(frame)
        grid.pack(fill="x", padx=10, pady=(0, 10))

        self._entry(grid, "Zlecenia", self.sto_jobs_var, 0, 0)
        self._entry(grid, "Czasy", self.sto_times_var, 1, 0)
        self._entry(grid, "Terminy", self.sto_deadlines_var, 2, 0)

        ctk.CTkButton(
            frame,
            text="Policz modele STO",
            command=self.run_sto_analysis,
            height=36
        ).pack(fill="x", padx=15, pady=(0, 12))

    def _build_actions_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(frame, text="Akcje ML", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 10))

        btns = ctk.CTkFrame(frame)
        btns.pack(fill="x", padx=10, pady=(0, 12))

        ctk.CTkButton(btns, text="Generuj dane", command=self.gen, height=38).pack(fill="x", padx=10, pady=6)
        ctk.CTkButton(btns, text="Wczytaj dane (CSV)", command=self.load_from_disk, height=38).pack(fill="x", padx=10, pady=6)
        ctk.CTkButton(btns, text="Trenuj wybrane modele", command=self.train, height=38).pack(fill="x", padx=10, pady=6)
        ctk.CTkButton(btns, text="▶ Rozwiąż istniejące modele", command=self.solve_existing_models, height=38).pack(fill="x", padx=10, pady=6)

    def _build_preview_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=8)

        ctk.CTkLabel(frame, text="Podgląd danych", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        self.preview_box = ctk.CTkTextbox(frame, height=220)
        self.preview_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.preview_box.configure(state="disabled")

    def _build_sto_results_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=8)

        ctk.CTkLabel(frame, text="Wyniki STO", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        self.sto_box = ctk.CTkTextbox(frame, height=260)
        self.sto_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.sto_box.configure(state="disabled")

    def _build_log_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=(8, 10))

        ctk.CTkLabel(frame, text="Log", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        self.logbox = ctk.CTkTextbox(frame, height=220)
        self.logbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _build_summary_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))

        ctk.CTkLabel(frame, text="Podsumowanie wyboru", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        self.summary_box = ctk.CTkTextbox(frame, height=300)
        self.summary_box.pack(fill="x", padx=15, pady=(0, 15))
        self.summary_box.configure(state="disabled")

    def _build_status_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=8)

        ctk.CTkLabel(frame, text="Szybki status", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        self.status_label = ctk.CTkLabel(frame, text="", font=("Arial", 13), justify="left")
        self.status_label.pack(anchor="w", padx=15, pady=(0, 12))

    def _build_right_hint_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)

        ctk.CTkLabel(frame, text="Panel boczny", font=("Arial", 18, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        hint = (
            "Po prawej masz stały podgląd aktualnych ustawień.\n\n"
            "Po lewej możesz przewijać pełną konfigurację,\n"
            "sekcję STO, dane oraz logi.\n\n"
            "Wyniki heurystyk STO pojawią się tylko wtedy,\n"
            "gdy wybierzesz przynajmniej jeden z modeli:\n"
            "MT / MO / MZO / GENETIC."
        )

        ctk.CTkLabel(
            frame,
            text=hint,
            justify="left",
            wraplength=280,
            font=("Arial", 12)
        ).pack(anchor="w", padx=15, pady=(0, 15))

    def _entry(self, parent, label, var, row, col):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=col, sticky="w", padx=10, pady=6)
        entry = ctk.CTkEntry(parent, textvariable=var, width=160)
        entry.grid(row=row, column=col + 1, sticky="w", padx=10, pady=6)

    def _ui_config(self):
        return {
            "selected_models": [name for name, var in self.model_vars.items() if var.get()],
            "selected_ksztalty": [name for name, var in self.ksztalt_vars.items() if var.get()],
            "selected_materialy": [name for name, var in self.material_vars.items() if var.get()],
            "n": self.n_var.get(),
            "n_machines": self.n_machines_var.get(),
            "test_size": self.test_size_var.get(),
            "seed": self.seed_var.get(),
            "prod_min": self.prod_min_var.get(),
            "prod_max": self.prod_max_var.get(),
            "deadline_min": self.deadline_min_var.get(),
            "deadline_max": self.deadline_max_var.get(),
        }

    def render_summary(self):
        summary = build_main_page_summary(self._ui_config())
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", summary)
        self.summary_box.configure(state="disabled")

    def render_status(self):
        self.status_label.configure(text=build_main_page_status(self.df_train, self.df_test))

    def render_preview(self):
        text = build_dataframe_preview_text(self.df_full, title="Podgląd pełnego zbioru", max_rows=15)
        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.insert("end", text)
        self.preview_box.configure(state="disabled")

    def render_sto_report(self, report: str):
        self.sto_box.configure(state="normal")
        self.sto_box.delete("1.0", "end")
        self.sto_box.insert("end", report)
        self.sto_box.configure(state="disabled")

    def log(self, msg: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def gen(self):
        try:
            cfg = self._ui_config()

            result = generate_and_store_datasets(
                n=int(cfg["n"]),
                n_machines=int(cfg["n_machines"]),
                test_size=float(cfg["test_size"]),
                seed=int(cfg["seed"]),
                ksztalty=cfg["selected_ksztalty"],
                materialy=cfg["selected_materialy"],
                production_time_range=(float(cfg["prod_min"]), float(cfg["prod_max"])),
                deadline_buffer_range=(float(cfg["deadline_min"]), float(cfg["deadline_max"])),
            )

            self.df_full = result["full_df"]
            self.df_train = result["train_df"]
            self.df_test = result["test_df"]

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

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def load_from_disk(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            initialdir=str(DATA_DIR),
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        try:
            result = load_training_data(path=path, train_ratio=0.8)

            self.df_full = result["full_df"]
            self.df_train = result["train_df"]
            self.df_test = result["test_df"]

            for line in result["messages"]:
                self.log(line)

            self.render_status()
            self.render_preview()

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def train(self):
        if self.df_train is None:
            messagebox.showerror("Błąd", "Brak danych treningowych")
            return

        selected_models = self._ui_config()["selected_models"]
        if not selected_models:
            messagebox.showerror("Błąd", "Wybierz przynajmniej jeden model")
            return

        threading.Thread(target=self.train_worker, args=(selected_models,), daemon=True).start()

    def train_worker(self, selected_models):
        try:
            result = train_models_flow(
                df_train=self.df_train,
                selected_models=selected_models,
                metadata=self.last_generation_metadata,
                progress_callback=lambda p: self.log(f"⏳ ScheduleModel: {p:.1f}%")
            )

            for line in result["messages"]:
                self.log(line)

        except Exception as e:
            self.log(f"❌ Błąd treningu: {e}")

    def solve_existing_models(self):
        threading.Thread(target=self._solve_existing_models_thread, daemon=True).start()

    def _solve_existing_models_thread(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Wybierz wytrenowany model",
                initialdir=str(MODELS_DIR),
                filetypes=[("Pickle", "*.pkl")]
            )
            if not model_path:
                return

            data_path = filedialog.askopenfilename(
                title="Wybierz dane do rozwiązania",
                initialdir=str(DATA_DIR),
                filetypes=[("CSV", "*.csv")]
            )
            if not data_path:
                return

            result = solve_models_flow(model_path=model_path, data_path=data_path)

            for line in result["messages"]:
                self.log(line)

        except Exception as e:
            self.log(f"❌ Błąd rozwiązywania: {e}")

    def run_sto_analysis(self):
        selected = [name for name, var in self.sto_vars.items() if var.get()]
        if not selected:
            messagebox.showinfo("Informacja", "Wybierz przynajmniej jeden model STO.")
            return

        try:
            result = analyze_sto_models(
                job_ids_text=self.sto_jobs_var.get(),
                processing_text=self.sto_times_var.get(),
                deadlines_text=self.sto_deadlines_var.get(),
                selected_methods=selected,
            )
            self.render_sto_report(result["report"])

        except Exception as e:
            messagebox.showerror("Błąd STO", str(e))

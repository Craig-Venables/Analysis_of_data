import os
from pathlib import Path
from typing import List, Callable

from PyQt5 import QtCore, QtWidgets

# Import the analysis function from the analysis module
from analysis import analyze_hdf5_levels


class LogEmitter(QtCore.QObject):
    message = QtCore.pyqtSignal(str)


class AnalysisWorker(QtCore.QThread):
    progressed = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        hdf5_file_path: str,
        solution_devices_excel_path: str,
        yield_dir_path: str,
        identifiers: List[str],
        match_all: bool,
        voltage_val: float,
        output_dir: str,
        valid_classifications: List[str],
        plot_options=None,
        save_options=None,
        classification_options=None,
    ) -> None:
        super().__init__()
        self.hdf5_file_path = hdf5_file_path
        self.solution_devices_excel_path = solution_devices_excel_path
        self.yield_dir_path = yield_dir_path
        self.identifiers = identifiers
        self.match_all = match_all
        self.voltage_val = voltage_val
        self.output_dir = output_dir
        self.valid_classifications = valid_classifications
        self.plot_options = plot_options
        self.save_options = save_options
        self.classification_options = classification_options

    def run(self) -> None:
        try:
            def _log(msg):
                try:
                    self.progressed.emit(str(msg))
                except Exception:
                    pass

            analyze_hdf5_levels(
                hdf5_file_path=self.hdf5_file_path,
                solution_devices_excel_path=self.solution_devices_excel_path,
                yield_dir_path=self.yield_dir_path,
                identifiers=self.identifiers,
                match_all=self.match_all,
                voltage_val=self.voltage_val,
                output_dir=self.output_dir,
                valid_classifications=self.valid_classifications,
                log_fn=_log,
                plot_options=self.plot_options,
                save_options=self.save_options,
                classification_options=self.classification_options,
            )
            self.finished_ok.emit()
        except Exception as exc:  # pragma: no cover - GUI runtime
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Memristor Analysis GUI")
        self.setMinimumSize(900, 600)

        home = str(Path.home())
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # --- Tab 1: Data & Paths ---
        tab_paths = QtWidgets.QWidget()
        tabs.addTab(tab_paths, "Data & Paths")
        paths_form = QtWidgets.QFormLayout(tab_paths)

        self.hdf5_edit = QtWidgets.QLineEdit()
        self.hdf5_edit.setToolTip("Path to the HDF5 data file (auto-detected to latest 'Memristor_data_YYYYMMDD.h5' if found)")
        self.hdf5_browse = QtWidgets.QPushButton("Browse…")
        self.hdf5_browse.setToolTip("Pick an HDF5 file to analyze")
        hdf5_row = QtWidgets.QHBoxLayout()
        hdf5_row.addWidget(self.hdf5_edit)
        hdf5_row.addWidget(self.hdf5_browse)
        paths_form.addRow("HDF5 File:", self._wrap(hdf5_row))

        self.excel_edit = QtWidgets.QLineEdit()
        self.excel_edit.setToolTip("Path to 'solutions and devices.xlsx' containing fabrication metadata")
        self.excel_browse = QtWidgets.QPushButton("Browse…")
        self.excel_browse.setToolTip("Pick the solutions/devices Excel workbook")
        excel_row = QtWidgets.QHBoxLayout()
        excel_row.addWidget(self.excel_edit)
        excel_row.addWidget(self.excel_browse)
        paths_form.addRow("Solutions Excel:", self._wrap(excel_row))

        self.yield_dir_edit = QtWidgets.QLineEdit()
        self.yield_dir_edit.setToolTip("Directory containing yield_dict.json and yield_dict_section.json")
        self.yield_dir_browse = QtWidgets.QPushButton("Browse…")
        self.yield_dir_browse.setToolTip("Pick the directory with yield JSON files")
        yield_row = QtWidgets.QHBoxLayout()
        yield_row.addWidget(self.yield_dir_edit)
        yield_row.addWidget(self.yield_dir_browse)
        paths_form.addRow("Yield JSON Dir:", self._wrap(yield_row))

        self.output_dir_edit = QtWidgets.QLineEdit("saved_files")
        self.output_dir_edit.setToolTip("Directory under this repo where results/plots will be saved")
        self.output_dir_browse = QtWidgets.QPushButton("Browse…")
        self.output_dir_browse.setToolTip("Pick a directory to save outputs")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.output_dir_edit)
        out_row.addWidget(self.output_dir_browse)
        paths_form.addRow("Output Dir:", self._wrap(out_row))

        # --- Tab 2: Analysis Options ---
        tab_opts = QtWidgets.QWidget()
        tabs.addTab(tab_opts, "Analysis Options")
        opts_form = QtWidgets.QFormLayout(tab_opts)

        self.identifiers_edit = QtWidgets.QLineEdit("PMMA")
        self.identifiers_edit.setToolTip("Comma-separated list of substrings to filter dataset keys (e.g., 'PMMA,Gold')")
        opts_form.addRow("Identifiers (comma-separated):", self.identifiers_edit)

        self.match_all_checkbox = QtWidgets.QCheckBox("Require all identifiers to match")
        self.match_all_checkbox.setChecked(True)
        self.match_all_checkbox.setToolTip("If checked, only datasets containing ALL identifiers are included. If unchecked, ANY identifier is sufficient.")
        opts_form.addRow("Match Mode:", self.match_all_checkbox)

        self.voltage_spin = QtWidgets.QDoubleSpinBox()
        self.voltage_spin.setRange(0.0, 1000.0)
        self.voltage_spin.setDecimals(4)
        self.voltage_spin.setSingleStep(0.01)
        self.voltage_spin.setValue(0.1)
        self.voltage_spin.setToolTip("Upper bound of voltage window [0,V] used to compute average initial resistance")
        opts_form.addRow("Voltage upper bound (V):", self.voltage_spin)

        self.classes_edit = QtWidgets.QLineEdit("Memristive")
        self.classes_edit.setToolTip("Valid classifications to include in analysis; comma-separated (e.g., 'Memristive,Ohmic')")
        opts_form.addRow("Valid classes (comma-separated):", self.classes_edit)

        # Classification-only option
        self.classify_checkbox = QtWidgets.QCheckBox("Run classification-only")
        self.classify_checkbox.setToolTip("If checked, run a simple sweep classification and export a CSV without resistance analysis")
        opts_form.addRow("Classification:", self.classify_checkbox)

        # --- Tab 3: Classification & Saving ---
        tab_class = QtWidgets.QWidget()
        tabs.addTab(tab_class, "Classification & Saving")
        class_form = QtWidgets.QFormLayout(tab_class)

        self.class_use_hdf5_checkbox = QtWidgets.QCheckBox("Filter by HDF5 labels (valid classes)")
        self.class_use_hdf5_checkbox.setChecked(True)
        self.class_use_hdf5_checkbox.setToolTip("If checked, only sweeps whose mapped HDF5 label is in 'Valid classes' will be analyzed")
        class_form.addRow("Use HDF5 Labels:", self.class_use_hdf5_checkbox)

        self.class_skip_capacitive_checkbox = QtWidgets.QCheckBox("Skip sweeps classified as capacitive (heuristic)")
        self.class_skip_capacitive_checkbox.setChecked(True)
        class_form.addRow("Skip Capacitive:", self.class_skip_capacitive_checkbox)

        self.class_skip_halfsweep_checkbox = QtWidgets.QCheckBox("Skip half-sweeps (don’t cross 0 current)")
        self.class_skip_halfsweep_checkbox.setChecked(False)
        class_form.addRow("Skip Half-Sweep:", self.class_skip_halfsweep_checkbox)

        self.class_skip_negative_checkbox = QtWidgets.QCheckBox("Skip negative-resistance in initial window")
        self.class_skip_negative_checkbox.setChecked(False)
        class_form.addRow("Skip Negative R:", self.class_skip_negative_checkbox)

        self.save_neg_checkbox = QtWidgets.QCheckBox("Save negative-resistance sweeps")
        self.save_neg_checkbox.setChecked(True)
        class_form.addRow("Save Negative R:", self.save_neg_checkbox)

        self.save_half_checkbox = QtWidgets.QCheckBox("Save half-sweeps")
        self.save_half_checkbox.setChecked(True)
        class_form.addRow("Save Half-Sweep:", self.save_half_checkbox)

        self.save_cap_checkbox = QtWidgets.QCheckBox("Save capacitive sweeps")
        self.save_cap_checkbox.setChecked(True)
        class_form.addRow("Save Capacitive:", self.save_cap_checkbox)

        self.save_let_checkbox = QtWidgets.QCheckBox("Save let-through (kept) sweeps")
        self.save_let_checkbox.setChecked(True)
        class_form.addRow("Save Let-through:", self.save_let_checkbox)

        self.class_csv_source = QtWidgets.QComboBox()
        self.class_csv_source.addItems(["heuristic", "hdf5"])
        self.class_csv_source.setToolTip("Source for classification-only CSV: heuristic rules or HDF5 labels")
        class_form.addRow("Classification CSV Source:", self.class_csv_source)

        # --- Tab 4: Plot Options ---
        tab_plots = QtWidgets.QScrollArea()
        tabs.addTab(tab_plots, "Plot Options")
        plots_container = QtWidgets.QWidget()
        plots_form = QtWidgets.QFormLayout(plots_container)
        tab_plots.setWidgetResizable(True)
        tab_plots.setWidget(plots_container)

        self.plot_yield_conc = QtWidgets.QCheckBox("Yield vs Concentration (linear x)")
        self.plot_yield_conc.setChecked(True)
        plots_form.addRow(self.plot_yield_conc)

        self.plot_yield_conc_logx = QtWidgets.QCheckBox("Yield vs Concentration (log x)")
        self.plot_yield_conc_logx.setChecked(True)
        plots_form.addRow(self.plot_yield_conc_logx)

        self.plot_yield_spacing = QtWidgets.QCheckBox("Yield vs Spacing (linear x)")
        self.plot_yield_spacing.setChecked(True)
        plots_form.addRow(self.plot_yield_spacing)

        self.plot_yield_spacing_logx = QtWidgets.QCheckBox("Yield vs Spacing (log x)")
        self.plot_yield_spacing_logx.setChecked(False)
        plots_form.addRow(self.plot_yield_spacing_logx)

        self.plot_3d = QtWidgets.QCheckBox("3D: Concentration vs Spacing vs Yield")
        self.plot_3d.setChecked(True)
        plots_form.addRow(self.plot_3d)

        self.plot_facet_conc = QtWidgets.QCheckBox("Facet: Concentration vs Yield by Polymer")
        self.plot_facet_conc.setChecked(True)
        plots_form.addRow(self.plot_facet_conc)

        self.plot_facet_spacing = QtWidgets.QCheckBox("Facet: Spacing vs Yield by Polymer")
        self.plot_facet_spacing.setChecked(True)
        plots_form.addRow(self.plot_facet_spacing)

        self.plot_corr_heatmap = QtWidgets.QCheckBox("Correlation heatmap (Yield, Concentration, Spacing, etc.)")
        self.plot_corr_heatmap.setChecked(True)
        plots_form.addRow(self.plot_corr_heatmap)

        self.plot_pairplot = QtWidgets.QCheckBox("Pairplot of numeric variables")
        self.plot_pairplot.setChecked(False)
        plots_form.addRow(self.plot_pairplot)

        self.plot_violin = QtWidgets.QCheckBox("Violin: Yield distribution by Polymer")
        self.plot_violin.setChecked(True)
        plots_form.addRow(self.plot_violin)

        self.plot_box = QtWidgets.QCheckBox("Box: Yield by Polymer")
        self.plot_box.setChecked(True)
        plots_form.addRow(self.plot_box)

        # Run / log
        run_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setToolTip("Run analysis or classification with the selected options")
        run_row.addStretch(1)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

        # Connections
        self.hdf5_browse.clicked.connect(self._pick_hdf5)
        self.excel_browse.clicked.connect(self._pick_excel)
        self.yield_dir_browse.clicked.connect(self._pick_yield_dir)
        self.output_dir_browse.clicked.connect(self._pick_output_dir)
        self.run_btn.clicked.connect(self._start)

        # Defaults and auto-detection
        self._auto_populate_defaults()

        self._worker = None  # type: ignore
    
    def _auto_populate_defaults(self) -> None:
        # Output dir under repo
        out = _cwd_output_dir()
        _ensure_dir(out)
        self.output_dir_edit.setText(out)

        # Latest HDF5
        latest = _find_latest_hdf5()
        if _exists_file(latest):
            self.hdf5_edit.setText(latest)

        # Excel and yield
        excel, ydir = _guess_excel_and_yield()
        if _exists_file(excel):
            self.excel_edit.setText(excel)
        if _exists_dir(ydir):
            self.yield_dir_edit.setText(ydir)

    def _wrap(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    def _pick_hdf5(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select HDF5 file", str(Path.home()), "HDF5 (*.h5 *.hdf5);;All files (*)")
        if path:
            self.hdf5_edit.setText(path)

    def _pick_excel(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Excel file", str(Path.home()), "Excel (*.xlsx *.xls);;All files (*)")
        if path:
            self.excel_edit.setText(path)

    def _pick_yield_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select yield JSON directory", str(Path.home()))
        if path:
            self.yield_dir_edit.setText(path)

    def _pick_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", str(Path.cwd()))
        if path:
            self.output_dir_edit.setText(path)

    def _append_log(self, msg: str) -> None:
        self.log.append(msg)
        self.log.ensureCursorVisible()

    def _start(self) -> None:
        hdf5 = self.hdf5_edit.text().strip()
        excel = self.excel_edit.text().strip()
        ydir = self.yield_dir_edit.text().strip()
        out = self.output_dir_edit.text().strip() or "saved_files"
        idents = [s.strip() for s in self.identifiers_edit.text().split(",") if s.strip()]
        match_all = self.match_all_checkbox.isChecked()
        voltage = float(self.voltage_spin.value())
        classes = [s.strip() for s in self.classes_edit.text().split(",") if s.strip()]
        run_classify = self.classify_checkbox.isChecked()

        if not (os.path.isfile(hdf5) and os.path.isfile(excel) and os.path.isdir(ydir)):
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please provide valid paths for HDF5, Excel and yield directory.")
            return

        os.makedirs(out, exist_ok=True)

        self.log.clear()
        self.run_btn.setEnabled(False)

        if run_classify:
            # Lightweight classification-only
            source = getattr(self, 'class_csv_source', None)
            source_text = source.currentText().strip() if source else 'heuristic'
            self._start_classification(hdf5, idents, match_all, out, source_text)
            return

        # Build options
        plot_opts = self._build_plot_options()
        save_opts = self._build_save_options()
        class_opts = self._build_classification_options()

        worker = AnalysisWorker(
            hdf5_file_path=hdf5,
            solution_devices_excel_path=excel,
            yield_dir_path=ydir,
            identifiers=idents if idents else ["PMMA"],
            match_all=match_all,
            voltage_val=voltage,
            output_dir=out,
            valid_classifications=classes if classes else ["Memristive"],
            plot_options=plot_opts,
            save_options=save_opts,
            classification_options=class_opts,
        )
        worker.progressed.connect(self._append_log)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _start_classification(self, hdf5: str, idents: list, match_all: bool, out: str, source: str) -> None:
        from analysis import classify_sweeps
        self._append_log("Running classification-only…")
        try:
            if source == 'hdf5':
                path = self._classify_from_hdf5(hdf5, idents if idents else ["PMMA"], match_all, out)
            else:
                path = classify_sweeps(
                    hdf5_file_path=hdf5,
                    identifiers=idents if idents else ["PMMA"],
                    match_all=match_all,
                    output_dir=out,
                    log_fn=self._append_log,
                )
            self._append_log(f"Done. CSV: {path}")
        except Exception as exc:
            self._append_log(f"Error: {exc}")
        finally:
            self.run_btn.setEnabled(True)

    def _build_plot_options(self):
        from analysis import PlotOptions
        return PlotOptions(
            yield_vs_concentration=self.plot_yield_conc.isChecked(),
            yield_vs_concentration_logx=self.plot_yield_conc_logx.isChecked(),
            yield_vs_spacing=self.plot_yield_spacing.isChecked(),
            yield_vs_spacing_logx=self.plot_yield_spacing_logx.isChecked(),
            plot_3d=self.plot_3d.isChecked(),
            facet_by_polymer_conc_yield=self.plot_facet_conc.isChecked(),
            facet_by_polymer_spacing_yield=self.plot_facet_spacing.isChecked(),
            correlation_heatmap=self.plot_corr_heatmap.isChecked(),
            pairplot_numeric=self.plot_pairplot.isChecked(),
            violin_yield_by_polymer=self.plot_violin.isChecked(),
            box_yield_by_polymer=self.plot_box.isChecked(),
            resistance_summary_plots=True,
        )

    def _build_save_options(self):
        from analysis import SaveOptions
        return SaveOptions(
            save_negative_resistance=self.save_neg_checkbox.isChecked(),
            save_half_sweep=self.save_half_checkbox.isChecked(),
            save_capacitive=self.save_cap_checkbox.isChecked(),
            save_let_through=self.save_let_checkbox.isChecked(),
        )

    def _build_classification_options(self):
        from analysis import ClassificationOptions
        return ClassificationOptions(
            filter_by_hdf5_labels=self.class_use_hdf5_checkbox.isChecked(),
            apply_capacitive_skip=self.class_skip_capacitive_checkbox.isChecked(),
            apply_half_sweep_skip=self.class_skip_halfsweep_checkbox.isChecked(),
            apply_negative_resistance_skip=self.class_skip_negative_checkbox.isChecked(),
            csv_source=self.class_csv_source.currentText() if hasattr(self, 'class_csv_source') else 'heuristic',
        )

    def _classify_from_hdf5(self, hdf5: str, idents: list, match_all: bool, out: str) -> str:
        import h5py, pandas as pd, os
        from analysis import return_data
        from Key_functions import filter_keys_by_identifiers
        results = []
        with h5py.File(hdf5, 'r') as store:
            grouped_keys = filter_keys_by_identifiers(store, idents, match_all=match_all)
            base_keys = sorted(set(k.rsplit('_', 2)[0] for k in grouped_keys))
            for base_key in base_keys:
                df_raw, _, _ = return_data(base_key, store)
                label = df_raw['classification'].iloc[0] if 'classification' in df_raw.columns and not df_raw.empty else 'Unknown'
                results.append({"key": base_key, "classification": label})
        out_csv = os.path.join(out, 'classification_report_hdf5.csv')
        pd.DataFrame(results).to_csv(out_csv, index=False)
        return out_csv

    def _on_finished(self) -> None:
        self._append_log("\nAnalysis complete.")
        self.run_btn.setEnabled(True)
        self._worker = None

    def _on_failed(self, err: str) -> None:
        self._append_log(f"\nError: {err}")
        self.run_btn.setEnabled(True)
        self._worker = None


def launch_gui() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()


    
def _find_latest_hdf5() -> str:
    base = Path(r"C:\Users\Craig-Desktop\OneDrive - The University of Nottingham\Documents\Phd\2) Data\1) Devices\1) Memristors")
    if not base.exists():
        return ""
    candidates = sorted(base.glob("Memristor_data_*.h5"))
    if not candidates:
        return ""
    # pick by lexicographic order of filename; relies on YYYYMMDD
    return str(candidates[-1])


def _guess_excel_and_yield() -> tuple:
    home = Path.home()
    excel = home / Path("OneDrive - The University of Nottingham/Documents/Phd/solutions and devices.xlsx")
    ydir = home / Path("OneDrive - The University of Nottingham/\Documents/Phd/1) Projects/1) Memristors/1) Curated Data")
    return (str(excel) if excel.exists() else "", str(ydir) if ydir.exists() else "")


def _norm(s: str) -> str:
    return s.replace("/", os.sep).replace("\\", os.sep)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _populate_if_exists(edit: QtWidgets.QLineEdit, value: str) -> None:
    if value:
        edit.setText(_norm(value))


def _cwd_output_dir() -> str:
    return _norm(str(Path.cwd() / "saved_files"))


def _cwd_repo_dir() -> str:
    return _norm(str(Path.cwd()))


def _exists_file(path: str) -> bool:
    return bool(path and os.path.isfile(path))


def _exists_dir(path: str) -> bool:
    return bool(path and os.path.isdir(path))



if __name__ == "__main__":
    launch_gui()

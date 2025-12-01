# DICStudio

**DICStudio** is a Windows-only, Python 3.8–based 2D digital image correlation (DIC) toolbox with a modern Qt GUI. It provides a full port and extension of the Ncorr 2D DIC software into Python, using a compiled backend (`ncorr.pyd`) plus additional preprocessing, automation, and post-processing tools.

> DICStudio is an independent research project and is **not officially affiliated** with the original Ncorr authors. It is a Python port and extension built on top of their open-source work.

---

## Key features

### Ncorr-based DIC core

- Uses the original Ncorr DIC algorithms through a compiled Python extension: `ncorr.pyd`.
- Supports running full-field displacement and strain analysis.
- Can read and write Ncorr `.bin` files:
  - `save_DIC_input`, `save_DIC_output`
  - `save_strain_input`, `save_strain_output`
  - `load_DIC_output`, `load_strain_output`
- Long-running jobs run in a background `QThread` (`NcorrWorker`) with logging and progress reporting.

---

### Preprocessing & experiment setup

The `Preprocessing.MainWindow` provides a multi-tab GUI for:

- **Frame extraction** (`FrameExtractorTab`)
  - Extract images from video files or sequences.
  - Manage reference and deformed frames.

- **ROI definition & mask generation** (`ROITab`, `ROIDrawer`)
  - Draw and edit regions of interest.
  - Generate binary ROI masks compatible with Ncorr.

- **Calibration**
  - Unit / scale calibration.
  - Camera-related utilities and pattern quality checks.

- **Pattern generation & virtual experiments**
  - Tools such as `PatternGeneratorTab`, `PatternMaster`, `VirtualLabGeneratorTab`.
  - Generate speckle patterns and perform virtual experiments / comparisons.

- **Quality assessment & uncertainty**
  - Tabs like `QualityAssessmentTab`, `UQComparisonTab` for evaluating pattern, ROI and analysis quality.

- **Background removal (optional)**
  - Integration with `rembg` for automatic background removal.
  - On first use, DICStudio can download the required ONNX model (~176 MB) with user consent.

---

### Manual DIC mode

`ManualModeWindow` groups the main workflows in tabs:

- **Preprocessing** – all the ROI, pattern and calibration tools.
- **DIC / strain analysis** – the main processing window (`MainWindow` in `main_app.py`) to run Ncorr-based DIC and strain.
- **Point inspection & analysis** – `PointInspectorApp` for detailed post-processing.

This mode is intended for users who want full manual control over each step.

---

### Automation controller

The `AutomationController` provides high-level automation:

- Automatically detects ROIs (e.g. using largest contour from frames).
- Prepares masks and launches Ncorr analysis in the background.
- Integrates with `FrameExtractorTab`, `NcorrWorker` and the GUI.
- Designed for batch / repeated experiments where the workflow is similar.

---

### Point inspector & post-processing

`point_inspector.py` implements an extensive **Point Inspector** GUI (`PointInspectorApp`) with multiple analysis tabs, including:

- **PointInspectorTab** – inspect displacement/strain time series at selected points.
- **RelativeDisplacementTab** – relative displacements between points or regions.
- **SDICalculatorTab** – scalar metrics derived from DIC/strain data.
- **PoissonRatioTab** – compute and visualize Poisson-like quantities.
- **AdvancedPlottingTab** – flexible plotting (filters, smoothing, transforms).
- **CMODTab** – crack mouth opening displacement tools.
- **FractureEnergyTab** – fracture energy and related metrics.
- **BinToCsvTab** – convert Ncorr `.bin` files into CSV.

Most operations use NumPy/SciPy and Matplotlib, and results can be exported as CSV or image files.

---

### Theming & launcher

- `ThemeManager` reads `dark_theme.qss` and `light_theme.qss`, detects system theme and applies appropriate styling.
- `LauncherWindow` provides a simple start screen:
  - Launch **Manual DIC Mode**.
  - Launch **Automation / Batch Mode**.
  - Open plotting / inspection windows.
- Custom icons:
  - `logo.png`, `icon_dark.png`, `icon_light.png`.

---

## Repository layout

```text
DICStudio/
  src/
    analysis_worker.py
    automatic_config_dialog.py
    automation_controller.py
    custom_widgets.py
    dialogs.py
    launcher.py
    main_app.py
    manual_mode_window.py
    plot_window.py
    point_inspector.py
    Preprocessing.py
    theme_manager.py
    __init__.py
  dependencies/
    ncorr.pyd           # compiled Ncorr backend for Python 3.8, Windows
    D3Dcompiler_47.dll
    libblas.dll
    libEGL.dll
    libfftw3-3.dll
    libgcc_s_sjlj-1.dll
    libgfortran-3.dll
    libGLESv2.dll
    liblapack.dll
    libquadmath-0.dll
    opencv_core300.dll
    opencv_highgui300.dll
    opencv_imgcodecs300.dll
    opencv_imgproc300.dll
    opencv_videoio300.dll
    python38.dll
    ncorr.exp
    ncorr.lib
    ncorr_lib.lib
  run_app.py
  requirements.txt
  dark_theme.qss
  light_theme.qss
  logo.png
  icon_dark.png
  icon_light.png
  force_data.csv
  LICENSE
```

> The `dependencies/` folder bundles the Windows-specific runtime libraries needed for the compiled `ncorr.pyd` and related functionality. These binaries are provided as part of this research prototype; if you intend to redistrib­ute them in other contexts (e.g. commercial software), you should review the corresponding third-party licenses.

---

## Installation

### 1. Environment

- **OS:** Windows 10/11, 64-bit
- **Python:** **3.8.x only**  
  (`ncorr.pyd` and the DLLs in `dependencies/` are built for Python 3.8 on Windows.)

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scriptsctivate
```

### 2. Install Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

This will install NumPy, SciPy, scikit-image, PySide6, OpenCV, rembg, Matplotlib and other required packages.

---

## Running DICStudio

From the repository root:

```bash
python run_app.py
```

What this script does:

- Adds `src/` and `dependencies/` to `sys.path`.
- Configures Qt high-DPI behaviour.
- Creates a `QApplication`, applies the current theme, and shows `LauncherWindow`.

From the launcher you can:

- Start **Manual DIC Mode**.
- Start **Automation / Batch Mode**.
- Access analysis / plotting tools.

---

## Basic workflow

1. **Preprocessing**
   - Load images or video, extract frames.
   - Define ROI and masks.
   - Calibrate units and camera if needed.
   - Optionally remove background and assess pattern / ROI quality.

2. **Run DIC (Ncorr backend)**
   - Configure subset size, step, strain radius and other parameters.
   - Start the analysis and monitor the progress dialog.
   - Save or load Ncorr `.bin` outputs as needed.

3. **Post-processing**
   - Use the Point Inspector to:
     - Select points or regions of interest.
     - Compute time-series, relative displacements, CMOD, fracture metrics, etc.
   - Export CSV files and plots for further analysis or publication.

---

## Citing DICStudio and Ncorr

If you use DICStudio in academic work, please cite:

### DICStudio (this project)

> A. Mirjafari, “DICStudio: A Python 3.8 Port and Extension of Ncorr for 2D Digital Image Correlation,” *[Journal/Conference]*, [Year].  
> (Update this section once the paper is accepted and has a final reference.)

### Original Ncorr work

The original Ncorr project and its algorithms are described in:

> J. Blaber, B. Adair, and A. Antoniou,  
> “Ncorr: Open-Source 2D Digital Image Correlation Matlab Software,”  
> *Experimental Mechanics*, 55(6), 1105–1122, 2015.  
> https://doi.org/10.1007/s11340-015-0009-1

and the accompanying code is hosted at:  
`https://github.com/justinblaber/ncorr_2D_cpp`

---

## License

DICStudio is released under the **BSD 3-Clause License** (see `LICENSE`).

- The Python source code in `src/` and the project infrastructure are © Ali Mirjafari.
- The compiled backend `ncorr.pyd` and corresponding `ncorr_lib` artifacts are derived from the original Ncorr sources, which are also licensed under BSD 3-Clause. For full licensing details of Ncorr, please refer to the original Ncorr repository and documentation.

Third-party libraries (Python itself, FFTW, LAPACK, OpenCV, etc.) and their DLLs are subject to their own licenses. They are included here solely as runtime dependencies for research and reproducibility purposes.

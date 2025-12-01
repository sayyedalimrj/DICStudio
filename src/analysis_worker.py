# -*- coding: utf-8 -*-
"""
DICStudio - Python-based 2D Digital Image Correlation (DIC) software.

This file is part of DICStudio, a full Python port of the Ncorr 2D DIC MATLAB
software with additional tools and GUI features.

Copyright (c) 2025, Seyed Ali Mirjafari
All rights reserved.

Portions of this work are based on:
Blaber, J., Adair, B., & Antoniou, A. (2015).
"Ncorr: Open-Source 2D Digital Image Correlation Matlab Software",
Experimental Mechanics, 55(6), 1105–1122.

DICStudio is distributed under the BSD 3-Clause License.
See the LICENSE file in the repository root for details.

Developed for research and educational use. No warranty is provided;
use at your own risk.

"""
import os
import traceback
from typing import List, Optional

import numpy as np
import cv2

try:
    import ncorr
except Exception as e:  
    ncorr = None

from PySide6.QtCore import QThread, Signal

# ---- helpers ----
def _to_numpy(arr):
    """Try to convert ncorr Array2D-like to numpy, gracefully handling Python/ncorr types."""
    try:
        return arr.to_numpy()
    except Exception:
        try:
            a = arr.get_array()
            try:
                return a.to_numpy()
            except Exception:
                return np.array(a)
        except Exception:
            return np.array(arr)

def _postprocess_field(field_data: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    processed_field = field_data.copy()
    processed_field[~roi_mask] = np.nan
    valid_mask = np.isfinite(processed_field)
    if not np.any(valid_mask):
        return processed_field
    median_val = np.nanmedian(processed_field)
    if np.isnan(median_val):
        median_val = 0.0
    field_for_smoothing = np.nan_to_num(processed_field, nan=median_val)
    smoothed_field = cv2.GaussianBlur(field_for_smoothing.astype(np.float32), (21, 21), 0)
    final_field = np.full_like(processed_field, np.nan, dtype=np.float32)
    final_field[roi_mask] = smoothed_field[roi_mask]
    return final_field


class NcorrWorker(QThread):
    """Runs a full ncorr DIC (+ optional strain) analysis in a QThread."""
    log = Signal(str)
    progress = Signal(int)
    done = Signal(bool, str, object, object)

    def __init__(self, cfg: dict, img_paths: List[str], roi_mask: np.ndarray, seeds: Optional[List] = None):
        super().__init__()
        self.cfg = dict(cfg or {})
        self.img_paths = list(img_paths or [])
        self.roi_mask = (roi_mask.astype(bool) if roi_mask is not None else None)
        self.seeds = list(seeds or [])
        self.DIC_in = None  # saved for downstream use (e.g., external strain calc)

    def _log(self, s: str):
        self.log.emit(str(s))

    def _save_bins(self, DIC_in_obj, DIC_out_obj, S_in_obj, S_out_obj, out_dir: str):
        try:
            os.makedirs(out_dir, exist_ok=True)
            if ncorr is None:
                self._log("ncorr is not available; cannot save .bin files.")
                return
            f_dic_in = os.path.join(out_dir, "DIC_input.bin")
            f_dic_out = os.path.join(out_dir, "DIC_output.bin")
            f_str_in = os.path.join(out_dir, "strain_input.bin")
            f_str_out = os.path.join(out_dir, "strain_output.bin")
            if hasattr(ncorr, "save_DIC_input") and DIC_in_obj is not None:
                ncorr.save_DIC_input(DIC_in_obj, f_dic_in)
            if hasattr(ncorr, "save_DIC_output") and DIC_out_obj is not None:
                ncorr.save_DIC_output(DIC_out_obj, f_dic_out)
            if S_in_obj is not None and S_out_obj is not None:
                if hasattr(ncorr, "save_strain_input"):
                    ncorr.save_strain_input(S_in_obj, f_str_in)
                if hasattr(ncorr, "save_strain_output"):
                    ncorr.save_strain_output(S_out_obj, f_str_out)
            self._log(f"Saved binaries to: {out_dir}")
        except Exception as e:
            self._log(f"Warning: failed to save .bin files: {e}")

    def run(self):
        dic_out = None
        s_in = None
        s_out = None
        try:
            if ncorr is None:
                raise RuntimeError("ncorr module is not installed or failed to import.")
            cfg = self.cfg
            out_dir = cfg.get('out_dir', './ncorr_output') or './ncorr_output'
            os.makedirs(out_dir, exist_ok=True)
            if self.roi_mask is None or not np.any(self.roi_mask):
                raise ValueError("ROI mask is empty or None.")
            if not self.img_paths or len(self.img_paths) < 2:
                raise ValueError("img_paths must include reference and at least one current image.")
            self._log(f"Images: {len(self.img_paths)}")
            self.progress.emit(5)
            # Build ROI
            self._log("Building ROI…")
            roi = ncorr.ROI2D(self.roi_mask.astype(bool))
            self.progress.emit(10)
            # Prepare DIC input
            self._log("Preparing DIC input…")
            interp = getattr(ncorr.INTERP, str(cfg.get('interp', 'QUINTIC_BSPLINE_PRECOMPUTE')))
            subreg = getattr(ncorr.SUBREGION, str(cfg.get('subregion', 'CIRCLE')))
            dic_cfg_enum = getattr(ncorr.DIC_analysis_config, str(cfg.get('dic_config', 'KEEP_MOST_POINTS')))
            scale_factor = int(cfg.get('scale_factor', 1))
            radius = int(cfg.get('radius', 20))
            threads = int(cfg.get('threads', max(1, os.cpu_count() or 1)))
            debug = bool(cfg.get('debug', False))
            self.DIC_in = ncorr.DIC_analysis_input(self.img_paths, roi, scale_factor, interp, subreg, radius, threads, dic_cfg_enum, debug)
            self.progress.emit(20)
            # Run DIC
            self._log("Running DIC…")
            dic_out, correlation_maps = None, []
            if hasattr(ncorr, "DIC_analysis_with_cc"):
                dic_out, correlation_maps = ncorr.DIC_analysis_with_cc(self.DIC_in)
            else:
                dic_out = ncorr.DIC_analysis(self.DIC_in)
                correlation_maps = []
            self.progress.emit(70)
            # Optional perspective and units
            if cfg.get('change_perspective', cfg.get('perspective', '').upper() == "EULERIAN"):
                self._log("Changing perspective…")
                dic_out = ncorr.change_perspective(dic_out, interp)
            if float(cfg.get('units_per_pixel', 0.0)) > 0.0 and cfg.get('units'):
                self._log("Setting units…")
                dic_out = ncorr.set_units(dic_out, str(cfg['units']), float(cfg['units_per_pixel']))
            self.progress.emit(75)
            # Optional strain
            strain_radius = int(cfg.get('strain_radius', cfg.get('strain', {}).get('radius', 0) if isinstance(cfg.get('strain'), dict) else 0))
            if strain_radius > 0:
                self._log("Running strain analysis…")
                s_in = ncorr.strain_analysis_input(self.DIC_in, dic_out, subreg, strain_radius)
                s_out = ncorr.strain_analysis(s_in)
                self.progress.emit(85)
                try:
                    self._log("Post-processing strain fields…")
                    num_frames = len(getattr(s_out, "strains", []))
                    if correlation_maps:
                        cmask0 = _to_numpy(correlation_maps[min(0, len(correlation_maps)-1)])
                        use_cut = float(cfg.get('cutoff_corrcoef', 1.8))
                        default_corr_mask = cmask0 < use_cut
                    else:
                        default_corr_mask = self.roi_mask.astype(bool)
                    for i, strain_frame in enumerate(getattr(s_out, "strains", [])):
                        self._log(f"Cleaning frame {i+1}/{num_frames}…")
                        self.progress.emit(85 + int(((i + 1) / max(1, num_frames)) * 10))
                        roi_m = strain_frame.get_roi().get_mask().to_numpy() if hasattr(strain_frame.get_roi(), "get_mask") else default_corr_mask
                        roi_m = np.asarray(roi_m, dtype=bool)
                        exx = _to_numpy(strain_frame.get_exx().get_array()) if hasattr(strain_frame, "get_exx") else None
                        eyy = _to_numpy(strain_frame.get_eyy().get_array()) if hasattr(strain_frame, "get_eyy") else None
                        exy = _to_numpy(strain_frame.get_exy().get_array()) if hasattr(strain_frame, "get_exy") else None
                        if exx is not None: exx[:] = _postprocess_field(exx, roi_m)
                        if eyy is not None: eyy[:] = _postprocess_field(eyy, roi_m)
                        if exy is not None: exy[:] = _postprocess_field(exy, roi_m)
                except Exception as e:
                    self._log(f"Post-processing warning: {e}")
            self._save_bins(self.DIC_in, dic_out, s_in, s_out, out_dir)
            self.progress.emit(100)
            self.done.emit(True, "Analysis finished", dic_out, s_out)
        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            self._log(msg)
            self.progress.emit(100)
            self.done.emit(False, msg, None, None)

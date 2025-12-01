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
import io
import traceback

import cv2
import imageio.v2 as iio
import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    import ncorr
except Exception:
    ncorr = None

from analysis_worker import NcorrWorker
from Preprocessing import REMBG_AVAILABLE, FrameExtractorTab
from custom_widgets import natural_key

if REMBG_AVAILABLE:
    from rembg import new_session, remove


class AutomationController(QThread):
    progress_update = Signal(int, str)
    finished = Signal(bool, dict)

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config or {})
    
    def _progress_bridge(self, pct, msg):
        try: self.progress_update.emit(int(pct), str(msg))
        except Exception: pass

    def _log(self, text):
        self.progress_update.emit(-1, str(text))

    def _make_roi_from_first_frame(self, frame_path: str) -> np.ndarray:
        if REMBG_AVAILABLE:
            with open(frame_path, 'rb') as f_in:
                output_bytes = remove(f_in.read(), session=new_session())
            rgba = iio.imread(io.BytesIO(output_bytes))
            if rgba.ndim == 3 and rgba.shape[-1] >= 4:
                alpha = rgba[..., 3]; mask = (alpha > 50).astype(np.uint8) * 255
            else:
                gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY) if rgba.ndim == 3 else rgba
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            img = iio.imread(frame_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: raise RuntimeError("Automatic ROI failed: no contours detected.")
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(clean, [largest], 0, 255, -1)
        return clean > 0

    def run(self):
        try:
            input_type = self.config.get('input_type', 'video')
            output_dir = self.config.get('output_dir', '')
            frame_files = []

            if input_type == 'files': 
                self.progress_update.emit(10, "Processing selected image files...")
                image_paths = self.config.get('image_file_paths', [])
                if not image_paths:
                    raise FileNotFoundError("No image files were provided for analysis.")
                
                processing_params = {
                    "source_mode": "files",
                    "image_paths": image_paths, # Use the provided list
                    "output_dir": output_dir,
                    "enhance": True,
                    "rembg": False,
                }
                processing_res = FrameExtractorTab._run_processing_task(self._progress_bridge, processing_params)
                frames_dir = processing_res.get("final_folder")
                if not frames_dir: raise RuntimeError("Image processing/copying failed.")
                frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)], key=natural_key)

            else: # Video mode (existing logic)
                self.progress_update.emit(5, "Extracting video frames…")
                extraction_params = {
                    "source_mode": "video",
                    "video_path": self.config.get('video_path', ''),
                    "output_dir": output_dir,
                    "enhance": True, "rembg": False,
                    "method": self.config.get('extraction_method', 'Interval'),
                    "interval": self.config.get('extraction_interval', 1.0),
                    "frames": self.config.get('extraction_frames', ""),
                    "start_frame": self.config.get('extraction_start_frame', 0),
                    "end_frame": self.config.get('extraction_end_frame', 100),
                    "step_frame": self.config.get('extraction_step_frame', 1),
                }
                extraction_res = FrameExtractorTab._run_processing_task(self._progress_bridge, extraction_params)
                frames_dir = extraction_res.get("final_folder")
                if not frames_dir: raise RuntimeError("Frame extraction failed.")
                frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))], key=natural_key)

            if len(frame_files) < 2:
                raise RuntimeError("Need at least 2 frames (1 reference + 1 current).")
            
            # --- The rest of the logic is common for both modes and is unchanged ---
            self._log(f"Frames processed to: {frames_dir}")
            self._log("Generating ROI from the first frame…")
            roi_mask = self._make_roi_from_first_frame(frame_files[0])
            iio.imwrite(os.path.join(output_dir, 'roi.png'), (roi_mask.astype(np.uint8) * 255))
            dic_cfg = dict(self.config.get('dic_config', {}).get('dic', {}))
            if not dic_cfg and isinstance(self.config.get('dic_config'), dict): dic_cfg = dict(self.config['dic_config'])
            dic_cfg['out_dir'] = output_dir
            if 'units' in self.config: dic_cfg['units'] = self.config.get('units')
            if 'units_per_pixel' in self.config: dic_cfg['units_per_pixel'] = self.config.get('units_per_pixel')
            if 'change_perspective' in self.config: dic_cfg['change_perspective'] = self.config.get('change_perspective')
            ys, xs = np.where(roi_mask)
            if xs.size == 0: raise RuntimeError("Generated ROI mask is empty.")
            n_threads = int(dic_cfg.get('threads', max(1, os.cpu_count() or 1)))
            seeds = [[int(xs[i]), int(ys[i])] for i in np.random.choice(xs.size, min(xs.size, n_threads), replace=False)]
            self._log("Starting DIC analysis…")
            if ncorr is None: raise RuntimeError("ncorr module is not available.")
            worker = NcorrWorker(dic_cfg, frame_files, roi_mask, seeds)
            worker.log.connect(lambda m: self.progress_update.emit(-1, m))
            worker.run()
            self._log("DIC stage finished.")
            dic_out_path = os.path.join(dic_cfg['out_dir'], "DIC_output.bin")
            dic_out = ncorr.load_DIC_output(dic_out_path) if os.path.exists(dic_out_path) else None
            s_out = None
            s_out_path = os.path.join(dic_cfg['out_dir'], "strain_output.bin")
            if os.path.exists(s_out_path):
                try: s_out = ncorr.load_strain_output(s_out_path) if hasattr(ncorr, "load_strain_output") else None
                except Exception: s_out = None
            if s_out is None and dic_out is not None:
                try:
                    subreg = getattr(ncorr.SUBREGION, str(dic_cfg.get('subregion', 'CIRCLE')))
                    s_in = ncorr.strain_analysis_input(worker.DIC_in, dic_out, subreg, int(dic_cfg.get('strain_radius', 15)))
                    s_out = ncorr.strain_analysis(s_in)
                except Exception: s_out = None
            ref_img = iio.imread(frame_files[0])
            if ref_img.ndim == 3: ref_img = ref_img[..., 0]
            strain_params = {}
            if s_out is not None and dic_cfg.get('strain_radius', 0) > 0:
                strain_params['strain_radius'] = dic_cfg.get('strain_radius')
            result_data = { "dic_results": dic_out, "strain_results": s_out, "ref_path": frame_files[0], "cur_paths": frame_files[1:], "ref_img": ref_img, "roi_mask": roi_mask, "dic_params": dic_cfg, "strain_params": strain_params, "seeds": seeds, }
            self.finished.emit(True, result_data)
        except Exception as e:
            self.finished.emit(False, {"error": f"{e}\n{traceback.format_exc()}"})
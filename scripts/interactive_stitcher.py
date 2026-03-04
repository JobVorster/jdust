# Interactive Spectrum Stitching Tool
# Phase 3: Uncertainty panel, save merged spectrum with metadata
#         + Overlap masking between subbands (nomask / upper / lower / average)
#
# Usage: python interactive_stitcher.py
#
# Applied flux per subcube: flux * scale + offset + global_offset
# Uncertainty scales with multiplicative factor only.
#
# Overlap masking modes (applied AFTER per-subband transforms):
#   nomask  — keep all subband data, concatenate as-is
#   upper   — in overlap region keep the higher-wavelength subband
#   lower   — in overlap region keep the lower-wavelength subband
#   average — resample lower onto upper's grid in overlap, average both
#             flux_avg = (f_lo + f_hi)/2
#             unc_avg  = sqrt(σ_lo² + σ_hi²) 
#
# Saved CSV columns:
#   um (μm)                — per-subband wavelengths, sorted, all data
#   flux_original (Jy)     — untransformed flux, same grid
#   unc_original (Jy)      — untransformed uncertainty, same grid
#   flux_stitched_raw (Jy) — transformed flux before overlap masking, same grid
#   unc_stitched_raw (Jy)  — transformed unc before overlap masking, same grid
#   um_stitched (μm)       — wavelengths AFTER overlap masking (may be shorter)
#   flux_stitched (Jy)     — flux after overlap masking
#   unc_stitched (Jy)      — uncertainty after overlap masking
#   (shorter arrays are NaN-padded to match the longer one)
#
# Author: Job Vorster — jobvorster8@gmail.com
#
# Dependencies: tkinter (stdlib), matplotlib, numpy, pandas, jdspextract

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import getpass
import re
from datetime import datetime

from ifu_analysis.jdspextract import load_spectra

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_TITLE         = 'Interactive Spectrum Stitcher'
FIGURE_SIZE          = (14, 7)
XLIM                 = (4.5, 28.0)
GLOBAL_SLIDER_RANGE  = 0.1
ADDITIVE_RANGE       = 0.01
MULTIPLICATIVE_RANGE = 2.0
SCRIPT_NAME          = 'interactive_stitcher.py'
AUTHOR_EMAIL         = 'jobvorster8@gmail.com'

OVERLAP_MASK_OPTIONS = ['nomask', 'upper', 'lower', 'average']

# =============================================================================
# HELPER — parse source/aperture from filename
# =============================================================================

def parse_filename_metadata(filepath):
    '''
    Try to extract source_name and aperture_name from filename.
    Expected format: SOURCE_aperAPER_*.spectra
    e.g. L1448MM1_aperB1_unstitched.spectra
    Returns (source_name, aperture_name) or (None, None) if parsing fails.
    '''
    basename = os.path.basename(filepath)
    match = re.match(r'^(.+?)_aper([^_]+)_', basename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def ask_metadata(parent):
    '''Popup dialog asking user for source and aperture name.'''
    dialog = tk.Toplevel(parent)
    dialog.title('Source metadata')
    dialog.grab_set()

    tk.Label(dialog, text='Could not parse source/aperture from filename.',
             font=('Helvetica', 10)).grid(row=0, column=0, columnspan=2, padx=10, pady=8)

    tk.Label(dialog, text='Source name:').grid(row=1, column=0, sticky='e', padx=8, pady=4)
    source_var = tk.StringVar()
    tk.Entry(dialog, textvariable=source_var, width=20).grid(row=1, column=1, padx=8)

    tk.Label(dialog, text='Aperture name:').grid(row=2, column=0, sticky='e', padx=8, pady=4)
    aper_var = tk.StringVar()
    tk.Entry(dialog, textvariable=aper_var, width=20).grid(row=2, column=1, padx=8)

    confirmed = tk.BooleanVar(value=False)

    def confirm():
        confirmed.set(True)
        dialog.destroy()

    tk.Button(dialog, text='OK', command=confirm).grid(row=3, column=0, columnspan=2, pady=8)
    parent.wait_window(dialog)

    if confirmed.get():
        return source_var.get() or 'unknown', aper_var.get() or 'unknown'
    return 'unknown', 'unknown'


# =============================================================================
# OVERLAP MASKING
# =============================================================================

def get_overlap_intervals(um_list):
    '''
    Return a list of (lo, hi) wavelength intervals where any two subbands overlap.

    Parameters
    ----------
    um_list : list of 1-D numpy arrays

    Returns
    -------
    intervals : list of (float, float)
        Each tuple is an overlapping wavelength range.  Empty list when there
        is no overlap.
    '''
    # Sort bands by minimum wavelength
    bands = sorted([np.array(u) for u in um_list], key=np.min)
    intervals = []
    for i in range(len(bands) - 1):
        lo = np.min(bands[i + 1])   # start of the upper band
        hi = np.max(bands[i])       # end of the lower band
        if hi > lo:
            intervals.append((lo, hi))
    return intervals


def apply_overlap_mask(um_list, flux_list, unc_list, mode='nomask'):
    '''
    Merge per-subband arrays according to the chosen overlap-masking mode.

    Parameters
    ----------
    um_list, flux_list, unc_list : lists of 1-D numpy arrays
        Per-subband wavelength, flux, and uncertainty arrays (already
        scale/offset-transformed).
    mode : str
        One of 'nomask', 'upper', 'lower', 'average'.

    Returns
    -------
    um_out, flux_out, unc_out : 1-D numpy arrays
        Merged, wavelength-sorted arrays after overlap masking.
    '''
    # Sort subbands by median wavelength
    order = sorted(range(len(um_list)), key=lambda i: np.nanmedian(um_list[i]))
    bands = [(np.array(um_list[i]), np.array(flux_list[i]), np.array(unc_list[i]))
             for i in order]

    if mode == 'nomask':
        um_all   = np.concatenate([b[0] for b in bands])
        flux_all = np.concatenate([b[1] for b in bands])
        unc_all  = np.concatenate([b[2] for b in bands])
        idx = np.argsort(um_all)
        return um_all[idx], flux_all[idx], unc_all[idx]

    # ---- modes that modify overlap ---
    # Accumulate result as lists of segments (each segment is a 1-D array triple)
    seg_um   = []
    seg_flux = []
    seg_unc  = []

    for k, (um_i, flux_i, unc_i) in enumerate(bands):
        if k == 0:
            seg_um.append(um_i.copy())
            seg_flux.append(flux_i.copy())
            seg_unc.append(unc_i.copy())
            continue

        # Current upper bound of everything accumulated so far
        prev_max = np.max(np.concatenate(seg_um)) if seg_um else -np.inf
        curr_min = np.min(um_i)

        if prev_max <= curr_min:
            # No overlap — just append
            seg_um.append(um_i)
            seg_flux.append(flux_i)
            seg_unc.append(unc_i)
            continue

        # Overlap region: [curr_min, prev_max]
        ov_lo = curr_min
        ov_hi = prev_max

        if mode == 'lower':
            # Keep what we have, discard current band in the overlap
            non_ov = um_i > ov_hi
            seg_um.append(um_i[non_ov])
            seg_flux.append(flux_i[non_ov])
            seg_unc.append(unc_i[non_ov])

        elif mode == 'upper':
            # Trim all accumulated segments so their wavelengths are < ov_lo
            for j in range(len(seg_um)):
                keep = seg_um[j] < ov_lo
                seg_um[j]   = seg_um[j][keep]
                seg_flux[j] = seg_flux[j][keep]
                seg_unc[j]  = seg_unc[j][keep]
            # Add entire current band (it defines the overlap region from its side)
            seg_um.append(um_i)
            seg_flux.append(flux_i)
            seg_unc.append(unc_i)

        elif mode == 'average':
            # Pull out the accumulated "lower" data in the overlap
            all_prev_um   = np.concatenate(seg_um)
            all_prev_flux = np.concatenate(seg_flux)
            all_prev_unc  = np.concatenate(seg_unc)

            mask_prev_ov  = (all_prev_um >= ov_lo) & (all_prev_um <= ov_hi)
            mask_prev_non = all_prev_um < ov_lo

            um_ov_prev   = all_prev_um[mask_prev_ov]
            flux_ov_prev = all_prev_flux[mask_prev_ov]
            unc_ov_prev  = all_prev_unc[mask_prev_ov]

            # Current (upper) band in overlap
            mask_curr_ov  = (um_i >= ov_lo) & (um_i <= ov_hi)
            mask_curr_non = um_i > ov_hi

            um_ov_curr   = um_i[mask_curr_ov]
            flux_ov_curr = flux_i[mask_curr_ov]
            unc_ov_curr  = unc_i[mask_curr_ov]

            # Resample lower (prev) onto upper's (curr) wavelength grid
            if len(um_ov_prev) > 1 and len(um_ov_curr) > 0:
                flux_ov_prev_r = np.interp(um_ov_curr, um_ov_prev, flux_ov_prev)
                unc_ov_prev_r  = np.interp(um_ov_curr, um_ov_prev, unc_ov_prev)
            else:
                flux_ov_prev_r = flux_ov_curr.copy()
                unc_ov_prev_r  = unc_ov_curr.copy()

            flux_avg = (flux_ov_prev_r + flux_ov_curr) / 2.0
            unc_avg  = np.sqrt(unc_ov_prev_r**2 + unc_ov_curr**2) 

            # Rebuild accumulated segments: keep only non-overlap from prev
            seg_um   = [all_prev_um[mask_prev_non]]
            seg_flux = [all_prev_flux[mask_prev_non]]
            seg_unc  = [all_prev_unc[mask_prev_non]]

            if len(um_ov_curr) > 0:
                seg_um.append(um_ov_curr)
                seg_flux.append(flux_avg)
                seg_unc.append(unc_avg)

            seg_um.append(um_i[mask_curr_non])
            seg_flux.append(flux_i[mask_curr_non])
            seg_unc.append(unc_i[mask_curr_non])

    # Concatenate non-empty segments and sort
    non_empty = [(u, f, e) for u, f, e in zip(seg_um, seg_flux, seg_unc) if len(u) > 0]
    if not non_empty:
        return np.array([]), np.array([]), np.array([])

    um_out, flux_out, unc_out = map(np.concatenate, zip(*non_empty))
    idx = np.argsort(um_out)
    return um_out[idx], flux_out[idx], unc_out[idx]


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class SpectraStitcher:

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry('1400x950')

        # Application state
        self.results         = None
        self.loaded_filepath = None
        self.source_name     = None
        self.aperture_name   = None
        self.colors          = []
        self.zoom_locked     = False
        self.locked_xlim     = None
        self.locked_ylim     = None
        self.locked_ylim_unc = None
        self.locked_yscale   = 'log'   # saved scale at lock time

        # Log/linear toggle (default: log)
        self.log_scale = True

        # Per-subband state
        self.offsets = {}
        self.scales  = {}

        # Overlap masking
        self.overlap_mask_var = tk.StringVar(value='nomask')

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # --- Top toolbar ---
        toolbar_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        tk.Button(toolbar_frame, text='Load spectrum',
                  command=self._load_spectrum).pack(side=tk.LEFT, padx=4)

        tk.Button(toolbar_frame, text='Save stitched spectrum',
                  command=self._save_spectrum).pack(side=tk.LEFT, padx=4)

        self.zoom_lock_btn = tk.Button(
            toolbar_frame,
            text='Lock zoom',
            command=self._toggle_zoom_lock,
            bg='lightgrey',
        )
        self.zoom_lock_btn.pack(side=tk.LEFT, padx=4)

        # --- Overlap mask selector ---
        tk.Label(toolbar_frame, text='  Overlap:',
                 font=('Helvetica', 9)).pack(side=tk.LEFT)

        self._overlap_btn = tk.Button(
            toolbar_frame,
            text='nomask',
            width=8,
            relief=tk.RAISED,
            command=self._cycle_overlap_mask,
            bg='#ddeeff',
            font=('Helvetica', 9, 'bold'),
        )
        self._overlap_btn.pack(side=tk.LEFT, padx=2)

        # --- Log / linear toggle ---
        tk.Label(toolbar_frame, text='  Scale:',
                 font=('Helvetica', 9)).pack(side=tk.LEFT)

        self._scale_btn = tk.Button(
            toolbar_frame,
            text='log',
            width=6,
            relief=tk.RAISED,
            command=self._toggle_log_scale,
            bg='#e8e8e8',
            font=('Helvetica', 9, 'bold'),
        )
        self._scale_btn.pack(side=tk.LEFT, padx=2)

        self.status_label = tk.Label(toolbar_frame, text='No spectrum loaded.',
                                     font=('Helvetica', 9), fg='grey')
        self.status_label.pack(side=tk.LEFT, padx=12)

        # --- Main pane: left = plot, right = controls ---
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: matplotlib canvas
        canvas_frame = tk.Frame(main_pane)
        main_pane.add(canvas_frame, width=950)

        # Two-panel figure: flux (top) and uncertainty (bottom)
        self.fig = plt.figure(figsize=FIGURE_SIZE)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        self.ax      = self.fig.add_subplot(gs[0])
        self.ax_unc  = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.fig.tight_layout(pad=3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav_toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        nav_toolbar.update()

        # Global slider
        global_frame = tk.LabelFrame(canvas_frame, text='Global additive offset (Jy)',
                                     padx=4, pady=4)
        global_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)

        self.global_slider_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            global_frame,
            variable=self.global_slider_var,
            from_=-GLOBAL_SLIDER_RANGE,
            to=GLOBAL_SLIDER_RANGE,
            resolution=GLOBAL_SLIDER_RANGE / 1000,
            orient=tk.HORIZONTAL,
            length=600,
            command=self._on_change,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(global_frame, text='Reset',
                  command=self._reset_global).pack(side=tk.LEFT, padx=4)

        # Right: scrollable per-subband controls
        right_frame = tk.Frame(main_pane, bd=1, relief=tk.SUNKEN)
        main_pane.add(right_frame, width=420)

        tk.Label(right_frame, text='Per-subband controls',
                 font=('Helvetica', 11, 'bold')).pack(pady=4)

        scroll_canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL,
                                  command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.subband_scroll_frame = tk.Frame(scroll_canvas)
        scroll_canvas.create_window((0, 0), window=self.subband_scroll_frame, anchor='nw')

        self.subband_scroll_frame.bind(
            '<Configure>',
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))
        )
        scroll_canvas.bind_all(
            '<MouseWheel>',
            lambda e: scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        )

    # -----------------------------------------------------------------------
    # PER-SUBBAND SLIDER CONSTRUCTION
    # -----------------------------------------------------------------------

    def _build_subband_sliders(self):
        for widget in self.subband_scroll_frame.winfo_children():
            widget.destroy()
        self.offsets = {}
        self.scales  = {}

        for i, subcube_name in enumerate(self.results['subcube_name']):
            color     = self.colors[i]
            color_hex = '#%02x%02x%02x' % tuple(int(c * 255) for c in color[:3])

            frame = tk.LabelFrame(
                self.subband_scroll_frame,
                text=subcube_name,
                fg=color_hex,
                font=('Helvetica', 9, 'bold'),
                padx=4, pady=2,
            )
            frame.pack(fill=tk.X, padx=4, pady=2)

            # Additive
            add_var = tk.DoubleVar(value=0.0)
            self.offsets[subcube_name] = add_var
            tk.Label(frame, text='+ offset (Jy)', font=('Helvetica', 8)).pack(anchor='w')
            add_row = tk.Frame(frame)
            add_row.pack(fill=tk.X)
            tk.Scale(
                add_row,
                variable=add_var,
                from_=-ADDITIVE_RANGE,
                to=ADDITIVE_RANGE,
                resolution=ADDITIVE_RANGE / 2000,
                orient=tk.HORIZONTAL,
                length=600,
                command=self._on_change,
                showvalue=True,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(
                add_row, text='0',
                command=lambda v=add_var: (v.set(0.0), self._plot_spectra())
            ).pack(side=tk.LEFT, padx=2)

            # Multiplicative
            mul_var = tk.DoubleVar(value=1.0)
            self.scales[subcube_name] = mul_var
            tk.Label(frame, text='× scale', font=('Helvetica', 8)).pack(anchor='w')
            mul_row = tk.Frame(frame)
            mul_row.pack(fill=tk.X)
            tk.Scale(
                mul_row,
                variable=mul_var,
                from_=0.0,
                to=MULTIPLICATIVE_RANGE,
                resolution=MULTIPLICATIVE_RANGE / 1000,
                orient=tk.HORIZONTAL,
                length=300,
                command=self._on_change,
                showvalue=True,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(
                mul_row, text='1',
                command=lambda v=mul_var: (v.set(1.0), self._plot_spectra())
            ).pack(side=tk.LEFT, padx=2)

    # -----------------------------------------------------------------------
    # EVENT HANDLERS
    # -----------------------------------------------------------------------

    def _cycle_overlap_mask(self):
        '''Cycle through overlap masking options on button click.'''
        current = self.overlap_mask_var.get()
        idx = OVERLAP_MASK_OPTIONS.index(current)
        new = OVERLAP_MASK_OPTIONS[(idx + 1) % len(OVERLAP_MASK_OPTIONS)]
        self.overlap_mask_var.set(new)

        # Update button appearance
        colours = {
            'nomask':  '#ddeeff',
            'upper':   '#d4f4d4',
            'lower':   '#fde8c0',
            'average': '#f0d4f4',
        }
        self._overlap_btn.config(text=new, bg=colours.get(new, '#ddeeff'))

        if self.results is not None:
            self._plot_spectra()

    def _load_spectrum(self):
        filename = filedialog.askopenfilename(
            title='Load spectrum',
            filetypes=[('Spectra files', '*.spectra'), ('All files', '*.*')],
        )
        if not filename:
            return
        try:
            self.results = load_spectra(filename)
            if self.results is None:
                messagebox.showerror('Load error', 'load_spectra returned None')
                return
            if 'subcube_name' not in self.results:
                messagebox.showerror('Load error',
                    f'Unexpected format. Keys: {list(self.results.keys())}')
                return
        except Exception as e:
            import traceback
            messagebox.showerror('Load error', traceback.format_exc())
            return

        self.loaded_filepath = filename

        # Parse source/aperture from filename
        source, aper = parse_filename_metadata(filename)
        if source is None:
            source, aper = ask_metadata(self.root)
        self.source_name   = source
        self.aperture_name = aper

        n_subcubes  = len(self.results['subcube_name'])
        cmap        = cm.tab20 if n_subcubes > 10 else cm.tab10
        self.colors = [cmap(i / max(n_subcubes - 1, 1)) for i in range(n_subcubes)]

        self.global_slider_var.set(0.0)
        self.zoom_locked = False
        self.locked_xlim = None
        self.locked_ylim = None
        self.zoom_lock_btn.config(text='Lock zoom', bg='lightgrey')

        self.status_label.config(
            text=f'Loaded: {self.source_name} / {self.aperture_name}',
            fg='black'
        )

        self._build_subband_sliders()
        self._plot_spectra()

    def _save_spectrum(self):
        if self.results is None:
            messagebox.showwarning('Nothing to save', 'Load a spectrum first.')
            return

        # Suggest default filename
        default_name = f'{self.source_name}_aper{self.aperture_name}_stitched.csv'
        filepath = filedialog.asksaveasfilename(
            title='Save stitched spectrum',
            initialfile=default_name,
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        if not filepath:
            return

        global_offset = self.global_slider_var.get()
        mask_mode     = self.overlap_mask_var.get()

        # ---- Collect transformed arrays per subcube ----
        all_um            = []
        all_flux_ori      = []
        all_unc_ori       = []
        all_flux_raw      = []   # transformed, before overlap masking
        all_unc_raw       = []
        stitch_params     = {}

        um_for_mask   = []
        flux_for_mask = []
        unc_for_mask  = []

        for subcube_name in self.results['subcube_name']:
            idx = self.results['subcube_name'].index(subcube_name)
            um       = np.array(self.results['um'][idx])
            flux     = np.array(self.results['flux'][idx])
            unc      = np.array(self.results['flux_unc'][idx])

            offset = self.offsets[subcube_name].get() if subcube_name in self.offsets else 0.0
            scale  = self.scales[subcube_name].get()  if subcube_name in self.scales  else 1.0

            additive_shift = offset + global_offset
            flux_s = flux * scale + additive_shift
            unc_s  = scale * (unc / flux) * (flux + additive_shift)

            all_um.append(um)
            all_flux_ori.append(flux)
            all_unc_ori.append(unc)
            all_flux_raw.append(flux_s)
            all_unc_raw.append(unc_s)

            um_for_mask.append(um)
            flux_for_mask.append(flux_s)
            unc_for_mask.append(unc_s)

            stitch_params[subcube_name] = {
                'additive_Jy':    round(offset, 6),
                'multiplicative': round(scale, 6),
            }

        # ---- Merge & sort the "raw" per-subband data (no overlap masking) ----
        um_raw_all   = np.concatenate(all_um)
        flux_ori_all = np.concatenate(all_flux_ori)
        unc_ori_all  = np.concatenate(all_unc_ori)
        flux_raw_all = np.concatenate(all_flux_raw)
        unc_raw_all  = np.concatenate(all_unc_raw)

        sort_raw    = np.argsort(um_raw_all)
        um_raw_all  = um_raw_all[sort_raw]
        flux_ori_all = flux_ori_all[sort_raw]
        unc_ori_all  = unc_ori_all[sort_raw]
        flux_raw_all = flux_raw_all[sort_raw]
        unc_raw_all  = unc_raw_all[sort_raw]

        # ---- Apply overlap masking to get the final stitched arrays ----
        um_st, flux_st, unc_st = apply_overlap_mask(
            um_for_mask, flux_for_mask, unc_for_mask, mode=mask_mode
        )

        # ---- NaN-pad shorter array so everything fits one DataFrame ----
        n_raw = len(um_raw_all)
        n_st  = len(um_st)
        n_max = max(n_raw, n_st)

        def pad(arr, n):
            if len(arr) < n:
                return np.concatenate([arr, np.full(n - len(arr), np.nan)])
            return arr

        um_raw_p    = pad(um_raw_all,   n_max)
        flux_ori_p  = pad(flux_ori_all, n_max)
        unc_ori_p   = pad(unc_ori_all,  n_max)
        flux_raw_p  = pad(flux_raw_all, n_max)
        unc_raw_p   = pad(unc_raw_all,  n_max)
        um_st_p     = pad(um_st,        n_max)
        flux_st_p   = pad(flux_st,      n_max)
        unc_st_p    = pad(unc_st,       n_max)

        # ---- Build header ----
        now      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        username = getpass.getuser()
        header_lines = [
            f'# Stitched spectrum',
            f'# Saved: {now}',
            f'# User: {username} ({AUTHOR_EMAIL})',
            f'# Script: {SCRIPT_NAME}',
            f'# Source: {self.source_name}',
            f'# Aperture: {self.aperture_name}',
            f'# Original file: {self.loaded_filepath}',
            f'# Global offset (Jy): {global_offset:.6f}',
            f'# Overlap masking mode: {mask_mode}',
            f'#   nomask  — all subband data kept (concatenated)',
            f'#   upper   — overlap region uses higher-wavelength subband',
            f'#   lower   — overlap region uses lower-wavelength subband',
            f'#   average — overlap averaged after resampling to upper grid;',
            f'#             flux_avg = (f_lo+f_hi)/2,  unc_avg = sqrt(σ_lo²+σ_hi²)/2',
            f'# Uncertainty formula (per-subband):',
            f'#   unc_stitched_raw = scale * (unc_original / flux_original)',
            f'#                      * (flux_original + additive_shift)',
            f'#   where additive_shift = per_subband_offset + global_offset',
            f'#   Use unc_original if constant relative uncertainty is not desired.',
            f'# Per-subband stitching parameters:',
        ]
        for k, v in stitch_params.items():
            header_lines.append(
                f'#   {k}: additive={v["additive_Jy"]} Jy, '
                f'multiplicative={v["multiplicative"]}'
            )
        header_lines.append(
            '# Columns:\n'
            '#   um (um)                  — per-subband wavelengths sorted, all subbands\n'
            '#   flux_original (Jy)       — untransformed flux on same grid\n'
            '#   unc_original (Jy)        — untransformed uncertainty on same grid\n'
            '#   flux_stitched_raw (Jy)   — transformed flux before overlap masking\n'
            '#   unc_stitched_raw (Jy)    — transformed unc before overlap masking\n'
            '#   um_stitched (um)         — wavelengths after overlap masking (may differ in length)\n'
            '#   flux_stitched (Jy)       — flux after overlap masking\n'
            '#   unc_stitched (Jy)        — uncertainty after overlap masking\n'
            '#   (shorter columns are NaN-padded to match the longer one)'
        )

        with open(filepath, 'w') as f:
            for line in header_lines:
                f.write(line + '\n')

        df = pd.DataFrame({
            'um':                 um_raw_p,
            'flux_original':      flux_ori_p,
            'unc_original':       unc_ori_p,
            'flux_stitched_raw':  flux_raw_p,
            'unc_stitched_raw':   unc_raw_p,
            'um_stitched':        um_st_p,
            'flux_stitched':      flux_st_p,
            'unc_stitched':       unc_st_p,
        })
        df.to_csv(filepath, mode='a', index=False)

        messagebox.showinfo('Saved', f'Spectrum saved to:\n{filepath}')

    def _on_change(self, value=None):
        if self.results is not None:
            self._plot_spectra()

    def _reset_global(self):
        self.global_slider_var.set(0.0)
        if self.results is not None:
            self._plot_spectra()

    def _toggle_log_scale(self):
        self.log_scale = not self.log_scale
        if self.log_scale:
            self._scale_btn.config(text='log',    bg='#e8e8e8')
        else:
            self._scale_btn.config(text='linear', bg='#ffe8cc')
        if self.results is not None:
            self._plot_spectra()

    def _toggle_zoom_lock(self):
        if not self.zoom_locked:
            self.locked_xlim     = self.ax.get_xlim()
            self.locked_ylim     = self.ax.get_ylim()
            self.locked_ylim_unc = self.ax_unc.get_ylim()
            self.locked_yscale   = 'log' if self.log_scale else 'linear'
            self.zoom_locked     = True
            self.zoom_lock_btn.config(text='Unlock zoom', bg='#ffcc00')
        else:
            self.zoom_locked     = False
            self.locked_xlim     = None
            self.locked_ylim     = None
            self.locked_ylim_unc = None
            self.zoom_lock_btn.config(text='Lock zoom', bg='lightgrey')
            if self.results is not None:
                self._plot_spectra()

    # -----------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------

    def _plot_spectra(self):
        self.ax.cla()
        self.ax_unc.cla()

        if self.results is None:
            self.canvas.draw()
            return

        global_offset = self.global_slider_var.get()
        mask_mode     = self.overlap_mask_var.get()

        um_for_mask   = []
        flux_for_mask = []
        unc_for_mask  = []

        for i, subcube_name in enumerate(self.results['subcube_name']):
            um       = np.array(self.results['um'][i])
            flux     = np.array(self.results['flux'][i])
            flux_unc = np.array(self.results['flux_unc'][i])
            color    = self.colors[i]

            offset = self.offsets[subcube_name].get() if subcube_name in self.offsets else 0.0
            scale  = self.scales[subcube_name].get()  if subcube_name in self.scales  else 1.0

            additive_shift       = offset + global_offset
            flux_transformed     = flux * scale + additive_shift
            flux_unc_transformed = scale * (flux_unc / flux) * (flux + additive_shift)

            um_for_mask.append(um)
            flux_for_mask.append(flux_transformed)
            unc_for_mask.append(flux_unc_transformed)

            # --- Top panel: original (faded) and transformed (solid) per subband ---
            self.ax.plot(um, flux,
                         color=color, lw=0.8, alpha=0.2, zorder=1)
            self.ax.plot(um, flux_transformed,
                         color=color, lw=1.2, alpha=0.5, zorder=2,
                         label=subcube_name)
            self.ax.fill_between(um,
                                 flux_transformed - flux_unc_transformed,
                                 flux_transformed + flux_unc_transformed,
                                 color=color, alpha=0.10, zorder=1)

            # --- Bottom panel: per-subband uncertainty (coloured) ---
            self.ax_unc.plot(um, flux_unc_transformed,
                             color=color, lw=0.8, alpha=0.5, zorder=2)

        # --- Merged / masked spectrum overlay — overlap regions only ---
        um_merged, flux_merged, unc_merged = apply_overlap_mask(
            um_for_mask, flux_for_mask, unc_for_mask, mode=mask_mode
        )
        overlap_ivs = get_overlap_intervals(um_for_mask)
        if len(um_merged) > 0 and overlap_ivs:
            # Build a boolean mask for every point that falls in any overlap interval
            in_overlap = np.zeros(len(um_merged), dtype=bool)
            for (lo, hi) in overlap_ivs:
                in_overlap |= (um_merged >= lo) & (um_merged <= hi)

            # Draw each contiguous segment separately so lines don't bridge gaps
            # between non-adjacent overlap windows
            changes   = np.diff(in_overlap.astype(int))
            starts    = np.where(changes == 1)[0] + 1
            ends      = np.where(changes == -1)[0] + 1
            if in_overlap[0]:
                starts = np.concatenate([[0], starts])
            if in_overlap[-1]:
                ends = np.concatenate([ends, [len(um_merged)]])

            first = True
            for s, e in zip(starts, ends):
                self.ax.plot(um_merged[s:e], flux_merged[s:e],
                             color='black', lw=1.2, alpha=0.5, zorder=3,
                             label=f'overlap ({mask_mode})' if first else '_nolegend_')
                self.ax_unc.plot(um_merged[s:e], unc_merged[s:e],
                                 color='black', lw=1.2, alpha=0.5, zorder=3)
                first = False

        self.ax.set_ylabel(r'$F_\nu$ (Jy)', fontsize=11)
        self.ax.tick_params(labelbottom=False)
        self.ax.legend(fontsize=7, ncol=4, loc='upper left')

        self.ax_unc.set_ylabel(r'$\sigma_{F_\nu}$ (Jy)', fontsize=11)
        self.ax_unc.set_xlabel(r'Wavelength ($\mu$m)', fontsize=11)
        self.ax_unc.set_xlim(XLIM)

        # Apply y-axis scale BEFORE restoring zoom limits so matplotlib
        # computes the correct tick positions for the chosen scale.
        if self.zoom_locked and self.locked_xlim is not None:
            # Restore the scale that was active when the user locked zoom
            _yscale = self.locked_yscale
        else:
            _yscale = 'log' if self.log_scale else 'linear'

        self.ax.set_yscale(_yscale)
        self.ax_unc.set_yscale(_yscale)

        if self.zoom_locked and self.locked_xlim is not None:
            self.ax.set_xlim(self.locked_xlim)
            self.ax.set_ylim(self.locked_ylim)
            self.ax_unc.set_ylim(self.locked_ylim_unc)
        else:
            self.ax.set_xlim(XLIM)

        self.fig.tight_layout(pad=3)
        self.canvas.draw()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    app  = SpectraStitcher(root)
    root.mainloop()
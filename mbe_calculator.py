"""
Material Balance Equation (MBE) Calculator
===========================================
Professional desktop tool for petroleum reservoir engineering.
Features single-point and multi-timestep analysis with plots.
"""

import sys
import os
import csv
import math
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QFrame,
    QSizePolicy, QScrollArea, QComboBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QTextEdit, QSplitter, QHeaderView,
    QMessageBox, QSpacerItem
)
from PyQt6.QtGui import QFont, QDoubleValidator, QCursor, QColor
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Utility Widgets
# ═══════════════════════════════════════════════════════════════

class ScrolllessComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class MBEPlotCanvas(FigureCanvas):
    """Reusable matplotlib canvas for embedding in PyQt6."""
    def __init__(self, parent=None, width=6, height=4.5):
        self.fig = Figure(figsize=(width, height), dpi=100, facecolor='#f8f9fa')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


# ═══════════════════════════════════════════════════════════════
# MBE Calculation Engine
# ═══════════════════════════════════════════════════════════════

class MBEEngine:
    """Pure calculation logic — no UI dependencies."""

    @staticmethod
    def compute(p, pi, Np, Rp, Rs, Bo, Bg, Wp, Bw,
                Boi, Bgi, Rsi, Swi, cw, cf, m,
                We=0, Winj=0, Ginj=0, Bginj=0.001,
                res_type_idx=3, dp=None, solving_for_We=False):
        # 0: Undersaturated, 1: Gas Cap, 2: Water Drive, 3: Full
        if dp is None:
            dp = pi - p
        
        # ──────── STRUCTURAL NUMERATOR CONSTRUCTION ────────
        # 1. Produced Oil & Gas (Always active)
        numerator = Np * (Bo + (Rp - Rs) * Bg)
        
        # 2. Net Water Influx (Active in Water Drive and Full MBE)
        # Note: res_type_idx == 0 (Undersat) and 1 (Gas Cap) physically exclude water influx entirely
        if res_type_idx in (2, 3):
            if solving_for_We:
                numerator += Wp * Bw
            else:
                numerator -= (We - Wp * Bw)
            
        # 3. Injection mechanisms (Active in Full MBE)
        if res_type_idx == 3:
            numerator -= (Ginj * Bginj)
            numerator -= (Winj * Bw)
            
        m_eff = m if res_type_idx in (1, 3) else 0.0
        
        # ──────── STRUCTURAL DENOMINATOR CONSTRUCTION ────────
        # 4. Evolved Gas Expansion (Always active)
        denominator = (Bo - Boi) + (Rsi - Rs) * Bg
        
        # 5. Gas Cap Expansion
        gas_cap_expansion_base = Boi * ((Bg / Bgi) - 1.0) if Bgi > 0 else 0.0
        denominator += m_eff * gas_cap_expansion_base
            
        # 6. Rock & Connate Water Expansion (Uses structured m_eff consistency)
        rock_water_expansion_base = Boi * ((Swi * cw + cf) / (1.0 - Swi)) * dp if Swi < 1.0 else 0.0
        denominator += (1.0 + m_eff) * rock_water_expansion_base

        # ──────── PRESERVE ORIGINAL FORMS FOR COMPATIBILITY ────────
        Eo = (Bo - Boi) + (Rsi - Rs) * Bg
        Eg = gas_cap_expansion_base
        Efw_star = rock_water_expansion_base
        Efw = (1.0 + m_eff) * Efw_star
        Bt = Bo + (Rsi - Rs) * Bg
        
        produced_oil_gas = Np * (Bo + (Rp - Rs) * Bg)
        F = numerator # F aligns exactly with constructed numerator per client 
        Et = denominator
        F_adj = numerator 
        
        gas_inj_val = (Ginj * Bginj) if res_type_idx == 3 else 0.0
        winj_val = (Winj * Bw) if res_type_idx == 3 else 0.0
        
        return dict(dp=dp, Bt=Bt, F=F, F_adj=F_adj,
                    Eo=Eo, Eg=Eg, Efw_star=Efw_star, Efw=Efw, Et=Et,
                    num=numerator, den=denominator,
                    produced_oil_gas=produced_oil_gas,
                    gas_inj=gas_inj_val, water_inj=winj_val)

    @staticmethod
    def solve_N(F_adj, Et):
        return F_adj / Et if Et != 0 else float('nan')

    @staticmethod
    def solve_m(F_adj, N, Eo, Eg, Efw_star):
        d = Eg + Efw_star
        return (F_adj / N - Eo - Efw_star) / d if (d != 0 and N != 0) else float('nan')

    @staticmethod
    def solve_We(F_adj, N, Et):
        return F_adj - N * Et

    @staticmethod
    def driving_indexes(N, Eo, m, Eg, Efw, We, Wp, Bw, Winj, Ginj, Bginj, res_type_idx=3):
        term_depletion = N * Eo
        
        term_gas_cap = 0.0
        if res_type_idx in (1, 3):
            term_gas_cap = N * m * Eg
            
        term_water = 0.0
        if res_type_idx in (2, 3):
            term_water += (We - Wp * Bw)
        if res_type_idx == 3:
            term_water += Winj * Bw
            term_water += Ginj * Bginj
            
        term_expansion = N * Efw
        
        A = term_depletion + term_gas_cap + term_water + term_expansion
        if abs(A) < 1e-20:
            return dict(DDI=0, SDI=0, WDI=0, EDI=0, Sum=0)
            
        return dict(
            DDI=term_depletion / A,
            SDI=term_gas_cap / A,
            WDI=term_water / A,
            EDI=term_expansion / A,
            Sum=1.0 # By definition A/A is 1
        )


# ═══════════════════════════════════════════════════════════════
# Tab 1 — Single-Point Calculator  (preserves original)
# ═══════════════════════════════════════════════════════════════

class SinglePointTab(QWidget):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.input_containers = {}
        self.setup_ui()

    # ---- field definitions ----
    @staticmethod
    def _prod_fields():
        return [
            ("N",   "Original Oil in Place (STB)",          "1000000"),
            ("Np",  "Cumulative oil produced (STB)",        "1000000"),
            ("Rp",  "Cumulative gas-oil ratio (scf/STB)",   "800"),
            ("Rsi", "Initial gas solubility (scf/STB)",     "850"),
            ("Rs",  "Current gas solubility (scf/STB)",     "600"),
            ("Bo",  "Oil FV factor (rb/STB)",               "1.35"),
            ("Boi", "Initial oil FV factor (rb/STB)",       "1.40"),
            ("Bg",  "Gas FV factor (rb/scf)",               "0.0012"),
            ("Bgi", "Initial gas FV factor (rb/scf)",       "0.0009"),
            ("We",  "Cumulative water influx (rb)",         "50000"),
            ("Wp",  "Cumulative water produced (bbl)",      "10000"),
            ("Bw",  "Water FV factor (rb/STB)",             "1.02"),
            ("m",   "Gas cap ratio",                        "0.2"),
        ]

    @staticmethod
    def _exp_fields():
        return [
            ("dp",  "Change in reservoir pressure (pi - p)", "0"),
            ("Swi", "Initial water saturation (fraction)",   "0.2"),
            ("cw",  "Water compressibility (1/psi)",         "0.000003"),
            ("cf",  "Formation compressibility (1/psi)",     "0.000004"),
        ]

    @staticmethod
    def _inj_fields():
        return [
            ("Winj",  "Cumulative water injected (bbl)",      "0"),
            ("Ginj",  "Cumulative gas injected (scf)",        "0"),
            ("Bginj", "Injected gas FV factor (rb/scf)",      "0.001"),
        ]

    # ---- UI ----
    def setup_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        inner.setObjectName("main_content_widget")
        scroll_area.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setSpacing(20)
        lay.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Material Balance Equation (MBE)")
        title.setObjectName("header_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        sub = QLabel("Select target calculation and enter the required reservoir parameters.")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)

        # Target selector
        tg = QGroupBox("Configuration")
        tl = QVBoxLayout(tg)

        self.res_type_selector = ScrolllessComboBox()
        self.res_type_selector.addItems([
            "Undersaturated Oil Reservoir",
            "Gas Cap Reservoir",
            "Water Drive Reservoir",
            "Full MBE"
        ])
        self.res_type_selector.currentIndexChanged.connect(self._on_setup_change)
        self.res_type_selector.setFixedHeight(35)
        self.res_type_selector.setObjectName("res_type_selector")
        self.res_type_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        self.target_selector = ScrolllessComboBox()
        self.target_selector.addItems([
            "Original Oil in Place (N)",
            "Size of Gas Cap (m)",
            "Cumulative Water Influx (We)",
            "Primary Driving Indexes",
        ])
        self.target_selector.currentIndexChanged.connect(self._on_setup_change)
        self.target_selector.setFixedHeight(35)
        self.target_selector.setObjectName("target_selector")
        self.target_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        tl.addWidget(QLabel("1. Reservoir Type:"))
        tl.addWidget(self.res_type_selector)
        tl.addSpacing(10)
        tl.addWidget(QLabel("2. Target Calculation:"))
        tl.addWidget(self.target_selector)
        lay.addWidget(tg)

        # Input groups
        for label, fields in [("Production & PVT Parameters", self._prod_fields()),
                              ("Expansion Parameters",        self._exp_fields()),
                              ("Injection Parameters",        self._inj_fields())]:
            grp = QGroupBox(label)
            gl = QGridLayout(grp)
            gl.setSpacing(15)
            gl.setContentsMargins(20, 30, 20, 20)
            self._create_fields(gl, fields)
            lay.addWidget(grp)

        # Result area
        self.result_frame = QFrame()
        self.result_frame.setObjectName("result_frame")
        self.result_frame.setMinimumHeight(120)
        rl = QVBoxLayout(self.result_frame)
        self.result_label = QLabel("Result will be displayed here.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("result_text")
        rl.addWidget(self.result_label)
        lay.addStretch()
        lay.addWidget(self.result_frame)

        # Buttons
        blay = QHBoxLayout()
        blay.setSpacing(15)
        self.reset_btn = QPushButton("Reset Fields")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.reset_btn.clicked.connect(self._reset)

        self.calc_btn = QPushButton("Calculate N")
        self.calc_btn.setObjectName("calc_btn")
        self.calc_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.calc_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.calc_btn.clicked.connect(self._calculate)

        blay.addWidget(self.reset_btn)
        blay.addWidget(self.calc_btn)
        lay.addLayout(blay)

        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll_area)
        self.res_type_selector.setCurrentIndex(3) # Set Full MBE default
        self._on_setup_change()

    def _create_fields(self, layout, fields):
        v = QDoubleValidator(-1e15, 1e15, 6, self)
        v.setNotation(QDoubleValidator.Notation.StandardNotation)
        row = col = 0
        for name, desc, default in fields:
            c = QWidget()
            b = QVBoxLayout(c)
            b.setContentsMargins(0, 0, 0, 0)
            b.setSpacing(5)
            b.addWidget(QLabel(f"<b>{name}</b> - {desc}"))
            le = QLineEdit(default)
            le.setValidator(v)
            le.setAlignment(Qt.AlignmentFlag.AlignRight)
            le.setFixedHeight(32)
            b.addWidget(le)
            self.inputs[name] = le
            self.input_containers[name] = c
            layout.addWidget(c, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

    # ---- logic ----
    def _on_setup_change(self):
        res_idx = self.res_type_selector.currentIndex()
        tgt_idx = self.target_selector.currentIndex()

        # Enforce target validity
        model = self.target_selector.model()
        if res_idx == 0:  # Undersaturated
            model.item(1).setEnabled(False)
            model.item(2).setEnabled(False)
            if tgt_idx in (1, 2):
                self.target_selector.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx == 1:  # Gas Cap
            model.item(1).setEnabled(True)
            model.item(2).setEnabled(False)
            if tgt_idx == 2:
                self.target_selector.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx == 2:  # Water Drive
            model.item(1).setEnabled(False)
            model.item(2).setEnabled(True)
            if tgt_idx == 1:
                self.target_selector.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx == 3:  # Full MBE
            for row in range(4):
                model.item(row).setEnabled(True)

        for k in ("N", "m", "We", "Ginj", "Winj", "Bginj"):
            if k in self.input_containers:
                self.input_containers[k].setVisible(True)

        # Hide structurally irrelevant inputs
        if res_idx == 0:
            for k in ("m", "We", "Ginj", "Winj", "Bginj"):
                if k in self.input_containers: self.input_containers[k].setVisible(False)
        elif res_idx == 1:
            for k in ("We", "Winj"):
                if k in self.input_containers: self.input_containers[k].setVisible(False)
        elif res_idx == 2:
            for k in ("m", "Ginj", "Bginj"):
                if k in self.input_containers: self.input_containers[k].setVisible(False)

        # Hide inputs handled by target logic
        if tgt_idx == 0:
            if "N" in self.input_containers: self.input_containers["N"].setVisible(False)
            self.calc_btn.setText("Calculate N")
        elif tgt_idx == 1:
            if "m" in self.input_containers: self.input_containers["m"].setVisible(False)
            self.calc_btn.setText("Calculate m")
        elif tgt_idx == 2:
            if "We" in self.input_containers: self.input_containers["We"].setVisible(False)
            self.calc_btn.setText("Calculate We")
        else:
            self.calc_btn.setText("Calculate Driving Indexes")

    def _val(self, name):
        if not self.input_containers[name].isVisible():
            return 0.0
        t = self.inputs[name].text().strip()
        if not t:
            raise ValueError(f"Please enter a valid numeric value for {name}")
        v = float(t.replace(',', ''))
        if v < 0 and name != "dp":
            raise ValueError(f"Please enter a valid positive value for {name}")
        return v

    def _calculate(self):
        try:
            v = {n: self._val(n) for n in self.inputs}
            if v["Swi"] >= 1.0:
                raise ValueError("Swi must be strictly less than 1.0")

            res_type_idx = self.res_type_selector.currentIndex()
            tgt_idx = self.target_selector.currentIndex()
            inter = MBEEngine.compute(
                p=v.get("pi", 0) - v["dp"], pi=v.get("pi", 0), dp=v["dp"],
                Np=v["Np"], Rp=v["Rp"], Rs=v["Rs"], Bo=v["Bo"], Bg=v["Bg"], Wp=v["Wp"], Bw=v["Bw"],
                Boi=v["Boi"], Bgi=v["Bgi"], Rsi=v["Rsi"], Swi=v["Swi"], cw=v["cw"], cf=v["cf"],
                m=v["m"], We=v["We"], Winj=v["Winj"], Ginj=v["Ginj"], Bginj=v["Bginj"],
                res_type_idx=res_type_idx, solving_for_We=(tgt_idx == 2)
            )
            
            F = inter["F"]
            Eo = inter["Eo"]
            Eg = inter["Eg"]
            Efw_star = inter["Efw_star"]
            Efw = inter["Efw"]
            idx = tgt_idx

            if idx == 0:
                d = inter["den"]
                if d == 0:
                    raise ZeroDivisionError("Denominator in expanded equation evaluates to zero.")
                r = inter["num"] / d
                self._show_ok(f"Original Oil in Place (N):\n{r:,.2f} STB")
            elif idx == 1:
                if v["N"] == 0:
                    raise ZeroDivisionError("N cannot be zero.")
                nm = ((inter["F_adj"]) / v["N"]) - Eo - Efw_star
                dm = Eg + Efw_star
                if dm == 0:
                    raise ZeroDivisionError("Denominator is zero.")
                self._show_ok(f"Size of Gas Cap (m):\n{nm / dm:,.4f}")
            elif idx == 2:
                # We from expanded intrinsically matches exactly
                inner_subtracts = inter["produced_oil_gas"] - (v["Wp"] * v["Bw"]) - inter["gas_inj"] - inter["water_inj"]
                We_c = v["N"] * inter["Et"] - inner_subtracts
                self._show_ok(f"Cumulative Water Influx (We):\n{We_c:,.2f} bbl")
            else:
                di = MBEEngine.driving_indexes(
                    v["N"], inter["Eo"], v["m"], inter["Eg"], inter["Efw"],
                    v["We"], v["Wp"], v["Bw"], v["Winj"], v["Ginj"], v["Bginj"], 
                    res_type_idx=res_type_idx
                )
                if abs(di["Sum"]) < 1e-10:
                    raise ZeroDivisionError("Net A is zero.")
                self._show_ok(f"Depletion-Drive (DDI): {di['DDI']*100:,.2f}%\n"
                              f"Segregation-Drive (SDI): {di['SDI']*100:,.2f}%\n"
                              f"Water-Drive (WDI): {di['WDI']*100:,.2f}%\n"
                              f"Expansion-Drive (EDI): {di['EDI']*100:,.2f}%")
        except (ValueError, ZeroDivisionError) as e:
            self._show_err(str(e))
        except Exception as e:
            self._show_err(f"Unexpected error: {e}")

    def _show_ok(self, txt):
        self.result_label.setText(txt)
        self.result_frame.setStyleSheet("QFrame#result_frame{background:#d1ecf1;border:1px solid #bee5eb;border-radius:8px;}")
        self.result_label.setStyleSheet("color:#0c5460;font-size:22px;font-weight:bold;")

    def _show_err(self, txt):
        self.result_label.setText(f"Error:\n{txt}")
        self.result_frame.setStyleSheet("QFrame#result_frame{background:#f8d7da;border:1px solid #f5c6cb;border-radius:8px;}")
        self.result_label.setStyleSheet("color:#721c24;font-size:16px;font-weight:bold;")

    def _reset(self):
        for le in self.inputs.values():
            le.clear()
        self.result_label.setText("Result will be displayed here.")
        self.result_label.setStyleSheet("color:#6c757d;font-size:16px;font-weight:normal;")
        self.result_frame.setStyleSheet("QFrame#result_frame{background:#e9ecef;border:1px dashed #ced4da;border-radius:8px;}")
        self.inputs["Np"].setFocus()


# ═══════════════════════════════════════════════════════════════
# Tab 2 — Multi-Timestep Analysis  (NEW)
# ═══════════════════════════════════════════════════════════════

CSV_COLUMNS = ["Timestep", "p", "Np", "Rp", "Rs", "Bo", "Bg", "Wp", "Bw", "Winj", "Ginj"]

CONST_FIELDS = [
    # (key, label, default, group)
    ("pi",   "Initial Pressure (psi)",              "4000",      "Initial Conditions"),
    ("Boi",  "Initial Oil FVF (rb/STB)",            "1.400",     "Initial Conditions"),
    ("Bgi",  "Initial Gas FVF (rb/scf)",            "0.00090",   "Initial Conditions"),
    ("Rsi",  "Initial Solution GOR (scf/STB)",      "850",       "Initial Conditions"),
    ("Swi",  "Initial Water Saturation (frac)",     "0.20",      "Rock / Fluid"),
    ("cw",   "Water Compressibility (1/psi)",       "0.000003",  "Rock / Fluid"),
    ("cf",   "Formation Compressibility (1/psi)",   "0.000004",  "Rock / Fluid"),
    ("N",    "Original Oil in Place (STB)",          "10000000",  "Reservoir"),
    ("m",    "Gas Cap Ratio",                       "0.2",       "Reservoir"),
    ("Bginj","Injected Gas FVF (rb/scf)",           "0.001",     "Injection"),
]

TARGET_OPTIONS = [
    "Solve for N (OOIP)",
    "Solve for m (Gas Cap)",
    "Solve for We (Water Influx)",
    "Driving Indexes",
]


class MultiStepTab(QWidget):
    def __init__(self):
        super().__init__()
        self.raw_data = []
        self.results = []
        self.const_inputs = {}
        self.const_containers = {}
        self._build_ui()

    # ──────── UI construction ────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── toolbar ──
        tb = QHBoxLayout()
        for txt, slot, obj in [
            ("📂  Import CSV",        self._import_csv,        "import_btn"),
            ("📄  Generate Template",  self._gen_template,      "template_btn"),
        ]:
            b = QPushButton(txt)
            b.setObjectName(obj)
            b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.clicked.connect(slot)
            tb.addWidget(b)

        tb.addSpacing(15)
        tb.addWidget(QLabel("Type:"))
        self.res_type_cb = ScrolllessComboBox()
        self.res_type_cb.addItems([
            "Undersaturated Oil",
            "Gas Cap",
            "Water Drive",
            "Full MBE"
        ])
        self.res_type_cb.setFixedHeight(32)
        self.res_type_cb.setMinimumWidth(160)
        self.res_type_cb.setObjectName("res_type_selector")
        self.res_type_cb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.res_type_cb.currentIndexChanged.connect(self._on_setup_change)
        tb.addWidget(self.res_type_cb)

        tb.addWidget(QLabel("  Target:"))
        self.target_cb = ScrolllessComboBox()
        self.target_cb.addItems(TARGET_OPTIONS)
        self.target_cb.setFixedHeight(32)
        self.target_cb.setMinimumWidth(180)
        self.target_cb.setObjectName("target_selector")
        self.target_cb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.target_cb.currentIndexChanged.connect(self._on_setup_change)
        tb.addWidget(self.target_cb)

        tb.addStretch()
        calc_b = QPushButton("▶  Calculate")
        calc_b.setObjectName("calc_btn")
        calc_b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        calc_b.clicked.connect(self._calculate)
        tb.addWidget(calc_b)

        exp_b = QPushButton("💾  Export")
        exp_b.setObjectName("reset_btn")
        exp_b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        exp_b.clicked.connect(self._export)
        tb.addWidget(exp_b)
        root.addLayout(tb)

        # ── splitter: left constants | right content ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # Left – constants panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(220)
        left_scroll.setMaximumWidth(300)
        left_inner = QWidget()
        left_inner.setObjectName("main_content_widget")
        left_lay = QVBoxLayout(left_inner)
        left_lay.setSpacing(10)
        left_lay.setContentsMargins(10, 10, 10, 10)
        left_lay.addWidget(QLabel("<b>Constant Parameters</b>"))

        val = QDoubleValidator(-1e15, 1e15, 8, self)
        val.setNotation(QDoubleValidator.Notation.StandardNotation)

        current_group = None
        grp_layout = None
        for key, label, default, group in CONST_FIELDS:
            if group != current_group:
                current_group = group
                grp = QGroupBox(group)
                grp_layout = QVBoxLayout(grp)
                grp_layout.setSpacing(6)
                grp_layout.setContentsMargins(10, 20, 10, 10)
                left_lay.addWidget(grp)
            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            cl.addWidget(QLabel(f"<b>{key}</b> — {label}"))
            le = QLineEdit(default)
            le.setValidator(val)
            le.setFixedHeight(28)
            le.setAlignment(Qt.AlignmentFlag.AlignRight)
            cl.addWidget(le)
            self.const_inputs[key] = le
            self.const_containers[key] = container
            grp_layout.addWidget(container)

        left_lay.addStretch()
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        # Right – content tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.data_table, "📊 Data")

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.results_table, "📋 Results")

        # Havlena-Odeh plot
        ho_w = QWidget()
        ho_l = QVBoxLayout(ho_w)
        ho_l.setContentsMargins(0, 0, 0, 0)
        self.ho_canvas = MBEPlotCanvas(ho_w)
        self.ho_toolbar = NavigationToolbar(self.ho_canvas, ho_w)
        ho_l.addWidget(self.ho_toolbar)
        ho_l.addWidget(self.ho_canvas)
        self.tabs.addTab(ho_w, "📈 Havlena-Odeh")

        # Driving Index plot
        di_w = QWidget()
        di_l = QVBoxLayout(di_w)
        di_l.setContentsMargins(0, 0, 0, 0)
        self.di_canvas = MBEPlotCanvas(di_w)
        self.di_toolbar = NavigationToolbar(self.di_canvas, di_w)
        di_l.addWidget(self.di_toolbar)
        di_l.addWidget(self.di_canvas)
        self.tabs.addTab(di_w, "📉 Drive Indexes")

        # Analysis text
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setObjectName("analysis_box")
        self.tabs.addTab(self.analysis_text, "📝 Analysis")

        splitter.addWidget(self.tabs)
        splitter.setSizes([250, 650])
        root.addWidget(splitter, 1)

        # Status
        self.status = QLabel("📂 Import a CSV file or generate a template to begin.")
        self.status.setObjectName("subtitle")
        root.addWidget(self.status)
        self.res_type_cb.setCurrentIndex(3)
        self._on_setup_change()

    # ──────── setup change ────────
    def _on_setup_change(self):
        res_idx = self.res_type_cb.currentIndex()
        tgt_idx = self.target_cb.currentIndex()

        # Enforce target validity
        model = self.target_cb.model()
        if res_idx == 0:  # Undersaturated
            model.item(1).setEnabled(False)
            model.item(2).setEnabled(False)
            if tgt_idx in (1, 2):
                self.target_cb.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx == 1:  # Gas Cap
            model.item(1).setEnabled(True)
            model.item(2).setEnabled(False)
            if tgt_idx == 2:
                self.target_cb.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx == 2:  # Water Drive
            model.item(1).setEnabled(False)
            model.item(2).setEnabled(True)
            if tgt_idx == 1:
                self.target_cb.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx == 3:  # Full
            for row in range(4):
                model.item(row).setEnabled(True)

        for k in ("N", "m", "Bginj"):
            if k in self.const_containers:
                self.const_containers[k].setVisible(True)

        if res_idx == 0:
            if "m" in self.const_containers: self.const_containers["m"].setVisible(False)
            if "Bginj" in self.const_containers: self.const_containers["Bginj"].setVisible(False)
        elif res_idx == 2:
            if "m" in self.const_containers: self.const_containers["m"].setVisible(False)
            if "Bginj" in self.const_containers: self.const_containers["Bginj"].setVisible(False)

        if tgt_idx == 0 and "N" in self.const_containers:
            self.const_containers["N"].setVisible(False)
        elif tgt_idx == 1 and "m" in self.const_containers:
            self.const_containers["m"].setVisible(False)

    # ──────── CSV template ────────
    def _gen_template(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Template CSV", "mbe_template.csv", "CSV Files (*.csv)")
        if not path:
            return
        sample = [
            [0, 4000,       0,   0, 850, 1.400, 0.00090,     0, 1.02, 0, 0],
            [1, 3900,  500000, 820, 830, 1.390, 0.00095,  5000, 1.02, 0, 0],
            [2, 3800, 1200000, 830, 800, 1.380, 0.00100, 15000, 1.02, 0, 0],
            [3, 3600, 2500000, 850, 750, 1.360, 0.00110, 35000, 1.02, 0, 0],
            [4, 3400, 4000000, 870, 700, 1.340, 0.00120, 60000, 1.02, 0, 0],
            [5, 3200, 5500000, 900, 650, 1.320, 0.00130, 90000, 1.02, 0, 0],
        ]
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            w.writerows(sample)
        self.status.setText(f"✅ Template saved → {os.path.basename(path)}")

    # ──────── CSV import ────────
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All (*)")
        if not path:
            return
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                self.raw_data = []
                for i, row in enumerate(reader):
                    parsed = {}
                    for col in CSV_COLUMNS:
                        if col not in row:
                            raise KeyError(f"Missing column: {col}")
                        parsed[col] = float(row[col])
                    self.raw_data.append(parsed)
            if not self.raw_data:
                raise ValueError("CSV file has no data rows.")
            self._populate_data_table()
            self.status.setText(f"✅ Loaded {len(self.raw_data)} timesteps from {os.path.basename(path)}")
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def _populate_data_table(self):
        t = self.data_table
        t.setColumnCount(len(CSV_COLUMNS))
        t.setHorizontalHeaderLabels(CSV_COLUMNS)
        t.setRowCount(len(self.raw_data))
        for r, row in enumerate(self.raw_data):
            for c, col in enumerate(CSV_COLUMNS):
                item = QTableWidgetItem(f"{row[col]:g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    # ──────── calculation ────────
    def _get_const(self, key):
        t = self.const_inputs[key].text().strip()
        if not t:
            raise ValueError(f"Enter a value for constant: {key}")
        return float(t.replace(',', ''))

    def _calculate(self):
        if not self.raw_data:
            QMessageBox.warning(self, "No Data", "Import a CSV file first.")
            return
        try:
            pi   = self._get_const("pi")
            Boi  = self._get_const("Boi")
            Bgi  = self._get_const("Bgi")
            Rsi  = self._get_const("Rsi")
            Swi  = self._get_const("Swi")
            cw   = self._get_const("cw")
            cf   = self._get_const("cf")
            Bginj = self._get_const("Bginj")

            idx = self.target_cb.currentIndex()
            m_val = self._get_const("m") if idx != 1 else 0.0
            N_val = self._get_const("N") if idx != 0 else 0.0

            res_type_idx = self.res_type_cb.currentIndex()

            self.results = []
            for row in self.raw_data:
                inter = MBEEngine.compute(
                    p=row["p"], pi=pi, Np=row["Np"], Rp=row["Rp"],
                    Rs=row["Rs"], Bo=row["Bo"], Bg=row["Bg"],
                    Wp=row["Wp"], Bw=row["Bw"], Boi=Boi, Bgi=Bgi,
                    Rsi=Rsi, Swi=Swi, cw=cw, cf=cf, m=m_val,
                    We=0.0,
                    Winj=row.get("Winj", 0), Ginj=row.get("Ginj", 0), Bginj=Bginj,
                    res_type_idx=res_type_idx, dp=None, solving_for_We=(idx == 2)
                )
                res = dict(Timestep=row["Timestep"], p=row["p"], **inter)
                res["Wp"] = row["Wp"]
                res["Winj"] = row.get("Winj", 0)
                res["Ginj"] = row.get("Ginj", 0)
                res["Bw"] = row["Bw"]

                if idx == 0: # Solve for N expanding form directly
                    res["N_calc"] = inter["num"] / inter["den"] if inter["den"] != 0 else float('nan')
                elif idx == 1:
                    res["m_calc"] = MBEEngine.solve_m(inter["F_adj"], N_val, inter["Eo"], inter["Eg"], inter["Efw_star"])
                elif idx == 2:
                    inner_subtracts = inter["produced_oil_gas"] - (row["Wp"] * row["Bw"]) - inter["gas_inj"] - inter["water_inj"]
                    res["We_calc"] = N_val * inter["Et"] - inner_subtracts
                
                # Driving Indexes
                We_for_di = res.get("We_calc", 0.0)
                N_for_di = res.get("N_calc", N_val)
                di = MBEEngine.driving_indexes(
                    N_for_di, inter["Eo"], m_val, inter["Eg"], inter["Efw"],
                    We_for_di, row["Wp"], row["Bw"], row.get("Winj", 0), row.get("Ginj", 0), Bginj,
                    res_type_idx=res_type_idx
                )
                res.update(di)
                self.results.append(res)

            # Do linear regression for H-O plot (skip timestep 0 where Et≈0)
            if idx == 0:
                valid = [(r["Et"], r["F_adj"]) for r in self.results if abs(r["Et"]) > 1e-15]
                if len(valid) >= 2:
                    Et_arr = np.array([v[0] for v in valid])
                    F_arr  = np.array([v[1] for v in valid])
                    slope = np.sum(Et_arr * F_arr) / np.sum(Et_arr ** 2)
                    self.ho_slope = slope
                    ss_res = np.sum((F_arr - slope * Et_arr) ** 2)
                    ss_tot = np.sum((F_arr - np.mean(F_arr)) ** 2)
                    self.ho_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
                else:
                    self.ho_slope = float('nan')
                    self.ho_r2 = 0.0

            self._fill_results_table(idx)
            self._draw_ho_plot(idx)
            self._draw_di_plot()
            self._write_analysis(idx, N_val, m_val)
            self.tabs.setCurrentIndex(1)
            self.status.setText(f"✅ Calculation complete — {len(self.results)} timesteps processed.")
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", str(e))

    # ──────── results table ────────
    def _fill_results_table(self, idx):
        cols = ["Timestep", "p", "dp", "F", "Eo", "Eg", "Efw", "Et"]
        if idx == 0:
            cols.append("N_calc")
        elif idx == 1:
            cols.append("m_calc")
        elif idx == 2:
            cols.append("We_calc")
        cols += ["DDI", "SDI", "WDI", "EDI"]

        t = self.results_table
        t.setColumnCount(len(cols))
        nice = {"N_calc": "N (STB)", "m_calc": "m", "We_calc": "We (bbl)",
                "DDI": "DDI %", "SDI": "SDI %", "WDI": "WDI %", "EDI": "EDI %"}
        t.setHorizontalHeaderLabels([nice.get(c, c) for c in cols])
        t.setRowCount(len(self.results))
        for r, row in enumerate(self.results):
            for c, col in enumerate(cols):
                v = row.get(col, 0)
                if col in ("DDI", "SDI", "WDI", "EDI"):
                    txt = f"{v * 100:.2f}"
                elif col in ("N_calc", "We_calc", "F"):
                    txt = f"{v:,.0f}" if not math.isnan(v) else "N/A"
                elif col in ("m_calc",):
                    txt = f"{v:.4f}" if not math.isnan(v) else "N/A"
                elif col in ("Eo", "Eg", "Efw", "Et"):
                    txt = f"{v:.6f}"
                else:
                    txt = f"{v:g}"
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    # ──────── Havlena-Odeh Plot ────────
    def _draw_ho_plot(self, idx):
        ax = self.ho_canvas.axes
        ax.clear()
        ax.set_facecolor('#ffffff')

        if idx == 0:
            Et = [r["Et"] for r in self.results]
            F_adj = [r["F_adj"] for r in self.results]
            ax.scatter(Et, F_adj, color='#2980b9', s=60, zorder=5, edgecolors='white', linewidths=1.2, label='Data Points')
            if hasattr(self, 'ho_slope') and not math.isnan(self.ho_slope):
                x_line = np.linspace(0, max(Et) * 1.1, 100)
                ax.plot(x_line, self.ho_slope * x_line, '--', color='#e74c3c', linewidth=2,
                        label=f'N = {self.ho_slope:,.0f} STB  (R² = {self.ho_r2:.4f})')
            ax.set_xlabel('Et  (Eo + m·Eg + Efw)', fontsize=11)
            ax.set_ylabel('F adjusted  (rb)', fontsize=11)
            ax.set_title('Havlena-Odeh: F vs Et  →  Slope = N', fontsize=13, fontweight='bold')
        elif idx == 1:
            valid = [(r["Eo"], r["Eg"], r["F_adj"]) for r in self.results if abs(r["Eo"]) > 1e-15]
            if valid:
                x = [v[1] / v[0] for v in valid]
                y = [v[2] / v[0] for v in valid]
                ax.scatter(x, y, color='#8e44ad', s=60, zorder=5, edgecolors='white', linewidths=1.2, label='Data Points')
                if len(valid) >= 2:
                    coeffs = np.polyfit(x, y, 1)
                    x_line = np.linspace(min(x), max(x) * 1.1, 100)
                    ax.plot(x_line, np.polyval(coeffs, x_line), '--', color='#e74c3c', linewidth=2,
                            label=f'Intercept(N)={coeffs[1]:,.0f}, Slope(Nm)={coeffs[0]:,.0f}')
                ax.set_xlabel('Eg / Eo', fontsize=11)
                ax.set_ylabel('F / Eo', fontsize=11)
                ax.set_title('Havlena-Odeh: F/Eo vs Eg/Eo', fontsize=13, fontweight='bold')
        elif idx == 2:
            ts = [r["Timestep"] for r in self.results]
            we = [r.get("We_calc", 0) for r in self.results]
            ax.bar(ts, we, color='#3498db', edgecolor='white', linewidth=0.8, label='We')
            ax.set_xlabel('Timestep', fontsize=11)
            ax.set_ylabel('We (bbl)', fontsize=11)
            ax.set_title('Calculated Water Influx Over Time', fontsize=13, fontweight='bold')
        else:
            ts = [r["Timestep"] for r in self.results]
            n_vals = [r.get("N_calc", self._get_const("N")) for r in self.results]
            ax.plot(ts, n_vals, 'o-', color='#2c3e50', linewidth=2, markersize=6, label='F/Et')
            ax.set_xlabel('Timestep', fontsize=11)
            ax.set_ylabel('F / Et  (STB)', fontsize=11)
            ax.set_title('Campbell Plot — N Consistency Check', fontsize=13, fontweight='bold')

        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        self.ho_canvas.fig.tight_layout()
        self.ho_canvas.draw()

    # ──────── Driving Index Plot ────────
    def _draw_di_plot(self):
        ax = self.di_canvas.axes
        ax.clear()
        ax.set_facecolor('#ffffff')

        valid = [r for r in self.results if abs(r.get("Sum", 0)) > 1e-10]
        if not valid:
            ax.text(0.5, 0.5, 'No valid driving index data', ha='center', va='center', fontsize=14, color='#999')
            self.di_canvas.draw()
            return

        ts   = [r["Timestep"] for r in valid]
        DDI  = np.array([r["DDI"] * 100 for r in valid])
        SDI  = np.array([r["SDI"] * 100 for r in valid])
        WDI  = np.array([r["WDI"] * 100 for r in valid])
        EDI  = np.array([r["EDI"] * 100 for r in valid])

        colors = {'DDI': '#e74c3c', 'SDI': '#f39c12', 'WDI': '#3498db', 'EDI': '#2ecc71'}
        ax.fill_between(ts, 0, DDI, color=colors['DDI'], alpha=0.85, label='Depletion (DDI)')
        ax.fill_between(ts, DDI, DDI + SDI, color=colors['SDI'], alpha=0.85, label='Gas Cap (SDI)')
        ax.fill_between(ts, DDI + SDI, DDI + SDI + WDI, color=colors['WDI'], alpha=0.85, label='Water (WDI)')
        ax.fill_between(ts, DDI + SDI + WDI, DDI + SDI + WDI + EDI, color=colors['EDI'], alpha=0.85, label='Expansion (EDI)')

        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Drive Contribution (%)', fontsize=11)
        ax.set_title('Primary Driving Indexes Over Time', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        self.di_canvas.fig.tight_layout()
        self.di_canvas.draw()

    # ──────── Analysis / Interpretation ────────
    def _write_analysis(self, idx, N_val, m_val):
        lines = []
        lines.append("=" * 55)
        lines.append("     MBE ANALYSIS REPORT")
        lines.append(f"     Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 55)
        lines.append(f"\nTarget: {TARGET_OPTIONS[idx]}")
        lines.append(f"Timesteps analysed: {len(self.results)}\n")

        if idx == 0 and hasattr(self, 'ho_slope'):
            lines.append("── Havlena-Odeh Straight-Line Analysis ──")
            lines.append(f"  Calculated N (slope): {self.ho_slope:,.0f} STB")
            lines.append(f"  R² value:             {self.ho_r2:.6f}")
            if self.ho_r2 > 0.99:
                lines.append("  Confidence:           ★★★ HIGH  (R² > 0.99)")
            elif self.ho_r2 > 0.95:
                lines.append("  Confidence:           ★★  MODERATE  (R² > 0.95)")
            else:
                lines.append("  Confidence:           ★   LOW  (R² < 0.95, possible water influx)")
            n_per_ts = [r["N_calc"] for r in self.results if not math.isnan(r.get("N_calc", float('nan'))) and r["Et"] != 0]
            if n_per_ts:
                avg_n = np.mean(n_per_ts)
                std_n = np.std(n_per_ts)
                cv = (std_n / avg_n * 100) if avg_n != 0 else 0
                lines.append(f"\n  N per-timestep average: {avg_n:,.0f} STB")
                lines.append(f"  Std deviation:          {std_n:,.0f} STB")
                lines.append(f"  Coeff. of variation:    {cv:.1f}%")

        elif idx == 1:
            m_vals = [r["m_calc"] for r in self.results if not math.isnan(r.get("m_calc", float('nan')))]
            if m_vals:
                lines.append("── Gas Cap Ratio Analysis ──")
                lines.append(f"  Average m:     {np.mean(m_vals):.4f}")
                lines.append(f"  Std deviation: {np.std(m_vals):.4f}")

        elif idx == 2:
            we_vals = [r["We_calc"] for r in self.results]
            if we_vals:
                lines.append("── Water Influx Analysis ──")
                lines.append(f"  Total We at last timestep: {we_vals[-1]:,.0f} bbl")
                if len(we_vals) > 1:
                    delta = we_vals[-1] - we_vals[-2]
                    lines.append(f"  Latest increment:          {delta:,.0f} bbl")

        # Driving Index summary
        valid_di = [r for r in self.results if abs(r.get("Sum", 0)) > 1e-10]
        if valid_di:
            lines.append("\n── Driving Mechanism Summary ──")
            avg_ddi = np.mean([r["DDI"] for r in valid_di]) * 100
            avg_sdi = np.mean([r["SDI"] for r in valid_di]) * 100
            avg_wdi = np.mean([r["WDI"] for r in valid_di]) * 100
            avg_edi = np.mean([r["EDI"] for r in valid_di]) * 100
            lines.append(f"  Avg DDI (Depletion):   {avg_ddi:.1f}%")
            lines.append(f"  Avg SDI (Gas Cap):     {avg_sdi:.1f}%")
            lines.append(f"  Avg WDI (Water):       {avg_wdi:.1f}%")
            lines.append(f"  Avg EDI (Expansion):   {avg_edi:.1f}%")

            mechs = {"Depletion Drive": avg_ddi, "Gas Cap Drive": avg_sdi,
                     "Water Drive": avg_wdi, "Expansion Drive": avg_edi}
            dominant = max(mechs, key=mechs.get)
            lines.append(f"\n  ➤ Dominant mechanism: {dominant} ({mechs[dominant]:.1f}%)")

            avg_sum = np.mean([r["Sum"] for r in valid_di]) * 100
            lines.append(f"  ➤ DI Sum check:       {avg_sum:.1f}% {'✓' if abs(avg_sum - 100) < 2 else '⚠ off by >2%'}")

        lines.append("\n" + "=" * 55)
        self.analysis_text.setPlainText("\n".join(lines))

    # ──────── export ────────
    def _export(self):
        if not self.results:
            QMessageBox.warning(self, "Nothing to export", "Run a calculation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Results CSV", "mbe_results.csv", "CSV Files (*.csv)")
        if not path:
            return
        keys = list(self.results[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.results)
        # Also save analysis text
        txt_path = path.rsplit('.', 1)[0] + '_analysis.txt'
        with open(txt_path, 'w') as f:
            f.write(self.analysis_text.toPlainText())
        self.status.setText(f"✅ Exported → {os.path.basename(path)}  +  analysis txt")


# ═══════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════

class MBECalculatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MBE Calculator — Reservoir Engineering")
        self.setMinimumSize(900, 650)
        self.resize(1100, 780)

        root = QWidget()
        root.setObjectName("root_widget")
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)

        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)
        self.main_tabs.addTab(SinglePointTab(), "🔢  Single-Point Calculator")
        self.main_tabs.addTab(MultiStepTab(),   "📈  Multi-Timestep Analysis")
        rl.addWidget(self.main_tabs)

        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QMainWindow, QWidget#root_widget, QWidget#main_content_widget {
                background-color: #f4f6f9;
            }
            QLabel {
                font-family: '.AppleSystemUIFont', 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #343a40; font-size: 13px;
            }
            QLabel#header_title { font-size: 24px; font-weight: bold; color: #2c3e50; }
            QLabel#subtitle { font-size: 13px; color: #6c757d; margin-bottom: 6px; }

            QGroupBox {
                font-size: 14px; font-weight: bold; color: #2c3e50;
                border: 1px solid #ced4da; border-radius: 8px;
                margin-top: 15px; background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 10px; left: 15px;
            }

            QLineEdit {
                padding: 0px 8px; border: 1px solid #ced4da; border-radius: 6px;
                background: #fff; font-size: 14px; color: #495057;
                selection-background-color: #007bff; min-height: 28px;
            }
            QLineEdit:focus { border: 2px solid #80bdff; }

            QFrame#result_frame {
                background: #e9ecef; border: 1px dashed #ced4da;
                border-radius: 8px; padding: 20px; margin-top: 10px;
            }
            QLabel#result_text { font-size: 16px; color: #6c757d; }

            QPushButton {
                font-family: '.AppleSystemUIFont', 'Helvetica Neue', Helvetica;
                font-size: 14px; font-weight: bold; padding: 10px 18px; border-radius: 6px;
            }
            QPushButton#calc_btn { background: #0069d9; color: white; border: none; }
            QPushButton#calc_btn:hover { background: #0056b3; }
            QPushButton#calc_btn:pressed { background: #004085; }
            QPushButton#reset_btn, QPushButton#import_btn, QPushButton#template_btn {
                background: #ffffff; color: #495057; border: 1px solid #ced4da;
            }
            QPushButton#reset_btn:hover, QPushButton#import_btn:hover, QPushButton#template_btn:hover {
                background: #e2e6ea; color: #212529;
            }

            QComboBox#target_selector {
                background: #495057; color: white; border: 1px solid #ced4da;
                border-radius: 6px; padding-left: 10px; min-height: 30px;
            }
            QComboBox#target_selector QAbstractItemView {
                background: #343a40; color: white; selection-background-color: #007bff;
            }

            QTabWidget::pane { border: 1px solid #ced4da; border-radius: 6px; background: #fff; }
            QTabBar::tab {
                padding: 8px 18px; margin-right: 3px; border-top-left-radius: 6px;
                border-top-right-radius: 6px; background: #e9ecef; color: #495057;
                font-weight: bold; font-size: 13px;
            }
            QTabBar::tab:selected { background: #fff; color: #0069d9; border-bottom: 2px solid #0069d9; }
            QTabBar::tab:hover { background: #dee2e6; }

            QTableWidget {
                gridline-color: #dee2e6; font-size: 13px; background: #fff;
                color: #212529;
                alternate-background-color: #f8f9fa; selection-background-color: #cce5ff;
                selection-color: #212529;
            }
            QTableWidget QTableWidgetItem {
                color: #212529;
            }
            QHeaderView::section {
                background: #e9ecef; color: #2c3e50; font-weight: bold;
                padding: 6px; border: 1px solid #dee2e6; font-size: 12px;
            }

            QTextEdit#analysis_box {
                font-family: 'Menlo', 'Courier New', monospace; font-size: 13px;
                background: #1e1e2e; color: #cdd6f4; border: none; padding: 15px;
                selection-background-color: #45475a;
            }

            QSplitter::handle { background: #ced4da; }
        """)


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MBECalculatorWindow()
    window.show()
    sys.exit(app.exec())

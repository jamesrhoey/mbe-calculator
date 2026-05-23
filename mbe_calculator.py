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
        self.fig = Figure(figsize=(width, height), dpi=100, facecolor='#0d1117')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0f1923')
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
        if res_type_idx in (5, 6, 7):
            numerator = Np * Bo
        else:
            numerator = Np * (Bo + (Rp - Rs) * Bg)
        
        # 2. Net Water Influx (Active in Water Drive, Full MBE, and Undersat. w/ Bottom Water modes)
        if res_type_idx in (2, 3, 4, 6, 7):
            if solving_for_We:
                numerator += Wp * Bw
            else:
                numerator -= (We - Wp * Bw)
            
        if res_type_idx == 3:
            numerator -= (Ginj * Bginj)
            numerator -= (Winj * Bw)
            
        m_eff = m if res_type_idx in (1, 3, 4, 9, 11) else 0.0
        
        # ──────── STRUCTURAL DENOMINATOR CONSTRUCTION ────────
        # 4. Evolved Gas Expansion (Always active)
        if res_type_idx in (0, 5, 6, 7):
            denominator = (Bo - Boi)
        else:
            denominator = (Bo - Boi) + (Rsi - Rs) * Bg
            
        # 5. Gas Cap Expansion
        gas_cap_expansion_base = Boi * ((Bg / Bgi) - 1.0) if Bgi > 0 else 0.0
        denominator += m_eff * gas_cap_expansion_base
            
        # 6. Rock & Connate Water Expansion (Uses structured m_eff consistency)
        rock_water_expansion_base = Boi * ((Swi * cw + cf) / (1.0 - Swi)) * dp if Swi < 1.0 and res_type_idx not in (5, 6, 8, 9, 10, 11) else 0.0
        denominator += (1.0 + m_eff) * rock_water_expansion_base

        # ──────── PRESERVE ORIGINAL FORMS FOR COMPATIBILITY ────────
        if res_type_idx in (0, 5, 6, 7):
            Eo = (Bo - Boi)
            if res_type_idx == 0:
                # F for index 0 includes gas term per user linear form
                produced_oil_gas = Np * (Bo + (Rp - Rs) * Bg)
            else:
                produced_oil_gas = Np * Bo
        else:
            Eo = (Bo - Boi) + (Rsi - Rs) * Bg
            produced_oil_gas = Np * (Bo + (Rp - Rs) * Bg)
            
        Eg = gas_cap_expansion_base
        Efw_star = rock_water_expansion_base
        Efw = (1.0 + m_eff) * Efw_star
        Bt = Bo + (Rsi - Rs) * Bg
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
        if res_type_idx in (1, 3, 4):
            term_gas_cap = N * m * Eg
            
        term_water = 0.0
        if res_type_idx in (2, 3, 4, 6, 7):
            term_water += (We - Wp * Bw)
        if res_type_idx == 3:
            term_water += Winj * Bw
            term_water += Ginj * Bginj
            
        term_expansion = N * Efw if res_type_idx not in (5, 6) else 0.0
        
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
# Formula Display Helper
# ═══════════════════════════════════════════════════════════════

RES_TITLES = {
    0: "Undersaturated Oil Reservoir",
    1: "Gas Cap Reservoir",
    2: "Water Drive Reservoir",
    3: "Full MBE (with Injection)",
    4: "Full MBE (No Injection)",
    5: "Undersat. (No Liquid/Rock Expansion)",
    6: "Undersat. with Bottom Water",
    7: "Undersat. Oil (Bottom Water & Expansion)",
    8: "Saturated Oil (No Gas Cap/Expansion)",
    9: "Saturated Oil (Gas Cap, No Expansion)",
    10: "Saturated Oil (Bottom Water, No Expansion)",
    11: "Combination Drive (No Expansion)",
}

def build_formula_html(res_idx, compact=False, notation='bo'):
    """Return rich-text HTML for the active MBE equation.
    notation: 'bo' for standard Bo form, 'bt' for two-phase Bt form.
    """
    title = RES_TITLES.get(res_idx, "Material Balance Equation")
    bt_mode = (notation == 'bt')
    notation_label = 'Two-Phase (B<sub>t</sub>) Form' if bt_mode else 'Standard (B<sub>o</sub>) Form'

    # ── Numerator ──
    if res_idx == 0:
        num = 'N<sub>p</sub>B<sub>o</sub>'
    elif bt_mode:
        num = 'N<sub>p</sub>(B<sub>t</sub> + (R<sub>p</sub> \u2212 R<sub>si</sub>)B<sub>g</sub>)'
    else:
        num = 'N<sub>p</sub>(B<sub>o</sub> + (R<sub>p</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>)'
    if res_idx in (2, 3, 4, 6, 7, 10, 11):
        num += ' \u2212 (W<sub>e</sub> \u2212 W<sub>p</sub>B<sub>w</sub>)'
    if res_idx == 3:
        num += ' \u2212 (G<sub>inj</sub>B<sub>ginj</sub>) \u2212 (W<sub>inj</sub>B<sub>w</sub>)'

    # ── Denominator ──
    # Base variable for FVF
    bv = 'B<sub>t</sub>' if bt_mode else 'B<sub>o</sub>'
    bvi = 'B<sub>ti</sub>' if bt_mode else 'B<sub>oi</sub>'
    if bt_mode or res_idx in (5, 6, 7):
        den = f'({bv} \u2212 {bvi})'
    else:
        den = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>'
    if res_idx in (1, 3, 4, 9, 11):
        den += f' + m{bvi}(B<sub>g</sub>/B<sub>gi</sub> \u2212 1)'
        if res_idx not in (9, 11):
            den += f' + {bvi}(1+m)(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'
    elif res_idx in (5, 6, 8, 9, 10, 11):
        # No expansion term
        pass
    else:
        den += f' + {bvi}(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'

    # ── Havlena-Odeh linear form ──
    if res_idx == 0:
        # User requested unified term for index 0
        ho_eq = 'F = N(E<sub>o,f,w</sub>)'
    elif res_idx == 7:
        # User requested specific form for index 7
        ho_eq = 'F = N(E<sub>o</sub> + E<sub>f,w</sub>) + W<sub>e</sub>'
    else:
        ho = ['E<sub>o</sub>']
        if res_idx in (1, 3, 4, 9):
            ho.append('mE<sub>g</sub>')
        if res_idx not in (5, 6, 8, 9):
            ho.append('E<sub>f,w</sub>')
        ho_eq = 'F = N(' + ' + '.join(ho) + ')'
    
    if res_idx in (2, 3, 4, 6, 10, 11) and res_idx != 7:
        ho_eq += ' + W<sub>e</sub>'

    if compact:
        return (f'<span style="font-size:12px;color:#6c757d;"><b>Active Equation</b> \u2014 {title} [{notation_label}]: </span>'
                f'<span style="font-size:13px;font-weight:bold;color:#1a5276;">{ho_eq}</span>')

    # ── Component definitions ──
    if bt_mode:
        f_def = f'N<sub>p</sub>({bv} + (R<sub>p</sub> \u2212 R<sub>si</sub>)B<sub>g</sub>) + W<sub>p</sub>B<sub>w</sub>'
    else:
        f_def = f'N<sub>p</sub>({bv} + (R<sub>p</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>) + W<sub>p</sub>B<sub>w</sub>'

    if res_idx == 7:
        # Simplified numerator for undersaturated bottom water
        num = 'N<sub>p</sub>B<sub>o</sub> \u2212 (W<sub>e</sub> \u2212 W<sub>p</sub>B<sub>w</sub>)'
        den = f'(B<sub>o</sub> \u2212 B<sub>oi</sub>) + B<sub>oi</sub>(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'
        if bt_mode:
            num = 'N<sub>p</sub>B<sub>t</sub> \u2212 (W<sub>e</sub> \u2212 W<sub>p</sub>B<sub>w</sub>)'
            den = f'(B<sub>t</sub> \u2212 B<sub>ti</sub>) + B<sub>ti</sub>(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'
    elif res_idx == 6:
        # Simplified numerator for undersaturated bottom water (no expansion)
        num = 'N<sub>p</sub>B<sub>o</sub> \u2212 (W<sub>e</sub> \u2212 W<sub>p</sub>B<sub>w</sub>)'
        den = f'(B<sub>o</sub> \u2212 B<sub>oi</sub>)'
        if bt_mode:
            num = 'N<sub>p</sub>B<sub>t</sub> \u2212 (W<sub>e</sub> \u2212 W<sub>p</sub>B<sub>w</sub>)'
            den = f'(B<sub>t</sub> \u2212 B<sub>ti</sub>)'
    elif res_idx == 5:
        # Simplified numerator for undersaturated no expansion
        num = 'N<sub>p</sub>B<sub>o</sub>'
        den = f'(B<sub>o</sub> \u2212 B<sub>oi</sub>)'
        if bt_mode:
            num = 'N<sub>p</sub>B<sub>t</sub>'
            den = f'(B<sub>t</sub> \u2212 B<sub>ti</sub>)'

    if res_idx == 3:
        f_def += ' + G<sub>inj</sub>B<sub>ginj</sub> + W<sub>inj</sub>B<sub>w</sub>'
    
    if res_idx == 0:
        # Unified term def for index 0
        f_def_unsat = f'N<sub>p</sub>(B<sub>o</sub> + (R<sub>p</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>)'
        eo_def = f'{bvi} \u00b7 c<sub>e</sub> \u00b7 \u0394p'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def_unsat}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o,f,w</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Oil, Formation Rock and (Connate) Water Expansion</td></tr>')
    elif res_idx == 7:
        f_def_simple = f'N<sub>p</sub>(B<sub>o</sub> + (R<sub>p</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>) + W<sub>p</sub>B<sub>w</sub>' if not bt_mode else f'N<sub>p</sub>(B<sub>t</sub> + (R<sub>p</sub> \u2212 R<sub>si</sub>)B<sub>g</sub>) + W<sub>p</sub>B<sub>w</sub>'
        eo_def = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>'
        efw_def = f'{bvi}(1+m)(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def_simple}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil Expansion</td></tr>'
                 f'<tr><td align="right"><b>E<sub>f,w</sub></b> =&nbsp;</td><td>{efw_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Formation Rock and (Connate) Water Expansion</td></tr>'
                 '<tr><td align="right"><b>W<sub>e</sub></b>&nbsp;</td><td></td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Water Influx</td></tr>')
    elif res_idx == 6:
        f_def_simple = f'N<sub>p</sub>B<sub>o</sub> + W<sub>p</sub>B<sub>w</sub>' if not bt_mode else f'N<sub>p</sub>B<sub>t</sub> + W<sub>p</sub>B<sub>w</sub>'
        eo_def = f'({bv} \u2212 {bvi})'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def_simple}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil Expansion</td></tr>'
                 '<tr><td align="right"><b>W<sub>e</sub></b>&nbsp;</td><td></td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Water Influx</td></tr>')
    elif res_idx == 5:
        f_def_simple = f'N<sub>p</sub>B<sub>o</sub>' if not bt_mode else f'N<sub>p</sub>B<sub>t</sub>'
        eo_def = f'({bv} \u2212 {bvi})'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def_simple}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil and Gas liberated Expansion</td></tr>')
    elif res_idx == 8:
        f_def_simple = f'N<sub>p</sub>({bv} + (R<sub>p</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>)' if not bt_mode else f'N<sub>p</sub>({bv} + (R<sub>p</sub> \u2212 R<sub>si</sub>)B<sub>g</sub>)'
        eo_def = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def_simple}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil and Gas liberated Expansion</td></tr>')
    elif res_idx == 9:
        eo_def = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>'
        eg_def = f'{bvi}(B<sub>g</sub>/B<sub>gi</sub> \u2212 1)'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil and Gas liberated Expansion</td></tr>'
                 f'<tr><td align="right"><b>E<sub>g</sub></b> =&nbsp;</td><td>{eg_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Gas Cap Expansion</td></tr>')
    elif res_idx == 10:
        f_def_simple = f'N<sub>p</sub>({bv} + (R<sub>p</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>) + W<sub>p</sub>B<sub>w</sub>' if not bt_mode else f'N<sub>p</sub>({bv} + (R<sub>p</sub> \u2212 R<sub>si</sub>)B<sub>g</sub>) + W<sub>p</sub>B<sub>w</sub>'
        eo_def = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def_simple}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil and Gas liberated Expansion</td></tr>'
                 '<tr><td align="right"><b>W<sub>e</sub></b>&nbsp;</td><td></td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Water Influx</td></tr>')
    elif res_idx == 11:
        eo_def = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>' if not bt_mode else f'({bv} \u2212 {bvi})'
        eg_def = f'{bvi}(B<sub>g</sub>/B<sub>gi</sub> \u2212 1)'
        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Remained Oil and Gas liberated Expansion</td></tr>'
                 f'<tr><td align="right"><b>E<sub>g</sub></b> =&nbsp;</td><td>{eg_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Gas Cap Expansion</td></tr>'
                 '<tr><td align="right"><b>W<sub>e</sub></b>&nbsp;</td><td></td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Water Influx</td></tr>')
    else:
        if bt_mode:
            eo_def = f'({bv} \u2212 {bvi})'
        else:
            eo_def = f'({bv} \u2212 {bvi}) + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub>'
        eg_def = f'{bvi}(B<sub>g</sub>/B<sub>gi</sub> \u2212 1)'
        if res_idx in (1, 3, 4):
            efw_def = f'{bvi}(1+m)(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'
        else:
            efw_def = f'{bvi}(S<sub>wi</sub>c<sub>w</sub>+c<sub>f</sub>)/(1\u2212S<sub>wi</sub>)\u00b7\u0394p'

        comps = (f'<tr><td align="right"><b>F</b> =&nbsp;</td><td>{f_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Withdrawal</td></tr>'
                 f'<tr><td align="right"><b>E<sub>o</sub></b> =&nbsp;</td><td>{eo_def}</td>'
                 '<td style="color:#27ae60;">&nbsp;\u2192 Oil &amp; Liberated Gas Expansion</td></tr>')
        if res_idx in (1, 3, 4):
            comps += (f'<tr><td align="right"><b>E<sub>g</sub></b> =&nbsp;</td><td>{eg_def}</td>'
                      '<td style="color:#27ae60;">&nbsp;\u2192 Gas Cap Expansion</td></tr>')
        if res_idx != 5 and res_idx != 6:
            comps += (f'<tr><td align="right"><b>E<sub>f,w</sub></b> =&nbsp;</td><td>{efw_def}</td>'
                      '<td style="color:#27ae60;">&nbsp;\u2192 Rock &amp; Water Expansion</td></tr>')
        if res_idx in (2, 3, 4, 6, 7):
            comps += ('<tr><td align="right"><b>W<sub>e</sub></b>&nbsp;</td><td></td>'
                      '<td style="color:#27ae60;">&nbsp;\u2192 Water Influx</td></tr>')

    # ── Notes definitions (show only in specific modes) ──
    notes = []
    if bt_mode:
        notes.append('B<sub>t</sub> = B<sub>o</sub> + (R<sub>si</sub> \u2212 R<sub>s</sub>)B<sub>g</sub> | B<sub>ti</sub> = B<sub>oi</sub>')
    if res_idx == 0:
        notes.append('c<sub>e</sub> = (c<sub>o</sub>S<sub>o</sub> + c<sub>w</sub>S<sub>wi</sub> + c<sub>f</sub>) / (1 \u2212 S<sub>wi</sub>)')
    
    note_html = ''
    if notes:
        note_text = '&nbsp;&nbsp;|&nbsp;&nbsp;'.join(notes)
        note_html = f'<div style="margin-top:8px;font-size:11px;color:#7f8c8d;"><b>Where:</b>&nbsp;&nbsp;{note_text}</div>'
    
    return f'''<div style="text-align:center;">
  <div style="font-size:15px;font-weight:bold;color:#1a5276;margin-bottom:4px;">{title}</div>
  <div style="font-size:11px;color:#8e44ad;margin-bottom:8px;">{notation_label}</div>
  <table cellspacing="0" cellpadding="0" align="center">
    <tr>
      <td rowspan="2" style="vertical-align:middle;padding-right:6px;font-size:20px;font-weight:bold;color:#2c3e50;">N&nbsp;=&nbsp;</td>
      <td style="border-bottom:2px solid #34495e;padding:5px 10px;text-align:center;font-size:13px;color:#2c3e50;">{num}</td>
    </tr>
    <tr>
      <td style="padding:5px 10px;text-align:center;font-size:13px;color:#2c3e50;">{den}</td>
    </tr>
  </table>
  <div style="margin-top:12px;padding-top:8px;border-top:1px dashed #ced4da;">
    <span style="font-size:11px;color:#7f8c8d;font-weight:bold;">Havlena-Odeh Linear Form: </span>
    <span style="font-size:14px;font-weight:bold;color:#1a5276;">{ho_eq}</span>
  </div>
  <table cellspacing="0" cellpadding="1" align="center" style="margin-top:6px;font-size:11px;color:#7f8c8d;">
    {comps}
  </table>
  {note_html}
</div>'''


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
            ("pi", "Initial reservoir pressure (psia)", "4000"),
            ("dp", "Pressure drop (\u0394p) (psi)", "200"),
            ("Np", "Cumulative oil produced (STB)", "500000"),
            ("Rp", "Cumulative gas-oil ratio (scf/STB)", "1200"),
            ("Wp", "Cumulative water produced (bbl)", "5000"),
            ("Rsi", "Initial gas solubility (scf/STB)", "850"),
            ("Rs", "Current gas solubility (scf/STB)", "800"),
            ("Boi", "Initial oil FV factor (rb/STB)", "1.40"),
            ("Bo", "Oil FV factor (rb/STB)", "1.38"),
            ("Bgi", "Initial gas FV factor (rb/scf)", "0.0011"),
            ("Bg", "Gas FV factor (rb/scf)", "0.0012"),
            ("Bw", "Water FV factor (rb/STB)", "1.02"),
        ]

    @staticmethod
    def _exp_fields():
        return [
            ("N",  "Original Oil in Place (STB)", "10000000"),
            ("m",  "Initial gas-cap ratio (m)", "0.5"),
            ("Swi", "Initial water saturation (fraction)", "0.25"),
            ("cw", "Water compressibility (1/psi)", "0.000003"),
            ("cf", "Formation compressibility (1/psi)", "0.000004"),
            ("We", "Cumulative water influx (bbl)", "100000"),
        ]

    @staticmethod
    def _inj_fields():
        return [
            ("Ginj", "Cumulative gas injected (scf)", "0"),
            ("Winj", "Cumulative water injected (bbl)", "0"),
            ("Bginj", "Injected gas FV factor (rb/scf)", "0.001"),
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

        title = QLabel("MATERIAL BALANCE EQUATION")
        title.setObjectName("header_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        sub = QLabel("Select reservoir type and target, then enter the required parameters below.")
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
            "Full MBE (with Injection)",
            "Full MBE (No Injection)",
            "Undersat. (No Liquid/Rock Expansion)",
            "Undersat. with Bottom Water",
            "Undersat. with Bottom Water & Expansion",
            "Saturated Oil (No Gas Cap/Expansion)",
            "Saturated Oil (Gas Cap, No Expansion)",
            "Saturated Oil (Bottom Water, No Expansion)",
            "Combination Drive (No Expansion)"
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
        
        self.notation_selector = ScrolllessComboBox()
        self.notation_selector.addItems([
            "Standard (Bo) Form",
            "Two-Phase (Bt) Form"
        ])
        self.notation_selector.currentIndexChanged.connect(self._on_setup_change)
        self.notation_selector.setFixedHeight(35)
        self.notation_selector.setObjectName("target_selector")
        self.notation_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        tl.addWidget(QLabel("1. Reservoir Type:"))
        tl.addWidget(self.res_type_selector)
        tl.addSpacing(10)
        tl.addWidget(QLabel("2. Target Calculation:"))
        tl.addWidget(self.target_selector)
        tl.addSpacing(10)
        tl.addWidget(QLabel("3. Formula Notation:"))
        tl.addWidget(self.notation_selector)
        lay.addWidget(tg)

        # Formula display
        self.formula_group = QGroupBox("Active Equation")
        self.formula_group.setObjectName("formula_group")
        fl = QVBoxLayout(self.formula_group)
        fl.setContentsMargins(15, 25, 15, 15)
        self.formula_label = QLabel()
        self.formula_label.setTextFormat(Qt.TextFormat.RichText)
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.setWordWrap(True)
        self.formula_label.setMinimumHeight(120)
        fl.addWidget(self.formula_label)
        lay.addWidget(self.formula_group)

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

        # Result area (Tabbed)
        self.results_tabs = QTabWidget()
        self.results_tabs.setObjectName("results_tabs")
        self.results_tabs.setMinimumHeight(400)
        
        # Sub-tab 1: Summary
        self.summary_tab = QWidget()
        sl = QVBoxLayout(self.summary_tab)
        
        self.result_frame = QFrame()
        self.result_frame.setObjectName("result_frame")
        self.result_frame.setMinimumHeight(150)
        rl = QVBoxLayout(self.result_frame)
        self.result_label = QLabel("Run calculation to see summary.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("result_text")
        rl.addWidget(self.result_label)
        sl.addWidget(self.result_frame)
        sl.addStretch()
        
        self.results_tabs.addTab(self.summary_tab, "Analysis Summary")
        
        # Sub-tab 2: Visualization
        self.viz_tab = QWidget()
        vl = QVBoxLayout(self.viz_tab)
        self.plot_canvas = MBEPlotCanvas()
        vl.addWidget(self.plot_canvas)
        self.results_tabs.addTab(self.viz_tab, "Visual Insights")
        
        # Sub-tab 3: Data Table
        self.table_tab = QWidget()
        tl = QVBoxLayout(self.table_tab)
        self.mechanism_table = QTableWidget()
        self.mechanism_table.setColumnCount(3)
        self.mechanism_table.setHorizontalHeaderLabels(["Component", "Value", "Contribution (%)"])
        self.mechanism_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tl.addWidget(self.mechanism_table)
        self.results_tabs.addTab(self.table_tab, "Mechanism Table")
        
        lay.addWidget(self.results_tabs)
        lay.addStretch()

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
        if res_idx in (0, 5, 6, 7, 8, 10):  # No gas cap types
            model.item(1).setEnabled(False)
            if res_idx in (6, 7, 10):
                model.item(2).setEnabled(True) # We calculation allowed
            else:
                model.item(2).setEnabled(False) # We calculation excluded
            
            if tgt_idx == 1 or (res_idx not in (6, 7, 10) and tgt_idx == 2):
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
        elif res_idx in (3, 4):  # Full MBE
            for row in range(4):
                model.item(row).setEnabled(True)

        # Handle Notation Selector Visibility
        # Undersaturated: 0, 5, 6, 7
        if hasattr(self, 'notation_selector'):
            if res_idx in (0, 5, 6, 7):
                self.notation_selector.setCurrentIndex(0) # Force Bo
                self.notation_selector.setEnabled(False)
            else:
                self.notation_selector.setEnabled(True)

        # Hide structurally irrelevant inputs
        if res_idx in (0, 5, 6, 7, 8, 9, 10, 11):
            if res_idx not in (1, 3, 4, 9, 11):
                if "m" in self.input_containers: self.input_containers["m"].setVisible(False)
            
            for k in ("Ginj", "Winj", "Bginj"):
                if k in self.input_containers: self.input_containers[k].setVisible(False)
            
            if res_idx in (0, 5, 8, 9):
                if "We" in self.input_containers: self.input_containers["We"].setVisible(False)
            
            if res_idx in (5, 6, 8, 9, 10, 11):
                for k in ("Swi", "cw", "cf"):
                    if k in self.input_containers: self.input_containers[k].setVisible(False)
        elif res_idx == 1:
            for k in ("We", "Winj"):
                if k in self.input_containers: self.input_containers[k].setVisible(False)
        elif res_idx == 2:
            for k in ("m", "Ginj", "Bginj"):
                if k in self.input_containers: self.input_containers[k].setVisible(False)
        elif res_idx == 4:
            for k in ("Ginj", "Winj", "Bginj"):
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

        self._update_formula()

    def _update_formula(self):
        notation = 'bt' if self.notation_selector.currentIndex() == 1 else 'bo'
        self.formula_label.setText(build_formula_html(self.res_type_selector.currentIndex(), notation=notation))

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

            r = 0
            if tgt_idx == 0:
                d = inter["den"]
                if d == 0: raise ZeroDivisionError("Denominator is zero.")
                r = inter["num"] / d
            elif tgt_idx == 1:
                if v["N"] == 0: raise ZeroDivisionError("N cannot be zero.")
                nm, dm = ((inter["F_adj"]) / v["N"]) - Eo - Efw_star, Eg + Efw_star
                if dm == 0: raise ZeroDivisionError("Denominator is zero.")
                r = nm / dm
            elif tgt_idx == 2:
                inner_s = inter["produced_oil_gas"] - (v["Wp"] * v["Bw"]) - inter["gas_inj"] - inter["water_inj"]
                r = v["N"] * inter["Et"] - inner_s
            
            # --- Professional Insight Layer ---
            N_final = r if tgt_idx == 0 else v.get("N", 1e-10)
            m_final = r if tgt_idx == 1 else v["m"]
            We_final = r if tgt_idx == 2 else v["We"]
            
            di = MBEEngine.driving_indexes(
                N_final, inter["Eo"], m_final, inter["Eg"], inter["Efw"],
                We_final, v["Wp"], v["Bw"], v["Winj"], v["Ginj"], v["Bginj"], 
                res_type_idx=res_type_idx
            )
            
            h_ddi, h_sdi, h_wdi, h_edi = di['DDI']*100, di['SDI']*100, di['WDI']*100, di['EDI']*100
            dominant = max([("Depletion", h_ddi), ("Segregation", h_sdi), ("Water", h_wdi), ("Expansion", h_edi)], key=lambda x: x[1])
            recovery = (v['Np'] / N_final * 100) if N_final > 1 else 0
            
            # --- Analysis Summary Tab ---
            sum_txt = f"<b style='font-size:18px; color:#2c3e50;'>CALCULATION COMPLETE</b><br><br>"
            if tgt_idx == 0: sum_txt += f"<b>Original Oil in Place (N):</b> {r:,.2f} STB<br>"
            elif tgt_idx == 1: sum_txt += f"<b>Gas Cap Ratio (m):</b> {r:,.4f}<br>"
            elif tgt_idx == 2: sum_txt += f"<b>Cumulative Water Influx (We):</b> {r:,.2f} bbl<br>"
            
            sum_txt += f"<br><b>EXECUTIVE DIAGNOSTIC:</b><br>"
            sum_txt += f"\u2022 Primary Drive: <span style='color:#3498db; font-weight:bold;'>{dominant[0]} Mechanism</span> ({dominant[1]:,.1f}%)<br>"
            sum_txt += f"\u2022 Recovery Factor: <b>{recovery:,.2f}%</b><br>"
            
            if dominant[0] == "Water":
                sum_txt += "<br><i>Note: Strong aquifer support detected. Watch for water breakthrough.</i>"
            elif dominant[0] == "Segregation":
                sum_txt += "<br><i>Note: Major gas cap energy. Maintain pressure above bubble point to keep gas in cap.</i>"
            elif dominant[0] == "Depletion":
                sum_txt += "<br><i>Note: Solution gas drive dominant; expect rapid pressure decline.</i>"
            self._show_ok(sum_txt)
            
            # --- Update Visuals & Table ---
            self._update_viz(N_final, m_final, We_final, res_type_idx, inter, dominant[0])
            self.mechanism_table.setRowCount(0)
            rows = [
                ("Total Withdrawal (F)", inter["F"], "bbl", "100.0%"),
                ("Oil Expansion (N*Eo)", N_final * inter["Eo"], "bbl", f"{h_ddi:.1f}%"),
                ("Gas Cap Expansion (N*mEg)", N_final * m_final * inter["Eg"], "bbl", f"{h_sdi:.1f}%"),
                ("Water Influx (We)", We_final, "bbl", f"{h_wdi:.1f}%"),
                ("Rock/Fluid Expansion (N*Efw)", N_final * inter["Efw"], "bbl", f"{h_edi:.1f}%")
            ]
            for name, val, unit, pct in rows:
                if abs(val) > 1e-6 or name.startswith("Total"):
                    ri = self.mechanism_table.rowCount()
                    self.mechanism_table.insertRow(ri)
                    item = QTableWidgetItem(name)
                    if name.startswith(dominant[0]): item.setFont(QFont("Arial", weight=QFont.Weight.Bold))
                    self.mechanism_table.setItem(ri, 0, item)
                    self.mechanism_table.setItem(ri, 1, QTableWidgetItem(f"{val:,.2f} {unit}"))
                    self.mechanism_table.setItem(ri, 2, QTableWidgetItem(pct))
            
        except (ValueError, ZeroDivisionError) as e:
            self._show_err(str(e))
        except Exception as e:
            self._show_err(f"Unexpected error: {e}")

    def _show_ok(self, txt):
        self.result_label.setText(txt)
        self.result_frame.setStyleSheet("QFrame#result_frame{background:#0a1a0f;border:1px solid #3fb950;border-radius:10px;}")
        self.result_label.setStyleSheet("color:#e6edf3;font-size:15px;font-family:'IBM Plex Mono','Courier New',monospace;line-height:1.4;background:transparent;")

    def _show_err(self, txt):
        self.result_label.setText(f"⚠  {txt}")
        self.result_frame.setStyleSheet("QFrame#result_frame{background:#1a0a0a;border:1px solid #f85149;border-radius:10px;}")
        self.result_label.setStyleSheet("color:#f85149;font-size:14px;font-weight:bold;font-family:'IBM Plex Mono','Courier New',monospace;background:transparent;")

    def _update_viz(self, N, m, We, res_type_idx, inter, dominant_name="Unknown"):
        """Generate a visual representation of drive mechanisms."""
        try:
            di = MBEEngine.driving_indexes(
                N, inter["Eo"], m, inter["Eg"], inter["Efw"],
                We, self._val("Wp"), self._val("Bw"), self._val("Winj"), 
                self._val("Ginj"), self._val("Bginj"), res_type_idx=res_type_idx
            )
            
            ax = self.plot_canvas.axes
            ax.clear()
            
            tgt_idx = self.target_selector.currentIndex()
            res_name = RES_TITLES.get(res_type_idx, "Reservoir")
            
            Et = inter.get("Et", 0)
            F_adj = inter.get("F_adj", 0)

            # Draw Havlena-Odeh X-Y line plot for Single-Point Analysis
            if tgt_idx == 1:  # Solve for m (F/Eo vs Eg/Eo plot)
                Eo = inter.get("Eo", 0)
                Eg = inter.get("Eg", 0)
                if Eo > 0:
                    x_val, y_val = Eg / Eo, F_adj / Eo
                    ax.plot([0, x_val], [N, y_val], 'o-', color='#fbbf24', linewidth=2, markersize=8, label=f'Current State')
                    
                    if x_val > 0:
                        x_line = np.linspace(0, x_val * 1.5, 100)
                        ax.plot(x_line, N + (N * m) * x_line, '--', color='#f87171', linewidth=1.5, label=f'Trend (Slope Nm = {N*m:,.0f})')
                    
                    ax.set_xlabel('Eg / Eo', fontsize=11, fontweight='bold', color='#8b949e')
                    ax.set_ylabel('F / Eo (STB)', fontsize=11, fontweight='bold', color='#8b949e')
                    ax.set_title(f"Havlena-Odeh Analysis (F/Eo vs Eg/Eo)\n{res_name}", fontsize=13, pad=15, fontweight='bold', color='#fbbf24')
                else:
                    ax.text(0.5, 0.5, "Insufficient expansion data to visualize", ha='center', va='center', fontsize=12, color='#7f8c8d')
            else:
                # Target N, We, or Drive Indexes: Use standard F vs Et plot
                ax.plot([0, Et], [0, F_adj], 'o-', color='#38bdf8', linewidth=2, markersize=8, label=f'Current State')
                
                if Et > 0:
                    x_line = np.linspace(0, Et * 1.5, 100)
                    ax.plot(x_line, N * x_line, '--', color='#f87171', linewidth=1.5, label=f'Trend (Slope N = {N:,.0f} STB)')
                
                ax.set_xlabel('Expansion Term, Et (bbl/STB)', fontsize=11, fontweight='bold', color='#8b949e')
                ax.set_ylabel('Total Withdrawal, F (rb)', fontsize=11, fontweight='bold', color='#8b949e')
                ax.set_title(f"Havlena-Odeh Analysis (F vs Et)\n{res_name}", fontsize=13, pad=15, fontweight='bold', color='#fbbf24')

            # Formatting
            ax.set_facecolor('#0f1923')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#374151')
            ax.spines['bottom'].set_color('#374151')
            ax.tick_params(axis='x', colors='#8b949e')
            ax.tick_params(axis='y', colors='#8b949e')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
            ax.grid(True, alpha=0.15, color='#374151')
            ax.legend(fontsize=10, loc='best', facecolor='#161b22', edgecolor='#374151',
                      labelcolor='#e6edf3')
            self.plot_canvas.fig.set_facecolor('#0d1117')

            self.plot_canvas.fig.tight_layout()
            self.plot_canvas.draw()
            
            # --- Update Table Tab ---
            self.mechanism_table.setRowCount(0)
            rows = [
                ("Total Withdrawal (F)", inter["F"], "100.0%"),
                ("Oil Expansion (N*Eo)", N * inter["Eo"], f"{di['DDI']*100:.1f}%"),
                ("Gas Cap Expansion (N*mEg)", N * m * inter["Eg"], f"{di['SDI']*100:.1f}%"),
                ("Water Influx (We)", We, f"{di['WDI']*100:.1f}%"),
                ("Rock/Fluid Expansion (N*Efw)", N * inter["Efw"], f"{di['EDI']*100:.1f}%")
            ]
            for name, val, pct in rows:
                if abs(val) > 1e-6 or name.startswith("Total"):
                    r_idx = self.mechanism_table.rowCount()
                    self.mechanism_table.insertRow(r_idx)
                    self.mechanism_table.setItem(r_idx, 0, QTableWidgetItem(name))
                    self.mechanism_table.setItem(r_idx, 1, QTableWidgetItem(f"{val:,.2f}"))
                    self.mechanism_table.setItem(r_idx, 2, QTableWidgetItem(pct))

            # Jump to visualization tab on success
            self.results_tabs.setCurrentIndex(1)
            
        except:
            pass

    def _reset(self):
        for le in self.inputs.values():
            le.clear()
        self.result_label.setText("Run a calculation to see results here.")
        self.result_label.setStyleSheet("color:#8b949e;font-size:14px;font-weight:normal;font-family:'IBM Plex Mono','Courier New',monospace;background:transparent;")
        self.result_frame.setStyleSheet("QFrame#result_frame{background:#0f1923;border:1px solid #374151;border-radius:10px;}")
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
            "Full MBE (with Injection)",
            "Full MBE (No Injection)",
            "Undersat. (No Liquid/Rock Expansion)",
            "Undersat. with Bottom Water",
            "Undersat. with Bottom Water & Expansion",
            "Saturated Oil (No Gas Cap/Expansion)",
            "Saturated Oil (Gas Cap, No Expansion)",
            "Saturated Oil (Bottom Water, No Expansion)",
            "Combination Drive (No Expansion)"
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

        tb.addWidget(QLabel("  Notation:"))
        self.notation_cb = ScrolllessComboBox()
        self.notation_cb.addItems(["Bo Form", "Bt Form"])
        self.notation_cb.setFixedHeight(32)
        self.notation_cb.setMinimumWidth(100)
        self.notation_cb.setObjectName("target_selector")
        self.notation_cb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.notation_cb.currentIndexChanged.connect(self._on_setup_change)
        tb.addWidget(self.notation_cb)

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

        # Formula display (compact)
        self.formula_label = QLabel()
        self.formula_label.setTextFormat(Qt.TextFormat.RichText)
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.setObjectName("formula_bar")
        self.formula_label.setFixedHeight(30)
        root.addWidget(self.formula_label)

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

        # Handle Notation Selector Visibility
        if res_idx in (0, 5, 6, 7):
            if hasattr(self, 'notation_cb'):
                self.notation_cb.setCurrentIndex(0)
                self.notation_cb.setEnabled(False)
        else:
            if hasattr(self, 'notation_cb'):
                self.notation_cb.setEnabled(True)

        # Enforce target validity
        model = self.target_cb.model()
        if res_idx in (0, 5, 6, 7, 8, 10):  # No gas cap types
            model.item(1).setEnabled(False)
            if res_idx in (6, 7, 10):
                model.item(2).setEnabled(True)
            else:
                model.item(2).setEnabled(False)
            
            if tgt_idx == 1 or (res_idx not in (6, 7, 10) and tgt_idx == 2):
                self.target_cb.setCurrentIndex(0)
                tgt_idx = 0
        elif res_idx in (9, 11): # Gas cap / Combination no expansion
             model.item(1).setEnabled(True)
             if res_idx == 11:
                 model.item(2).setEnabled(True)
             else:
                 model.item(2).setEnabled(False)
                 
             if (res_idx == 9 and tgt_idx == 2):
                 self.target_cb.setCurrentIndex(0)
                 tgt_idx = 0
        elif res_idx == 1:  # Gas Cap with expansion
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
        elif res_idx in (3, 4):  # Full
            for row in range(4):
                model.item(row).setEnabled(True)

        for k in ("N", "m", "Bginj"):
            if k in self.const_containers:
                self.const_containers[k].setVisible(True)

        if res_idx in (0, 5, 6, 7, 8, 9, 10, 11):
            if res_idx not in (1, 3, 4, 9, 11):
                if "m" in self.const_containers: self.const_containers["m"].setVisible(False)
            if "Bginj" in self.const_containers: self.const_containers["Bginj"].setVisible(False)
            if res_idx in (5, 6, 8, 9, 10, 11):
                for k in ("Swi", "cw", "cf"):
                    if k in self.const_containers: self.const_containers[k].setVisible(False)
        elif res_idx == 2:
            if "m" in self.const_containers: self.const_containers["m"].setVisible(False)
            if "Bginj" in self.const_containers: self.const_containers["Bginj"].setVisible(False)
        elif res_idx == 4:
            if "Bginj" in self.const_containers: self.const_containers["Bginj"].setVisible(False)

        if tgt_idx == 0 and "N" in self.const_containers:
            self.const_containers["N"].setVisible(False)
        elif tgt_idx == 1 and "m" in self.const_containers:
            self.const_containers["m"].setVisible(False)

        self._update_formula()

    def _update_formula(self):
        notation = 'bt' if self.notation_cb.currentIndex() == 1 else 'bo'
        self.formula_label.setText(build_formula_html(self.res_type_cb.currentIndex(), compact=True, notation=notation))

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
        ax.set_facecolor('#0f1923')
        self.ho_canvas.fig.set_facecolor('#0d1117')

        if idx == 0:
            Et = [r["Et"] for r in self.results]
            F_adj = [r["F_adj"] for r in self.results]
            ax.scatter(Et, F_adj, color='#38bdf8', s=60, zorder=5, edgecolors='#0d1117', linewidths=1.2, label='Data Points')
            if hasattr(self, 'ho_slope') and not math.isnan(self.ho_slope):
                x_line = np.linspace(0, max(Et) * 1.1, 100)
                ax.plot(x_line, self.ho_slope * x_line, '--', color='#f87171', linewidth=2,
                        label=f'N = {self.ho_slope:,.0f} STB  (R² = {self.ho_r2:.4f})')
            ax.set_xlabel('Et  (Eo + m·Eg + Efw)', fontsize=11, color='#8b949e')
            ax.set_ylabel('F adjusted  (rb)', fontsize=11, color='#8b949e')
            ax.set_title('Havlena-Odeh: F vs Et  →  Slope = N', fontsize=13, fontweight='bold', color='#fbbf24')
        elif idx == 1:
            valid = [(r["Eo"], r["Eg"], r["F_adj"]) for r in self.results if abs(r["Eo"]) > 1e-15]
            if valid:
                x = [v[1] / v[0] for v in valid]
                y = [v[2] / v[0] for v in valid]
                ax.scatter(x, y, color='#c084fc', s=60, zorder=5, edgecolors='#0d1117', linewidths=1.2, label='Data Points')
                if len(valid) >= 2:
                    coeffs = np.polyfit(x, y, 1)
                    x_line = np.linspace(min(x), max(x) * 1.1, 100)
                    ax.plot(x_line, np.polyval(coeffs, x_line), '--', color='#f87171', linewidth=2,
                            label=f'Intercept(N)={coeffs[1]:,.0f}, Slope(Nm)={coeffs[0]:,.0f}')
                ax.set_xlabel('Eg / Eo', fontsize=11, color='#8b949e')
                ax.set_ylabel('F / Eo', fontsize=11, color='#8b949e')
                ax.set_title('Havlena-Odeh: F/Eo vs Eg/Eo', fontsize=13, fontweight='bold', color='#fbbf24')
        elif idx == 2:
            ts = [r["Timestep"] for r in self.results]
            we = [r.get("We_calc", 0) for r in self.results]
            ax.bar(ts, we, color='#38bdf8', edgecolor='#0d1117', linewidth=0.8, label='We', alpha=0.85)
            ax.set_xlabel('Timestep', fontsize=11, color='#8b949e')
            ax.set_ylabel('We (bbl)', fontsize=11, color='#8b949e')
            ax.set_title('Calculated Water Influx Over Time', fontsize=13, fontweight='bold', color='#fbbf24')
        else:
            ts = [r["Timestep"] for r in self.results]
            n_vals = [r.get("N_calc", self._get_const("N")) for r in self.results]
            ax.plot(ts, n_vals, 'o-', color='#fbbf24', linewidth=2, markersize=6, label='F/Et')
            ax.set_xlabel('Timestep', fontsize=11, color='#8b949e')
            ax.set_ylabel('F / Et  (STB)', fontsize=11, color='#8b949e')
            ax.set_title('Campbell Plot — N Consistency Check', fontsize=13, fontweight='bold', color='#fbbf24')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#374151')
        ax.spines['bottom'].set_color('#374151')
        ax.tick_params(axis='x', colors='#8b949e')
        ax.tick_params(axis='y', colors='#8b949e')
        ax.legend(fontsize=9, loc='best', facecolor='#161b22', edgecolor='#374151', labelcolor='#e6edf3')
        ax.grid(True, alpha=0.12, color='#374151')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        self.ho_canvas.fig.tight_layout()
        self.ho_canvas.draw()

    # ──────── Driving Index Plot ────────
    def _draw_di_plot(self):
        ax = self.di_canvas.axes
        ax.clear()
        ax.set_facecolor('#0f1923')
        self.di_canvas.fig.set_facecolor('#0d1117')

        valid = [r for r in self.results if abs(r.get("Sum", 0)) > 1e-10]
        if not valid:
            ax.text(0.5, 0.5, 'No valid driving index data', ha='center', va='center',
                    fontsize=14, color='#8b949e')
            ax.set_facecolor('#0f1923')
            self.di_canvas.draw()
            return

        ts   = [r["Timestep"] for r in valid]
        DDI  = np.array([r["DDI"] * 100 for r in valid])
        SDI  = np.array([r["SDI"] * 100 for r in valid])
        WDI  = np.array([r["WDI"] * 100 for r in valid])
        EDI  = np.array([r["EDI"] * 100 for r in valid])

        colors = {'DDI': '#f87171', 'SDI': '#fbbf24', 'WDI': '#38bdf8', 'EDI': '#34d399'}
        ax.fill_between(ts, 0, DDI, color=colors['DDI'], alpha=0.9, label='Depletion (DDI)')
        ax.fill_between(ts, DDI, DDI + SDI, color=colors['SDI'], alpha=0.9, label='Gas Cap (SDI)')
        ax.fill_between(ts, DDI + SDI, DDI + SDI + WDI, color=colors['WDI'], alpha=0.9, label='Water (WDI)')
        ax.fill_between(ts, DDI + SDI + WDI, DDI + SDI + WDI + EDI, color=colors['EDI'], alpha=0.9, label='Expansion (EDI)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#374151')
        ax.spines['bottom'].set_color('#374151')
        ax.tick_params(axis='x', colors='#8b949e')
        ax.tick_params(axis='y', colors='#8b949e')
        ax.set_xlabel('Timestep', fontsize=11, color='#8b949e')
        ax.set_ylabel('Drive Contribution (%)', fontsize=11, color='#8b949e')
        ax.set_title('Primary Driving Indexes Over Time', fontsize=13, fontweight='bold', color='#fbbf24')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, loc='upper right', facecolor='#161b22',
                  edgecolor='#374151', labelcolor='#e6edf3')
        ax.grid(True, alpha=0.12, axis='y', color='#374151')
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
        self.setWindowTitle("⛽  MBE Calculator  ·  Petroleum Reservoir Engineering")
        self.setMinimumSize(920, 660)
        self.resize(1140, 800)

        root = QWidget()
        root.setObjectName("root_widget")
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # ── Brand header bar ──────────────────────────────────────────
        header_bar = QWidget()
        header_bar.setObjectName("header_bar")
        header_bar.setFixedHeight(52)
        hb_lay = QHBoxLayout(header_bar)
        hb_lay.setContentsMargins(24, 0, 24, 0)

        logo_lbl = QLabel("⛽  MBE CALCULATOR")
        logo_lbl.setObjectName("app_logo")
        hb_lay.addWidget(logo_lbl)
        hb_lay.addStretch()

        badge = QLabel("Petroleum Reservoir Engineering  ·  Havlena–Odeh Method")
        badge.setObjectName("header_badge")
        hb_lay.addWidget(badge)

        rl.addWidget(header_bar)

        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)
        self.main_tabs.addTab(SinglePointTab(), "  Single-Point Calculator  ")
        self.main_tabs.addTab(MultiStepTab(),   "  Multi-Timestep Analysis  ")
        rl.addWidget(self.main_tabs)

        self._apply_style()

    def _apply_style(self):
        # ── Palette ──────────────────────────────────────────────────
        # BG_DEEP   #0d1117   page background
        # BG_PANEL  #161b22   card/panel surface
        # BG_RAISED #1f2937   elevated groupbox
        # BORDER    #30363d   subtle dividers
        # AMBER     #d97706   primary accent (crude-oil amber)
        # AMBER_LT  #fbbf24   hover / glow
        # TEXT_PRI  #e6edf3   primary text
        # TEXT_SEC  #8b949e   muted / labels
        # GREEN     #3fb950   success
        # RED       #f85149   error
        self.setStyleSheet("""
            /* ── Global ─────────────────────────────────────────── */
            QMainWindow, QWidget#root_widget, QWidget, QScrollArea {
                background-color: #0d1117;
                color: #e6edf3;
            }
            QScrollArea { border: none; }
            QWidget#main_content_widget { background-color: #0d1117; }

            /* ── Header bar ──────────────────────────────────────── */
            QWidget#header_bar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d1117, stop:0.5 #161b22, stop:1 #0d1117);
                border-bottom: 1px solid #30363d;
            }
            QLabel#app_logo {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 16px; font-weight: bold;
                color: #d97706;
                letter-spacing: 3px;
            }
            QLabel#header_badge {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 11px; color: #8b949e;
                letter-spacing: 1px;
            }

            /* ── Labels ──────────────────────────────────────────── */
            QLabel {
                font-family: 'DM Sans', 'Segoe UI', 'Helvetica Neue', sans-serif;
                color: #e6edf3; font-size: 13px;
                background: transparent;
            }
            QLabel#header_title {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 22px; font-weight: bold;
                color: #fbbf24;
                letter-spacing: 1px;
            }
            QLabel#subtitle { font-size: 12px; color: #8b949e; margin-bottom: 4px; }

            /* ── GroupBox ────────────────────────────────────────── */
            QGroupBox {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 11px; font-weight: bold;
                color: #d97706;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 14px;
                background-color: #161b22;
                letter-spacing: 1px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 2px 10px; left: 14px;
                background: #0d1117;
                border-radius: 4px;
            }

            /* ── Formula equation display ────────────────────────── */
            QGroupBox#formula_group {
                background-color: #0f1923;
                border: 1px solid #d97706;
                border-radius: 8px;
            }
            QLabel#formula_bar {
                background: #0f1923;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 4px 14px;
                color: #fbbf24;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
            }

            /* ── Inputs ──────────────────────────────────────────── */
            QLineEdit {
                padding: 0px 10px;
                border: 1px solid #30363d;
                border-radius: 6px;
                background: #0d1117;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 13px;
                color: #fbbf24;
                selection-background-color: #d97706;
                min-height: 30px;
            }
            QLineEdit:focus {
                border: 1px solid #d97706;
                background: #0f1923;
            }
            QLineEdit:hover { border: 1px solid #4b5563; }

            /* ── Result frame ────────────────────────────────────── */
            QFrame#result_frame {
                background: #0f1923;
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 20px;
                margin-top: 8px;
            }
            QLabel#result_text {
                font-size: 15px;
                color: #8b949e;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
            }

            /* ── Buttons ─────────────────────────────────────────── */
            QPushButton {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
                padding: 10px 20px;
                border-radius: 6px;
            }
            QPushButton#calc_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #d97706, stop:1 #b45309);
                color: #0d1117;
                border: none;
            }
            QPushButton#calc_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fbbf24, stop:1 #d97706);
            }
            QPushButton#calc_btn:pressed { background: #92400e; }

            QPushButton#reset_btn, QPushButton#import_btn, QPushButton#template_btn {
                background: #161b22;
                color: #8b949e;
                border: 1px solid #30363d;
            }
            QPushButton#reset_btn:hover, QPushButton#import_btn:hover, QPushButton#template_btn:hover {
                background: #21262d;
                color: #e6edf3;
                border: 1px solid #8b949e;
            }

            /* ── ComboBox ────────────────────────────────────────── */
            QComboBox {
                background: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding-left: 10px;
                min-height: 30px;
                font-family: 'DM Sans', 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QComboBox:hover { border: 1px solid #8b949e; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow { width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
                selection-background-color: #d97706;
                selection-color: #0d1117;
                outline: none;
            }
            QComboBox#target_selector {
                background: #1f2937;
                color: #fbbf24;
                border: 1px solid #d97706;
                font-weight: bold;
            }
            QComboBox#res_type_selector {
                background: #1f2937;
                color: #e6edf3;
                border: 1px solid #374151;
            }

            /* ── Tab bar ─────────────────────────────────────────── */
            QTabWidget::pane {
                border: 1px solid #30363d;
                border-radius: 0px;
                background: #0d1117;
                top: -1px;
            }
            QTabBar::tab {
                padding: 10px 22px;
                margin-right: 2px;
                background: #0d1117;
                color: #8b949e;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                border: 1px solid transparent;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #161b22;
                color: #fbbf24;
                border: 1px solid #30363d;
                border-bottom: 2px solid #d97706;
            }
            QTabBar::tab:hover:!selected {
                background: #161b22;
                color: #e6edf3;
            }

            /* ── Table ───────────────────────────────────────────── */
            QTableWidget {
                gridline-color: #21262d;
                font-size: 12px;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                background: #0d1117;
                color: #e6edf3;
                alternate-background-color: #0f1923;
                selection-background-color: #d97706;
                selection-color: #0d1117;
                border: none;
            }
            QTableWidget QTableWidgetItem { color: #e6edf3; padding: 4px; }
            QHeaderView::section {
                background: #161b22;
                color: #d97706;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 1px;
                padding: 8px 6px;
                border: none;
                border-right: 1px solid #30363d;
                border-bottom: 1px solid #d97706;
            }
            QTableCornerButton::section { background: #161b22; border: none; }

            /* ── Analysis text box ───────────────────────────────── */
            QTextEdit#analysis_box {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 12px;
                background: #0a0f14;
                color: #fbbf24;
                border: none;
                padding: 16px;
                selection-background-color: #374151;
            }

            /* ── Scrollbars ──────────────────────────────────────── */
            QScrollBar:vertical {
                background: #0d1117; width: 8px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #374151; border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #d97706; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #0d1117; height: 8px; margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #374151; border-radius: 4px; min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover { background: #d97706; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

            /* ── Splitter ────────────────────────────────────────── */
            QSplitter::handle { background: #21262d; width: 2px; height: 2px; }
            QSplitter::handle:hover { background: #d97706; }

            /* ── Status label ────────────────────────────────────── */
            QLabel#subtitle {
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                font-size: 11px; color: #6b7280;
                letter-spacing: 0.5px;
                padding: 4px 8px;
            }
        """)


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette so native dialogs (file chooser, message box) stay dark
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor("#0d1117"))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor("#e6edf3"))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#161b22"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#0f1923"))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#161b22"))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor("#e6edf3"))
    palette.setColor(QPalette.ColorRole.Text,            QColor("#e6edf3"))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#21262d"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#e6edf3"))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor("#fbbf24"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#d97706"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d1117"))
    palette.setColor(QPalette.ColorRole.Link,            QColor("#fbbf24"))
    app.setPalette(palette)

    window = MBECalculatorWindow()
    window.show()
    sys.exit(app.exec())
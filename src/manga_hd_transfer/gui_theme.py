from __future__ import annotations

import re

"""Shared Qt palette and stylesheet for Folirina.

This module has no widget/state dependencies, so visual tokens can evolve without
turning ``gui_qt.py`` back into a style + workflow God Object.
"""

ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
BG = "#F5F7FB"
CARD = "#FFFFFF"
CARD_BLUE = "#F8FBFF"
TEXT = "#1F2937"
MUTED = "#64748B"
MUTED_2 = "#94A3B8"
BORDER = "#E5EAF1"
BORDER_STRONG = "#D7DEE8"
BLUE_SOFT = "#EEF4FF"
GREEN = "#2E8B6D"
GREEN_SOFT = "#ECF8F3"
ORANGE = "#B97828"
ORANGE_SOFT = "#FFF6E8"
RED = "#C94D5D"
RED_SOFT = "#FFF0F2"

STYLE = f"""
QMainWindow {{ background:{BG}; }}
QWidget {{ color:{TEXT}; font-size:13px; }}
QWidget#root {{ background:{BG}; }}
QWidget#contentShell {{ background:{BG}; }}
QFrame#workflowRail {{
    background:#F7F9FD;
    border-right:1px solid {BORDER};
}}
QLabel#railProductName {{
    color:{TEXT};
    font-size:19px;
    font-weight:850;
    padding:7px 8px 5px 8px;
}}
QLabel#railSection {{
    color:{MUTED_2};
    font-size:9px;
    font-weight:750;
    letter-spacing:.8px;
    padding:2px 10px 4px 10px;
}}
QFrame#card {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:12px;
}}
QFrame#cardBlue {{
    background:{CARD_BLUE};
    border:1px solid #DCE6F3;
    border-radius:11px;
}}
QFrame#typographyPanel {{
    background:#FBFCFE;
    border:1px solid #DEE6F0;
    border-radius:11px;
}}
QLabel#typographyTitle {{ font-size:12px; font-weight:750; color:{TEXT}; }}
QLabel#runtimeSectionTitle {{ font-size:12px; font-weight:750; color:{TEXT}; }}
QLabel#railBrand {{
    color:{ACCENT_HOVER};
    background:#E7EEF8;
    border:1px solid #D4E0F0;
    border-radius:10px;
    font-size:13px;
    font-weight:850;
    padding:4px 0;
}}
QLabel#railCaption {{
    color:{MUTED_2};
    font-size:10px;
    font-weight:700;
}}
QLabel#railVersion {{
    color:{MUTED};
    font-size:10px;
    padding:4px 0;
}}
QLabel#appTitle {{ font-size:17px; font-weight:780; letter-spacing:.1px; }}
QLabel#appSubtitle {{ color:{MUTED}; font-size:11px; }}
QLabel#workflowStep {{ color:{MUTED_2}; font-size:9px; font-weight:700; letter-spacing:.3px; }}
QLabel#pageTitle {{ font-size:17px; font-weight:750; }}
QLabel#sectionTitle {{ font-size:14px; font-weight:750; }}
QLabel#cardTitle {{ font-size:14px; font-weight:750; }}
QLabel#cardSubtitle {{ color:{MUTED}; font-size:11px; }}
QLabel#hint {{ color:{MUTED}; font-size:12px; }}
QLabel#quiet {{ color:{MUTED_2}; font-size:11px; }}
QLabel#badge {{
    color:{ACCENT_HOVER};
    background:{BLUE_SOFT};
    border:1px solid #D9E5F5;
    border-radius:10px;
    padding:3px 8px;
    font-weight:700;
    font-size:10px;
}}
QFrame#recognitionHero {{
    background:#F4F8FF;
    border:1px solid #D8E4F6;
    border-radius:13px;
}}
QLabel#heroIcon {{
    color:white; background:{ACCENT}; border-radius:16px;
    font-size:27px; font-weight:850;
}}
QLabel#heroTitle {{ font-size:18px; font-weight:800; color:#172033; }}
QLabel#heroStat {{
    background:#FFFFFF; border:1px solid #E4EAF3; border-radius:10px;
    padding:7px 8px; font-size:11px; font-weight:650; color:#334155;
}}
QLabel#successHint {{
    color:#21835F; background:#EDF9F3; border:1px solid #CBEBDD;
    border-radius:10px; padding:7px 9px; font-size:11px;
}}
QLabel#successBadge {{
    color:#21835F; background:#EDF9F3; border:1px solid #CBEBDD;
    border-radius:9px; padding:3px 8px; font-size:10px; font-weight:700;
}}
QFrame#heroPreviewFrame {{
    background:#FFFFFF; border:1px solid #DDE5F0; border-radius:14px;
}}
QLabel#heroPreview {{ background:#FCFDFE; border:1px solid #EDF1F6; border-radius:10px; color:{MUTED_2}; }}
QPushButton {{
    min-height:32px;
    border-radius:8px;
    border:1px solid {BORDER_STRONG};
    background:#FFFFFF;
    padding:0 12px;
    font-weight:600;
}}
QPushButton:hover {{ background:#F2F6FB; border-color:#C1CFDF; }}
QPushButton:pressed {{ background:#EAF0F7; }}
QPushButton:focus {{ border-color:#AFC4E4; }}
QPushButton:disabled {{ color:{MUTED_2}; background:#F7F9FC; border-color:{BORDER}; }}
QPushButton#primary {{
    color:white;
    background:{ACCENT};
    border:1px solid {ACCENT};
    font-weight:700;
    min-height:36px;
}}
QPushButton#primary:hover {{ background:{ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background:#1D4ED8; border-color:#1D4ED8; }}
QPushButton#softPrimary {{
    color:{ACCENT_HOVER};
    background:{BLUE_SOFT};
    border:1px solid #CBDBF6;
    font-weight:650;
}}
QPushButton#softPrimary:hover {{ background:#E7EEF8; border-color:#D4E0F0; }}
QPushButton#softPrimary:pressed {{ background:#DCE6F3; }}
QPushButton#pageProcessAction {{
    color:{ACCENT_HOVER};
    background:{BLUE_SOFT};
    border:1px solid #CBDBF6;
    font-weight:700;
    min-height:38px; max-height:38px;
    padding:0 12px;
}}
QPushButton#pageProcessAction:hover {{ background:#E7EEF8; border-color:#D4E0F0; }}
QPushButton#pageProcessAction:pressed {{ background:#DCE6F3; border-color:#C5D5EA; }}
QPushButton#pageProcessAction:disabled {{
    color:#8A9AB0; background:#EEF4FC; border-color:#D9E4F2;
}}
QPushButton#danger {{ color:{RED}; border-color:#E7AEB7; background:{RED_SOFT}; font-weight:650; }}
QPushButton#danger:hover {{ background:#FFE4E8; border-color:#DA8C99; }}
QPushButton#danger:pressed {{ background:#E7AEB7; }}
QPushButton#stopTask {{
    color:white;
    background:{RED};
    border:1px solid {RED};
    min-width:82px;
    min-height:31px;
    font-weight:700;
}}
QPushButton#stopTask:hover {{ background:#B94352; border-color:#B94352; }}
QPushButton#stopTask:disabled {{ color:#A7AFBA; background:#EEF1F5; border-color:#E0E5EB; }}
QPushButton#navButton {{
    border:1px solid transparent;
    border-radius:10px;
    background:transparent;
    color:{MUTED};
    min-width:138px;
    min-height:42px;
    max-height:42px;
    padding:0 12px;
    text-align:left;
    font-weight:650;
}}
QPushButton#navButton:hover {{
    background:#F2F6FB;
    border-color:#E3EAF3;
    color:{TEXT};
}}
QPushButton#navButton:pressed {{
    background:#EAF0F7;
}}
QPushButton#navButton:checked {{
    background:{BLUE_SOFT};
    color:{ACCENT_HOVER};
    border:1px solid #D9E5F5;
    font-weight:750;
}}
QPushButton#navButton:checked:hover {{
    background:#E7EEF8;
    border-color:#D4E0F0;
}}
QPushButton#railTool {{
    min-height:30px;
    padding:0 8px;
    border-radius:8px;
    background:#F8FAFD;
    border:1px solid #E3EAF3;
    color:{MUTED};
    font-size:11px;
    font-weight:650;
}}
QPushButton#railTool:hover {{ background:#F2F6FB; color:{TEXT}; border-color:#D7E2F0; }}
QPushButton#railTool:pressed {{ background:#EAF0F7; }}
QLabel#railPlatform {{
    color:{MUTED}; background:#F2F6FB; border:1px solid #E3EAF3;
    border-radius:8px; padding:5px 6px; font-size:9px; font-weight:700;
}}
QPushButton#collapseToggle {{
    min-width:58px; min-height:28px; padding:0 9px;
    background:transparent; color:{MUTED}; border:1px solid {BORDER}; border-radius:7px;
}}
QPushButton#collapseToggle:hover {{ background:#F2F6FB; color:{TEXT}; }}
QFrame#selectionPanel {{
    background:{CARD_BLUE}; border:1px solid #DCE6F3; border-radius:10px;
}}
QComboBox#routeSelector {{ font-weight:650; min-height:34px; }}
QComboBox QAbstractItemView::item {{ min-height:26px; padding:4px 8px; }}
QPushButton#segmented {{
    border:1px solid transparent;
    border-radius:7px;
    background:transparent;
    color:{MUTED};
    min-height:28px;
    padding:0 9px;
    font-weight:600;
}}
QPushButton#segmented:hover {{ background:#F2F6FB; border-color:#EDF1F6; color:{TEXT}; }}
QPushButton#segmented:checked {{ background:{BLUE_SOFT}; color:{ACCENT_HOVER}; border:1px solid #D9E5F5; font-weight:700; }}
QPushButton#pageNav {{ min-width:78px; min-height:30px; padding:0 10px; }}
QPushButton#pageRailCollapse {{
    min-width:25px; max-width:25px; min-height:25px; max-height:25px;
    padding:0; border-radius:7px; color:{MUTED}; background:#FFFFFF;
    border:1px solid {BORDER}; font-size:17px; font-weight:750;
}}
QPushButton#pageRailCollapse:hover {{ background:{BLUE_SOFT}; color:{ACCENT_HOVER}; border-color:#CBDBF6; }}
QLabel#activityBadge {{
    color:{GREEN}; background:{GREEN_SOFT}; border:1px solid #CBEBDD;
    border-radius:9px; padding:3px 8px; font-size:10px; font-weight:700;
}}
QLabel#activityBadge[busy="true"] {{
    color:{ACCENT_HOVER}; background:{BLUE_SOFT}; border-color:#CBDBF6;
}}
QFrame#workbenchPageRail {{
    background:#F8FAFD; border:1px solid {BORDER}; border-radius:11px;
}}
QLabel#pageRailTitle {{ font-size:12px; font-weight:750; }}
QLabel#pageRailCount {{
    color:{MUTED}; background:#FFFFFF; border:1px solid {BORDER};
    border-radius:8px; padding:2px 7px; font-size:9px; font-weight:700;
}}
QLineEdit#pageRailSearch {{ min-height:28px; font-size:10px; padding:0 7px; }}
QListWidget#workbenchPageList {{ background:transparent; border:0; outline:0; }}
QListWidget#workbenchPageList::item {{
    background:transparent; border:1px solid transparent; border-radius:8px;
    padding:4px 5px; margin:1px; font-size:10px;
}}
QListWidget#workbenchPageList::item:hover {{ background:#F2F6FB; border-color:#E3EAF3; }}
QListWidget#workbenchPageList::item:selected {{
    background:{BLUE_SOFT}; color:{ACCENT_HOVER}; border-color:#CBDBF6; font-weight:700;
}}
QLabel#pageRailFooter {{ color:{MUTED_2}; font-size:9px; padding:2px 0; }}
QFrame#inspectorShell {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:11px;
}}
QFrame#inspectorTabBar {{ background:#F8FAFD; border-bottom:1px solid {BORDER}; }}
QPushButton#inspectorTab {{
    min-height:28px; border:1px solid transparent; border-radius:7px;
    background:transparent; color:{MUTED}; padding:0 9px; font-size:10px; font-weight:650;
}}
QPushButton#inspectorTab:hover {{ background:#F2F6FB; color:{TEXT}; }}
QPushButton#inspectorTab:checked {{
    background:#FFFFFF; color:{ACCENT_HOVER}; border-color:#D9E5F5; font-weight:750;
}}
QLabel#stageChip {{
    color:{TEXT}; background:#F8FAFD; border:1px solid {BORDER}; border-radius:9px;
    padding:7px 10px; font-size:10px; font-weight:650;
}}
QPushButton#topAction {{
    min-height:31px;
    background:#F8FAFD;
    border-color:#E3EAF3;
    color:{MUTED};
    font-weight:650;
}}
QPushButton#topAction:hover {{ background:#F2F6FB; color:{TEXT}; border-color:#D7E2F0; }}
QPushButton#topAction:pressed {{ background:#EAF0F7; }}
QPushButton#compactAction {{ min-height:30px; padding:0 10px; }}
QLabel#pageCounter {{
    color:{TEXT};
    background:{BLUE_SOFT};
    border:1px solid #D7E3F4;
    border-radius:8px;
    padding:4px 10px;
    font-weight:650;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height:32px;
    border:1px solid {BORDER};
    border-radius:8px;
    background:white;
    padding:0 8px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color:#ABC1E2; }}
QLineEdit#lockedField {{ background:#F5F7FA; color:{MUTED}; border-color:#DDE3EB; }}
QComboBox::drop-down {{ border:0; width:24px; }}
QTableWidget {{
    background:white;
    border:1px solid {BORDER};
    border-radius:10px;
    gridline-color:#EEF2F6;
    selection-background-color:{BLUE_SOFT};
    selection-color:{TEXT};
}}
QTableWidget::item {{ padding:4px 6px; }}
QHeaderView::section {{
    background:#F7F9FC;
    color:{MUTED};
    border:0;
    border-bottom:1px solid {BORDER};
    padding:7px 8px;
    font-weight:600;
}}

QListWidget#pageGallery {{
    background:transparent;
    border:0;
    outline:0;
}}
QListWidget#pageGallery::item {{
    background:#FFFFFF;
    border:1px solid {BORDER};
    border-radius:10px;
    padding:6px;
    margin:2px;
}}
QListWidget#pageGallery::item:hover {{
    background:#F8FAFD;
    border-color:#D7E2F0;
}}
QListWidget#pageGallery::item:selected {{
    background:{BLUE_SOFT};
    border-color:#CBDBF6;
    color:{TEXT};
}}
QPlainTextEdit {{
    background:#FCFDFE;
    border:1px solid {BORDER};
    border-radius:9px;
    padding:7px;
    selection-background-color:{BLUE_SOFT};
    selection-color:{TEXT};
}}
QPushButton#modelDownload {{ min-height:28px; padding:0 8px; font-size:11px; }}
QScrollArea {{ border:0; background:transparent; }}
QCheckBox, QRadioButton {{ spacing:7px; min-height:27px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width:16px; height:16px; }}
QProgressBar {{
    border:1px solid {BORDER};
    border-radius:6px;
    background:#EDF2F7;
    text-align:center;
    min-height:20px;
    color:{MUTED};
}}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:5px; }}
QGraphicsView {{ background:#F2F5F9; border:0; border-radius:9px; }}
QGraphicsView#workbenchImage {{
    background:#E9EEF5;
    border:1px solid #DCE4EE;
    border-radius:12px;
}}
QSplitter::handle {{ background:{BG}; }}
QSplitter::handle:horizontal {{ width:5px; }}
QSplitter::handle:vertical {{ height:5px; }}
QStatusBar {{ background:{CARD}; border-top:1px solid {BORDER}; color:{MUTED}; }}
QToolTip {{ background:#FFFFFF; color:{TEXT}; border:1px solid {BORDER}; padding:5px; }}
QScrollBar:vertical {{ background:transparent; width:8px; margin:2px; }}
QScrollBar::handle:vertical {{ background:#CBD5E1; min-height:28px; border-radius:4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ background:transparent; height:8px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:#CBD5E1; min-width:28px; border-radius:4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QFrame#optionRow {{
    background:#FFFFFF;
    border:1px solid #E6EDF6;
    border-radius:10px;
}}
QFrame#optionRow[selected="true"] {{
    background:#F7FBFF;
    border:1px solid #C9DCF4;
}}
QLabel#optionHint {{ color:{MUTED_2}; font-size:11px; padding-left:23px; }}
QFrame#pageHero {{
    background:qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F8FBFF, stop:1 #F3F7FC);
    border:1px solid #DCE7F4;
    border-radius:13px;
}}
QLabel#heroTitle {{ font-size:18px; font-weight:780; }}
QLabel#heroHint {{ color:{MUTED}; font-size:12px; }}
QLabel#infoChip {{
    color:{ACCENT_HOVER};
    background:#FFFFFF;
    border:1px solid #DCE7F4;
    border-radius:11px;
    padding:5px 9px;
    font-size:11px;
    font-weight:650;
}}
"""



LIGHT_STYLE = STYLE

# Dark theme is derived from the locked light stylesheet rather than maintaining a
# second independent widget rule set.  This keeps spacing, sizing and control
# behavior byte-for-byte aligned while only replacing visual tokens.
_DARK_COLOR_REPLACEMENTS = {
    "#2563EB": "#3B82F6",
    "#1D4ED8": "#60A5FA",
    "#F5F7FB": "#0B1220",
    "#FFFFFF": "#111827",
    "#F8FBFF": "#111C2D",
    "#1F2937": "#E5E7EB",
    "#64748B": "#A7B4C8",
    "#94A3B8": "#7F8CA3",
    "#E5EAF1": "#263244",
    "#D7DEE8": "#334155",
    "#EEF4FF": "#16233B",
    "#2E8B6D": "#6EE7B7",
    "#ECF8F3": "#10271F",
    "#B97828": "#FBBF24",
    "#FFF6E8": "#302715",
    "#C94D5D": "#FB7185",
    "#FFF0F2": "#351922",
    "#F7F9FD": "#0D1524",
    "#DCE6F3": "#25344A",
    "#E7EEF8": "#17243A",
    "#D4E0F0": "#293B55",
    "#D9E5F5": "#2E4566",
    "#F4F8FF": "#101C2F",
    "#D8E4F6": "#263C5A",
    "#172033": "#F1F5F9",
    "#E4EAF3": "#2B374A",
    "#334155": "#CBD5E1",
    "#21835F": "#6EE7B7",
    "#EDF9F3": "#10271F",
    "#CBEBDD": "#255A48",
    "#DDE5F0": "#29384D",
    "#FCFDFE": "#0F172A",
    "#EDF1F6": "#233044",
    "#F2F6FB": "#182235",
    "#C1CFDF": "#3B4A60",
    "#EAF0F7": "#1D2A3D",
    "#AFC4E4": "#45658D",
    "#CBDBF6": "#31527F",
    "#E7AEB7": "#75404A",
    "#FFE4E8": "#43212A",
    "#DA8C99": "#8F4B59",
    "#B94352": "#E05B6F",
    "#A7AFBA": "#7B8798",
    "#EEF1F5": "#1A2433",
    "#E0E5EB": "#2B3646",
    "#F8FAFD": "#131D2C",
    "#E3EAF3": "#2A374A",
    "#D7E2F0": "#30425B",
    "#D7E3F4": "#304563",
    "#ABC1E2": "#4D6F9F",
    "#EEF2F6": "#202C3B",
    "#F7F9FC": "#111B2A",
    "#EDF2F7": "#172131",
    "#F2F5F9": "#0D1521",
    "#E9EEF5": "#111A28",
    "#DCE4EE": "#253246",
    "#CBD5E1": "#45556D",
    "#E6EDF6": "#263449",
    "#F7FBFF": "#121E30",
    "#C9DCF4": "#34567F",
    "#F3F7FC": "#0E1827",
    "#DCE7F4": "#2A3E59",
}


def _make_dark_style(light_style: str) -> str:
    dark = str(light_style)
    # Replace longer visual tokens first and only after the light stylesheet has
    # been fully interpolated.  CSS behavior/layout declarations are untouched.
    pattern = re.compile("|".join(re.escape(key) for key in sorted(_DARK_COLOR_REPLACEMENTS, key=len, reverse=True)))
    dark = pattern.sub(lambda match: _DARK_COLOR_REPLACEMENTS[match.group(0)], dark)
    dark = dark.replace("background:white;", "background:#111827;")
    dark += """
QDialog, QMessageBox { background:#0B1220; color:#E5E7EB; }
QPlainTextEdit, QTextEdit, QListWidget, QTreeWidget {
    background:#111827; color:#E5E7EB; border:1px solid #263244;
    selection-background-color:#16233B; selection-color:#E5E7EB;
}
QMenu { background:#111827; color:#E5E7EB; border:1px solid #334155; }
QMenu::item:selected { background:#16233B; }
QLineEdit#lockedField { background:#111827; color:#94A3B8; border-color:#334155; }
QComboBox QAbstractItemView {
    background:#111827; color:#E5E7EB; border:1px solid #334155;
    selection-background-color:#16233B; selection-color:#E5E7EB;
}
QPushButton:disabled, QLineEdit:disabled, QComboBox:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {
    color:#64748B; background:#111827; border-color:#263244;
}
"""
    return dark


DARK_STYLE = _make_dark_style(LIGHT_STYLE)
SUPPORTED_THEMES = ("light", "dark")


def normalize_theme(value: object) -> str:
    name = str(value or "light").strip().lower()
    return name if name in SUPPORTED_THEMES else "light"


_PX_VALUE_RE = re.compile(r"(?<![A-Za-z0-9_])(?P<value>(?:\d+(?:\.\d+)?)|(?:\.\d+))px")


def scale_stylesheet_pixels(stylesheet: str, scale: float) -> str:
    """Scale visual pixel metrics while keeping theme/color semantics intact.

    The default factor returns the original string object so existing callers and
    regression tests retain the exact v2.0.64 stylesheet.  Borders remain at
    least one device-independent pixel while fonts/padding/radii may shrink.
    """
    factor = max(0.20, min(1.0, float(scale)))
    if factor >= 0.9995:
        return stylesheet

    def repl(match: re.Match[str]) -> str:
        value = float(match.group("value"))
        scaled = value * factor
        if value >= 0.9 and value <= 2.0:
            scaled = max(1.0, scaled)
        elif value > 2.0:
            scaled = max(1.0, round(scaled))
        else:
            scaled = max(0.05, round(scaled, 2))
        if abs(scaled - round(scaled)) < 1e-9:
            text = str(int(round(scaled)))
        else:
            text = (f"{scaled:.2f}").rstrip("0").rstrip(".")
        return text + "px"

    return _PX_VALUE_RE.sub(repl, stylesheet)


def style_for_theme(theme: object, scale: float = 1.0) -> str:
    base = DARK_STYLE if normalize_theme(theme) == "dark" else LIGHT_STYLE
    return scale_stylesheet_pixels(base, scale)


def semantic_palette(theme: object) -> dict[str, str]:
    """Colors for the few legacy widgets that still use inline styles."""
    if normalize_theme(theme) == "dark":
        return {
            "accent": "#60A5FA", "green": "#6EE7B7", "green_soft": "#10271F",
            "green_border": "#255A48", "orange": "#FBBF24", "orange_soft": "#302715",
            "orange_border": "#6B5320", "red": "#FB7185", "red_soft": "#351922",
            "red_border": "#75404A", "muted": "#A7B4C8", "muted_2": "#7F8CA3",
            "border": "#334155", "thumb_bg": "#151F2E",
        }
    return {
        "accent": ACCENT_HOVER, "green": GREEN, "green_soft": GREEN_SOFT,
        "green_border": "#BFE3D5", "orange": ORANGE, "orange_soft": ORANGE_SOFT,
        "orange_border": "#F0D4A5", "red": RED, "red_soft": RED_SOFT,
        "red_border": "#E7AEB7", "muted": MUTED, "muted_2": MUTED_2,
        "border": BORDER_STRONG, "thumb_bg": "#F1F4F8",
    }


__all__ = [
    'ACCENT', 'ACCENT_HOVER', 'BG', 'CARD', 'CARD_BLUE', 'TEXT', 'MUTED', 'MUTED_2',
    'BORDER', 'BORDER_STRONG', 'BLUE_SOFT', 'GREEN', 'GREEN_SOFT', 'ORANGE',
    'ORANGE_SOFT', 'RED', 'RED_SOFT', 'STYLE', 'LIGHT_STYLE', 'DARK_STYLE',
    'SUPPORTED_THEMES', 'normalize_theme', 'scale_stylesheet_pixels', 'style_for_theme', 'semantic_palette',
]

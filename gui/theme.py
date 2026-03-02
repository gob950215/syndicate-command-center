"""
SYNDICATE — Dark Trading Terminal Theme
=========================================
Professional dark mode theme inspired by Bloomberg Terminal / TradingView.
All colors, fonts, and style constants centralized here.
"""

# ── Color Palette ──────────────────────────────────────────────────────────
# Primary backgrounds (darkest to lightest)
BG_VOID       = "#0A0A0F"      # Deepest background
BG_PRIMARY    = "#0D1117"      # Main panels
BG_SECONDARY  = "#161B22"      # Cards, elevated surfaces
BG_TERTIARY   = "#1C2333"      # Input fields, wells
BG_HOVER      = "#21262D"      # Hover state
BG_SELECTED   = "#1A2744"      # Selected items

# Borders
BORDER_SUBTLE = "#21262D"
BORDER_MEDIUM = "#30363D"
BORDER_ACCENT = "#E8590C33"    # Orange glow (20% opacity)

# Text
TEXT_PRIMARY   = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
TEXT_MUTED     = "#6E7681"
TEXT_INVERSE   = "#0D1117"

# Accent colors
ORANGE_PRIMARY  = "#E8590C"    # Primary action
ORANGE_LIGHT    = "#FF6B2B"    # Hover
ORANGE_DARK     = "#C44500"    # Pressed
ORANGE_GLOW     = "#E8590C22"  # Subtle glow

GREEN_PROFIT    = "#3FB950"    # Positive / Win
GREEN_DARK      = "#238636"
GREEN_BG        = "#0D3117"

RED_LOSS        = "#F85149"    # Negative / Loss
RED_DARK        = "#DA3633"
RED_BG          = "#3D1117"

BLUE_INFO       = "#58A6FF"    # Information
BLUE_DARK       = "#1F6FEB"
BLUE_BG         = "#0C2D6B"

YELLOW_WARN     = "#D29922"    # Warning
GOLD_DIAMOND    = "#FFD700"    # Diamond tier
PURPLE_RLM      = "#BC8CFF"    # RLM signals

# Risk levels
RISK_LOW    = GREEN_PROFIT
RISK_MEDIUM = YELLOW_WARN
RISK_HIGH   = RED_LOSS

# Tier colors
TIER_DIAMOND      = GOLD_DIAMOND
TIER_RLM_DIAMOND  = PURPLE_RLM
TIER_TOP3         = BLUE_INFO
TIER_STANDARD     = TEXT_SECONDARY

# ── Fonts ──────────────────────────────────────────────────────────────────
FONT_FAMILY_DISPLAY  = "JetBrains Mono"    # Monospace for data/numbers
FONT_FAMILY_UI       = "Segoe UI"           # System UI font
FONT_FAMILY_FALLBACK = "Consolas"

FONT_SIZE_TITLE  = 20
FONT_SIZE_H1     = 16
FONT_SIZE_H2     = 14
FONT_SIZE_BODY   = 12
FONT_SIZE_SMALL  = 10
FONT_SIZE_MONO   = 12

# ── Dimensions ─────────────────────────────────────────────────────────────
SIDEBAR_WIDTH    = 220
PANEL_PADDING    = 16
CARD_RADIUS      = 8
BUTTON_RADIUS    = 6
INPUT_HEIGHT     = 36
GAUGE_SIZE       = 140

# ── Complete Stylesheet ────────────────────────────────────────────────────

STYLESHEET = f"""
/* ── Global ───────────────────────────────────────────────── */
QMainWindow {{
    background-color: {BG_VOID};
}}

QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: "{FONT_FAMILY_UI}", "{FONT_FAMILY_FALLBACK}", sans-serif;
    font-size: {FONT_SIZE_BODY}px;
}}

/* ── Scrollbar ────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_PRIMARY};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_MEDIUM};
    min-height: 30px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {BG_PRIMARY};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_MEDIUM};
    min-width: 30px;
    border-radius: 4px;
}}

/* ── Labels ───────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    padding: 0px;
}}
QLabel[class="title"] {{
    font-size: {FONT_SIZE_TITLE}px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    font-family: "{FONT_FAMILY_DISPLAY}", monospace;
}}
QLabel[class="subtitle"] {{
    font-size: {FONT_SIZE_H2}px;
    color: {TEXT_SECONDARY};
}}
QLabel[class="mono"] {{
    font-family: "{FONT_FAMILY_DISPLAY}", monospace;
    font-size: {FONT_SIZE_MONO}px;
}}
QLabel[class="stat-value"] {{
    font-family: "{FONT_FAMILY_DISPLAY}", monospace;
    font-size: 22px;
    font-weight: bold;
    color: {ORANGE_PRIMARY};
}}
QLabel[class="diamond"] {{
    font-size: 14px;
    font-weight: bold;
    color: {GOLD_DIAMOND};
}}

/* ── Buttons ──────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {BUTTON_RADIUS}px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 32px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {TEXT_MUTED};
}}
QPushButton:pressed {{
    background-color: {BG_SELECTED};
}}
QPushButton[class="primary"] {{
    background-color: {ORANGE_PRIMARY};
    color: {TEXT_INVERSE};
    border: none;
    font-weight: bold;
}}
QPushButton[class="primary"]:hover {{
    background-color: {ORANGE_LIGHT};
}}
QPushButton[class="primary"]:pressed {{
    background-color: {ORANGE_DARK};
}}
QPushButton[class="danger"] {{
    background-color: {RED_BG};
    color: {RED_LOSS};
    border: 1px solid {RED_DARK};
}}
QPushButton[class="danger"]:hover {{
    background-color: {RED_DARK};
    color: white;
}}
QPushButton[class="success"] {{
    background-color: {GREEN_BG};
    color: {GREEN_PROFIT};
    border: 1px solid {GREEN_DARK};
}}

/* ── Inputs ───────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {BUTTON_RADIUS}px;
    padding: 8px 12px;
    font-family: "{FONT_FAMILY_DISPLAY}", monospace;
    selection-background-color: {ORANGE_PRIMARY};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {ORANGE_PRIMARY};
}}

/* ── ComboBox ─────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {BUTTON_RADIUS}px;
    padding: 6px 12px;
    min-height: 28px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    selection-background-color: {BG_SELECTED};
}}

/* ── SpinBox ──────────────────────────────────────────────── */
QSpinBox {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {BUTTON_RADIUS}px;
    padding: 4px 8px;
}}

/* ── Tables ───────────────────────────────────────────────── */
QTableWidget {{
    background-color: {BG_PRIMARY};
    alternate-background-color: {BG_SECONDARY};
    gridline-color: {BORDER_SUBTLE};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {CARD_RADIUS}px;
    font-family: "{FONT_FAMILY_DISPLAY}", monospace;
    font-size: {FONT_SIZE_SMALL}px;
}}
QTableWidget::item {{
    padding: 6px 8px;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
QTableWidget::item:selected {{
    background-color: {BG_SELECTED};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    font-weight: bold;
    font-size: {FONT_SIZE_SMALL}px;
    padding: 8px;
    border: none;
    border-bottom: 2px solid {ORANGE_PRIMARY};
}}

/* ── Tab Widget ───────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER_MEDIUM};
    background-color: {BG_PRIMARY};
    border-radius: {CARD_RADIUS}px;
}}
QTabBar::tab {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    padding: 8px 20px;
    border: 1px solid {BORDER_MEDIUM};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {BG_PRIMARY};
    color: {ORANGE_PRIMARY};
    border-bottom: 2px solid {ORANGE_PRIMARY};
}}
QTabBar::tab:hover {{
    color: {TEXT_PRIMARY};
}}

/* ── Frames / Cards ───────────────────────────────────────── */
QFrame[class="card"] {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {CARD_RADIUS}px;
}}
QFrame[class="card-diamond"] {{
    background-color: {BG_SECONDARY};
    border: 1px solid {GOLD_DIAMOND}44;
    border-radius: {CARD_RADIUS}px;
}}
QFrame[class="card-hover"]:hover {{
    border-color: {ORANGE_PRIMARY};
}}
QFrame[class="sidebar"] {{
    background-color: {BG_PRIMARY};
    border-right: 1px solid {BORDER_MEDIUM};
}}

/* ── GroupBox ──────────────────────────────────────────────── */
QGroupBox {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: {CARD_RADIUS}px;
    margin-top: 16px;
    padding-top: 20px;
    font-weight: bold;
}}
QGroupBox::title {{
    color: {ORANGE_PRIMARY};
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}}

/* ── Checkbox / Radio ─────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {BORDER_MEDIUM};
    border-radius: 3px;
    background-color: {BG_TERTIARY};
}}
QCheckBox::indicator:checked {{
    background-color: {ORANGE_PRIMARY};
    border-color: {ORANGE_PRIMARY};
}}

/* ── ProgressBar ──────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_TERTIARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-family: "{FONT_FAMILY_DISPLAY}", monospace;
    height: 22px;
}}
QProgressBar::chunk {{
    background-color: {ORANGE_PRIMARY};
    border-radius: 3px;
}}

/* ── Tooltips ─────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    padding: 6px 10px;
    font-size: {FONT_SIZE_SMALL}px;
}}
"""

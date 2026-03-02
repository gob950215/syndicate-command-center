# 🏀 SYNDICATE COMMAND CENTER v1.0

## Elite Sports Prediction Management Platform

### Architecture Overview

```
syndicate_app/
├── main.py                     # Entry point — launches the GUI
├── requirements.txt            # Dependencies
├── config.py                   # Global config & constants
│
├── core/                       # Business logic layer
│   ├── __init__.py
│   ├── base_sport.py           # Abstract base class (Factory pattern)
│   ├── nba_engine.py           # NBA adapter wrapping nba_syndicate_v8.py
│   ├── pick_manager.py         # Pick lifecycle: generate → validate → track
│   └── data_models.py          # Pydantic-style dataclasses
│
├── security/                   # Cybersecurity module
│   ├── __init__.py
│   ├── auth.py                 # Login, Fernet-encrypted user DB
│   ├── session.py              # Session management + IP/PC logging
│   └── totp_prep.py            # 2FA preparation (TOTP / Google Authenticator)
│
├── scheduler/                  # Automation module
│   ├── __init__.py
│   ├── scheduler_engine.py     # APScheduler-based smart scheduling
│   └── rules.py                # Rule definitions and calendar integration
│
├── sports/                     # Multi-sport factory (scalable)
│   ├── __init__.py
│   ├── sport_factory.py        # Factory pattern for sport modules
│   ├── nba/                    # NBA-specific
│   └── mlb/                    # Future: MLB stub
│
├── gui/                        # PyQt6 interface layer
│   ├── __init__.py
│   ├── theme.py                # Dark trading-terminal theme
│   ├── main_window.py          # Central window with navigation
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── dashboard.py        # Home dashboard with live picks
│   │   ├── war_room.py         # Detailed pick analysis panel
│   │   ├── scheduler_panel.py  # Scheduler / automation config
│   │   ├── settings_panel.py   # API keys & configuration
│   │   └── admin_panel.py      # Master admin: logs, users, access
│   └── widgets/
│       ├── __init__.py
│       ├── gauge.py            # Confidence gauge widget
│       ├── four_factors_chart.py  # Bar chart for Four Factors
│       ├── pick_card.py        # Clickable pick card
│       └── sidebar.py          # Navigation sidebar
│
└── assets/                     # Icons, fonts (optional)
```

### Installation

```bash
pip install PyQt6 cryptography pyotp apscheduler requests numpy pandas scikit-learn xgboost
```

### First Run

```bash
python main.py
```

Default admin credentials: `admin` / `Syndicate2026!`

### Environment Variables

```bash
export SPORTS_API_KEY=your_api_basketball_key
export ODDS_API_KEY=your_the_odds_api_key
```

### Adding a New Sport (e.g., MLB)

1. Create `sports/mlb/mlb_engine.py` inheriting from `core.base_sport.BaseSportEngine`
2. Implement the 5 required methods
3. Register in `sports/sport_factory.py`
4. The GUI will auto-detect and add a new tab

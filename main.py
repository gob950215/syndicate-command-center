#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ███████╗██╗   ██╗███╗   ██╗██████╗ ██╗ ██████╗          ║
║     ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██║██╔════╝          ║
║     ███████╗ ╚████╔╝ ██╔██╗ ██║██║  ██║██║██║               ║
║     ╚════██║  ╚██╔╝  ██║╚██╗██║██║  ██║██║██║               ║
║     ███████║   ██║   ██║ ╚████║██████╔╝██║╚██████╗          ║
║     ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝          ║
║                                                              ║
║              COMMAND CENTER v1.0                             ║
║     Elite Sports Prediction Management Platform              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Entry point for the Syndicate Command Center desktop application.

Usage:
    python main.py              # Launch GUI (normal mode)
    python main.py --headless   # CLI mode (run analysis, output to CSV)

Default admin credentials:
    Username: admin
    Password: Syndicate2026!
"""
import sys
import os
import argparse
import logging

# Ensure project root is in path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "data", "syndicate.log"), "a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Syndicate")


def run_gui():
    """Launch the PyQt6 GUI application."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon

    # High DPI support — MUST be called before QApplication instantiation
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Syndicate Command Center")
    app.setOrganizationName("Syndicate")

    # Set global application icon (appears in all window title bars)
    icon_path = os.path.join(PROJECT_DIR, "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Keep a reference so garbage collector doesn't destroy the window
    from gui.main_window import MainWindow
    window = MainWindow()

    # Show login → then show main window
    if window.show_login():
        window.showMaximized()
        logger.info("Application started")
        return_code = app.exec()
        sys.exit(return_code)
    else:
        logger.info("Login cancelled — exiting")
        sys.exit(0)


def run_headless():
    """Run analysis in headless/CLI mode (no GUI)."""
    from config import DB_PATH, MODEL_DIR
    from core.nba_engine import NBAEngine

    print("\n" + "=" * 60)
    print("  SYNDICATE COMMAND CENTER — Headless Mode")
    print("=" * 60 + "\n")

    engine = NBAEngine()
    if not engine.initialize(str(DB_PATH), str(MODEL_DIR)):
        print(f"  ERROR: {engine.last_error}")
        sys.exit(1)

    print("  Generating picks...\n")
    picks = engine.generate_picks()

    if not picks:
        print("  No picks generated today.")
        return

    # Display results
    diamonds = [p for p in picks if p.is_diamond]
    print(f"  Total picks: {len(picks)}")
    print(f"  Diamonds:    {len(diamonds)}")
    print()

    for p in sorted(picks, key=lambda x: -x.confidence):
        tier = "💎" if p.is_diamond else "  "
        print(
            f"  {tier}  {p.matchup:12s}  →  {p.pick:3s}  "
            f"Conf:{p.confidence:.1%}  EV:{p.ev:+.3f}  "
            f"RLM:{p.rlm:+d}  Risk:{p.risk_level.value}"
        )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Syndicate Command Center")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run in CLI mode without GUI"
    )
    args = parser.parse_args()

    # Ensure data directory exists
    os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)

    if args.headless:
        run_headless()
    else:
        run_gui()


if __name__ == "__main__":
    main()

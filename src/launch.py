import logging
import sys

from src.ui.main_ui import MainUI
from app_manager import get_app

logger = logging.getLogger()
logger.setLevel(0)

def launch_app():
    app = get_app()

    ui = MainUI()
    app.main_ui = ui

    ui.show()

    sys.exit(app.run())

if __name__ == '__main__':
    launch_app()


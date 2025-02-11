import logging
import os, sys

from ui.main_ui import MainUI
from main_app import uc_agriculture_app

logger = logging.getLogger()
logger.setLevel(0)

def launch_app():
    app = uc_agriculture_app()

    ui = MainUI()
    app.main_ui = ui

    ui.show()

    sys.exit(app.run())

if __name__ == '__main__':
    launch_app()


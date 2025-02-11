import sys
import traceback

from PySide6.QtWidgets import QApplication
from flask import Flask, render_template, request, url_for, redirect, get_flashed_messages, flash, jsonify, \
    send_from_directory, g, current_app
# from flask_classful import FlaskView
from threading import Thread

import logging
import asyncio
import plotly.express as px
import pandas as pd

# from helpers.dash_app.main_dash_app import dash_data_app
from tools.units import units, unit_system
from ui.widgets.drone_presets.drone_preset_window import DronePresetWindow
logger = logging.getLogger()
logger.setLevel(0)



'''
Do all of your connecting logic in this file
You can have other functions/helpers in different places, but everything should lead back here
for ease of access...
'''

# def get_app():
#     """ Returns the current QApplication instance """
#     return QApplication.instance()

class uc_agriculture_app(QApplication):
    def __init__(self, *args):
        QApplication.__init__(self, *args)

        self.setApplicationName("UC Agriculture App")
        self.setApplicationVersion("0.0.1")

        self.worker = None

        self.main_ui = None
        self.db = None

        self.flask_app = None
        self.flask_thread = None
        self.flask_server_status = "INACTIVE"

        self.dash_app = None

        self.units = units(unit_system.METRIC.value)
    def run(self):
        """ Start the primary Qt event loop for the interface """
        res = self.exec()
        return res

    def create_app_instance(self):
        # Initialize app instances

        self.flask_app = Flask(__name__)
        self.flask_app.secret_key = b'secret key'

        self.flask_app.app_context().current_app = self.flask_app

        self.flask_app.app_context().push()
        # TestView.register(self.flask_app, route_base='/')

        self.flask_thread = Thread(target=self.flask_app.run, kwargs={'host': '0.0.0.0'}, daemon=True)
        self.flask_thread.start()

        self.create_dash_urls()
        # Disable UI elements to prevent multiple processes being opened, change some values for visibility

        self.flask_server_status = "RUNNING"
        self.main_ui.create_instance_btn.setEnabled(False)
        self.main_ui.ip_label.setHidden(False)
        self.main_ui.status_label.setText(f"Web server is currently: {self.flask_server_status}")

        # Intialize thread that allows us to keep the UI moving

        # self.worker = LongProcessWorker(self.flask_app.run)
        # self.worker.signals.finished.connect(self.progress_report)
        # self.threadpool.start(self.worker)

    def create_drone_preset_window(self):
        self.drone_preset_window = DronePresetWindow()
        self.drone_preset_window.show()

    def create_dash_urls(self):
        self.dash_app = dash_data_app("/test_dash_app/")
        self.dash_app.main_dash_app_init()

#
# class TestView(FlaskView):
#     def __init__(self, *args):
#         self.app = get_app()
#     def index(self):
#
#         return "yo"

# TODO: Look into whether or not we need the below functions...

# class WorkerSignals(QObject):
#     '''
#     Defines the signals available from a running worker thread.
#
#     Supported signals are:
#
#     finished
#         No data
#
#     error
#         tuple (exctype, value, traceback.format_exc() )
#
#     result
#         object data returned from processing, anything
#
#     progress
#         int indicating % progress
#
#     '''
#     finished = Signal()
#     error = Signal(tuple)
#     result = Signal(object)
#     progress = Signal(int)
#
# class LongProcessWorker(QObject):
#     '''
#     Worker thread
#
#     Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
#
#     :param callback: The function callback to run on this worker thread. Supplied args and
#                      kwargs will be passed through to the runner.
#     :type callback: function
#     :param args: Arguments to pass to the callback function
#     :param kwargs: Keywords to pass to the callback function
#
#     '''
#
#     def __init__(self, fn, *args, **kwargs):
#         super().__init__()
#
#         # Store constructor arguments (re-used for processing)
#         self.fn = fn
#         self.args = args
#         self.kwargs = kwargs
#         self.signals = WorkerSignals()
#
#         # Add the callback to our kwargs
#         # self.kwargs['progress_callback'-] = self.signals.progress
#
#     @Slot()
#     def run(self):
#         '''
#         Initialise the runner function with passed args, kwargs.
#         '''
#
#         # Retrieve args/kwargs here; and fire processing using them
#         try:
#             result = self.fn(*self.args, **self.kwargs)
#         except:
#             traceback.print_exc()
#             exctype, value = sys.exc_info()[:2]
#             self.signals.error.emit((exctype, value, traceback.format_exc()))
#         else:
#             self.signals.result.emit(result)  # Return the result of the processing
#         finally:
#             self.signals.finished.emit()  # Done

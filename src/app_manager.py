from PySide6.QtWidgets import QApplication
app_instance = None

def get_app(app_instance=None):

    if QApplication.instance() is not None:
        app_instance = QApplication.instance()
    elif app_instance is None:
        from main_app import uc_agriculture_app
        app_instance = uc_agriculture_app([])

    return app_instance
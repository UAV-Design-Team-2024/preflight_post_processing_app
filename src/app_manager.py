from PySide6.QtWidgets import QApplication
app_instance = None

def get_app(app_instance=None):

    '''
    Generates an instance of the app for devs to pass around. Using this logic, you can relay commands back to your
    UI and vice versa, all while keeping logic and UI separated.

    '''
    if QApplication.instance() is not None:
        app_instance = QApplication.instance()
    elif app_instance is None:
        from main_app import uc_agriculture_app
        app_instance = uc_agriculture_app([])

    return app_instance
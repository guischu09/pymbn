import sys

from PyQt5.QtWidgets import QApplication, QDialog

from gui import InputDialog


def main():
    app = QApplication(sys.argv)

    dialog = InputDialog()
    if dialog.exec_() == QDialog.Accepted:
        name = dialog.name
        number = dialog.number
        print("Name:", name)
        print("Number:", number)

    app.quit()


if __name__ == "__main__":
    main()

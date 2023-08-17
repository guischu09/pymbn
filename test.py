import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Close Example")
        self.setGeometry(100, 100, 300, 200)

        self.close_button = QPushButton("Close", self)
        self.close_button.setGeometry(100, 100, 100, 50)
        self.close_button.clicked.connect(self.close_app)

    def close_app(self):
        self.close()


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
    print("Application closed")


if __name__ == "__main__":
    main()

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class InputDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Settings")

        self.title1_label = QLabel("MS-scheme parameters:")
        self.title1_label.setObjectName("titleLabel")
        self.title1_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")

        self.method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.setObjectName("methodCombo")
        self.method_combo.addItem("ms-scheme")
        self.method_combo.addItem("conventional")

        self.weights_label = QLabel("Weights:")
        self.weights_combo = QComboBox()
        self.weights_combo.setObjectName("weightsCombo")
        self.weights_combo.addItem("Pearson correlation")
        self.weights_combo.addItem("Spearman correlation")

        self.sampling_label = QLabel("Sampling method:")
        self.sampling_combo = QComboBox()
        self.sampling_combo.setObjectName("samplingCombo")
        self.sampling_combo.addItem("bootstrap")
        self.sampling_combo.addItem("subsampling")

        self.correction_label = QLabel("Correction type:")
        self.correction_combo = QComboBox()
        self.correction_combo.setObjectName("correctionCombo")
        self.correction_combo.addItem("FDR (B&H-1995)")
        self.correction_combo.addItem("Bonferroni")

        self.representation_label = QLabel("Group representation:")
        self.representation_combo = QComboBox()
        self.representation_combo.setObjectName("representationCombo")
        self.representation_combo.addItem("mean")
        self.representation_combo.addItem("mode")
        self.representation_combo.addItem("geodesic")

        self.data_balance_label = QLabel("Data balance method:")
        self.data_balance_combo = QComboBox()
        self.data_balance_combo.setObjectName("data_balanceCombo")
        self.data_balance_combo.addItem("imbalanced")
        self.data_balance_combo.addItem("undersampled")
        self.data_balance_combo.addItem("ADASYN")

        self.brain_type_label = QLabel("Brain type:")
        self.brain_type_combo = QComboBox()
        self.brain_type_combo.setObjectName("brain_typeCombo")
        self.brain_type_combo.addItem("human")
        self.brain_type_combo.addItem("rat/mouse")

        self.out_format_label = QLabel("Output format:")
        self.out_format_combo = QComboBox()
        self.out_format_combo.setObjectName("out_formatCombo")
        self.out_format_combo.addItem("png")
        self.out_format_combo.addItem("jpg")
        self.out_format_combo.addItem("svg")
        self.out_format_combo.addItem("pdf")

        self.name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        self.name_input.setObjectName("nameInput")
        self.name_input.setText("Will")  # Set default value

        self.alpha_label = QLabel("alpha:")
        self.alpha_input = QLineEdit()
        self.alpha_input.setObjectName("alphaInput")
        self.alpha_input.setText("0.05")  # Set default value
        self.alpha_slider = QSlider()
        self.alpha_slider.setOrientation(1)  # Vertical slider
        self.alpha_slider.setRange(0, 100)  # Range from 0 to 1, scaled by 100
        self.alpha_slider.setValue(30)  # Default value scaled by 100

        self.theta_label = QLabel("theta:")
        self.theta_input = QLineEdit()
        self.theta_input.setObjectName("thetaInput")
        self.theta_input.setText("0.95")  # Set default value
        self.theta_slider = QSlider()
        self.theta_slider.setOrientation(1)  # Vertical slider
        self.theta_slider.setRange(0, 100)  # Range from 0 to 1, scaled by 100
        self.theta_slider.setValue(30)  # Default value scaled by 100

        self.threshold_label = QLabel("threshold:")
        self.threshold_input = QLineEdit()
        self.threshold_input.setObjectName("thresholdInput")
        self.threshold_input.setText("0.3")  # Set default value
        self.threshold_slider = QSlider()
        self.threshold_slider.setOrientation(1)  # Vertical slider
        self.threshold_slider.setRange(0, 100)  # Range from 0 to 1, scaled by 100
        self.threshold_slider.setValue(30)  # Default value scaled by 100

        self.nsamples_label = QLabel("Number of samples:")
        self.nsamples_input = QLineEdit()
        self.nsamples_input.setObjectName("nsamplesInput")
        self.nsamples_input.setText("2000")  # Set default value

        self.error_label = QLabel("")
        self.error_label.setObjectName("errorLabel")
        self.error_label.setStyleSheet("color: red;")

        self.n_measures_label = QLabel("Number of measures:")
        self.n_measures = QCheckBox()
        self.n_measures.setObjectName("n_measures")
        self.n_measures.setChecked(True)  # Set initial state to checked

        self.n_measures_input = QLineEdit()
        self.n_measures_input.setObjectName("n_measuresInput")
        self.n_measures_input.setEnabled(True)  # Enable input by default
        self.n_measures_input.setText("30")  # Set default value

        self.n_measures.stateChanged.connect(self.update_input_state)

        self.plotting_options_label = QLabel("Plotting options:")
        self.plotting_options_label.setObjectName("plottingOptionsLabel")
        self.plotting_options_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 20px;")

        self.brain_3d_label = QLabel("Plot 3D brain")
        self.brain_3d = QCheckBox()
        self.brain_3d.setObjectName("brain_3d")
        self.brain_3d.setChecked(False)  # Set initial state to checked

        self.interactive_3d_label = QLabel("Interactive 3D brain")
        self.interactive_3d = QCheckBox()
        self.interactive_3d.setObjectName("interactive_3d")
        self.interactive_3d.setChecked(False)  # Set initial state to checked
        self.interactive_3d.setEnabled(False)

        self.brain_3d.stateChanged.connect(self.update_interactive_3d_state)

        self.space_label = QLabel("")
        self.space_label.setObjectName("plottingOptionsLabel")
        self.space_label.setStyleSheet("font-size: 10px")

        self.heatmap_label = QLabel("Plot heatmap")
        self.heatmap = QCheckBox()
        self.heatmap.setObjectName("heatmap")
        self.heatmap.setChecked(True)  # Set initial state to checked
        self.heatmap.setEnabled(False)

        self.circle_label = QLabel("Plot circle")
        self.circle = QCheckBox()
        self.circle.setObjectName("circle")
        self.circle.setChecked(True)  # Set initial state to checked
        self.circle.setEnabled(True)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setObjectName("submitButton")
        self.submit_button.clicked.connect(self.submit_clicked)

        layout = QVBoxLayout()

        layout.addWidget(self.title1_label)

        method_layout = QHBoxLayout()
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)

        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weights_label)
        weights_layout.addWidget(self.weights_combo)

        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(self.sampling_label)
        sampling_layout.addWidget(self.sampling_combo)

        correction_layout = QHBoxLayout()
        correction_layout.addWidget(self.correction_label)
        correction_layout.addWidget(self.correction_combo)

        representation_layout = QHBoxLayout()
        representation_layout.addWidget(self.representation_label)
        representation_layout.addWidget(self.representation_combo)

        data_balance_layout = QHBoxLayout()
        data_balance_layout.addWidget(self.data_balance_label)
        data_balance_layout.addWidget(self.data_balance_combo)

        brain_type_layout = QHBoxLayout()
        brain_type_layout.addWidget(self.brain_type_label)
        brain_type_layout.addWidget(self.brain_type_combo)

        out_format_layout = QHBoxLayout()
        out_format_layout.addWidget(self.out_format_label)
        out_format_layout.addWidget(self.out_format_combo)

        name_layout = QHBoxLayout()
        name_layout.addWidget(self.name_label)
        name_layout.addWidget(self.name_input)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_label)
        alpha_layout.addWidget(self.alpha_input)
        alpha_layout.addWidget(self.alpha_slider)

        theta_layout = QHBoxLayout()
        theta_layout.addWidget(self.theta_label)
        theta_layout.addWidget(self.theta_input)
        theta_layout.addWidget(self.theta_slider)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_input)
        threshold_layout.addWidget(self.threshold_slider)

        nsamples_layout = QHBoxLayout()
        nsamples_layout.addWidget(self.nsamples_label)
        nsamples_layout.addWidget(self.nsamples_input)

        n_measures_layout = QHBoxLayout()
        n_measures_layout.addWidget(self.n_measures)
        n_measures_layout.addWidget(self.n_measures_label)
        n_measures_layout.addWidget(self.n_measures_input)

        brain_3d_layout = QHBoxLayout()
        brain_3d_layout.addWidget(self.brain_3d)
        brain_3d_layout.addWidget(self.brain_3d_label)

        interactive_3d_layout = QHBoxLayout()
        interactive_3d_layout.addWidget(self.interactive_3d)
        interactive_3d_layout.addWidget(self.interactive_3d_label)

        heatmap_layout = QHBoxLayout()
        heatmap_layout.addWidget(self.heatmap)
        heatmap_layout.addWidget(self.heatmap_label)

        circle_layout = QHBoxLayout()
        circle_layout.addWidget(self.circle)
        circle_layout.addWidget(self.circle_label)

        space_layout = QHBoxLayout()
        space_layout.addWidget(self.space_label)

        # Layouts order
        layout.addLayout(method_layout)
        layout.addLayout(alpha_layout)
        layout.addLayout(theta_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(weights_layout)
        layout.addLayout(correction_layout)
        layout.addLayout(sampling_layout)
        layout.addLayout(nsamples_layout)
        layout.addLayout(representation_layout)
        layout.addLayout(data_balance_layout)
        layout.addLayout(n_measures_layout)
        layout.addWidget(self.plotting_options_label)
        layout.addWidget(self.space_label)
        layout.addLayout(brain_3d_layout)
        layout.addLayout(interactive_3d_layout)
        layout.addLayout(brain_type_layout)
        layout.addLayout(space_layout)
        layout.addLayout(heatmap_layout)
        layout.addLayout(circle_layout)
        layout.addLayout(out_format_layout)

        layout.addWidget(self.error_label)

        layout.addWidget(self.submit_button)

        self.setLayout(layout)

        self.alpha_slider.valueChanged.connect(
            lambda value: self.update_slider_value(value, self.alpha_input, self.alpha_slider, 0, 1)
        )
        self.alpha_input.textChanged.connect(
            lambda text: self.update_slider_from_input(text, self.alpha_input, self.alpha_slider, 0, 1)
        )

        self.theta_slider.valueChanged.connect(
            lambda value: self.update_slider_value(value, self.theta_input, self.theta_slider, 0, 1)
        )
        self.theta_input.textChanged.connect(
            lambda text: self.update_slider_from_input(text, self.theta_input, self.theta_slider, 0, 1)
        )
        self.threshold_slider.valueChanged.connect(
            lambda value: self.update_slider_value(value, self.threshold_input, self.threshold_slider, 0, 1)
        )
        self.threshold_input.textChanged.connect(
            lambda text: self.update_slider_from_input(text, self.threshold_input, self.threshold_slider, 0, 1)
        )

    def update_interactive_3d_state(self, state):
        if state == Qt.Checked:
            self.interactive_3d.setEnabled(True)
        else:
            self.interactive_3d.setEnabled(False)
            self.interactive_3d.setChecked(False)

    def update_input_state(self, state):
        self.n_measures_input.setEnabled(state == 2)  # 2 corresponds to checked state

    def update_slider_value(self, value, input_field, slider, min_value, max_value):
        scaled_value = self.scale_slider_value(value, min_value, max_value)
        input_field.setText(str(scaled_value))

    def scale_slider_value(self, slider_value, min_value, max_value):
        # if max_value > 1:
        # return int(min_value + (slider_value / 100.0) * (max_value - min_value))
        return min_value + (slider_value / 100.0) * (max_value - min_value)

    def update_slider_from_input(self, text, input_field, slider, min_value, max_value):
        try:
            scaled_value = float(text)
            if min_value <= scaled_value <= max_value:  # Check valid range
                input_field.setStyleSheet("")  # Reset border color
                slider_value = self.inverse_scale_slider_value(scaled_value, min_value, max_value)
                slider.setValue(slider_value)
                self.error_label.setText("")  # Clear any error message
            else:
                input_field.setStyleSheet("border: 1px solid red;")
                self.error_label.setText(
                    f"Invalid input: Value must be between {min_value} and {max_value} for {input_field.objectName()}"
                )
        except ValueError:
            input_field.setStyleSheet("border: 1px solid red;")
            self.error_label.setText(f"Invalid input: Not a valid number for {input_field.objectName()}")

    def inverse_scale_slider_value(self, scaled_value, min_value, max_value):
        return int((scaled_value - min_value) / (max_value - min_value) * 100)

    def submit_clicked(self):
        attribute_info = [
            ("alpha", self.alpha_input, self.alpha_slider),
            ("theta", self.theta_input, self.theta_slider),
            ("threshold", self.threshold_input, self.threshold_slider),
            ("nsamples", self.nsamples_input, self.nsamples_slider),
        ]
        name = self.name_input.text()
        valid_input = True

        for attr_name, input_field, slider in attribute_info:
            text = input_field.text()
            try:
                value = float(text)
            except ValueError:
                input_field.setStyleSheet("border: 1px solid red;")
                self.error_label.setText(f"Invalid input: Not a valid number for {attr_name}")
                valid_input = False
                break
            if not (0 <= value <= 1):
                input_field.setStyleSheet("border: 1px solid red;")
                self.error_label.setText(f"Invalid input: Value must be between 0 and 1 for {attr_name}")
                valid_input = False
                break
            input_field.setStyleSheet("")  # Reset border color
            slider_value = int(value * 100)
            slider.setValue(slider_value)

        if valid_input:
            self.name = name
            self.method = self.method_combo.currentText()
            self.n_measures_state = self.n_measures.isChecked()
            self.n_measures_text = self.n_measures_input.text() if self.n_measures_state else ""
            self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = InputDialog()
    dialog.exec_()
    print("Name:", dialog.name)
    print("alpha:", dialog.alpha)
    print("Method:", dialog.method)
    print("Checkbox State:", dialog.checkbox_state)
    print("Checkbox Text:", dialog.checkbox_text)
    sys.exit(app.exec_())

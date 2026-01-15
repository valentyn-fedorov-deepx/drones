from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit


class SavePhotoDialog(QDialog):
	def __init__(self, user_input_old: str) -> None:
		super().__init__()

		self.setWindowTitle("Save Photo")

		q_btn: QDialogButtonBox.StandardButton = QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel

		self.button_box: QDialogButtonBox = QDialogButtonBox(q_btn)
		self.button_box.accepted.connect(self.accept)
		self.button_box.rejected.connect(self.reject)

		self.layout: QVBoxLayout = QVBoxLayout()

		message: QLabel = QLabel("Please, enter your input:")

		self.user_input_save: QLineEdit = QLineEdit(self)
		self.user_input_save.setText(user_input_old)

		self.layout.addWidget(message)
		self.layout.addWidget(self.user_input_save)
		self.layout.addWidget(self.button_box)
		self.setLayout(self.layout)

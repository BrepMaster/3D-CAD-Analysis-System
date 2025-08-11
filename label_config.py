# label_config.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget,
    QHBoxLayout, QPushButton, QLineEdit, QFormLayout,
    QColorDialog, QListWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from constants import DEFAULT_COLORS


class LabelConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配置标签和颜色")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.current_colors = []
        self.setup_ui()

        # 从父窗口获取当前标签配置
        if hasattr(parent, 'logic'):
            label_info = parent.logic.get_label_info()
            self.set_config(label_info['names'], label_info['colors'])
        elif hasattr(parent, 'label_names') and hasattr(parent, 'colors'):
            self.set_config(parent.label_names, parent.colors)
        else:
            # 默认配置（不应该执行到这里）
            self.set_config(["类别 1", "类别 2"], [DEFAULT_COLORS[0], DEFAULT_COLORS[1]])

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 标题
        header = QLabel("为每个标签类别配置名称和显示颜色:")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # 可滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        self.form_layout = QFormLayout()
        self.form_layout.setVerticalSpacing(10)
        self.form_layout.setHorizontalSpacing(15)
        self.form_layout.setContentsMargins(5, 5, 5, 5)

        self.label_names = []  # 存储所有标签名称输入框
        self.color_buttons = []  # 存储所有颜色按钮

        scroll_content.setLayout(self.form_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # 添加/删除按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        add_btn = QPushButton("＋ 添加类别")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #e1f5fe; 
                color: #0288d1;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #b3e5fc;
            }
        """)
        add_btn.clicked.connect(self.add_new_row)

        remove_btn = QPushButton("－ 删除最后类别")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffebee; 
                color: #d32f2f;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ffcdd2;
            }
        """)
        remove_btn.clicked.connect(self.remove_last_row)

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 确定/取消按钮
        button_box = QHBoxLayout()
        button_box.setSpacing(10)

        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50; 
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        ok_btn.clicked.connect(self.accept)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        button_box.addStretch()
        button_box.addWidget(ok_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)

        self.setLayout(layout)

    def set_config(self, label_names, colors):
        """设置初始标签配置"""
        # 清除现有行
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)
        self.label_names = []
        self.color_buttons = []
        self.current_colors = []

        # 为每个标签添加行
        for name, color in zip(label_names, colors):
            self.add_row(name, color)

    def add_row(self, default_name, default_color):
        """添加一个新标签行"""
        row = len(self.label_names)

        # 确保颜色有效
        if not isinstance(default_color, list) or len(default_color) != 3:
            default_color = DEFAULT_COLORS[row % len(DEFAULT_COLORS)]
        default_color = [max(0, min(255, c)) for c in default_color]

        # 创建行部件
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        # 序号标签
        num_label = QLabel(f"{row + 1}.")
        num_label.setFixedWidth(30)
        num_label.setAlignment(Qt.AlignCenter)
        num_label.setStyleSheet("font-weight: bold;")
        row_layout.addWidget(num_label)

        # 名称输入框
        name_edit = QLineEdit(default_name)
        name_edit.setMinimumWidth(200)
        name_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        self.label_names.append(name_edit)
        row_layout.addWidget(name_edit)

        # 颜色按钮
        color_btn = QPushButton()
        color_btn.setFixedSize(60, 30)
        color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({default_color[0]},{default_color[1]},{default_color[2]});
                border: 1px solid #ddd;
                border-radius: 3px;
            }}
        """)
        color_btn.clicked.connect(lambda _, r=row: self.change_color(r))
        self.color_buttons.append(color_btn)
        row_layout.addWidget(color_btn)

        row_widget.setLayout(row_layout)
        self.form_layout.addRow(row_widget)
        self.current_colors.append(default_color.copy())

    def add_new_row(self):
        """添加一个新类别行"""
        next_idx = len(self.label_names)
        default_color = DEFAULT_COLORS[next_idx % len(DEFAULT_COLORS)]
        self.add_row(f"类别 {next_idx + 1}", default_color)

    def remove_last_row(self):
        """删除最后一个类别行"""
        if len(self.label_names) > 1:  # 至少保留一个类别
            row = self.form_layout.rowCount() - 1
            self.form_layout.removeRow(row)
            self.label_names.pop()
            self.color_buttons.pop()
            self.current_colors.pop()

    def change_color(self, row):
        """修改指定行的颜色"""
        initial_color = QColor(*self.current_colors[row])
        color = QColorDialog.getColor(initial_color, self, "选择颜色")

        if color.isValid():
            rgb = [color.red(), color.green(), color.blue()]
            self.current_colors[row] = rgb
            self.color_buttons[row].setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]});
                    border: 1px solid #ddd;
                    border-radius: 3px;
                }}
            """)

    def get_config(self):
        """获取当前配置"""
        config = {
            "label_names": [],
            "colors": []
        }

        for name_edit, color in zip(self.label_names, self.current_colors):
            name = name_edit.text().strip()
            if not name:
                name = "未命名类别"
            config["label_names"].append(name)
            config["colors"].append(color.copy())  # 返回颜色副本

        return config
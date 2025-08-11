# ui_app.py
import os
import sys
import json
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QLabel, QFileDialog, QMessageBox, QGroupBox, QListWidget, QFrame,
    QSizePolicy, QStatusBar, QDockWidget, QProgressDialog, QCheckBox,
    QListWidgetItem, QTabWidget, QStackedWidget, QColorDialog
)
from PyQt5.QtGui import QFont, QColor, QPixmap
from OCC.Display.backend import load_backend
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Extend.DataExchange import read_step_file, write_step_file
from graph_utils import build_graph
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Face

load_backend("pyqt5")
from OCC.Display.qtDisplay import qtViewer3d

import dgl
from occwl.io import load_step

from segmentation_ui import SegmentationUI
from label_config import LabelConfigDialog
from segmentation_logic import SegmentationLogic
from constants import DEFAULT_COLORS, LANGUAGE_STRINGS, STYLESHEET


class CADAnalysisSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = 'zh'
        self.title = "3D CAD 智能分析系统 (分类/分割)"
        self.setWindowTitle(self.title)
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)
        self.menuBar().setNativeMenuBar(False)
        self.menuBar().hide()

        # 初始化背景颜色
        self.bg_color_light = [240, 240, 240]  # 默认浅色背景
        self.bg_color_dark = [40, 40, 40]  # 默认深色背景

        # 初始化两种模式
        self.segmentation_system = SegmentationSystem(self)
        self.classification_system = ClassificationSystem(self)

        # 当前模式
        self.current_mode = "classification"  # 默认改为分类模式

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="cad_analysis_")

        self.setup_ui()
        self.switch_mode("classification")  # 默认设置为分类模式

    def setup_ui(self):
        """设置主界面布局"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主布局
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部工具栏
        top_toolbar = QHBoxLayout()
        top_toolbar.setSpacing(5)

        # 模式切换按钮
        self.mode_toolbar = QHBoxLayout()
        self.mode_toolbar.setSpacing(5)

        self.segmentation_btn = QPushButton("分割模式")
        self.segmentation_btn.setCheckable(True)
        self.segmentation_btn.setChecked(False)  # 默认不选中
        self.segmentation_btn.clicked.connect(lambda: self.switch_mode("segmentation"))

        self.classification_btn = QPushButton("分类模式")
        self.classification_btn.setCheckable(True)
        self.classification_btn.setChecked(True)  # 默认选中
        self.classification_btn.clicked.connect(lambda: self.switch_mode("classification"))

        self.mode_toolbar.addWidget(self.segmentation_btn)
        self.mode_toolbar.addWidget(self.classification_btn)
        self.mode_toolbar.addStretch()

        # 背景设置按钮
        self.bg_color_btn = QPushButton("背景设置")
        self.bg_color_btn.clicked.connect(self.change_background_color)
        self.bg_color_btn.setToolTip("点击设置背景颜色")

        # 语言切换按钮
        self.lang_btn = QPushButton("EN")
        self.lang_btn.setCheckable(True)
        self.lang_btn.setChecked(False)
        self.lang_btn.clicked.connect(self.toggle_language)
        self.lang_btn.setFixedWidth(50)
        self.lang_btn.setToolTip("切换中英文")

        top_toolbar.addLayout(self.mode_toolbar)
        top_toolbar.addWidget(self.bg_color_btn)
        top_toolbar.addWidget(self.lang_btn)

        main_layout.addLayout(top_toolbar)

        # 使用QStackedWidget替代手动切换
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.segmentation_system)
        self.stacked_widget.addWidget(self.classification_system)

        main_layout.addWidget(self.stacked_widget, 1)

        # 设置按钮样式
        self.update_mode_buttons_style()

    def toggle_language(self):
        """切换中英文界面"""
        self.current_lang = 'en' if self.current_lang == 'zh' else 'zh'
        self.update_ui_language()

    def update_ui_language(self):
        """更新界面语言"""
        lang = self.current_lang
        strings = LANGUAGE_STRINGS.get(lang, LANGUAGE_STRINGS['en'])

        # 更新主窗口标题
        self.setWindowTitle(strings.get("title", "3D CAD Analysis System"))

        # 更新模式按钮
        self.segmentation_btn.setText(strings.get("segmentation_mode", "Segmentation"))
        self.classification_btn.setText(strings.get("classification_mode", "Classification"))

        # 更新工具按钮
        self.bg_color_btn.setText(strings.get("bg_settings", "Background Settings"))
        self.lang_btn.setText("EN" if lang == 'zh' else "中文")

        # 更新子系统语言
        self.segmentation_system.update_language(strings)
        self.classification_system.update_language(strings)

        # 更新按钮样式
        self.update_mode_buttons_style()

    def change_background_color(self):
        """更改背景颜色"""
        # 选择浅色背景
        light_color = QColorDialog.getColor(
            QColor(*self.bg_color_light),
            self,
            "选择浅色背景"
        )
        if not light_color.isValid():
            return

        # 选择深色背景
        dark_color = QColorDialog.getColor(
            QColor(*self.bg_color_dark),
            self,
            "选择深色背景"
        )
        if not dark_color.isValid():
            return

        # 更新背景颜色
        self.set_background_color(
            [light_color.red(), light_color.green(), light_color.blue()],
            [dark_color.red(), dark_color.green(), dark_color.blue()]
        )

    def set_background_color(self, light_color, dark_color):
        """设置统一的背景颜色"""
        self.bg_color_light = light_color
        self.bg_color_dark = dark_color

        # 更新两个子系统的背景
        self.segmentation_system.display.set_bg_gradient_color(light_color, dark_color)
        self.classification_system.display.set_bg_gradient_color(light_color, dark_color)

        # 强制重绘
        self.repaint_current_viewer()

    def switch_mode(self, mode):
        """切换分类/分割模式"""
        self.current_mode = mode

        if mode == "segmentation":
            self.stacked_widget.setCurrentIndex(0)
            self.segmentation_btn.setChecked(True)
            self.classification_btn.setChecked(False)
            # 确保显示正确初始化
            self.segmentation_system.canvas.InitDriver()
            self.segmentation_system.display = self.segmentation_system.canvas._display
            # 应用当前背景设置
            self.segmentation_system.display.set_bg_gradient_color(
                self.bg_color_light, self.bg_color_dark
            )
        else:
            self.stacked_widget.setCurrentIndex(1)
            self.segmentation_btn.setChecked(False)
            self.classification_btn.setChecked(True)
            # 确保显示正确初始化
            self.classification_system.canvas.InitDriver()
            self.classification_system.display = self.classification_system.canvas._display
            # 应用当前背景设置
            self.classification_system.display.set_bg_gradient_color(
                self.bg_color_light, self.bg_color_dark
            )
            # 强制重绘
            self.classification_system.display.Repaint()

        self.update_mode_buttons_style()
        QTimer.singleShot(100, self.repaint_current_viewer)

    def repaint_current_viewer(self):
        """强制重绘当前查看器"""
        if self.current_mode == "segmentation":
            self.segmentation_system.display.Repaint()
        else:
            self.classification_system.display.Repaint()

    def update_mode_buttons_style(self):
        """更新模式按钮样式"""
        active_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """

        inactive_style = """
            QPushButton {
                background-color: #f1f1f1;
                color: #333;
                border: 1px solid #ddd;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e1e1e1;
            }
        """

        self.segmentation_btn.setStyleSheet(active_style if self.current_mode == "segmentation" else inactive_style)
        self.classification_btn.setStyleSheet(active_style if self.current_mode == "classification" else inactive_style)

    def closeEvent(self, event):
        """清理临时目录"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        event.accept()


class SegmentationSystem(QWidget, SegmentationUI):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        SegmentationUI.__init__(self)
        self.parent_system = parent
        self.logic = SegmentationLogic()
        self.current_step_file = None
        self.current_model = None
        self.ais_list = []
        self.model_loaded = False
        self.labels_loaded = False
        self.step_loaded = False
        self.face_items = []

        self.category_buttons_layout = QVBoxLayout()
        self.setup_ui()
        self.setAcceptDrops(True)
        self.setMinimumSize(1200, 800)

    def update_language(self, strings):
        """更新子系统的语言"""
        # 通过对象名称查找组件
        file_group = self.findChild(QGroupBox, "fileOperationsGroup")
        seg_group = self.findChild(QGroupBox, "segmentationOperationsGroup")
        output_group = self.findChild(QGroupBox, "outputOperationsGroup")
        face_group = self.findChild(QGroupBox, "faceListGroup")
        category_group = self.findChild(QGroupBox, "categoryControlGroup")

        # 更新组标题
        if file_group:
            file_group.setTitle(strings.get("file_operations", "File Operations"))
        if seg_group:
            seg_group.setTitle(strings.get("segmentation_operations", "Segmentation Operations"))
        if output_group:
            output_group.setTitle(strings.get("output_operations", "Output Operations"))
        if face_group:
            face_group.setTitle(strings.get("face_list", "Face List"))
        if category_group:
            category_group.setTitle(strings.get("category_control", "Categories"))

        # 文件操作组
        self.loadModelButton.setText(strings.get("load_model", "Load Model"))
        self.loadLabelsButton.setText(strings.get("load_labels", "Load Labels"))
        self.loadButton.setText(strings.get("load_step", "Load STEP"))
        self.clearButton.setText(strings.get("clear", "Clear"))

        # 分割操作组
        self.segmentButton.setText(strings.get("segment", "Segment"))
        self.configButton.setText(strings.get("config", "Configure"))
        self.batchProcessButton.setText(strings.get("batch", "Batch Process"))

        # 输出操作组
        self.exportButton.setText(strings.get("export", "Export"))
        self.statsButton.setText(strings.get("stats", "Statistics"))
        self.helpButton.setText(strings.get("help", "Help"))

        # 状态标签
        if hasattr(self, 'statusLabel'):
            self.statusLabel.setText(strings.get("ready", "Ready"))

        # 拖放标签
        if hasattr(self, 'drop_label'):
            self.drop_label.setText(strings.get("drop_file", "Drop file here"))

    def setup_ui(self):
        """设置分割模式界面"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_group.setObjectName("fileOperationsGroup")
        file_layout = QVBoxLayout()

        self.loadModelButton = self.create_tool_button("加载模型")
        self.loadLabelsButton = self.create_tool_button("加载标签")
        self.loadButton = self.create_tool_button("加载STEP")
        self.clearButton = self.create_tool_button("清除")

        for btn in [self.loadModelButton, self.loadLabelsButton, self.loadButton, self.clearButton]:
            file_layout.addWidget(btn)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 分割操作组
        seg_group = QGroupBox("分割操作")
        seg_group.setObjectName("segmentationOperationsGroup")
        seg_layout = QVBoxLayout()

        self.configButton = self.create_tool_button("配置标签")
        self.segmentButton = self.create_tool_button("开始分割")
        self.batchProcessButton = self.create_tool_button("批量处理")

        for btn in [self.configButton, self.segmentButton, self.batchProcessButton]:
            seg_layout.addWidget(btn)

        seg_group.setLayout(seg_layout)
        left_layout.addWidget(seg_group)

        # 输出操作组
        output_group = QGroupBox("输出操作")
        output_group.setObjectName("outputOperationsGroup")
        output_layout = QVBoxLayout()

        self.exportButton = self.create_tool_button("导出结果")
        self.statsButton = self.create_tool_button("统计信息")
        self.helpButton = self.create_tool_button("帮助")

        for btn in [self.exportButton, self.statsButton, self.helpButton]:
            output_layout.addWidget(btn)

        output_group.setLayout(output_layout)
        left_layout.addWidget(output_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 中间画布区域
        self.canvas = qtViewer3d(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.InitDriver()
        self.display = self.canvas._display
        self.canvas.setMinimumSize(800, 600)

        # 设置背景颜色为父窗口的统一设置
        self.display.set_bg_gradient_color(
            self.parent_system.bg_color_light,
            self.parent_system.bg_color_dark
        )

        self.drop_label = QLabel("拖放文件到此处")
        self.drop_label.setObjectName("drop_label")
        self.drop_label.setVisible(False)

        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.drop_label)

        canvas_container = QWidget()
        canvas_container.setLayout(canvas_layout)
        main_layout.addWidget(canvas_container, 1)

        # 右侧面板
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # 面列表
        face_group = QGroupBox("面列表")
        face_group.setObjectName("faceListGroup")
        face_layout = QVBoxLayout()

        self.faceListWidget = QListWidget()
        self.faceListWidget.setSelectionMode(QListWidget.SingleSelection)
        self.faceListWidget.itemClicked.connect(self.on_face_selected)
        face_layout.addWidget(self.faceListWidget)

        face_group.setLayout(face_layout)
        right_layout.addWidget(face_group)

        # 类别控制
        category_group = QGroupBox("类别控制")
        category_group.setObjectName("categoryControlGroup")
        category_group.setLayout(self.category_buttons_layout)
        right_layout.addWidget(category_group)

        # 状态显示
        self.statusLabel = QLabel("准备就绪")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet("""
            QLabel {
                background-color: #f8fafc;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                color: #1e293b;
            }
        """)
        right_layout.addWidget(self.statusLabel)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

        # 连接按钮信号
        self.loadModelButton.clicked.connect(self.load_model)
        self.loadLabelsButton.clicked.connect(self.load_label_mapping)
        self.loadButton.clicked.connect(self.load_step)
        self.clearButton.clicked.connect(self.clear_all)
        self.configButton.clicked.connect(self.configure_labels)
        self.segmentButton.clicked.connect(self.start_segmentation)
        self.batchProcessButton.clicked.connect(self.batch_process_step_files)
        self.exportButton.clicked.connect(self.export_results)
        self.statsButton.clicked.connect(self.show_statistics)
        self.helpButton.clicked.connect(self.show_help)

        # 初始状态
        self.segmentButton.setEnabled(False)

        QTimer.singleShot(100, lambda: self.display.Repaint())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display.Repaint()

    def configure_labels(self):
        """配置标签映射"""
        dialog = LabelConfigDialog(self)
        if dialog.exec_():
            try:
                config = dialog.get_config()
                if config:
                    self.logic.update_label_config(config)
                    self.update_status("标签配置已更新")
                else:
                    self.show_error("获取的标签配置为空")
            except Exception as e:
                self.show_error(f"配置标签出错: {str(e)}")

    def show_error(self, message):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
        self.update_status(message, is_error=True)

    def update_status(self, message, is_error=False):
        """更新状态信息"""
        self.statusLabel.setText(message)
        if is_error:
            self.statusLabel.setStyleSheet("""
                QLabel {
                    background-color: #FFEBEE;
                    color: #F44336;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 14px;
                }
            """)
        else:
            self.statusLabel.setStyleSheet("""
                QLabel {
                    background-color: #f8fafc;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 14px;
                    color: #1e293b;
                }
            """)

    def create_tool_button(self, text):
        """创建工具按钮"""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f4f8;
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #e1e7ed;
            }
            QPushButton:pressed {
                background-color: #d1d9e6;
            }
            QPushButton[loaded="true"] {
                background-color: #d4edda;
                border-color: #c3e6cb;
            }
            QPushButton[loaded="true"]:hover {
                background-color: #c3e6cb;
            }
        """)
        return btn

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        <b>分割模式使用说明:</b>
        <ol>
            <li><b>实时分割模式</b>
                <ul>
                    <li>加载模型文件(.ckpt/.pt/.pth)</li>
                    <li>配置标签和颜色(可选)</li>
                    <li>加载STEP文件(.step/.stp)</li>
                    <li>点击"开始分割"按钮进行实时分割</li>
                </ul>
            </li>
            <li><b>批量处理模式</b>
                <ul>
                    <li>点击"批量处理"按钮</li>
                    <li>选择包含STEP文件的文件夹</li>
                    <li>选择输出文件夹</li>
                    <li>系统会自动处理所有STEP文件</li>
                </ul>
            </li>
        </ol>
        <b>支持拖拽操作</b>"""

        QMessageBox.information(self, "帮助", help_text)

    def create_category_buttons(self):
        while self.category_buttons_layout.count():
            child = self.category_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        label_info = self.logic.get_label_info()
        for i, (name, color) in enumerate(zip(label_info["names"], label_info["colors"])):
            container = QWidget()
            hbox = QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(5)

            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_category_visibility(idx, state))
            hbox.addWidget(checkbox)

            btn = QPushButton(f"类别 {i + 1}: {name}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({color[0]}, {color[1]}, {color[2]});
                    color: white;
                    border: none;
                    padding: 5px;
                    text-align: left;
                    border-radius: 3px;
                }}
                QPushButton:hover {{
                    background-color: rgba({color[0]}, {color[1]}, {color[2]}, 200);
                }}
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self.on_category_selected(idx))
            hbox.addWidget(btn, 1)

            container.setLayout(hbox)
            self.category_buttons_layout.addWidget(container)

    def toggle_category_visibility(self, category_idx, state):
        if not self.step_loaded or not hasattr(self, 'ais_list'):
            return

        context = self.display.GetContext()
        if not context:
            return

        label_info = self.logic.get_label_info()
        predicted_labels = self.logic.get_predicted_labels()

        try:
            for i, ais in enumerate(self.ais_list):
                if i >= len(predicted_labels) or not ais:
                    continue

                label_num = predicted_labels[i]
                if label_num == category_idx:
                    if state == Qt.Checked:
                        color_rgb = label_info["colors"][label_num]
                        context.SetColor(ais, Quantity_Color(
                            color_rgb[0] / 255.0,
                            color_rgb[1] / 255.0,
                            color_rgb[2] / 255.0,
                            Quantity_TOC_RGB), False)
                        context.SetTransparency(ais, 0.0, False)
                    else:
                        context.SetColor(ais, Quantity_Color(
                            1.0, 1.0, 1.0,
                            Quantity_TOC_RGB), False)
                        context.SetTransparency(ais, 0.7, False)

            context.UpdateCurrentViewer()
            self.display.Repaint()
        except Exception as e:
            print(f"切换类别可见性出错: {str(e)}")

    def on_category_selected(self, category_idx):
        if not self.step_loaded or not hasattr(self, 'ais_list'):
            return

        context = self.display.GetContext()
        if not context:
            return

        label_info = self.logic.get_label_info()
        predicted_labels = self.logic.get_predicted_labels()

        try:
            for i, ais in enumerate(self.ais_list):
                if i >= len(predicted_labels) or not ais:
                    continue

                label_num = predicted_labels[i]
                if label_num == category_idx:
                    color_rgb = label_info["colors"][label_num]
                    context.SetColor(ais, Quantity_Color(
                        color_rgb[0] / 255.0,
                        color_rgb[1] / 255.0,
                        color_rgb[2] / 255.0,
                        Quantity_TOC_RGB), False)
                    context.SetTransparency(ais, 0.0, False)
                else:
                    context.SetColor(ais, Quantity_Color(
                        1.0, 1.0, 1.0,
                        Quantity_TOC_RGB), False)
                    context.SetTransparency(ais, 0.7, False)

            context.UpdateCurrentViewer()
            self.display.Repaint()
        except Exception as e:
            print(f"按类别渲染出错: {str(e)}")

    def on_face_selected(self, item):
        if not hasattr(item, 'face_index'):
            return

        face_index = item.face_index
        context = self.display.GetContext()
        if not context:
            return

        try:
            for i, ais in enumerate(self.ais_list):
                if not ais:
                    continue
                context.SetDisplayMode(ais, 1, False)
                context.SetColor(ais,
                                 Quantity_Color(150 / 255.0, 150 / 255.0, 150 / 255.0, Quantity_TOC_RGB),
                                 False)
                context.SetTransparency(ais, 0.8, False)

            if face_index < len(self.ais_list) and self.ais_list[face_index]:
                selected_ais = self.ais_list[face_index]
                label_info = self.logic.get_label_info()
                label_color = label_info["colors"][self.logic.get_predicted_labels()[face_index]]

                context.SetDisplayMode(selected_ais, 1, False)
                context.SetColor(selected_ais,
                                 Quantity_Color(
                                     label_color[0] / 255.0,
                                     label_color[1] / 255.0,
                                     label_color[2] / 255.0,
                                     Quantity_TOC_RGB
                                 ),
                                 False)
                context.SetTransparency(selected_ais, 0.0, False)

            context.UpdateCurrentViewer()
            self.display.FitAll()
            self.display.Repaint()
        except Exception as e:
            print(f"设置显示模式出错: {str(e)}")

    def populate_face_list(self):
        self.faceListWidget.clear()
        self.face_items = []

        label_info = self.logic.get_label_info()
        predicted_labels = self.logic.get_predicted_labels()

        for i, label_num in enumerate(predicted_labels):
            if i >= len(predicted_labels):
                continue

            label_name = label_info["names"][label_num] if label_num < len(label_info["names"]) else f"未知标签 {label_num}"
            color = label_info["colors"][label_num] if label_num < len(label_info["colors"]) else [150, 150, 150]

            item_text = f"面 {i + 1}: {label_name}"
            item = QListWidgetItem(item_text)
            item.face_index = i

            item.setForeground(QColor(*color))
            self.faceListWidget.addItem(item)
            self.face_items.append(item)

    def start_segmentation(self):
        if not self.labels_loaded:
            self.show_error("请先加载标签文件")
            return
        if not self.step_loaded:
            self.show_error("请先加载STEP文件")
            return

        try:
            self.clear_display()
            self.update_status("正在处理文件...")

            if not self.model_loaded:
                self.show_error("请先加载模型文件")
                return
            QTimer.singleShot(100, lambda: self.process_step_file(self.current_step_file))

        except Exception as e:
            self.show_error(f"分割出错: {str(e)}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet("background-color: rgba(234, 242, 255, 0.9);")
            self.drop_label.setVisible(True)

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("")
        self.drop_label.setVisible(False)

    def dropEvent(self, event):
        self.drop_label.setStyleSheet("")
        self.drop_label.setVisible(False)
        urls = event.mimeData().urls()
        if not urls:
            return

        file_path = urls[0].toLocalFile()
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.step', '.stp']:
            self.handle_dropped_step(file_path)
        elif ext in ['.ckpt', '.pt', '.pth']:
            self.handle_dropped_model(file_path)
        elif ext == '.json':
            self.handle_dropped_labels(file_path)
        else:
            self.show_error("不支持的文件类型")

    def handle_dropped_model(self, file_path):
        try:
            model_name = self.logic.load_model(file_path)
            self.current_model = file_path
            self.model_loaded = True
            self.loadModelButton.setProperty("loaded", "true")
            self.loadModelButton.style().polish(self.loadModelButton)
            self.update_status(f"模型已加载: {model_name}")

            label_path = os.path.splitext(file_path)[0] + ".json"
            if os.path.exists(label_path):
                QTimer.singleShot(1000, lambda: self.handle_dropped_labels(label_path))
        except Exception as e:
            self.show_error(f"加载模型出错: {str(e)}")

    def handle_dropped_labels(self, file_path):
        try:
            label_name = self.logic.load_labels(file_path)
            self.labels_loaded = True
            self.loadLabelsButton.setProperty("loaded", "true")
            self.loadLabelsButton.style().polish(self.loadLabelsButton)
            self.update_status(f"标签已加载: {label_name}")
            self.segmentButton.setEnabled(True)
            self.create_category_buttons()
        except Exception as e:
            self.show_error(f"加载标签出错: {str(e)}")

    def handle_dropped_step(self, file_path):
        self.current_step_file = file_path
        self.step_loaded = True
        self.update_status(f"STEP文件已加载: {os.path.basename(file_path)}")
        self.loadButton.setProperty("loaded", "true")
        self.loadButton.style().polish(self.loadButton)

    def process_step_file(self, file_path):
        try:
            self.logic.process_step_file(file_path, 1)
            self.display_segmentation(file_path)
            self.update_status("分割完成")
        except Exception as e:
            self.show_error(f"处理文件出错: {str(e)}")

    def display_segmentation(self, step_file):
        self.clear_display()
        self.face_items = []

        shape = read_step_file(step_file)
        if not shape:
            print("Failed to load shapes")
            return

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        index = 0
        self.ais_list = []
        label_info = self.logic.get_label_info()
        predicted_labels = self.logic.get_predicted_labels()

        while explorer.More():
            face = explorer.Current()
            if face.IsNull():
                explorer.Next()
                continue

            if not isinstance(face, TopoDS_Face):
                face = TopoDS_Face(face)

            if index < len(predicted_labels):
                label_num = min(max(0, int(predicted_labels[index])), len(label_info["colors"]) - 1)
                color_rgb = label_info["colors"][label_num]
                color_rgb = [max(0, min(255, c)) for c in color_rgb]

                color = Quantity_Color(
                    color_rgb[0] / 255.0,
                    color_rgb[1] / 255.0,
                    color_rgb[2] / 255.0,
                    Quantity_TOC_RGB
                )

                ais_shape = self.display.DisplayShape(face, color=color, update=True)
                if ais_shape:
                    if isinstance(ais_shape, list):
                        self.ais_list.append(ais_shape[0])
                    else:
                        self.ais_list.append(ais_shape)
                index += 1

            explorer.Next()

        self.populate_face_list()
        self.create_category_buttons()
        self.display.FitAll()
        self.display.Repaint()

    def clear_display(self):
        context = self.display.GetContext()
        if not context:
            return

        context.RemoveAll(False)
        self.ais_list = []
        context.UpdateCurrentViewer()
        self.display.FitAll()
        self.display.Repaint()
        self.faceListWidget.clear()
        self.face_items = []

    def clear_all(self):
        self.clear_display()
        self.current_model = None
        self.current_step_file = None
        self.logic = SegmentationLogic()
        self.model_loaded = False
        self.labels_loaded = False
        self.step_loaded = False
        self.segmentButton.setEnabled(False)

        while self.category_buttons_layout.count():
            child = self.category_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for btn in [self.loadModelButton, self.loadLabelsButton, self.loadButton]:
            btn.setProperty("loaded", "false")
            btn.style().polish(btn)

        self.update_status("准备就绪")

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "模型文件 (*.ckpt *.pt *.pth)"
        )
        if file_name:
            self.handle_dropped_model(file_name)

    def load_label_mapping(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择标签文件",
            "",
            "JSON文件 (*.json)"
        )
        if file_name:
            self.handle_dropped_labels(file_name)

    def load_step(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择STEP文件",
            "",
            "STEP文件 (*.step *.stp)"
        )
        if file_name:
            self.handle_dropped_step(file_name)

    def export_results(self):
        if not self.step_loaded:
            self.show_error("没有可导出的结果")
            return

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "保存结果", "",
            "JSON文件 (*.json);;文本文件 (*.txt);;纯文本SEG (*.seg);;所有文件 (*)"
        )

        if not file_name:
            return

        try:
            label_info = self.logic.get_label_info()
            predicted_labels = self.logic.get_predicted_labels()

            # Convert numpy int64 to native Python int for JSON serialization
            predicted_labels = [int(label) for label in predicted_labels]
            label_counts = [int(count) for count in label_info["counts"]]

            if selected_filter == "纯文本SEG (*.seg)":
                with open(file_name, 'w', encoding='utf-8') as f:
                    for label in predicted_labels:
                        f.write(f"{label}\n")
            elif file_name.endswith('.txt'):
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(f"3D CAD 分割结果报告\n")
                    f.write(f"=" * 40 + "\n")
                    f.write(f"模型文件: {os.path.basename(self.current_model) if self.current_model else '未知'}\n")
                    f.write(f"STEP文件: {os.path.basename(self.current_step_file)}\n")
                    f.write(f"总面数: {len(predicted_labels)}\n\n")
                    f.write("类别分布:\n")
                    for i, count in enumerate(label_counts):
                        if count > 0:
                            percentage = count / len(predicted_labels) * 100
                            f.write(f"{label_info['names'][i]}: {count} ({percentage:.1f}%)\n")
            else:
                results = {
                    "model": os.path.basename(self.current_model) if self.current_model else "未知",
                    "step_file": os.path.basename(self.current_step_file),
                    "total_faces": len(predicted_labels),
                    "label_distribution": {
                        label_info['names'][i]: count for i, count in enumerate(label_counts)
                    },
                    "face_labels": predicted_labels,
                    "label_colors": label_info["colors"],
                    "label_names": label_info["names"]
                }
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

            self.update_status(f"结果已保存到: {os.path.basename(file_name)}")
        except Exception as e:
            self.show_error(f"导出结果出错: {str(e)}")

    def show_statistics(self):
        if not self.step_loaded:
            self.show_error("没有可显示的统计数据")
            return

        label_info = self.logic.get_label_info()
        stats_text = f"""
        <b>分割统计信息</b>
        <table border="0" cellspacing="5" cellpadding="3">
            <tr><td><b>总面数:</b></td><td>{label_info['total_faces']}</td></tr>
        """

        for i, count in enumerate(label_info["counts"]):
            if count > 0:
                percentage = count / label_info['total_faces'] * 100
                stats_text += f"""
                <tr>
                    <td><b>类别 {i + 1}: {label_info['names'][i]}</b></td>
                    <td>{count} ({percentage:.1f}%)</td>
                </tr>
                """

        stats_text += "</table>"

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("统计信息")
        msg.setTextFormat(Qt.RichText)
        msg.setText(stats_text)
        msg.exec_()

    def batch_process_step_files(self):
        input_dir = QFileDialog.getExistingDirectory(
            self,
            "选择包含STEP文件的文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not input_dir:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "选择输出文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not output_dir:
            return

        step_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.step', '.stp')):
                    step_files.append(os.path.join(root, file))

        if not step_files:
            self.show_error("没有找到STEP文件")
            return

        if not self.model_loaded:
            self.show_error("请先加载模型文件")
            return

        progress_dialog = QProgressDialog(
            "正在批量处理STEP文件...",
            "取消",
            0,
            len(step_files),
            self
        )
        progress_dialog.setWindowTitle("批量处理")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)

        for i, step_file in enumerate(step_files):
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在处理: {os.path.basename(step_file)}")
            QApplication.processEvents()

            if progress_dialog.wasCanceled():
                break

            try:
                self.logic.process_step_file(step_file, 1)
                predicted_labels = self.logic.get_predicted_labels()

                base_name = os.path.splitext(os.path.basename(step_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}.seg")

                with open(output_file, 'w', encoding='utf-8') as f:
                    for label in predicted_labels:
                        f.write(f"{label}\n")

            except Exception as e:
                self.show_error(f"处理文件 {os.path.basename(step_file)} 时出错: {str(e)}")
                continue

        progress_dialog.setValue(len(step_files))
        self.update_status(f"批量处理完成，共处理 {len(step_files)} 个STEP文件")


class ClassificationSystem(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_system = parent
        self.current_model = None
        self.label_mapping = None
        self.ais_list = []
        self.model_loaded = False
        self.labels_loaded = False
        self.step_loaded = False
        self.current_prediction = None
        self.current_confidence = None
        self.setMinimumSize(1200, 800)

        self.setup_ui()
        self.setAcceptDrops(True)

        # 确保显示正确初始化
        QTimer.singleShot(100, self.initialize_display)

    def update_language(self, strings):
        """更新子系统的语言"""
        # 通过对象名称查找组件
        file_group = self.findChild(QGroupBox, "fileOperationsGroup")
        classify_group = self.findChild(QGroupBox, "classificationOperationsGroup")
        results_group = self.findChild(QGroupBox, "classificationResultsGroup")

        # 更新组标题
        if file_group:
            file_group.setTitle(strings.get("file_operations", "File Operations"))
        if classify_group:
            classify_group.setTitle(strings.get("classification_operations", "Classification Operations"))
        if results_group:
            results_group.setTitle(strings.get("classification_results", "Results"))

        # 文件操作组
        self.loadModelButton.setText(strings.get("load_model", "Load Model"))
        self.loadLabelsButton.setText(strings.get("load_labels", "Load Labels"))
        self.loadButton.setText(strings.get("load_step", "Load STEP"))
        self.clearButton.setText(strings.get("clear", "Clear"))

        # 分类操作组
        self.classifyButton.setText(strings.get("classify", "Classify"))
        self.helpButton.setText(strings.get("help", "Help"))

        # 结果显示
        if not self.current_prediction:
            self.resultDisplay.setText(
                strings.get("load_prompt", "Load model and STEP file\nThen click classify button"))

        # 拖放标签
        if hasattr(self, 'drop_label'):
            self.drop_label.setText(strings.get("drop_file", "Drop file here"))

    def initialize_display(self):
        """确保显示正确初始化"""
        self.canvas.InitDriver()
        self.display = self.canvas._display
        self.display.set_bg_gradient_color(
            self.parent_system.bg_color_light,
            self.parent_system.bg_color_dark
        )
        self.display.Repaint()

    def setup_ui(self):
        """设置分类模式界面"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_group.setObjectName("fileOperationsGroup")
        file_layout = QVBoxLayout()

        self.loadModelButton = self.create_tool_button("加载模型")
        self.loadLabelsButton = self.create_tool_button("加载标签")
        self.loadButton = self.create_tool_button("加载STEP")
        self.clearButton = self.create_tool_button("清除")

        for btn in [self.loadModelButton, self.loadLabelsButton, self.loadButton, self.clearButton]:
            file_layout.addWidget(btn)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 分类操作组
        classify_group = QGroupBox("分类操作")
        classify_group.setObjectName("classificationOperationsGroup")
        classify_layout = QVBoxLayout()

        self.classifyButton = self.create_tool_button("开始分类")
        self.helpButton = self.create_tool_button("帮助")

        for btn in [self.classifyButton, self.helpButton]:
            classify_layout.addWidget(btn)

        classify_group.setLayout(classify_layout)
        left_layout.addWidget(classify_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 中间画布区域
        self.canvas = qtViewer3d(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.InitDriver()
        self.display = self.canvas._display
        self.canvas.setMinimumSize(800, 600)

        # 设置背景颜色为父窗口的统一设置
        self.display.set_bg_gradient_color(
            self.parent_system.bg_color_light,
            self.parent_system.bg_color_dark
        )

        self.drop_label = QLabel("拖放文件到此处")
        self.drop_label.setObjectName("drop_label")
        self.drop_label.setVisible(False)

        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.drop_label)

        canvas_container = QWidget()
        canvas_container.setLayout(canvas_layout)
        main_layout.addWidget(canvas_container, 1)

        # 右侧结果面板
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # 结果展示
        results_group = QGroupBox("分类结果")
        results_group.setObjectName("classificationResultsGroup")
        results_layout = QVBoxLayout()

        self.resultDisplay = QLabel("加载模型和STEP文件后\n点击开始分类按钮")
        self.resultDisplay.setAlignment(Qt.AlignCenter)
        self.resultDisplay.setWordWrap(True)
        self.resultDisplay.setStyleSheet("""
            QLabel {
                background-color: #f8fafc;
                border-radius: 6px;
                padding: 20px;
                font-size: 18px;
                color: #1e293b;
                min-height: 100px;
                font-family: "Microsoft YaHei", sans-serif;
                font-weight: normal;
            }
        """)
        results_layout.addWidget(self.resultDisplay)

        # 置信度显示
        self.confidenceMeter = QLabel()
        self.confidenceMeter.setAlignment(Qt.AlignCenter)
        self.confidenceMeter.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: normal;
                padding: 10px;
                border-radius: 6px;
                font-family: "Microsoft YaHei", sans-serif;
            }
        """)
        results_layout.addWidget(self.confidenceMeter)

        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

        # 连接按钮信号
        self.loadModelButton.clicked.connect(self.load_model)
        self.loadLabelsButton.clicked.connect(self.load_labels)
        self.loadButton.clicked.connect(self.load_step)
        self.clearButton.clicked.connect(self.clear_all)
        self.classifyButton.clicked.connect(self.classify)
        self.helpButton.clicked.connect(self.show_help)

        # 初始状态
        self.classifyButton.setEnabled(False)

        QTimer.singleShot(100, lambda: self.display.Repaint())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display.Repaint()

    def load_step(self):
        """加载STEP文件"""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "选择STEP文件", "", "STEP Files (*.step *.stp)"
        )
        if fileName:
            # 清除之前显示的模型
            self.clear_display()

            # 加载新模型
            shapes = read_step_file(fileName)
            ais = self.display.DisplayShape(shapes, update=True)[0]
            self.display.FitAll()
            self.ais_list = [ais]  # 重置列表，只保留当前模型
            self.step_loaded = True
            self.loadButton.setProperty("loaded", "true")
            self.loadButton.style().polish(self.loadButton)
            self.update_status(f"STEP文件已加载: {os.path.basename(fileName)}")
            self.check_ready_state()

    def clear_display(self):
        """清除显示的所有模型"""
        context = self.display.GetContext()
        if not context:
            return

        # 清除所有显示的图形
        context.RemoveAll(False)
        self.ais_list = []

        # 更新视图
        context.UpdateCurrentViewer()
        self.display.FitAll()
        self.display.Repaint()

    def create_tool_button(self, text):
        """创建工具按钮"""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f4f8;
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
                font-family: "Microsoft YaHei", sans-serif;
            }
            QPushButton:hover {
                background-color: #e1e7ed;
            }
            QPushButton:pressed {
                background-color: #d1d9e6;
            }
            QPushButton[loaded="true"] {
                background-color: #d4edda;
                border-color: #c3e6cb;
            }
            QPushButton[loaded="true"]:hover {
                background-color: #c3e6cb;
            }
        """)
        return btn

    def load_model(self):
        """加载分类模型"""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "Model Files (*.ckpt *.pt *.pth)"
        )
        if fileName:
            self.current_model = fileName
            self.model_loaded = True
            self.loadModelButton.setProperty("loaded", "true")
            self.loadModelButton.style().polish(self.loadModelButton)
            self.update_status(f"模型已加载: {os.path.basename(fileName)}")
            self.check_ready_state()

    def load_labels(self):
        """加载标签文件"""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "选择标签文件", "", "JSON Files (*.json)"
        )
        if fileName:
            try:
                with open(fileName, 'r', encoding='utf-8') as f:
                    self.label_mapping = json.load(f)
                self.labels_loaded = True
                self.loadLabelsButton.setProperty("loaded", "true")
                self.loadLabelsButton.style().polish(self.loadLabelsButton)
                self.update_status(f"标签已加载: {os.path.basename(fileName)}")
                self.check_ready_state()
            except Exception as e:
                self.show_error(f"加载标签出错: {str(e)}")

    def classify(self):
        """执行分类操作"""
        if not all([self.model_loaded, self.labels_loaded, self.step_loaded]):
            self.show_error("请先加载模型、标签和STEP文件")
            return

        try:
            # 1. Get the last loaded STEP file from display context
            if not self.ais_list:
                self.show_error("没有可用的STEP文件")
                return

            # Get the shape from the last AIS object
            ais = self.ais_list[-1]
            shape = ais.Shape()

            if shape.IsNull():
                self.show_error("无法获取当前STEP文件的形状")
                return

            # 2. Create a temporary STEP file from the displayed shape
            temp_step_path = os.path.join(self.parent_system.temp_dir, "temp_classify.step")
            write_step_file(shape, temp_step_path)

            # 3. Convert to BIN format
            bin_path = os.path.join(self.parent_system.temp_dir, "temp_classify.bin")
            if not self.convert_step_to_bin(temp_step_path, bin_path):
                return

            # 4. Classify using the model
            from preprocessor import load_one_graph
            from classification_model import init

            sample = load_one_graph(bin_path)
            inputs = sample["graph"]
            inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
            inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)

            with torch.no_grad():
                logits = init(bin_path, self.current_model)
                preds = F.softmax(logits, dim=-1)
                max_index = torch.argmax(preds, dim=-1).item()
                confidence = torch.max(preds).item() * 100

            prediction_text = self.label_mapping.get(str(max_index), f'Class {max_index}')
            self.current_prediction = prediction_text
            self.current_confidence = confidence

            # Display results
            self.display_results(prediction_text, confidence)

        except Exception as e:
            self.show_error(f"分类出错: {str(e)}")

    def display_results(self, prediction, confidence):
        """显示分类结果"""
        # Set prediction text with appropriate color
        if confidence > 80:
            pred_color = "#10b981"  # Green
            meter_color = "background-color: #d1fae5; color: #065f46;"
        elif confidence > 60:
            pred_color = "#f59e0b"  # Yellow
            meter_color = "background-color: #fef3c7; color: #92400e;"
        else:
            pred_color = "#ef4444"  # Red
            meter_color = "background-color: #fee2e2; color: #991b1b;"

        # Update result display with larger, softer font
        self.resultDisplay.setText(
            f'<span style="font-size: 24px; color: {pred_color}; '
            f'font-family: "Microsoft YaHei", sans-serif;">'
            f'{prediction}</span>'
        )

        # Update confidence meter with larger, softer font
        self.confidenceMeter.setText(f'{confidence:.1f}%')
        self.confidenceMeter.setStyleSheet(f"""
            QLabel {{
                {meter_color}
                font-size: 28px;
                font-weight: normal;
                padding: 10px;
                border-radius: 6px;
                font-family: "Microsoft YaHei", sans-serif;
            }}
        """)

    def clear_all(self):
        """清除所有内容"""
        for ais in self.ais_list:
            self.display.Context.Erase(ais, True)
        self.ais_list = []
        self.display.FitAll()

        self.current_model = None
        self.label_mapping = None
        self.model_loaded = False
        self.labels_loaded = False
        self.step_loaded = False
        self.current_prediction = None
        self.current_confidence = None

        for btn in [self.loadModelButton, self.loadLabelsButton, self.loadButton]:
            btn.setProperty("loaded", "false")
            btn.style().polish(btn)

        self.classifyButton.setEnabled(False)

        self.resultDisplay.setText("加载模型和STEP文件后\n点击开始分类按钮")
        self.confidenceMeter.setText("")
        self.confidenceMeter.setStyleSheet("")

        self.update_status("准备就绪")

    def check_ready_state(self):
        """检查是否可以开始分类"""
        if all([self.model_loaded, self.labels_loaded, self.step_loaded]):
            self.classifyButton.setEnabled(True)
        else:
            self.classifyButton.setEnabled(False)

    def update_status(self, message, is_error=False):
        """更新状态信息"""
        if is_error:
            self.resultDisplay.setStyleSheet("""
                QLabel {
                    background-color: #FFEBEE;
                    color: #F44336;
                    border-radius: 6px;
                    padding: 20px;
                    font-size: 18px;
                    min-height: 100px;
                }
            """)
        else:
            self.resultDisplay.setStyleSheet("""
                QLabel {
                    background-color: #f8fafc;
                    border-radius: 6px;
                    padding: 20px;
                    font-size: 18px;
                    color: #1e293b;
                    min-height: 100px;
                }
            """)
        self.resultDisplay.setText(message)

    def show_error(self, message):
        """显示错误信息"""
        self.update_status(message, is_error=True)
        QMessageBox.critical(self, "错误", message)

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        <div style="font-family: 'Microsoft YaHei', sans-serif; font-size: 12px; line-height: 1.6;">
        <b style="font-size: 14px;">使用说明:</b>
        <ol style="margin-top: 5px; padding-left: 20px;">
            <li><b>分类模式</b>
                <ul style="margin-top: 5px; padding-left: 15px;">
                    <li>加载模型文件(.ckpt/.pt/.pth)</li>
                    <li>加载标签映射文件(.json)</li>
                    <li>加载STEP文件(.step/.stp)</li>
                    <li>点击"开始分类"按钮进行分类</li>
                </ul>
            </li>
        </ol>
        </div>"""

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("帮助")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.exec_()

    def convert_step_to_bin(self, step_path, bin_path):
        """Convert STEP file to DGL graph binary format"""
        try:
            # Load the STEP file
            solid = load_step(step_path)[0]

            # Build the graph with default parameters
            graph = build_graph(solid,
                                curv_num_u_samples=10,
                                surf_num_u_samples=10,
                                surf_num_v_samples=10)

            # Save the graph
            dgl.data.utils.save_graphs(str(bin_path), [graph])
            return True
        except Exception as e:
            self.show_error(f"转换出错: {str(e)}")
            return False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = CADAnalysisSystem()

    # 设置自定义背景颜色 (RGB格式)
    window.set_background_color(
        light_color=[40, 40, 40],  # 浅灰色
        dark_color=[40, 40, 40]  # 深灰色
    )

    window.show()
    sys.exit(app.exec_())
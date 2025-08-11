# segmentation_ui.py
import os
import numpy as np
import json
from PyQt5.QtWidgets import (
    QWidget, QMessageBox, QFileDialog, QProgressDialog,
    QCheckBox, QPushButton, QListWidgetItem, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Extend.DataExchange import read_step_file
from segmentation_logic import SegmentationLogic
from PyQt5.QtWidgets import QApplication

class SegmentationUI:
    def convert_to_python_types(self, data):
        """Convert numpy arrays and other non-JSON-serializable types to Python native types"""
        if isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        elif isinstance(data, list):
            return [self.convert_to_python_types(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.convert_to_python_types(v) for k, v in data.items()}
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        else:
            return str(data)

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
            label_counts = label_info["counts"]

            predicted_labels = self.convert_to_python_types(predicted_labels)
            label_counts = self.convert_to_python_types(label_counts)

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

    def configure_labels(self):
        """配置标签映射"""
        dialog = LabelConfigDialog(self)
        if dialog.exec_():
            try:
                config = dialog.get_config()
                if config:
                    self.logic.update_label_config(config)
                    self.update_status("标签配置已更新")
                    self.create_category_buttons()
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
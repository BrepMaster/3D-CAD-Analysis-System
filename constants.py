LANGUAGE_STRINGS = {
    "zh": {
        "title": "3D CAD 智能分析系统",
        "segmentation_mode": "分割模式",
        "classification_mode": "分类模式",
        "bg_settings": "背景设置",
        "file_operations": "文件操作",
        "segmentation_operations": "分割操作",
        "classification_operations": "分类操作",
        "output_operations": "输出操作",
        "load_model": "加载模型",
        "load_labels": "加载标签",
        "load_step": "加载STEP",
        "clear": "清除",
        "segment": "开始分割",
        "classify": "开始分类",
        "config": "配置标签",
        "batch": "批量处理",
        "export": "导出结果",
        "stats": "统计信息",
        "help": "帮助",
        "face_list": "面列表",
        "category_control": "类别控制",
        "classification_results": "分类结果",
        "ready": "准备就绪",
        "drop_file": "拖放文件到此处",
        "load_prompt": "加载模型和STEP文件后\n点击开始分类按钮"
    },
    "en": {
        "title": "3D CAD Analysis System",
        "segmentation_mode": "Segmentation",
        "classification_mode": "Classification",
        "bg_settings": "Background",
        "file_operations": "File Operations",
        "segmentation_operations": "Segmentation Operations",
        "classification_operations": "Classification Operations",
        "output_operations": "Output Operations",
        "load_model": "Load Model",
        "load_labels": "Load Labels",
        "load_step": "Load STEP",
        "clear": "Clear",
        "segment": "Segment",
        "classify": "Classify",
        "config": "Configure",
        "batch": "Batch Process",
        "export": "Export",
        "stats": "Statistics",
        "help": "Help",
        "face_list": "Face List",
        "category_control": "Categories",
        "classification_results": "Results",
        "ready": "Ready",
        "drop_file": "Drop file here",
        "load_prompt": "Load model and STEP file\nThen click classify button"
    }
}

DEFAULT_COLORS = [
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
    [255, 255, 0],  # 黄色
    [255, 0, 255],  # 紫色
    [0, 255, 255],  # 青色
    [255, 128, 0],  # 橙色
    [128, 0, 255],  # 紫罗兰色
    [0, 255, 128],  # 春绿色
    [128, 255, 0]  # 黄绿色
]

STYLESHEET = """
    QWidget {
        font-family: "Microsoft YaHei", "Arial";
        font-size: 12px;
    }
    QGroupBox {
        border: 1px solid #d1d9e6;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 15px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
    }
    QPushButton {
        background-color: #f0f4f8;
        border: 1px solid #d1d9e6;
        border-radius: 4px;
        padding: 5px 10px;
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
    QListWidget {
        border: 1px solid #d1d9e6;
        border-radius: 4px;
    }

    /* 艺术字体状态标签 */
    #artistic_status_label {
        font-family: "Arial Rounded MT Bold", "Segoe UI", "Microsoft YaHei";
        font-size: 14px;
        font-weight: bold;
        color: #2c3e50;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #f8f9fa, stop:1 #e9ecef);
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 6px 15px;
        margin-right: 8px;
        min-width: 120px;
        text-align: center;
    }
    #artistic_status_label[error="true"] {
        color: #721c24;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #f8d7da, stop:1 #f5c6cb);
        border-color: #f5c6cb;
        text-shadow: none;
    }

    #drop_label {
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 20px;
        font-size: 16px;
        color: #6c757d;
        background-color: rgba(248, 249, 250, 0.7);
    }
    #display_frame {
        border: 1px solid #d1d9e6;
        border-radius: 4px;
    }
"""
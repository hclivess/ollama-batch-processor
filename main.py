import sys
import os
import asyncio
import json
import aiohttp
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLineEdit, QPushButton, QLabel, 
    QProgressBar, QFileDialog, QComboBox, QMessageBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox, QTabWidget,
    QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont
from qasync import QEventLoop, asyncSlot

from Translator import OllamaProcessor


class ModularProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load configuration
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config.json: {e}")
            sys.exit(1)
        
        # Set window properties from config
        app_config = self.config.get('app', {})
        self.setWindowTitle(app_config.get('title', 'Ollama Processor'))
        window_size = app_config.get('window_size', {'width': 1100, 'height': 900})
        self.setMinimumSize(window_size['width'], window_size['height'])
        
        # Processor instance
        self.processor = OllamaProcessor(
            ollama_host=app_config.get('ollama_host', 'http://localhost:11434'),
            config=self.config
        )
        
        # Store operation widgets and chunking widgets
        self.operation_widgets = {}
        self.chunking_widgets = {}
        self.available_models = []
        
        # Initialize pipeline order based on config
        operations = self.config.get('operations', {})
        self.pipeline_order = sorted(operations.keys(), 
                                     key=lambda x: operations[x].get('order', 999))
        
        self.setup_ui()
        self.connect_signals()
        self.apply_styles()
        
        self.input_files = []
        self.output_directory = None
        self.translation_start_time = None
        self.current_file_index = 0
        self.total_files = 0
        
        # Schedule model fetch after event loop starts
        QTimer.singleShot(100, lambda: asyncio.ensure_future(self.fetch_models()))

    async def fetch_models(self):
        """Fetch available models from Ollama API"""
        try:
            ollama_host = self.config['app'].get('ollama_host', 'http://localhost:11434')
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_host}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.available_models = [model['name'] for model in data.get('models', [])]
                        
                        if self.available_models:
                            # Update all model combos
                            for op_id, widgets in self.operation_widgets.items():
                                if 'model_combo' in widgets:
                                    combo = widgets['model_combo']
                                    combo.clear()
                                    combo.addItem("(Use first operation's model)", None)
                                    combo.addItems(self.available_models)
                                    combo.setCurrentIndex(1)
                            
                            self.log_message(f"✓ Loaded {len(self.available_models)} models from Ollama")
                        else:
                            self.log_message("⚠️ No models found in Ollama")
                    else:
                        self.log_message(f"⚠️ Failed to fetch models: HTTP {response.status}")
        except Exception as e:
            self.log_message(f"⚠️ Could not connect to Ollama: {e}")
            # Add some default models as fallback
            self.available_models = [
                "mistral:latest",
                "llama3.2:latest",
                "qwen2.5:latest"
            ]
            for op_id, widgets in self.operation_widgets.items():
                if 'model_combo' in widgets:
                    combo = widgets['model_combo']
                    combo.clear()
                    combo.addItem("(Use first operation's model)", None)
                    combo.addItems(self.available_models)
                    combo.setCurrentIndex(1)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left side - Pipeline Order and Chunking
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # Title
        title_label = QLabel(self.config['app']['title'])
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)

        # Pipeline Order Group
        pipeline_group = QGroupBox("📋 Pipeline Order")
        pipeline_layout = QVBoxLayout()
        
        info_label = QLabel("Drag to reorder, or use buttons.\nOnly enabled operations will run.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10pt;")
        pipeline_layout.addWidget(info_label)
        
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.setDefaultDropAction(Qt.MoveAction)
        self.pipeline_list.model().rowsMoved.connect(self.on_pipeline_reordered)
        pipeline_layout.addWidget(self.pipeline_list)
        
        # Populate pipeline list
        operations = self.config.get('operations', {})
        for op_id in self.pipeline_order:
            if op_id in operations:
                op_config = operations[op_id]
                icon = op_config.get('tab_icon', '')
                name = op_config.get('tab_name', op_id.title())
                item = QListWidgetItem(f"{icon} {name}")
                item.setData(Qt.UserRole, op_id)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.pipeline_list.addItem(item)
        
        # Order buttons
        btn_layout = QHBoxLayout()
        self.move_up_btn = QPushButton("⬆️ Up")
        self.move_down_btn = QPushButton("⬇️ Down")
        self.move_up_btn.clicked.connect(self.move_operation_up)
        self.move_down_btn.clicked.connect(self.move_operation_down)
        btn_layout.addWidget(self.move_up_btn)
        btn_layout.addWidget(self.move_down_btn)
        pipeline_layout.addLayout(btn_layout)
        
        pipeline_group.setLayout(pipeline_layout)
        left_panel.addWidget(pipeline_group)
        
        # Global Chunking Settings
        chunking_group = QGroupBox("✂️ Chunking Settings (Global)")
        chunking_layout = QVBoxLayout()
        
        # Process entire file checkbox
        self.chunking_widgets['process_entire_file'] = QCheckBox("📄 Process entire file (no chunking)")
        self.chunking_widgets['process_entire_file'].setToolTip("Process the entire file as a single chunk (ignores chunk size and overlap)")
        self.chunking_widgets['process_entire_file'].setChecked(False)
        self.chunking_widgets['process_entire_file'].stateChanged.connect(self.toggle_global_chunking)
        chunking_layout.addWidget(self.chunking_widgets['process_entire_file'])
        
        # Preset combo
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.chunking_widgets['preset_combo'] = QComboBox()
        chunking_config = self.config.get('chunking', {})
        presets = chunking_config.get('presets', [])
        for preset in presets:
            self.chunking_widgets['preset_combo'].addItem(preset['name'])
        self.chunking_widgets['preset_combo'].setCurrentIndex(chunking_config.get('default_preset', 1))
        self.chunking_widgets['preset_combo'].currentIndexChanged.connect(
            lambda idx: self.update_global_chunk_preset(idx, presets)
        )
        preset_layout.addWidget(self.chunking_widgets['preset_combo'])
        chunking_layout.addLayout(preset_layout)
        
        # Chunk size
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk:"))
        self.chunking_widgets['chunk_size'] = QSpinBox()
        self.chunking_widgets['chunk_size'].setRange(500, 50000)
        self.chunking_widgets['chunk_size'].setValue(chunking_config.get('default_chunk_size', 2500))
        self.chunking_widgets['chunk_size'].setSuffix(" chars")
        self.chunking_widgets['chunk_size'].setToolTip("Maximum characters per processing chunk")
        chunk_layout.addWidget(self.chunking_widgets['chunk_size'])
        chunking_layout.addLayout(chunk_layout)
        
        # Overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap:"))
        self.chunking_widgets['overlap'] = QSpinBox()
        self.chunking_widgets['overlap'].setRange(0, 2000)
        self.chunking_widgets['overlap'].setValue(chunking_config.get('default_overlap', 200))
        self.chunking_widgets['overlap'].setSuffix(" chars")
        self.chunking_widgets['overlap'].setToolTip("Context overlap between chunks for consistency")
        overlap_layout.addWidget(self.chunking_widgets['overlap'])
        chunking_layout.addLayout(overlap_layout)
        
        chunking_group.setLayout(chunking_layout)
        left_panel.addWidget(chunking_group)
        
        # File Selection Group
        file_group = QGroupBox("📁 File Selection")
        file_layout = QVBoxLayout()
        
        # Input files
        input_header = QHBoxLayout()
        input_header.addWidget(QLabel("Input Files:"))
        input_header.addStretch()
        self.add_files_btn = QPushButton("➕ Add Files")
        self.add_files_btn.clicked.connect(self.add_input_files)
        self.clear_files_btn = QPushButton("🗑️ Clear All")
        self.clear_files_btn.clicked.connect(self.clear_input_files)
        input_header.addWidget(self.add_files_btn)
        input_header.addWidget(self.clear_files_btn)
        file_layout.addLayout(input_header)
        
        # File list
        self.input_files_list = QListWidget()
        self.input_files_list.setMaximumHeight(120)
        self.input_files_list.setSelectionMode(QListWidget.ExtendedSelection)
        file_layout.addWidget(self.input_files_list)
        
        # File list buttons
        list_btn_layout = QHBoxLayout()
        self.remove_selected_btn = QPushButton("➖ Remove Selected")
        self.remove_selected_btn.clicked.connect(self.remove_selected_files)
        list_btn_layout.addWidget(self.remove_selected_btn)
        list_btn_layout.addStretch()
        file_layout.addLayout(list_btn_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        output_label.setMinimumWidth(90)
        output_layout.addWidget(output_label)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Same as input files (auto)")
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("📂 Choose")
        self.output_dir_btn.clicked.connect(self.select_output_directory)
        self.output_dir_btn.setMinimumWidth(100)
        output_layout.addWidget(self.output_dir_edit, 3)
        output_layout.addWidget(self.output_dir_btn, 1)
        file_layout.addLayout(output_layout)
        
        # File info label
        self.file_info_label = QLabel("No files selected")
        file_layout.addWidget(self.file_info_label)
        
        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)
        
        left_panel.addStretch()
        
        # Right side - Settings and Progress
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # Dynamic Settings Tabs
        settings_group = QGroupBox("⚙️ Processing Settings")
        settings_layout = QVBoxLayout()
        
        self.settings_tabs = QTabWidget()
        
        # Create tabs dynamically from config
        operations = self.config.get('operations', {})
        for op_id in self.pipeline_order:
            if op_id in operations:
                op_config = operations[op_id]
                if op_config.get('enabled', True):
                    tab = self.create_operation_tab(op_id, op_config)
                    icon = op_config.get('tab_icon', '')
                    name = op_config.get('tab_name', op_id.title())
                    self.settings_tabs.addTab(tab, f"{icon} {name}")
        
        settings_layout.addWidget(self.settings_tabs)
        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)

        # Progress Group
        progress_group = QGroupBox("📊 Processing Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(30)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to process. Ensure Ollama is running.")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)
        
        # Stats row
        stats_layout = QHBoxLayout()
        self.time_label = QLabel("Time: --:--")
        self.chunks_label = QLabel("Chunks: 0/0")
        self.speed_label = QLabel("Speed: -- chunks/min")
        
        stats_layout.addWidget(self.time_label)
        stats_layout.addWidget(self.chunks_label)
        stats_layout.addWidget(self.speed_label)
        stats_layout.addStretch()
        progress_layout.addLayout(stats_layout)
        
        progress_group.setLayout(progress_layout)
        right_panel.addWidget(progress_group)

        # Log Group
        log_group = QGroupBox("📝 Activity Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

        # Control Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 Start Processing")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setMinimumHeight(40)
        
        self.stop_btn = QPushButton("🛑 Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        
        btn_layout.addWidget(self.start_btn, 3)
        btn_layout.addWidget(self.stop_btn, 1)
        right_panel.addLayout(btn_layout)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        
        # Timer for elapsed time
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        
        self.log_message("Application started. Ready to process.")
        self.log_message("ℹ️ Progressive saving enabled: Each step will be saved to a separate file")

    def toggle_global_chunking(self, state: int):
        """Enable/disable chunk size and overlap when process_entire_file is toggled"""
        enabled = not bool(state)
        self.chunking_widgets['chunk_size'].setEnabled(enabled)
        self.chunking_widgets['overlap'].setEnabled(enabled)
        self.chunking_widgets['preset_combo'].setEnabled(enabled)

    def update_global_chunk_preset(self, index: int, presets: list):
        """Update chunk size and overlap based on global preset selection"""
        if index < len(presets):
            preset = presets[index]
            chunk_size = preset.get('chunk_size', 2500)
            overlap = preset.get('overlap', 200)
            
            self.chunking_widgets['chunk_size'].blockSignals(True)
            self.chunking_widgets['chunk_size'].setValue(chunk_size)
            self.chunking_widgets['chunk_size'].blockSignals(False)
            
            self.chunking_widgets['overlap'].blockSignals(True)
            self.chunking_widgets['overlap'].setValue(overlap)
            self.chunking_widgets['overlap'].blockSignals(False)

    def on_pipeline_reordered(self):
        """Called when user drags to reorder pipeline"""
        self.update_pipeline_order_from_list()

    def move_operation_up(self):
        """Move selected operation up in the pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row > 0:
            item = self.pipeline_list.takeItem(current_row)
            self.pipeline_list.insertItem(current_row - 1, item)
            self.pipeline_list.setCurrentRow(current_row - 1)
            self.update_pipeline_order_from_list()

    def move_operation_down(self):
        """Move selected operation down in the pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row < self.pipeline_list.count() - 1:
            item = self.pipeline_list.takeItem(current_row)
            self.pipeline_list.insertItem(current_row + 1, item)
            self.pipeline_list.setCurrentRow(current_row + 1)
            self.update_pipeline_order_from_list()

    def update_pipeline_order_from_list(self):
        """Update internal pipeline order from list widget"""
        new_order = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            op_id = item.data(Qt.UserRole)
            new_order.append(op_id)
        self.pipeline_order = new_order
        self.log_message(f"Pipeline order updated: {' → '.join([self.config['operations'][op]['tab_icon'] for op in new_order])}")

    def create_operation_tab(self, op_id: str, op_config: dict) -> QWidget:
        """Dynamically create a tab based on operation config"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Store widgets for this operation
        self.operation_widgets[op_id] = {}
        
        # Description if available
        if 'description' in op_config:
            desc_label = QLabel(op_config['description'])
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(desc_label)
        
        # Create options dynamically
        options = op_config.get('options', {})
        
        for option_id, option_config in options.items():
            option_type = option_config.get('type')
            
            if option_type == 'text':
                row = QHBoxLayout()
                row.addWidget(QLabel(option_config.get('label', option_id)))
                edit = QLineEdit(option_config.get('default', ''))
                edit.setMinimumHeight(24)
                if 'width' in option_config:
                    edit.setMaximumWidth(option_config['width'])
                if 'tooltip' in option_config:
                    edit.setToolTip(option_config['tooltip'])
                row.addWidget(edit)
                row.addStretch()
                layout.addLayout(row)
                self.operation_widgets[op_id][option_id] = edit
                
            elif option_type == 'combo':
                row = QHBoxLayout()
                row.addWidget(QLabel(option_config.get('label', option_id)))
                combo = QComboBox()
                combo_options = option_config.get('options', [])
                for opt in combo_options:
                    combo.addItem(opt.get('name', opt))
                combo.setCurrentIndex(option_config.get('default_index', 0))
                if 'tooltip' in option_config:
                    combo.setToolTip(option_config['tooltip'])
                
                row.addWidget(combo)
                row.addStretch()
                layout.addLayout(row)
                self.operation_widgets[op_id][option_id] = combo
                
            elif option_type == 'spinbox':
                row = QHBoxLayout()
                row.addWidget(QLabel(option_config.get('label', option_id)))
                spinbox = QSpinBox()
                
                # Check if this is a decimal spinbox
                if option_config.get('decimal', False):
                    from PySide6.QtWidgets import QDoubleSpinBox
                    spinbox = QDoubleSpinBox()
                    spinbox.setDecimals(option_config.get('decimals', 1))
                    spinbox.setSingleStep(option_config.get('single_step', 0.1))
                else:
                    spinbox.setSingleStep(option_config.get('step', 1))
                
                spinbox.setRange(option_config.get('min', 0), option_config.get('max', 100000))
                spinbox.setValue(option_config.get('default', 0))
                if 'suffix' in option_config:
                    spinbox.setSuffix(option_config['suffix'])
                if 'tooltip' in option_config:
                    spinbox.setToolTip(option_config['tooltip'])
                row.addWidget(spinbox)
                row.addStretch()
                layout.addLayout(row)
                self.operation_widgets[op_id][option_id] = spinbox
                
            elif option_type == 'checkbox':
                checkbox = QCheckBox(option_config.get('label', option_id))
                checkbox.setChecked(option_config.get('default', False))
                if 'tooltip' in option_config:
                    checkbox.setToolTip(option_config['tooltip'])
                
                layout.addWidget(checkbox)
                self.operation_widgets[op_id][option_id] = checkbox
        
        # Add model selector if operation requires it
        if op_config.get('requires_model', False):
            model_layout = QHBoxLayout()
            model_layout.addWidget(QLabel("Model:"))
            model_combo = QComboBox()
            model_combo.addItem("(Use first operation's model)", None)
            model_combo.setMinimumWidth(150)
            model_layout.addWidget(model_combo)
            model_layout.addStretch()
            layout.addLayout(model_layout)
            self.operation_widgets[op_id]['model_combo'] = model_combo
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def apply_styles(self):
        """Apply global styles to the application"""
        try:
            with open('styles.qss', 'r') as f:
                style_sheet = f.read()
                self.setStyleSheet(style_sheet)
        except Exception as e:
            print(f"Warning: Could not load styles.qss: {e}")

    def connect_signals(self):
        """Connects the processor's signals to the GUI's slots."""
        self.processor.processing_progress.connect(self.update_progress)
        self.processor.processing_finished.connect(self.processing_finished)
        self.processor.processing_error.connect(self.display_error)
        self.processor.step_status.connect(self.update_step_status)
        self.processor.step_saved.connect(self.on_step_saved)
    
    @Slot(str)
    def on_step_saved(self, file_path: str):
        """Called when a processing step is saved"""
        filename = os.path.basename(file_path)
        self.log_message(f"💾 Step saved: {filename}")
    
    @Slot()
    def add_input_files(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Input Files", 
            "", 
            "Text Files (*.txt);;All Files (*)"
        )
        if file_names:
            for file_name in file_names:
                if file_name not in self.input_files:
                    self.input_files.append(file_name)
                    self.input_files_list.addItem(os.path.basename(file_name))
            
            self.update_file_info()
            self.log_message(f"Added {len(file_names)} file(s). Total: {len(self.input_files)}")
    
    @Slot()
    def clear_input_files(self):
        self.input_files.clear()
        self.input_files_list.clear()
        self.file_info_label.setText("No files selected")
        self.log_message("Cleared all input files")
    
    @Slot()
    def remove_selected_files(self):
        selected_items = self.input_files_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.input_files_list.row(item)
            self.input_files_list.takeItem(row)
            if row < len(self.input_files):
                removed_file = self.input_files.pop(row)
                self.log_message(f"Removed: {os.path.basename(removed_file)}")
        
        self.update_file_info()
    
    @Slot()
    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        if directory:
            self.output_directory = directory
            self.output_dir_edit.setText(directory)
            self.log_message(f"Output directory: {directory}")
    
    def update_file_info(self):
        """Update file info label with statistics"""
        if not self.input_files:
            self.file_info_label.setText("No files selected")
            return
        
        total_size = 0
        total_words = 0
        total_chars = 0
        
        for file_path in self.input_files:
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_words += len(content.split())
                    total_chars += len(content)
            except:
                pass
        
        size_kb = total_size / 1024
        self.file_info_label.setText(
            f"📄 {len(self.input_files)} file(s) | {size_kb:.1f} KB | {total_words:,} words | {total_chars:,} characters"
        )

    @asyncSlot() 
    async def start_processing(self):
        self.log_message("=" * 60)
        self.log_message("🚀 START PROCESSING CLICKED")
        self.log_message("=" * 60)
        
        if not self.input_files:
            self.log_message("❌ ERROR: No input files selected")
            QMessageBox.warning(self, "Warning", "Please select at least one input file.")
            return

        self.log_message(f"✓ Input files: {len(self.input_files)}")
        for i, f in enumerate(self.input_files, 1):
            self.log_message(f"  {i}. {os.path.basename(f)}")
        
        self.progress_bar.setValue(0)
        self.progress_label.setText("Initializing processing...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.log_message("⚙️ Reading chunking settings...")
        # Get global chunking settings
        global_chunk_size = self.chunking_widgets['chunk_size'].value()
        global_overlap = self.chunking_widgets['overlap'].value()
        process_entire_file = self.chunking_widgets['process_entire_file'].isChecked()
        
        self.log_message(f"✓ Chunk size: {global_chunk_size}")
        self.log_message(f"✓ Overlap: {global_overlap}")
        self.log_message(f"✓ Process entire file: {process_entire_file}")
        
        # Override if process entire file is checked
        if process_entire_file:
            global_chunk_size = -1
            global_overlap = 0
            self.log_message("→ Overriding: Will process entire file")
        
        self.log_message("🔨 Building pipeline...")
        # Build pipeline from UI order
        pipeline = []
        operations = self.config.get('operations', {})
        first_model = None
        
        # Get the order from the list widget
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            op_id = item.data(Qt.UserRole)
            
            # Skip if not checked/enabled
            if item.checkState() != Qt.Checked:
                continue
            
            op_config = operations.get(op_id)
            if not op_config or not op_config.get('enabled', True):
                continue
            
            widgets = self.operation_widgets.get(op_id, {})
            op_settings = {
                'operation_id': op_id, 
                'config': op_config,
                'chunk_size': global_chunk_size,
                'overlap': global_overlap
            }
            
            # Check if this operation should be executed
            should_execute = False
            
            # Collect all option values
            for option_id, widget in widgets.items():
                if option_id == 'model_combo':
                    continue
                
                if isinstance(widget, QLineEdit):
                    op_settings[option_id] = widget.text()
                elif isinstance(widget, QComboBox):
                    op_settings[option_id] = widget.currentIndex()
                elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    op_settings[option_id] = widget.value()
                elif isinstance(widget, QCheckBox):
                    checked = widget.isChecked()
                    op_settings[option_id] = checked
                    if checked:
                        should_execute = True
            
            # Special handling for target_tone combo box in paraphrase operation
            if op_id == 'paraphrase' and 'target_tone' in widgets:
                tone_index = op_settings.get('target_tone', 0)
                tone_options = op_config['options']['target_tone']['options']
                
                if tone_index > 0 and tone_index < len(tone_options):
                    tone_value = tone_options[tone_index].get('value')
                    
                    # Map tone selection to the appropriate sub-operation
                    tone_mapping = {
                        'formal': 'adjust_tone_formal',
                        'casual': 'adjust_tone_casual',
                        'professional': 'adjust_tone_professional',
                        'conversational': 'adjust_tone_conversational'
                    }
                    
                    if tone_value in tone_mapping:
                        sub_op = tone_mapping[tone_value]
                        op_settings[sub_op] = True
                        should_execute = True
            
            # Get model selection
            if 'model_combo' in widgets:
                combo = widgets['model_combo']
                
                if combo.currentIndex() == 0:
                    # "(Use first operation's model)" selected
                    if first_model is None:
                        # This IS the first model, use the next selection
                        if combo.count() > 1:
                            op_settings['model'] = combo.itemText(1)
                            first_model = op_settings['model']
                        else:
                            op_settings['model'] = self.available_models[0] if self.available_models else 'mistral:latest'
                            first_model = op_settings['model']
                    else:
                        op_settings['model'] = first_model
                else:
                    # Specific model selected
                    op_settings['model'] = combo.currentText()
                    if first_model is None:
                        first_model = op_settings['model']
            
            # For translation, always execute. For others, check if any sub-operation is enabled
            if op_id == 'translation' or should_execute:
                pipeline.append(op_settings)
        
        if not pipeline:
            QMessageBox.warning(self, "Warning", "No operations enabled. Please check at least one operation in the pipeline list.")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return
        
        # Start timer
        self.translation_start_time = asyncio.get_event_loop().time()
        self.timer.start(1000)
        
        self.total_files = len(self.input_files)
        self.current_file_index = 0
        
        self.log_message("=" * 50)
        self.log_message(f"Starting batch processing: {self.total_files} file(s)")
        self.log_message(f"Pipeline: {len(pipeline)} operation(s)")
        
        if process_entire_file:
            self.log_message("📄 Mode: Processing entire file (no chunking)")
        else:
            self.log_message(f"✂️ Chunking: {global_chunk_size} chars, {global_overlap} overlap")
        
        for op_settings in pipeline:
            op_id = op_settings['operation_id']
            op_config = op_settings['config']
            self.log_message(f"✓ {op_config.get('tab_icon', '')} {op_config.get('tab_name', op_id)}")
            if 'model' in op_settings:
                self.log_message(f"  Model: {op_settings['model']}")
        
        self.log_message("📁 Progressive file saving: Each step will create its own file")
        self.log_message("=" * 50)
        
        # Process each file
        for idx, input_file in enumerate(self.input_files):
            self.current_file_index = idx + 1
            
            # Determine output path
            if self.output_directory:
                output_file = os.path.join(
                    self.output_directory,
                    os.path.basename(input_file).replace('.txt', '_processed.txt')
                )
            else:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_processed{ext}"
            
            self.log_message(f"\n📄 Processing file {self.current_file_index}/{self.total_files}: {os.path.basename(input_file)}")
            self.progress_label.setText(f"File {self.current_file_index}/{self.total_files}: {os.path.basename(input_file)}")
            
            print(f"\n[DEBUG] ===== STARTING FILE PROCESSING =====")
            print(f"[DEBUG] File: {input_file}")
            print(f"[DEBUG] Output: {output_file}")
            print(f"[DEBUG] Pipeline operations: {len(pipeline)}")
            for op in pipeline:
                print(f"[DEBUG]   - {op['operation_id']}: model={op.get('model', 'N/A')}")
            
            # Process this file
            print(f"[DEBUG] Calling processor.process_pipeline()...")
            await self.processor.process_pipeline(
                input_file,
                output_file,
                pipeline
            )
        
        # Processing complete for all files
        if self.processor.is_running:
            self.timer.stop()
            elapsed = asyncio.get_event_loop().time() - self.translation_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            self.progress_label.setText(
                f"✅ Batch Complete! {self.total_files} file(s) processed. Time: {minutes:02d}:{seconds:02d}"
            )
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            self.log_message("=" * 50)
            self.log_message(f"✅ Batch processing completed!")
            self.log_message(f"Files processed: {self.total_files}")
            self.log_message(f"Total time: {minutes:02d}:{seconds:02d}")
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Batch processing complete!\n\nFiles: {self.total_files}\nTime: {minutes:02d}:{seconds:02d}"
            )

    @Slot(int, int, str)
    def update_progress(self, current: int, total: int, phase: str):
        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        self.progress_label.setText(f"{phase}: {current}/{total} ({percentage}%)")
        self.chunks_label.setText(f"Chunks: {current}/{total}")
        
        # Calculate speed
        if self.translation_start_time and current > 0:
            elapsed = asyncio.get_event_loop().time() - self.translation_start_time
            speed = (current / elapsed) * 60
            self.speed_label.setText(f"Speed: {speed:.1f} chunks/min")
        
        if current % 5 == 0 or current == total:
            self.log_message(f"Progress: {current}/{total} chunks ({percentage}%)")

    @Slot(str)
    def update_step_status(self, status: str):
        """Update the UI when step status changes"""
        self.progress_label.setText(status)
        self.log_message(status)

    def update_elapsed_time(self):
        """Update the elapsed time display"""
        if self.translation_start_time:
            elapsed = asyncio.get_event_loop().time() - self.translation_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.time_label.setText(f"Time: {minutes:02d}:{seconds:02d}")

    @Slot(str)
    def processing_finished(self, output_path: str):
        """Called when a single file finishes processing"""
        # Don't stop timer or show final message if we're in batch mode
        if self.current_file_index < self.total_files:
            self.log_message(f"✅ Completed: {os.path.basename(output_path)}")
            return
        
        # This was the last file
        self.timer.stop()
        elapsed = asyncio.get_event_loop().time() - self.translation_start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        self.progress_label.setText(
            f"✅ Processing Complete! Time: {minutes:02d}:{seconds:02d}"
        )
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_message("=" * 50)
        self.log_message(f"✅ Processing completed successfully!")
        self.log_message(f"Total time: {minutes:02d}:{seconds:02d}")
        self.log_message(f"Output: {os.path.basename(output_path)}")
        
        # List all step files created for the last file
        base, ext = os.path.splitext(output_path)
        self.log_message("📁 Step files created:")
        
        step_files = []
        for filename in os.listdir(os.path.dirname(output_path) or '.'):
            if filename.startswith(os.path.basename(base) + "_step_"):
                step_files.append(filename)
        
        for step_file in sorted(step_files):
            self.log_message(f"  • {step_file}")
    
    @Slot(str)
    def display_error(self, message: str):
        self.timer.stop()
        self.progress_label.setText(f"❌ Error: {message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_message(f"❌ ERROR: {message}")
        QMessageBox.critical(self, "Error", message)

    @Slot()
    def stop_processing(self):
        self.timer.stop()
        self.processor.stop_processing()
        self.progress_label.setText("🛑 Processing stopped by user.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_message("🛑 Processing stopped by user")

    def log_message(self, message: str):
        """Add a message to the activity log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        self.processor.stop_processing()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = ModularProcessorApp()
    window.show()
    
    with loop:
        sys.exit(loop.run_forever())
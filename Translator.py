import asyncio
from ollama import AsyncClient, ResponseError
import re
from PySide6.QtCore import QObject, Signal
import os


class TextChunker:
    """Handles intelligent text chunking with boundary detection"""
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = 3000, overlap: int = 300) -> list:
        """
        Splits text into chunks with smart boundary detection.        
        Returns list of tuples: (chunk_text, overlap_context, is_first_chunk)       
        If max_chars is -1, treats the entire text as a single chunk (no chunking).
        """
        chunks = []
        text_length = len(text)
        
        # Handle whole file mode (no chunking)
        if max_chars == -1:
            return [(text, "", True)]
        
        if text_length <= max_chars:
            return [(text, "", True)]
        
        position = 0
        previous_end = ""
        
        while position < text_length:
            chunk_end = min(position + max_chars, text_length)
            
            if chunk_end < text_length:
                search_start = max(position, chunk_end - 200)
                search_text = text[search_start:chunk_end + 100]
                
                # Try to find sentence endings
                sentence_breaks = []
                for match in re.finditer(r'[.!?]\s+', search_text):
                    actual_pos = search_start + match.end()
                    if position < actual_pos <= chunk_end + 100:
                        sentence_breaks.append(actual_pos)
                
                for match in re.finditer(r'[.!?]["»"]\s+', search_text):
                    actual_pos = search_start + match.end()
                    if position < actual_pos <= chunk_end + 100:
                        sentence_breaks.append(actual_pos)
                
                if sentence_breaks:
                    best_break = max([b for b in sentence_breaks if b <= chunk_end], 
                                     default=sentence_breaks[0])
                    chunk_end = best_break
                else:
                    # Try paragraph breaks
                    para_breaks = []
                    for match in re.finditer(r'\n\s*\n', search_text):
                        actual_pos = search_start + match.end()
                        if position < actual_pos <= chunk_end + 100:
                            para_breaks.append(actual_pos)
                    
                    if para_breaks:
                        best_break = max([b for b in para_breaks if b <= chunk_end],
                                         default=para_breaks[0])
                        chunk_end = best_break
                    else:
                        # Break at space
                        space_pos = text.rfind(' ', chunk_end - 100, chunk_end)
                        if space_pos > position:
                            chunk_end = space_pos + 1
            
            chunk_text = text[position:chunk_end].strip()
            is_first = (position == 0)
            chunks.append((chunk_text, previous_end, is_first))
            
            context_size = min(overlap, len(chunk_text))
            if len(chunk_text) > context_size:
                previous_end = chunk_text[-context_size:]
            else:
                previous_end = chunk_text
            
            position = chunk_end
        
        return chunks


class OllamaProcessor(QObject):
    # Signals for communicating with the GUI
    processing_progress = Signal(int, int, str)  # (current, total, phase)
    processing_finished = Signal(str)            # (output_path)
    processing_error = Signal(str)               # (error_message)
    step_status = Signal(str)                    # (status_message)
    step_saved = Signal(str)                     # (file_path)

    def __init__(self, ollama_host='http://localhost:11434', config=None, parent=None):
        super().__init__(parent)
        self.client = AsyncClient(host=ollama_host)
        self.is_running = False
        self.config = config or {}
        self.output_base_path = None
        self.chunker = TextChunker()

    def generate_step_filename(self, step_name: str) -> str:
        """Generate filename for a specific processing step"""
        if not self.output_base_path:
            return None
        
        base, ext = os.path.splitext(self.output_base_path)
        step_file = f"{base}_step_{step_name}{ext}"
        return step_file

    def save_step(self, content: str, step_name: str):
        """Save content for a specific processing step"""
        try:
            step_file = self.generate_step_filename(step_name)
            if step_file:
                with open(step_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.step_saved.emit(step_file)
        except Exception as e:
            self.processing_error.emit(f"Error saving step {step_name}: {e}")

    async def process_with_llm(self, text: str, system_prompt: str, user_prompt: str, 
                               model: str, temperature: float = 0.3) -> str:
        """Generic LLM processing function"""
        # Ensure model is a string
        if not isinstance(model, str):
            model = str(model)
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        # Diagnostic logging
        print(f"[DEBUG] ===== OLLAMA CALL START =====")
        print(f"[DEBUG] Model: '{model}' (type: {type(model)})")
        print(f"[DEBUG] Text length: {len(text)} chars")
        print(f"[DEBUG] Temperature: {temperature}")
        print(f"[DEBUG] System prompt length: {len(system_prompt)} chars")
        print(f"[DEBUG] User prompt length: {len(user_prompt)} chars")
        
        try:
            # No timeout - let it run as long as needed
            response = await self.client.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'top_p': 0.9,
                    'num_predict': -1,
                }
            )
            
            # Handle both dict and object responses from ollama library
            print(f"[DEBUG] Response type: {type(response)}")
            
            try:
                if isinstance(response, dict):
                    result = response['message']['content'].strip()
                elif hasattr(response, 'message'):
                    # Response is an object (ChatResponse)
                    if hasattr(response.message, 'content'):
                        result = response.message.content.strip()
                    else:
                        result = str(response.message).strip()
                else:
                    # Fallback: try to convert to string
                    result = str(response).strip()
            except (KeyError, AttributeError) as e:
                print(f"[DEBUG] Error accessing response: {e}")
                print(f"[DEBUG] Response structure: {dir(response)}")
                raise Exception(f"Cannot parse Ollama response: {type(response)}. Error: {e}")
            
            print(f"[DEBUG] Got response: {len(result)} chars")
            print(f"[DEBUG] ===== OLLAMA CALL SUCCESS =====")
            result
            
            # Remove common unwanted prefixes
            unwanted_prefixes = [
                "here is the translation:",
                "translation:",
                "here's the translation:",
                "translated text:",
                "here is the complete translation:",
                "complete translation:",
                "here is the processed text:",
                "processed text:",
            ]
            
            result_lower = result.lower()
            for prefix in unwanted_prefixes:
                if result_lower.startswith(prefix):
                    result = result[len(prefix):].strip()
                    break
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"⏱️ Timeout: Ollama took too long to respond (>2 min). Check if model '{model}' is loaded."
            print(f"[DEBUG] ===== TIMEOUT ERROR =====")
            print(f"[DEBUG] Model: {model}")
            self.processing_error.emit(error_msg)
            return f"[TIMEOUT ERROR: {text[:100]}...]"
        except ConnectionError as e:
            error_msg = f"🔌 Connection Error: Cannot connect to Ollama. Is it running?\nDetails: {e}"
            print(f"[DEBUG] ===== CONNECTION ERROR =====")
            print(f"[DEBUG] {e}")
            self.processing_error.emit(error_msg)
            return f"[CONNECTION ERROR: {text[:100]}...]"
        except ResponseError as e:
            error_msg = f"❌ Ollama Error: {e.error}"
            print(f"[DEBUG] ===== RESPONSE ERROR =====")
            print(f"[DEBUG] {e.error}")
            self.processing_error.emit(error_msg)
            return f"[PROCESSING FAILED: {text[:100]}...]"
        except Exception as e:
            error_msg = f"❌ Unexpected Error: {type(e).__name__}: {str(e)}"
            print(f"[DEBUG] ===== UNEXPECTED ERROR =====")
            print(f"[DEBUG] Type: {type(e).__name__}")
            print(f"[DEBUG] Message: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            self.processing_error.emit(error_msg)
            return f"[ERROR: {text[:100]}...]"

    def deduplicate_paragraphs(self, text: str) -> str:
        """Remove duplicate paragraphs that may occur at chunk boundaries"""
        paragraphs = text.split('\n\n')
        seen = set()
        deduplicated = []
        
        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue
            
            para_normalized = ' '.join(para_stripped.lower().split())
            
            if para_normalized not in seen:
                seen.add(para_normalized)
                deduplicated.append(para_stripped)
        
        return '\n\n'.join(deduplicated)

    async def execute_translation(self, text: str, op_settings: dict) -> str:
        """Execute translation operation"""
        op_config = op_settings['config']
        
        model = op_settings.get('model', 'mistral:latest')
        
        src_lang = op_settings.get('source_language', 'English')
        target_lang = op_settings.get('target_language', 'Czech')
        chunk_size = op_settings.get('chunk_size', 2500)
        overlap = op_settings.get('overlap', 200)
        deduplicate = op_settings.get('deduplicate', True)
        
        prompts = op_config.get('prompts', {})
        
        # Create chunks using abstracted chunker
        chunks = self.chunker.chunk_text(text, chunk_size, overlap)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return text
        
        # Status update
        self.step_status.emit(f"📝 Translation: {src_lang} → {target_lang} | {total_chunks} chunks | Model: {model}")
        
        translated_parts = []
        last_translation = ""
        
        for i, (chunk, context, is_first) in enumerate(chunks):
            if not self.is_running:
                break
            
            print(f"[DEBUG] Processing chunk {i+1}/{total_chunks}")
            
            # Build prompts
            if not is_first and context:
                context_snippet = context[-150:] if len(context) > 150 else context
                system_prompt = prompts.get('system_continuation', '').format(
                    src_lang=src_lang, target_lang=target_lang
                )
                user_prompt = prompts.get('user_continuation', '').format(
                    src_lang=src_lang, 
                    target_lang=target_lang,
                    context_snippet=context_snippet, 
                    chunk=chunk
                )
            else:
                system_prompt = prompts.get('system_first', '').format(
                    src_lang=src_lang, target_lang=target_lang
                )
                user_prompt = prompts.get('user_first', '').format(
                    src_lang=src_lang, 
                    target_lang=target_lang, 
                    chunk=chunk
                )
            
            # Process chunk
            translated_chunk = await self.process_with_llm(
                chunk, system_prompt, user_prompt, model, temperature=0.3
            )
            
            translated_parts.append(translated_chunk)
            
            # Save progress
            current_translation = '\n\n'.join(translated_parts)
            self.save_step(current_translation, f"{op_config.get('step_name', 'translation')}_progress")
            
            # Update context
            context_size = min(overlap, len(translated_chunk))
            last_translation = translated_chunk[-context_size:] if len(translated_chunk) > context_size else translated_chunk
            
            # Emit progress
            phase = f"Translating ({src_lang} → {target_lang})"
            self.processing_progress.emit(i + 1, total_chunks, phase)
            
            await asyncio.sleep(0.1)
        
        result = '\n\n'.join(translated_parts)
        
        # Deduplicate if enabled
        if deduplicate:
            result = self.deduplicate_paragraphs(result)
        
        return result

    def build_combined_prompt(self, op_settings: dict, enabled_sub_ops: list) -> tuple:
        """
        Build a single combined system and user prompt from multiple enabled sub-operations.
        This is the key optimization - instead of processing text multiple times,
        we combine all instructions into one prompt.
        """
        op_config = op_settings['config']
        sub_ops = op_config.get('sub_operations', {})
        
        if not enabled_sub_ops:
            return "", ""
        
        # Start with a base system prompt
        system_parts = [
            "You are a professional content rewriter and editor. You will perform MULTIPLE tasks on the provided text in a SINGLE pass.",
            "\nYour tasks are:"
        ]
        
        user_parts = []
        task_descriptions = []
        
        # Build instructions for each enabled sub-operation
        for idx, sub_op_id in enumerate(enabled_sub_ops, 1):
            if sub_op_id not in sub_ops:
                continue
            
            sub_op_config = sub_ops[sub_op_id]
            
            # Map sub-operation IDs to human-readable task names
            task_name_map = {
                'improve_flow': 'improve flow and readability',
                'simplify_language': 'simplify complex language',
                'remove_idioms': 'replace idioms with literal language',
                'adjust_tone_formal': 'adjust tone to be more formal',
                'adjust_tone_casual': 'adjust tone to be more casual',
                'adjust_tone_professional': 'adjust tone to be more professional',
                'adjust_tone_conversational': 'adjust tone to be more conversational'
            }
            
            task_name = task_name_map.get(sub_op_id, sub_op_id.replace('_', ' '))
            task_descriptions.append(task_name)
            
            # Extract the core instruction from the original system prompt
            original_system = sub_op_config.get('system', '')
            # Remove the common prefixes/suffixes to get just the core task
            core_instruction = original_system.replace(
                "You are a professional content rewriter. Your task is to ", ""
            ).replace(
                "You are a professional content rewriter specializing in ", ""
            ).replace(
                " DO NOT translate or change the language of the text.", ""
            ).replace(
                " Do NOT omit any content, change the meaning, or alter factual information.", ""
            ).replace(
                " Output ONLY the processed text without any explanations.", ""
            ).replace(
                " Output ONLY the rewritten text without any explanations.", ""
            ).replace(
                " Output ONLY the simplified text without any explanations.", ""
            ).split('\n')[0]  # Take first line as main instruction
            
            system_parts.append(f"\n{idx}. {core_instruction}")
        
        # Add common rules
        system_parts.extend([
            "\n\nCRITICAL REQUIREMENTS:",
            "• Perform ALL tasks listed above in a single pass - do not process the text multiple times",
            "• DO NOT translate or change the language of the text",
            "• Do NOT omit any content, change the meaning, or alter factual information",
            "• Preserve all key points, arguments, and details",
            "• All tasks should work together harmoniously in the output",
            "\nOUTPUT FORMAT:",
            "Output ONLY the fully processed text with all tasks applied. No explanations, no meta-comments, no introductory remarks. Start immediately with the processed content."
        ])
        
        system_prompt = ''.join(system_parts)
        
        # Build user prompt
        tasks_list = ', '.join(task_descriptions[:-1]) + (f', and {task_descriptions[-1]}' if len(task_descriptions) > 1 else task_descriptions[0])
        user_prompt = f"Process this text by applying these tasks: {tasks_list}.\n\nText to process:\n\n{{text}}"
        
        return system_prompt, user_prompt

    async def execute_combined_operation(self, text: str, op_settings: dict, enabled_sub_ops: list) -> str:
        """
        Execute multiple sub-operations in a SINGLE pass using a combined prompt.
        This is much faster than executing each operation separately.
        """
        if not enabled_sub_ops:
            return text
        
        op_config = op_settings['config']
        model = op_settings.get('model', 'mistral:latest')
        temperature = op_settings.get('temperature', 0.5)
        
        # Get chunk size and overlap from operation settings
        chunk_size = op_settings.get('chunk_size', 5000)
        overlap = op_settings.get('overlap', 200)
        
        # Build the combined prompt
        system_prompt, user_prompt_template = self.build_combined_prompt(op_settings, enabled_sub_ops)
        
        # Use abstracted chunker with smart boundary detection
        chunks = self.chunker.chunk_text(text, chunk_size, overlap)
        
        processed_chunks = []
        total = len(chunks)
        
        # Create a nice display name for progress
        op_name = op_config.get('tab_name', 'Processing')
        task_count = len(enabled_sub_ops)
        
        # Status update
        tasks_str = ', '.join(enabled_sub_ops)
        self.step_status.emit(f"🔧 {op_name}: {task_count} tasks combined | {total} chunks | Model: {model}")
        
        for idx, (chunk, context, is_first) in enumerate(chunks):
            if not self.is_running:
                break
            
            print(f"[DEBUG] Combined operation chunk {idx+1}/{total}, model: {model}")
            
            # Format prompts
            formatted_user = user_prompt_template.format(text=chunk)
            
            print(f"[DEBUG] About to call process_with_llm...")
            # Process with combined prompt (ONE API call instead of multiple)
            processed = await self.process_with_llm(
                chunk, system_prompt, formatted_user, model, temperature
            )
            processed_chunks.append(processed)
            
            # Emit progress
            phase = f"{op_name} ({task_count} tasks combined)"
            self.processing_progress.emit(idx + 1, total, phase)
            
            await asyncio.sleep(0.05)
        
        result = '\n\n'.join(processed_chunks)
        
        # Apply deduplication
        result = self.deduplicate_paragraphs(result)
        
        return result

    async def process_pipeline(self, input_path: str, output_path: str, pipeline: list):
        """Process the entire pipeline of operations"""
        if self.is_running:
            self.processing_error.emit("A process is already running.")
            return

        self.is_running = True
        self.output_base_path = output_path
        
        # Test Ollama connection first
        try:
            self.step_status.emit("🔍 Testing Ollama connection...")
            print("[DEBUG] Testing Ollama connection...")
            result = await self.client.list()
            print(f"[DEBUG] Connection successful. Models available: {result}")
            self.step_status.emit("✅ Ollama connected successfully")
        except asyncio.TimeoutError:
            self.processing_error.emit("⏱️ Timeout: Cannot connect to Ollama (5 sec timeout). Is Ollama running?")
            self.is_running = False
            return
        except Exception as e:
            self.processing_error.emit(f"🔌 Cannot connect to Ollama: {e}\n\nMake sure Ollama is running!")
            self.is_running = False
            return
        
        # Read input file
        try:
            self.step_status.emit("📖 Reading input file...")
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.step_status.emit(f"✅ Loaded {len(text)} characters")
        except Exception as e:
            self.processing_error.emit(f"File reading error: {e}")
            self.is_running = False
            return
        
        step_counter = 1
        
        for op_settings in pipeline:
            if not self.is_running:
                break
            
            op_id = op_settings['operation_id']
            op_config = op_settings['config']
            
            self.step_status.emit(f"🔄 Starting: {op_config.get('tab_name', op_id)}")
            
            if op_id == 'translation':
                # Execute translation
                text = await self.execute_translation(text, op_settings)
                step_name = op_config.get('step_name', 'translated')
                self.save_step(text, f"{step_counter:02d}_{step_name}")
                step_counter += 1
                
            else:
                # NEW OPTIMIZED APPROACH: Collect all enabled sub-operations for this tab
                enabled_sub_ops = []
                sub_ops = op_config.get('sub_operations', {})
                
                for sub_op_id in sub_ops.keys():
                    if op_settings.get(sub_op_id, False):
                        enabled_sub_ops.append(sub_op_id)
                
                # Execute ALL enabled sub-operations in a SINGLE pass
                if enabled_sub_ops:
                    text = await self.execute_combined_operation(text, op_settings, enabled_sub_ops)
                    
                    # Create a step name that reflects what was done
                    step_names = [sub_ops[sub_op_id].get('step_name', sub_op_id) 
                                 for sub_op_id in enabled_sub_ops]
                    combined_step_name = '_'.join(step_names)
                    
                    self.save_step(text, f"{step_counter:02d}_{combined_step_name}")
                    step_counter += 1
        
        if self.is_running:
            try:
                # Save final output
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                self.processing_finished.emit(output_path)
            except Exception as e:
                self.processing_error.emit(f"File writing error: {e}")
        
        self.is_running = False

    def stop_processing(self):
        """Stop the processing pipeline"""
        self.is_running = False

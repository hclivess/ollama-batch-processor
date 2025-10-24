# Ollama Batch Processor 📚

A powerful PySide6-based GUI application for batch processing text files using Ollama LLM models. Supports translation, audiobook formatting, and intelligent text paraphrasing with customizable pipeline operations.

## Features

- **🌐 Translation**: Professional translation between languages with context preservation
- **🎧 Audiobook Formatting**: Optimize text for text-to-speech systems
- **✍️ Paraphrasing**: Improve flow, simplify language, remove idioms, adjust tone
- **📊 Pipeline Processing**: Chain multiple operations in custom order
- **✂️ Smart Chunking**: Intelligent text splitting with overlap and boundary detection
- **💾 Progressive Saving**: Each pipeline step saved to separate files
- **🔄 Batch Processing**: Process multiple files sequentially
- **🎯 Model Selection**: Use different Ollama models per operation

## Requirements

- Python 3.8+
- Ollama installed and running
- At least one Ollama model installed

## Installation

### 1. Install Ollama

Download and install from [ollama.ai](https://ollama.ai)

```bash
# Install a model
ollama pull mistral
# or
ollama pull llama3.2
ollama pull aya-expanse:32b
```

### 2. Install Python Dependencies

```bash
pip install PySide6 aiohttp qasync ollama
```

### 3. Run the Application

```bash
# Start Ollama server (in separate terminal)
ollama serve

# Run the application
python main.py
```

## Configuration

Edit `config.json` to customize:

- **Ollama host**: Default `http://localhost:11434`
- **Chunking presets**: Adjust chunk sizes and overlap
- **Operation settings**: Modify prompts, icons, defaults
- **UI settings**: Window size, titles

## Usage

### Basic Workflow

1. **Start Ollama**: Run `ollama serve` in a terminal
2. **Launch App**: Run `python main.py`
3. **Add Files**: Click "📁 Add File" and select .txt files
4. **Configure Pipeline**:
   - Check operations to enable (Translation, Audiobook, Paraphrase)
   - Drag operations to reorder
   - Configure each operation's settings
5. **Select Models**: Choose Ollama model for each operation
6. **Set Chunking**: Select preset or enable "Process entire file"
7. **Start Processing**: Click "🚀 START"
8. **Monitor Progress**: Watch Activity Log and progress bar
9. **Access Outputs**: Find processed files and step files in output directory

### Translation

- Set source and target languages
- Handles idioms intelligently
- Maintains consistency across chunks
- Temperature: 0.2-0.4 recommended

### Audiobook Formatting

Enable options:
- **Expand Contractions**: "don't" → "do not"
- **Spell Out Numbers**: "123" → "one hundred twenty-three"
- **Remove Special Characters**: Clean non-standard symbols
- **Add Reading Markers**: Insert TTS-friendly markers

### Paraphrasing

Enable sub-operations:
- **Improve Flow**: Better sentence structure and transitions
- **Simplify Language**: Make complex text accessible
- **Remove Idioms**: Convert figurative to literal language
- **Adjust Tone**: Formal, casual, professional, or conversational

### Chunking Settings

**Presets**:
- Fast (2000/150): Quick processing
- Balanced (2500/200): Default, good quality
- High Context (3000/250): Better continuity
- Large (4000/300): Fewer API calls
- Extra Large (6000/400): Maximum context

**Process Entire File**: Disable chunking for small files (< 2500 chars)

## Output Files

The application creates multiple output files:

```
input.txt                          # Original file
input_step_01_translated.txt       # After translation
input_step_02_audiobook.txt        # After audiobook formatting
input_processed.txt                # Final output
```

## Troubleshooting

### "Nothing Happens" When Clicking Start

**Check:**
1. Is Ollama running? → `ollama serve`
2. Is a model installed? → `ollama list`
3. Did you add input files?
4. Is at least one operation checked?
5. Is a model selected in dropdown?

### Connection Errors

```bash
# Test Ollama
ollama list

# If empty, install a model
ollama pull mistral

# Test inference
ollama run mistral "hello"
```

### Model Not Loading

Large models (30B+) take 1-2 minutes to load on first use. Watch console output for `[DEBUG]` messages.

### Performance Tips

- **Smaller chunks**: Faster processing, less context
- **Larger chunks**: Slower but better quality
- **Combined operations**: More efficient than separate runs
- **Fast models**: Use smaller quantized models for speed
- **GPU**: Ensure Ollama uses GPU for better performance

## Advanced Configuration

### Custom Prompts

Edit `config.json` operation prompts:

```json
{
  "operations": {
    "translation": {
      "prompts": {
        "system_first": "Your custom system prompt...",
        "user_first": "Your custom user prompt..."
      }
    }
  }
}
```

### Temperature Settings

- **Translation**: 0.2-0.4 (deterministic)
- **Paraphrasing**: 0.4-0.6 (creative)
- **Creative Writing**: 0.7-1.0 (very creative)

### Pipeline Order

Operations execute in order from top to bottom. Typical workflows:

1. **Translation → Audiobook**: Translate then optimize for TTS
2. **Paraphrase → Simplify → Remove Idioms**: Multi-step text cleanup
3. **Translation → Paraphrase (tone)**: Translate and adjust formality

## Keyboard Shortcuts

- **Ctrl+O**: Add files
- **Ctrl+S**: Select output directory
- **Ctrl+R**: Start processing
- **Esc**: Stop processing

## Diagnostic Tools

### Test Ollama Connection

```bash
python test_ollama.py
```

Shows detailed connection diagnostics and model availability.

### Debug Mode

Console output shows `[DEBUG]` messages for:
- Model calls with parameters
- Response types and content length
- Error tracebacks
- Pipeline execution flow

## Known Limitations

- Only .txt files supported
- No real-time progress within chunks (model inference time varies)
- Large models require significant RAM/VRAM
- Async operations prevent UI responsiveness during heavy processing

## Tips for Best Results

1. **Pre-process text**: Remove excessive whitespace, fix encoding
2. **Use appropriate models**: Match model size to task complexity
3. **Test with small files first**: Verify settings before batch processing
4. **Monitor first chunk**: Shows model loading time and quality
5. **Enable deduplication**: Removes duplicate paragraphs at chunk boundaries
6. **Save intermediate steps**: Useful for debugging and iterative refinement

## Architecture

- **main.py**: PySide6 GUI and application logic
- **Translator.py**: Ollama API interface and text processing
- **config.json**: Configuration and prompts
- **qasync**: Async event loop integration with Qt

## License

MIT License - feel free to modify and distribute.

## Contributing

Improvements welcome! Focus areas:
- Additional operation types
- Better error recovery
- Real-time streaming output
- Support for other file formats
- Memory optimization for large files

## Credits

Built with:
- [Ollama](https://ollama.ai) - Local LLM inference
- [PySide6](https://doc.qt.io/qtforpython-6/) - Qt GUI framework
- [qasync](https://github.com/CabbageDevelopment/qasync) - Qt async support

---

**Version**: 1.0  
**Last Updated**: October 2025

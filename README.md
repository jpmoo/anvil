# Anvil: AI Model Training and Interaction Platform

Anvil is a comprehensive Python-based application for training and interacting with custom AI language models. It provides a streamlined interface for fine-tuning models using Vast.ai cloud GPU instances and the Axolotl training framework, with support for local model interaction via Ollama.

## Features

### 1. Model Profile Management
- Create and manage multiple model profiles
- Each profile tracks its own training history and versions
- Support for various base models (LLaMA, Mistral, Gemma, Phi, Qwen, etc.)
- Automatic versioning of fine-tuned models

### 2. Training Mode

#### Tab 1: Training Management
- **Queue Management**: View all queued training files grouped by YAML configuration
- **Job Launch**: Launch training jobs on Vast.ai with configurable parameters:
  - GPU selection (name, RAM, quantity)
  - Price limits
  - Disk space requirements
  - Training epochs and learning rate
  - Optional Hugging Face model override
- **Sequential Job Processing**: Multiple training jobs are processed sequentially through a single Vast.ai instance
- **Phase-Based Workflow**:
  - **Phase 1**: Starting Instance - Launch and wait for Vast.ai instance to be ready
  - **Phase 2**: Upload File - Upload training data and configuration files via SSH/SCP
  - **Phase 3**: Do Training - Monitor training progress with real-time status checks
  - **Phase 4**: Finalize - Download trained weights, clean up, and destroy instance
- **Comprehensive Debugging**: 
  - YAML configuration verification at start of Phase 3
  - File processing verification at end of job
  - Terminal output for all operations

#### Tab 2: Training Data Management
- **File Upload**: Upload training data files (.txt, .json, .jsonl, .docx)
- **YAML Configuration Management**:
  - Upload YAML configuration files
  - Attach YAML configs to training data files
  - Group files by YAML for batch training
  - Delete YAML configs (with safety checks)
- **File Queue**: All uploaded files are queued until training is launched
- **Metadata Tracking**: Each file has associated metadata for tracking

### 3. Converse Mode (Interact Tab)
- Real-time chat interface with your models
- Support for both base Ollama models and fine-tuned models
- **Prompt Settings**:
  - Prepend text (invisibly added to all prompts)
  - Conversation summary request option
  - Automatic summary extraction from responses
- Conversation history management
- Model status indicators

### 4. Vast.ai Integration
- Automatic instance management
- SSH/SCP file transfer
- Real-time training monitoring
- Cost tracking (price per hour)
- Instance location information
- Automatic cleanup after training

### 5. Axolotl Training Framework
- Automatic Axolotl configuration generation
- Support for YAML-based training configurations
- LoRA fine-tuning support
- Automatic dataset preparation in JSONL format
- Model mapping from Ollama names to Hugging Face identifiers

## Installation

### Prerequisites
1. **Python 3.11+**
2. **Ollama** installed and running
   - Download from [ollama.ai](https://ollama.ai)
   - Pull a base model: `ollama pull llama2` (or mistral, llama3, gemma3, etc.)
3. **SSH/SCP access** configured (for Vast.ai file transfers)
4. **Vast.ai API key** (get from [vast.ai](https://vast.ai))

### Setup

1. **Clone or download this repository**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure Vast.ai API key:**
   - The API key can be entered in the application sidebar
   - It will be saved securely for future use

4. **Start the application:**
```bash
streamlit run main.py
```

## Usage

### Creating a Model Profile

1. Select "Create New Model Profile" in the sidebar
2. Enter a model name
3. Select a base model from your installed Ollama models
4. Click "Create Model Profile"

### Training a Model

1. **Upload Training Data (Tab 2)**:
   - Upload training files (.txt, .json, .jsonl, .docx)
   - Optionally upload and attach YAML configuration files
   - Files are queued until training is launched

2. **Launch Training (Tab 1)**:
   - Configure training parameters (GPU, price, epochs, learning rate)
   - Click "Launch Training Job"
   - Monitor progress through the 4 phases

3. **Monitor Training**:
   - Phase 1: Wait for instance to be ready
   - Phase 2: Upload files (automatic directory creation, manual upload button)
   - Phase 3: Monitor training progress with status checks
   - Phase 4: Finalize and download weights

### Interacting with Models

1. Navigate to the "Interact" tab
2. Configure prompt settings (optional):
   - Add prepend text for all prompts
   - Enable conversation summary requests
3. Start chatting with your model
4. View conversation history

## Project Structure

```
Anvil/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── run.sh                       # Launch script
├── assets/
│   └── logo.png                # Application logo
├── models/                      # Model profiles and training data
│   └── [model_name]/
│       ├── training/
│       │   ├── queue/          # Queued training files
│       │   └── jobs/            # Training job metadata
│       └── V[version]/          # Versioned fine-tuned models
│           ├── training/       # Training data used
│           └── weights/        # Trained model weights
├── utils/
│   ├── config.py               # Configuration and paths
│   ├── model_manager.py        # Model profile management
│   ├── file_processor.py      # File processing utilities
│   ├── training_data.py        # Training data management
│   ├── vast_ai_client.py      # Vast.ai API client
│   ├── vast_training_manager.py # Training job management
│   ├── axolotl_prep.py         # Axolotl data preparation
│   ├── ollama_client.py        # Ollama integration
│   ├── fine_tuner.py           # Fine-tuning logic
│   ├── fine_tuned_client.py   # Fine-tuned model client
│   └── model_status.py         # Model status queries
└── ui/
    └── pages/
        ├── training_mode.py    # Training Mode interface
        └── converse_mode.py    # Converse Mode interface
```

## Tech Stack

- **Language**: Python 3.11+
- **Front-end**: Streamlit
- **Core Libraries**:
  - `streamlit` - Web interface
  - `transformers`, `datasets`, `torch` - Model training
  - `ollama` - Local model inference
  - `accelerate`, `peft`, `bitsandbytes` - Efficient training
  - `pyyaml` - Configuration management
  - `requests` - API communication
  - `python-docx` - DOCX file support
  - `tinydb` - Lightweight metadata storage

## Training Workflow

1. **File Upload**: Training files are uploaded and queued in Tab 2
2. **YAML Configuration** (optional): YAML configs can be attached to group files
3. **Job Launch**: Training jobs are launched on Vast.ai in Tab 1
4. **Sequential Processing**: Multiple jobs are processed sequentially through one instance
5. **Phase Execution**: Each job goes through 4 phases automatically
6. **Weight Download**: Trained weights are downloaded to local storage
7. **Version Creation**: Each successful training creates a new model version

## YAML Configuration

YAML configuration files allow fine-grained control over training parameters:
- Training hyperparameters (learning rate, batch size, etc.)
- Model-specific settings
- LoRA configuration
- Dataset preparation settings

**Note**: YAML files should NOT include:
- Dataset paths (injected automatically)
- Output directories (injected automatically)
- Base model names (injected automatically)

These are automatically set by the system based on your model profile and job configuration.

## Model Support

Anvil supports a wide range of base models through Ollama:
- **LLaMA**: llama2, llama3, llama3.1, llama3.2
- **Mistral**: mistral, mixtral
- **Gemma**: gemma, gemma2, gemma3
- **Phi**: phi, phi-2, phi3
- **Qwen**: qwen, qwen2
- **CodeLlama**: codellama

## Troubleshooting

### Training Issues
- Check terminal output in each phase for detailed error messages
- Verify SSH connectivity to Vast.ai instances
- Ensure training files are properly formatted
- Check YAML configuration syntax if using custom configs

### Model Issues
- Verify Ollama is running: `ollama list`
- Ensure base model is installed: `ollama pull [model_name]`
- Check model profile configuration in sidebar

### File Upload Issues
- Verify file formats are supported (.txt, .json, .jsonl, .docx)
- Check file size limits
- Ensure proper file encoding (UTF-8)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Training framework
- [Vast.ai](https://vast.ai) - Cloud GPU platform
- [Ollama](https://ollama.ai) - Local model inference
- [Streamlit](https://streamlit.io) - Web framework

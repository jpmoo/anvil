# Anvil ChatBot - Inference Server Setup Companion App

This Electron app is a companion to the main Anvil application. It helps you set up and manage a FastAPI inference server on remote Vast.ai instances with your trained models.

## Features

- **Profile Selection**: Select a model profile from the parent Anvil app
- **Inference Environment Initialization**: Comprehensive setup that tests SSH, installs dependencies, downloads models, uploads adapters, and configures the inference server
- **Version Management**: View all available model versions for the selected profile
- **Inference Server Preparation**: Upload model adapters and prepare FastAPI inference server configuration

## Prerequisites

1. **Node.js and npm** installed on your system
2. **SSH keys** configured for connecting to Vast.ai instances (typically in `~/.ssh/id_rsa`)
3. **Parent Anvil app** with at least one model profile created
4. **FastAPI Inference Server** will be started automatically on the target Vast.ai instance

## Installation

1. Navigate to the ChatBot directory:
   ```bash
   cd ChatBot
   ```

2. Run the setup script:
   ```bash
   ./setup.sh
   ```
   
   Or manually install dependencies:
   ```bash
   npm install
   ```
   
   Or use npm script:
   ```bash
   npm run setup
   ```

## Usage

1. **Start the app** (choose one method):
   ```bash
   # Method 1: Use the run script
   ./run.sh
   
   # Method 2: Use npm script
   npm run run
   # or
   npm start
   ```
   
   The run script will automatically check for dependencies and run setup if needed.

2. **Select a Profile**:
   - Choose a model profile from the dropdown
   - The app will automatically load available versions

3. **Configure SSH**:
   - Enter the SSH host/IP address of your Vast.ai instance
   - Enter the SSH port (default: 22)
   - Click "Initialize Inference Environment" to set up SSH, install dependencies, download models, upload adapters, and start the inference server

4. **Configure Inference Server URL**:
   - Enter the HTTP API URL for your inference server (e.g., `http://your-server-ip:8000`)
   - Click the test button to verify connectivity
   - The app will automatically retrieve authentication tokens if needed

5. **Prepare Inference Server**:
   - Once SSH and URL are tested successfully, click "Prepare Inference Server with Selected Model"
   - The app will:
     - Upload the inference server script to the remote instance
     - Create remote model directories
     - Upload all version adapters
     - Configure and start the FastAPI inference server via supervisor
     - Monitor server startup and model loading

## How It Works

1. **Profile Discovery**: Reads model profiles from `../models/` directory
2. **Version Detection**: Scans for version directories (V1, V2, etc.) and their adapter weights
3. **SSH Connection**: Uses NodeSSH library to connect via SSH (requires SSH keys)
4. **File Upload**: Transfers adapter directories to the remote instance
5. **Inference Server Setup**: Uploads FastAPI server script and configures supervisor to run it
6. **Server Management**: Uses supervisor to manage the inference server process

## Remote Directory Structure

After preparation, the remote instance will have:

```
/workspace/models/{profileName}/
├── V1/
│   └── adapter/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
├── V2/
│   └── adapter/
│       └── ...
└── model_info.json
```

## Inference Server Integration

The app uploads a FastAPI inference server (`utils/inference_server.py`) that automatically loads:
- The base model (specified in the profile)
- All available adapter versions (V1, V2, etc.) from `/workspace/models/{profileName}/V*/adapter`

The server exposes REST API endpoints:
- `GET /health` - Health check endpoint
- `GET /models` - List available models
- `POST /chat` - Chat completion endpoint

The server is managed by supervisor and automatically loads models on startup.

## Troubleshooting

- **SSH Connection Fails**: Ensure SSH keys are configured in Vast.ai account settings
- **Inference Server Not Starting**: Check supervisor logs with `supervisorctl tail -200 inference`
- **No Profiles Found**: Make sure the parent Anvil app has created at least one model profile
- **Upload Fails**: Check disk space on the remote instance and network connectivity
- **Model Loading Fails**: Verify the base model name is correct and accessible from HuggingFace
- **Authentication Errors**: The app automatically retrieves `OPEN_BUTTON_TOKEN` for Basic Auth

## SSH Key Configuration

The app automatically detects and uses SSH keys in the following order:

1. **From parent app config**: If `ssh_key_path` is set in `../app_config.json`, it will use that key
2. **Default locations**: Falls back to common SSH key locations:
   - `~/.ssh/id_rsa`
   - `~/.ssh/id_ed25519`
   - `~/.ssh/id_ecdsa`
   - `~/.ssh/id_dsa`

To specify a custom SSH key path, add it to the parent app's `app_config.json`:

```json
{
  "vast_api_key": "...",
  "hf_token": "...",
  "ssh_key_path": "/path/to/your/ssh/key"
}
```

The app will display which SSH key is being used in the SSH Configuration section.

## Notes

- The app uses the Vast.ai API key from the parent app's `app_config.json` (for future features)
- SSH authentication uses the SSH key from parent app config or default locations
- The app automatically starts and manages the FastAPI inference server via supervisor
- The inference server uses PEFT (Parameter-Efficient Fine-Tuning) to load adapters on top of the base model
- Models are loaded on server startup - first startup may take several minutes for large models


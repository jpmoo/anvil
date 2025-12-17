# Anvil ChatBot - vLLM Setup Companion App

This Electron app is a companion to the main Anvil application. It helps you set up and manage vLLM on remote Vast.ai instances with your trained models.

## Features

- **Profile Selection**: Select a model profile from the parent Anvil app
- **SSH Configuration**: Configure and test SSH connection to your Vast.ai instance
- **Version Management**: View all available model versions for the selected profile
- **vLLM Preparation**: Upload model adapters and prepare vLLM configuration

## Prerequisites

1. **Node.js and npm** installed on your system
2. **SSH keys** configured for connecting to Vast.ai instances (typically in `~/.ssh/id_rsa`)
3. **Parent Anvil app** with at least one model profile created
4. **vLLM** already running on the target Vast.ai instance

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
   - Click "Test SSH Connection" to verify connectivity

4. **Prepare vLLM**:
   - Once SSH is tested successfully, click "Prepare vLLM with Selected Model"
   - The app will:
     - Check if vLLM is running
     - Create remote model directories
     - Upload all version adapters
     - Create preparation scripts and model info files

## How It Works

1. **Profile Discovery**: Reads model profiles from `../models/` directory
2. **Version Detection**: Scans for version directories (V1, V2, etc.) and their adapter weights
3. **SSH Connection**: Uses NodeSSH library to connect via SSH (requires SSH keys)
4. **File Upload**: Transfers adapter directories to the remote instance
5. **vLLM Setup**: Creates scripts and configuration files for vLLM usage

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
├── prepare_vllm.sh
└── model_info.json
```

## vLLM Integration

The app creates a `prepare_vllm.sh` script with example commands for using the models with vLLM. You can use the adapters with vLLM's LoRA support:

```bash
vllm serve {baseModel} \
  --enable-lora \
  --lora-modules adapter1=/workspace/models/{profileName}/V1/adapter \
                 adapter2=/workspace/models/{profileName}/V2/adapter
```

## Troubleshooting

- **SSH Connection Fails**: Ensure SSH keys are configured in Vast.ai account settings
- **vLLM Not Running**: Start vLLM on the instance before running preparation
- **No Profiles Found**: Make sure the parent Anvil app has created at least one model profile
- **Upload Fails**: Check disk space on the remote instance and network connectivity

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
- The app assumes vLLM is already running - it only prepares the model files, not the vLLM service itself


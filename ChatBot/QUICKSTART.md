# Quick Start Guide

## First Time Setup

```bash
cd ChatBot
./setup.sh
```

This will:
- Check for Node.js and npm
- Install all required dependencies
- Verify the installation

## Running the App

```bash
./run.sh
```

Or:

```bash
npm start
```

## What You Need

1. **Node.js 16+** - [Download here](https://nodejs.org/)
2. **SSH keys** configured for Vast.ai (typically `~/.ssh/id_rsa`)
3. **Parent Anvil app** with at least one model profile
4. **vLLM running** on your target Vast.ai instance

## Troubleshooting

### "node_modules not found"
Run `./setup.sh` to install dependencies.

### "Electron not found"
Run `npm install` or `./setup.sh`.

### "Permission denied" on scripts
Make scripts executable:
```bash
chmod +x setup.sh run.sh
```

### SSH connection fails
- Ensure SSH keys are configured in Vast.ai account settings
- Check that the SSH host and port are correct
- Verify the instance is running and accessible
















const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { NodeSSH } = require('node-ssh');
const http = require('http');
const https = require('https');
const { URL } = require('url');

// Get parent app directory (one level up)
const PARENT_APP_DIR = path.resolve(__dirname, '..');
const MODELS_DIR = path.join(PARENT_APP_DIR, 'models');
const APP_CONFIG_FILE = path.join(PARENT_APP_DIR, 'app_config.json');
const CHATBOT_CONFIG_FILE = path.join(__dirname, 'config.json');
const os = require('os');

// Helper function to find SSH key
function findSSHKey() {
  // First, check if SSH key path is in app config
  try {
    if (fs.existsSync(APP_CONFIG_FILE)) {
      const config = JSON.parse(fs.readFileSync(APP_CONFIG_FILE, 'utf8'));
      if (config.ssh_key_path) {
        // Handle both absolute and relative paths
        let keyPath;
        if (path.isAbsolute(config.ssh_key_path)) {
          keyPath = config.ssh_key_path;
        } else {
          // Resolve relative to home directory (for ~/ paths) or parent app directory
          if (config.ssh_key_path.startsWith('~/') || config.ssh_key_path.startsWith('~\\')) {
            keyPath = path.join(os.homedir(), config.ssh_key_path.substring(2));
          } else {
            // Resolve relative to parent app directory
            keyPath = path.resolve(PARENT_APP_DIR, config.ssh_key_path);
          }
        }
        
        if (fs.existsSync(keyPath)) {
          return keyPath;
        }
      }
    }
  } catch (error) {
    console.error('Error reading SSH key from config:', error);
  }
  
  // Fall back to common default locations
  const homeDir = os.homedir();
  const defaultKeys = [
    path.join(homeDir, '.ssh', 'id_rsa'),
    path.join(homeDir, '.ssh', 'id_ed25519'),
    path.join(homeDir, '.ssh', 'id_ecdsa'),
    path.join(homeDir, '.ssh', 'id_dsa'),
  ];
  
  for (const keyPath of defaultKeys) {
    if (fs.existsSync(keyPath)) {
      return keyPath;
    }
  }
  
  return null;
}

let mainWindow;

// Store SSH config and model info after successful preparation
let storedSSHConfig = null;
let storedModelInfo = null;
let storedVLLMUrl = null;
let storedOpenButtonToken = null; // Vast.ai OPEN_BUTTON_TOKEN for Basic Auth

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadFile('index.html');
  
  // Forward renderer console logs to main process terminal (for debugging)
  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    const prefix = level === 0 ? '[RENDERER]' : level === 1 ? '[RENDERER WARN]' : '[RENDERER ERROR]';
    console.log(`${prefix} ${message}`);
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers

// Get list of available profiles
ipcMain.handle('get-profiles', async () => {
  try {
    console.log('[APP] Loading profiles from:', MODELS_DIR);
    if (!fs.existsSync(MODELS_DIR)) {
      console.log('[APP] Models directory does not exist');
      return [];
    }

    const profiles = [];
    const entries = fs.readdirSync(MODELS_DIR, { withFileTypes: true });
    console.log(`[APP] Found ${entries.length} entries in models directory`);

    for (const entry of entries) {
      if (entry.isDirectory()) {
        const metadataPath = path.join(MODELS_DIR, entry.name, 'metadata.json');
        if (fs.existsSync(metadataPath)) {
          try {
            const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            profiles.push({
              name: entry.name,
              baseModel: metadata.base_model || 'unknown',
              createdDate: metadata.created_date || null
            });
            console.log(`[APP] Loaded profile: ${entry.name} (${metadata.base_model || 'unknown'})`);
          } catch (e) {
            console.error(`[APP] Error reading metadata for ${entry.name}:`, e);
          }
        }
      }
    }

    console.log(`[APP] Total profiles loaded: ${profiles.length}`);
    return profiles;
  } catch (error) {
    console.error('[APP] Error getting profiles:', error);
    throw error;
  }
});

// Get versions for a profile
ipcMain.handle('get-profile-versions', async (event, profileName) => {
  try {
    console.log(`[APP] Loading versions for profile: ${profileName}`);
    const profileDir = path.join(MODELS_DIR, profileName);
    if (!fs.existsSync(profileDir)) {
      console.log(`[APP] Profile directory does not exist: ${profileDir}`);
      return [];
    }

    const versions = [];
    const entries = fs.readdirSync(profileDir, { withFileTypes: true });

    for (const entry of entries) {
      if (entry.isDirectory() && entry.name.match(/^V\d+$/)) {
        const versionNum = parseInt(entry.name.substring(1));
        const weightsPath = path.join(profileDir, entry.name, 'weights', 'adapter');
        
        if (fs.existsSync(weightsPath)) {
          versions.push({
            version: versionNum,
            path: weightsPath,
            exists: true
          });
          console.log(`[APP] Found version ${versionNum} at: ${weightsPath}`);
        } else {
          console.log(`[APP] Version ${versionNum} directory exists but no adapter weights found`);
        }
      }
    }

    const sortedVersions = versions.sort((a, b) => a.version - b.version);
    console.log(`[APP] Total versions found: ${sortedVersions.length}`);
    return sortedVersions;
  } catch (error) {
    console.error('[APP] Error getting profile versions:', error);
    throw error;
  }
});

// Get Vast.ai API key
ipcMain.handle('get-vast-api-key', async () => {
  try {
    if (!fs.existsSync(APP_CONFIG_FILE)) {
      return null;
    }

    const config = JSON.parse(fs.readFileSync(APP_CONFIG_FILE, 'utf8'));
    return config.vast_api_key || null;
  } catch (error) {
    console.error('Error reading app config:', error);
    return null;
  }
});

// Get SSH key path
ipcMain.handle('get-ssh-key-path', async () => {
  return findSSHKey();
});

// Load ChatBot configuration
ipcMain.handle('load-config', async () => {
  try {
    if (fs.existsSync(CHATBOT_CONFIG_FILE)) {
      const config = JSON.parse(fs.readFileSync(CHATBOT_CONFIG_FILE, 'utf8'));
      console.log('[CONFIG] Loaded configuration from file');
      return {
        success: true,
        config: config
      };
    } else {
      console.log('[CONFIG] No configuration file found, using defaults');
      return {
        success: true,
        config: {}
      };
    }
  } catch (error) {
    console.error('[CONFIG] Error loading configuration:', error.message);
    return {
      success: false,
      error: error.message,
      config: {}
    };
  }
});

// Save ChatBot configuration
ipcMain.handle('save-config', async (event, config) => {
  try {
    // Ensure we only save the fields we want (no token - it's retrieved during setup)
    // Use explicit checks to preserve 0 values (which are falsy but valid)
    const configToSave = {
      sshHost: config.sshHost !== undefined && config.sshHost !== null ? config.sshHost : '',
      sshPort: config.sshPort !== undefined && config.sshPort !== null ? config.sshPort : 22,
      vllmUrl: config.vllmUrl !== undefined && config.vllmUrl !== null ? config.vllmUrl : '',
      prependedText: config.prependedText !== undefined && config.prependedText !== null ? config.prependedText : '',
      useSummary: config.useSummary !== undefined ? config.useSummary : false,
      conversationSummary: config.conversationSummary !== undefined && config.conversationSummary !== null ? config.conversationSummary : '',
      selectedProfile: config.selectedProfile !== undefined && config.selectedProfile !== null ? config.selectedProfile : '',
      selectedVersion: config.selectedVersion !== undefined && config.selectedVersion !== null ? config.selectedVersion : 'base'
    };
    
    console.log('[CONFIG] Saving configuration to:', CHATBOT_CONFIG_FILE);
    console.log('[CONFIG] Config values:', JSON.stringify(configToSave, null, 2));
    
    // Ensure directory exists
    const configDir = path.dirname(CHATBOT_CONFIG_FILE);
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
      console.log('[CONFIG] Created config directory:', configDir);
    }
    
    fs.writeFileSync(CHATBOT_CONFIG_FILE, JSON.stringify(configToSave, null, 2), 'utf8');
    console.log('[CONFIG] ✓ Configuration saved successfully');
    
    // Verify it was written
    if (fs.existsSync(CHATBOT_CONFIG_FILE)) {
      const verify = JSON.parse(fs.readFileSync(CHATBOT_CONFIG_FILE, 'utf8'));
      console.log('[CONFIG] Verified saved config:', JSON.stringify(verify, null, 2));
    } else {
      console.error('[CONFIG] ✗ Config file was not created!');
    }
    
    return {
      success: true
    };
  } catch (error) {
    console.error('[CONFIG] Error saving configuration:', error.message);
    console.error('[CONFIG] Error stack:', error.stack);
    return {
      success: false,
      error: error.message
    };
  }
});

// Check Caddy Basic Auth configuration
ipcMain.handle('check-caddy-auth', async (event, { host, port, username = 'root' }) => {
  const ssh = new NodeSSH();
  
  try {
    // Get SSH key path
    const sshKeyPath = findSSHKey();
    const connectOptions = {
      host: host,
      port: port || 22,
      username: username,
      readyTimeout: 10000,
    };
    
    if (sshKeyPath) {
      try {
        connectOptions.privateKey = fs.readFileSync(sshKeyPath, 'utf8');
      } catch (error) {
        console.error('[SSH] Error reading SSH key file:', error);
      }
    }
    
    await ssh.connect(connectOptions);
    console.log('[SSH] ✓ Connection established for Caddy check');
    
    // Check common Caddyfile locations
    const caddyfilePaths = [
      '/etc/caddy/Caddyfile',
      '/opt/caddy/Caddyfile',
      '~/.config/caddy/Caddyfile',
      '/root/.config/caddy/Caddyfile',
      '/workspace/Caddyfile',
      './Caddyfile'
    ];
    
    let foundCaddyfile = null;
    let caddyfileContent = null;
    
    for (const caddyfilePath of caddyfilePaths) {
      try {
        const result = await ssh.execCommand(`test -f ${caddyfilePath} && echo "exists" || echo "not_found"`);
        if (result.stdout.trim() === 'exists') {
          foundCaddyfile = caddyfilePath;
          const readResult = await ssh.execCommand(`cat ${caddyfilePath}`);
          caddyfileContent = readResult.stdout;
          break;
        }
      } catch (error) {
        // Continue to next path
      }
    }
    
    if (!foundCaddyfile) {
      // Try to find Caddyfile using find
      try {
        const findResult = await ssh.execCommand('find /etc /opt /workspace /root -name "Caddyfile" 2>/dev/null | head -1');
        if (findResult.stdout.trim()) {
          foundCaddyfile = findResult.stdout.trim();
          const readResult = await ssh.execCommand(`cat ${foundCaddyfile}`);
          caddyfileContent = readResult.stdout;
        }
      } catch (error) {
        // Couldn't find Caddyfile
      }
    }
    
    await ssh.dispose();
    
    if (!foundCaddyfile || !caddyfileContent) {
      return {
        success: false,
        message: 'Could not find Caddyfile on the server',
        details: 'Caddyfile not found in common locations. Basic Auth may be configured elsewhere or Caddy may not be installed.'
      };
    }
    
    // Parse Caddyfile for Basic Auth
    const basicAuthMatch = caddyfileContent.match(/basicauth\s+[^\n]+/gi);
    const basicAuthLines = basicAuthMatch || [];
    
    if (basicAuthLines.length === 0) {
      return {
        success: true,
        message: 'No Basic Auth found in Caddyfile',
        details: `Caddyfile found at ${foundCaddyfile}, but no basicauth directives found. Basic Auth may be configured elsewhere.`,
        caddyfilePath: foundCaddyfile
      };
    }
    
    // Extract username:password hashes
    const authConfigs = [];
    for (const line of basicAuthLines) {
      // Caddy Basic Auth format: basicauth username hash
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 3) {
        authConfigs.push({
          username: parts[1],
          hash: parts[2],
          rawLine: line.trim()
        });
      }
    }
    
    return {
      success: true,
      message: 'Basic Auth found in Caddyfile',
      details: `Found ${authConfigs.length} Basic Auth configuration(s) in ${foundCaddyfile}`,
      caddyfilePath: foundCaddyfile,
      authConfigs: authConfigs,
      note: 'The password is hashed in the Caddyfile. You need the original password to use it. If you don\'t have it, you may need to reset the Basic Auth configuration.'
    };
    
  } catch (error) {
    console.error('[SSH] Error checking Caddy config:', error);
    return {
      success: false,
      message: 'Failed to check Caddy configuration',
      details: error.message
    };
  }
});

// Disable Basic Auth or set up new Basic Auth in Caddy
ipcMain.handle('configure-caddy-auth', async (event, { host, port, username = 'root', action, newUsername, newPassword }) => {
  const ssh = new NodeSSH();
  
  try {
    // Get SSH key path
    const sshKeyPath = findSSHKey();
    const connectOptions = {
      host: host,
      port: port || 22,
      username: username,
      readyTimeout: 10000,
    };
    
    if (sshKeyPath) {
      try {
        connectOptions.privateKey = fs.readFileSync(sshKeyPath, 'utf8');
      } catch (error) {
        console.error('[SSH] Error reading SSH key file:', error);
      }
    }
    
    await ssh.connect(connectOptions);
    console.log('[SSH] ✓ Connection established for Caddy config');
    
    // Find Caddyfile (same logic as check-caddy-auth)
    const caddyfilePaths = [
      '/etc/caddy/Caddyfile',
      '/opt/caddy/Caddyfile',
      '~/.config/caddy/Caddyfile',
      '/root/.config/caddy/Caddyfile',
      '/workspace/Caddyfile',
      './Caddyfile'
    ];
    
    let foundCaddyfile = null;
    let caddyfileContent = null;
    
    for (const caddyfilePath of caddyfilePaths) {
      try {
        const result = await ssh.execCommand(`test -f ${caddyfilePath} && echo "exists" || echo "not_found"`);
        if (result.stdout.trim() === 'exists') {
          foundCaddyfile = caddyfilePath;
          const readResult = await ssh.execCommand(`cat ${caddyfilePath}`);
          caddyfileContent = readResult.stdout;
          break;
        }
      } catch (error) {
        // Continue to next path
      }
    }
    
    if (!foundCaddyfile) {
      // Try to find Caddyfile using find
      try {
        const findResult = await ssh.execCommand('find /etc /opt /workspace /root -name "Caddyfile" 2>/dev/null | head -1');
        if (findResult.stdout.trim()) {
          foundCaddyfile = findResult.stdout.trim();
          const readResult = await ssh.execCommand(`cat ${foundCaddyfile}`);
          caddyfileContent = readResult.stdout;
        }
      } catch (error) {
        // Couldn't find Caddyfile
      }
    }
    
    if (!foundCaddyfile || !caddyfileContent) {
      await ssh.dispose();
      return {
        success: false,
        message: 'Could not find Caddyfile',
        details: 'Cannot modify Caddy configuration without finding the Caddyfile.'
      };
    }
    
    let newCaddyfileContent = caddyfileContent;
    
    if (action === 'disable') {
      // Remove all basicauth lines
      const lines = caddyfileContent.split('\n');
      const filteredLines = lines.filter(line => !line.trim().toLowerCase().startsWith('basicauth'));
      newCaddyfileContent = filteredLines.join('\n');
      
      // Write backup first
      await ssh.execCommand(`cp ${foundCaddyfile} ${foundCaddyfile}.backup.${Date.now()}`);
      
      // Write new content
      const tempLocalFile = path.join(os.tmpdir(), `caddyfile_${Date.now()}`);
      fs.writeFileSync(tempLocalFile, newCaddyfileContent);
      await ssh.putFile(tempLocalFile, foundCaddyfile);
      fs.unlinkSync(tempLocalFile);
      
      // Reload Caddy
      try {
        await ssh.execCommand('systemctl reload caddy || caddy reload --config ' + foundCaddyfile + ' || killall -USR1 caddy');
      } catch (error) {
        console.log('[SSH] Note: Caddy reload command may have failed, but config was updated');
      }
      
      await ssh.dispose();
      
      return {
        success: true,
        message: 'Basic Auth disabled',
        details: `Removed Basic Auth from ${foundCaddyfile}. Backup saved. Caddy should reload automatically.`
      };
      
    } else if (action === 'setup' && newUsername && newPassword) {
      // Generate bcrypt hash for new password (Caddy uses bcrypt)
      // We'll use caddy hash-password command if available, or provide instructions
      let passwordHash = null;
      
      try {
        const hashResult = await ssh.execCommand(`caddy hash-password --plaintext "${newPassword}"`);
        if (hashResult.stdout.trim()) {
          passwordHash = hashResult.stdout.trim();
        }
      } catch (error) {
        // caddy hash-password might not be available, try alternative
        try {
          const hashResult = await ssh.execCommand(`echo -n "${newPassword}" | caddy hash-password`);
          if (hashResult.stdout.trim()) {
            passwordHash = hashResult.stdout.trim();
          }
        } catch (error2) {
          // If we can't generate hash, we'll need to provide instructions
        }
      }
      
      if (!passwordHash) {
        await ssh.dispose();
        return {
          success: false,
          message: 'Could not generate password hash',
          details: 'Caddy hash-password command not available. You may need to generate the hash manually using: caddy hash-password'
        };
      }
      
      // Remove existing basicauth lines and add new one
      const lines = caddyfileContent.split('\n');
      const filteredLines = lines.filter(line => !line.trim().toLowerCase().startsWith('basicauth'));
      
      // Find where to insert (usually after the site block starts)
      let insertIndex = 0;
      for (let i = 0; i < filteredLines.length; i++) {
        if (filteredLines[i].includes(':') && filteredLines[i].includes('{')) {
          insertIndex = i + 1;
          break;
        }
      }
      
      // Insert new basicauth line
      filteredLines.splice(insertIndex, 0, `    basicauth ${newUsername} ${passwordHash}`);
      newCaddyfileContent = filteredLines.join('\n');
      
      // Write backup first
      await ssh.execCommand(`cp ${foundCaddyfile} ${foundCaddyfile}.backup.${Date.now()}`);
      
      // Write new content
      const tempLocalFile = path.join(os.tmpdir(), `caddyfile_${Date.now()}`);
      fs.writeFileSync(tempLocalFile, newCaddyfileContent);
      await ssh.putFile(tempLocalFile, foundCaddyfile);
      fs.unlinkSync(tempLocalFile);
      
      // Reload Caddy
      try {
        await ssh.execCommand('systemctl reload caddy || caddy reload --config ' + foundCaddyfile + ' || killall -USR1 caddy');
      } catch (error) {
        console.log('[SSH] Note: Caddy reload command may have failed, but config was updated');
      }
      
      await ssh.dispose();
      
      return {
        success: true,
        message: 'Basic Auth configured',
        details: `Added Basic Auth user "${newUsername}" to ${foundCaddyfile}. Backup saved. Use credentials: ${newUsername}:${newPassword}`,
        credentials: {
          username: newUsername,
          password: newPassword
        }
      };
    } else {
      await ssh.dispose();
      return {
        success: false,
        message: 'Invalid action or missing parameters',
        details: 'Action must be "disable" or "setup", and "setup" requires newUsername and newPassword.'
      };
    }
    
  } catch (error) {
    console.error('[SSH] Error configuring Caddy:', error);
    return {
      success: false,
      message: 'Failed to configure Caddy',
      details: error.message
    };
  }
});

// Test SSH connection
ipcMain.handle('test-ssh', async (event, { host, port, username = 'root' }) => {
  const ssh = new NodeSSH();
  
  // Get SSH key path
  const sshKeyPath = findSSHKey();
  const connectOptions = {
    host: host,
    port: port || 22,
    username: username,
    readyTimeout: 10000,
  };
  
  console.log('\n[SSH] Testing SSH connection...');
  console.log(`[SSH] Host: ${host}`);
  console.log(`[SSH] Port: ${port || 22}`);
  console.log(`[SSH] Username: ${username}`);
  
  // Add private key if found
  if (sshKeyPath) {
    try {
      connectOptions.privateKey = fs.readFileSync(sshKeyPath, 'utf8');
      console.log(`[SSH] Using SSH key: ${sshKeyPath}`);
    } catch (error) {
      console.error('[SSH] Error reading SSH key file:', error);
      // Continue without explicit key - NodeSSH will try default locations
    }
  } else {
    console.log('[SSH] Using default SSH key location');
  }
  
  try {
    console.log('[SSH] Attempting connection...');
    await ssh.connect(connectOptions);
    console.log('[SSH] ✓ Connection established');

    // Test with a simple command
    const testCommand = 'echo "SSH connection successful"';
    console.log(`[SSH] Executing command: ${testCommand}`);
    const result = await ssh.execCommand(testCommand);
    
    console.log(`[SSH] Command exit code: ${result.code}`);
    if (result.stdout) {
      console.log(`[SSH] stdout: ${result.stdout.trim()}`);
    }
    if (result.stderr) {
      console.log(`[SSH] stderr: ${result.stderr.trim()}`);
    }
    
    ssh.dispose();
    console.log('[SSH] Connection closed\n');
    
    if (result.code === 0) {
      return { success: true, message: 'SSH connection successful' };
    } else {
      return { 
        success: false, 
        message: `SSH connection failed: ${result.stderr || 'Unknown error'}` 
      };
    }
  } catch (error) {
    console.error('[SSH] Connection error:', error.message);
    if (ssh.isConnected) {
      ssh.dispose();
      console.log('[SSH] Connection closed after error\n');
    }
    return { 
      success: false, 
      message: `SSH connection error: ${error.message}` 
    };
  }
});

// Prepare vLLM with model and versions
ipcMain.handle('prepare-vllm', async (event, { host, port, username = 'root', profileName, baseModel, versions, vllmUrl }) => {
  // Add http:// if protocol is missing
  if (vllmUrl && !vllmUrl.match(/^https?:\/\//i)) {
    vllmUrl = 'http://' + vllmUrl;
    console.log(`[vLLM] Added http:// prefix. New URL: ${vllmUrl}`);
  }
  
  const ssh = new NodeSSH();
  const results = {
    success: false,
    steps: [],
    errors: []
  };

  try {
    // Connect
    results.steps.push('Connecting via SSH...');
    console.log('\n[vLLM] Starting vLLM preparation...');
    console.log(`[vLLM] Profile: ${profileName}`);
    console.log(`[vLLM] Base Model: ${baseModel}`);
    console.log(`[vLLM] Versions: ${versions.map(v => `V${v.version}`).join(', ')}`);
    
    // Get SSH key path
    const sshKeyPath = findSSHKey();
    const connectOptions = {
      host: host,
      port: port || 22,
      username: username,
      readyTimeout: 10000,
    };
    
    console.log(`[SSH] Connecting to ${username}@${host}:${port || 22}`);
    
    // Add private key if found
    if (sshKeyPath) {
      try {
        connectOptions.privateKey = fs.readFileSync(sshKeyPath, 'utf8');
        results.steps.push(`Using SSH key: ${sshKeyPath}`);
        console.log(`[SSH] Using SSH key: ${sshKeyPath}`);
      } catch (error) {
        results.errors.push(`Error reading SSH key file: ${error.message}`);
        console.error(`[SSH] Error reading SSH key: ${error.message}`);
        // Continue without explicit key - NodeSSH will try default locations
      }
    } else {
      results.steps.push('Using default SSH key location');
      console.log('[SSH] Using default SSH key location');
    }
    
    console.log('[SSH] Attempting connection...');
    await ssh.connect(connectOptions);
    results.steps.push('✓ SSH connection established');
    console.log('[SSH] ✓ Connection established');

    // Check if vllm is running and detect how to run vLLM
    results.steps.push('Checking if vLLM is running...');
    const vllmCheckCommand = 'pgrep -f vllm || echo "not_running"';
    console.log(`[SSH] Executing: ${vllmCheckCommand}`);
    const vllmCheck = await ssh.execCommand(vllmCheckCommand);
    console.log(`[SSH] Exit code: ${vllmCheck.code}`);
    console.log(`[SSH] stdout: ${vllmCheck.stdout.trim()}`);
    if (vllmCheck.stderr) {
      console.log(`[SSH] stderr: ${vllmCheck.stderr.trim()}`);
    }
    
    // Detect vLLM command format (we'll need this later)
    let vllmCommand = 'vllm';
    const vllmCommands = [
      'which vllm',
      'python3 -m vllm --help 2>&1 | head -1',
      'python -m vllm --help 2>&1 | head -1'
    ];
    
    for (const cmd of vllmCommands) {
      const checkResult = await ssh.execCommand(cmd);
      if (checkResult.code === 0 && checkResult.stdout.trim()) {
        if (cmd.includes('vllm')) {
          if (cmd.includes('python')) {
            vllmCommand = cmd.split(' ')[0] === 'python3' ? 'python3 -m vllm' : 'python -m vllm';
          }
          console.log(`[vLLM] Found vLLM: ${vllmCommand}`);
          break;
        }
      }
    }
    
    // Check for GPU availability (Ray needs GPU)
    console.log('[vLLM] Checking GPU availability...');
    const gpuCheck = await ssh.execCommand('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no_gpu"');
    const gpuInfo = gpuCheck.stdout.trim();
    console.log(`[vLLM] GPU info: ${gpuInfo}`);
    
    if (vllmCheck.stdout.trim() === 'not_running') {
      console.log('[vLLM] vLLM is not running - will start it after adapter upload');
      results.steps.push('vLLM is not running (will start after adapter upload)');
      if (gpuInfo === 'no_gpu' || gpuInfo.includes('error')) {
        results.errors.push('⚠️ WARNING: No GPU detected. Ray/vLLM requires a GPU to run. Check nvidia-smi output.');
      }
    }
    
    // Check if vLLM is stuck waiting for Ray
    results.steps.push('Checking vLLM status...');
    const supervisorStatusCheck = await ssh.execCommand('supervisorctl status vllm 2>/dev/null || echo "not_managed"');
    const supervisorStatus = supervisorStatusCheck.stdout.trim();
    console.log(`[vLLM] Supervisor status: ${supervisorStatus}`);
    
    if (supervisorStatus.includes('RUNNING')) {
      // Check if it's stuck on Ray
      const rayCheck = await ssh.execCommand('supervisorctl tail -50 vllm 2>/dev/null | grep -i "waiting for ray" | tail -1 || echo "no_ray_wait"');
      const rayWait = rayCheck.stdout.trim();
      if (rayWait && !rayWait.includes('no_ray_wait')) {
        console.warn('[vLLM] ⚠ vLLM appears to be stuck waiting for Ray to start');
        console.warn(`[vLLM] Last Ray wait message: ${rayWait}`);
        results.errors.push('vLLM is running but stuck waiting for Ray. This prevents model download. Common causes:');
        results.errors.push('  - GPU not available or not properly configured');
        results.errors.push('  - Insufficient GPU memory');
        results.errors.push('  - Ray port conflicts');
        results.errors.push('Try: supervisorctl restart vllm');
      }
    }
    results.steps.push('✓ vLLM is running');
    console.log('[vLLM] ✓ vLLM is running');

    // Create model directory structure on remote
    const remoteModelDir = `/workspace/models/${profileName}`;
    results.steps.push(`Creating remote model directory: ${remoteModelDir}`);
    const mkdirCommand = `mkdir -p ${remoteModelDir}`;
    console.log(`[SSH] Executing: ${mkdirCommand}`);
    const mkdirResult = await ssh.execCommand(mkdirCommand);
    console.log(`[SSH] Exit code: ${mkdirResult.code}`);
    if (mkdirResult.stdout) {
      console.log(`[SSH] stdout: ${mkdirResult.stdout.trim()}`);
    }
    if (mkdirResult.stderr) {
      console.log(`[SSH] stderr: ${mkdirResult.stderr.trim()}`);
    }
    results.steps.push('✓ Remote directory created');
    console.log(`[vLLM] ✓ Created directory: ${remoteModelDir}`);

    // Read base model from first adapter's config and HF token from app_config.json
    let actualBaseModel = baseModel; // Default to provided baseModel
    let hfToken = null;
    
    results.steps.push('Determining base model from adapter configuration...');
    console.log('[vLLM] Determining base model...');
    
    // Try to read base model from first adapter's adapter_config.json
    if (versions.length > 0 && versions[0].exists) {
      try {
        const firstAdapterConfigPath = path.join(versions[0].path, 'adapter_config.json');
        if (fs.existsSync(firstAdapterConfigPath)) {
          const adapterConfig = JSON.parse(fs.readFileSync(firstAdapterConfigPath, 'utf8'));
          if (adapterConfig.base_model_name_or_path) {
            actualBaseModel = adapterConfig.base_model_name_or_path;
            console.log(`[vLLM] ✓ Found base model in adapter config: ${actualBaseModel}`);
            results.steps.push(`✓ Base model determined: ${actualBaseModel}`);
          } else {
            console.log(`[vLLM] ⚠ No base_model_name_or_path in adapter config, using provided: ${baseModel}`);
            results.steps.push(`⚠ Using provided base model: ${baseModel}`);
          }
        } else {
          console.log(`[vLLM] ⚠ adapter_config.json not found, using provided: ${baseModel}`);
          results.steps.push(`⚠ adapter_config.json not found, using provided: ${baseModel}`);
        }
      } catch (error) {
        console.warn(`[vLLM] ⚠ Could not read adapter config: ${error.message}`);
        results.steps.push(`⚠ Could not read adapter config, using provided: ${baseModel}`);
      }
    } else {
      console.log(`[vLLM] No adapters found, using provided base model: ${baseModel}`);
      results.steps.push(`Using provided base model: ${baseModel}`);
    }
    
    // Read HF token from app_config.json
    results.steps.push('Loading HuggingFace token...');
    console.log('[vLLM] Loading HuggingFace token...');
    try {
      if (fs.existsSync(APP_CONFIG_FILE)) {
        const appConfig = JSON.parse(fs.readFileSync(APP_CONFIG_FILE, 'utf8'));
        if (appConfig.hf_token) {
          hfToken = appConfig.hf_token;
          console.log(`[vLLM] ✓ Found HF token in app_config.json`);
          results.steps.push('✓ HuggingFace token loaded (will be used for model download)');
        } else {
          console.warn(`[vLLM] ⚠ No hf_token in app_config.json`);
          results.steps.push('⚠ No HuggingFace token found - model download may fail if authentication is required');
        }
      } else {
        console.warn(`[vLLM] ⚠ app_config.json not found`);
        results.steps.push('⚠ app_config.json not found - model download may fail if authentication is required');
      }
    } catch (error) {
      console.warn(`[vLLM] ⚠ Could not read app_config.json: ${error.message}`);
      results.steps.push(`⚠ Could not read app_config.json: ${error.message}`);
    }
    
    // Upload all version adapters (only if not already present and matching)
    for (const version of versions) {
      if (!version.exists) continue;
      
      const versionName = `V${version.version}`;
      const remoteVersionDir = `${remoteModelDir}/${versionName}`;
      const remoteAdapterPath = `${remoteVersionDir}/adapter`;
      
      results.steps.push(`Checking if ${versionName} adapter already exists...`);
      console.log(`\n[vLLM] Checking ${versionName} adapter...`);
      console.log(`[vLLM] Local path: ${version.path}`);
      console.log(`[vLLM] Remote path: ${remoteAdapterPath}`);
      
      // Check if remote adapter exists
      const checkAdapterCommand = `test -d ${remoteAdapterPath} && test -f ${remoteAdapterPath}/adapter_config.json && echo "exists" || echo "not_found"`;
      const adapterCheckResult = await ssh.execCommand(checkAdapterCommand);
      const adapterExists = adapterCheckResult.stdout.trim() === 'exists';
      
      let shouldUpload = true;
      let skipReason = null;
      
      if (adapterExists) {
        console.log(`[vLLM] ${versionName} adapter exists on remote, comparing with local...`);
        
        try {
          // Read local adapter_config.json
          const localAdapterConfigPath = path.join(version.path, 'adapter_config.json');
          let localConfig = null;
          let localConfigHash = null;
          
          if (fs.existsSync(localAdapterConfigPath)) {
            const localConfigContent = fs.readFileSync(localAdapterConfigPath, 'utf8');
            localConfig = JSON.parse(localConfigContent);
            // Create a simple hash of the config (excluding paths that might differ)
            const configForHash = { ...localConfig };
            delete configForHash.base_model_name_or_path; // Path might be different
            localConfigHash = JSON.stringify(configForHash, Object.keys(configForHash).sort());
          }
          
          // Read remote adapter_config.json
          const remoteConfigResult = await ssh.execCommand(`cat ${remoteAdapterPath}/adapter_config.json 2>/dev/null || echo "not_found"`);
          let remoteConfig = null;
          let remoteConfigHash = null;
          
          if (remoteConfigResult.stdout.trim() !== 'not_found') {
            try {
              remoteConfig = JSON.parse(remoteConfigResult.stdout);
              const configForHash = { ...remoteConfig };
              delete configForHash.base_model_name_or_path;
              remoteConfigHash = JSON.stringify(configForHash, Object.keys(configForHash).sort());
            } catch (e) {
              console.log(`[vLLM] Could not parse remote config: ${e.message}`);
            }
          }
          
          // Compare configs
          if (localConfigHash && remoteConfigHash && localConfigHash === remoteConfigHash) {
            // Configs match, check file sizes/dates
            console.log(`[vLLM] Configs match, checking file sizes...`);
            
            // Get local adapter_model file size
            const localModelFiles = ['adapter_model.safetensors', 'adapter_model.bin'];
            let localModelSize = null;
            for (const modelFile of localModelFiles) {
              const localModelPath = path.join(version.path, modelFile);
              if (fs.existsSync(localModelPath)) {
                const stats = fs.statSync(localModelPath);
                localModelSize = stats.size;
                console.log(`[vLLM] Local ${modelFile} size: ${localModelSize} bytes`);
                break;
              }
            }
            
            // Get remote adapter_model file size
            if (localModelSize !== null) {
              const remoteSizeCheck = await ssh.execCommand(`stat -c%s ${remoteAdapterPath}/adapter_model.safetensors 2>/dev/null || stat -c%s ${remoteAdapterPath}/adapter_model.bin 2>/dev/null || echo "0"`);
              const remoteModelSize = parseInt(remoteSizeCheck.stdout.trim(), 10);
              console.log(`[vLLM] Remote model file size: ${remoteModelSize} bytes`);
              
              if (remoteModelSize > 0 && remoteModelSize === localModelSize) {
                // Sizes match, likely the same file
                console.log(`[vLLM] ✓ File sizes match - adapter appears to be up to date`);
                shouldUpload = false;
                skipReason = 'Adapter already exists and matches local version';
              } else if (remoteModelSize > 0) {
                console.log(`[vLLM] File sizes differ (local: ${localModelSize}, remote: ${remoteModelSize}) - will re-upload`);
              } else {
                console.log(`[vLLM] Remote model file not found or size check failed - will upload`);
              }
            } else {
              console.log(`[vLLM] Could not determine local model file size - will upload to be safe`);
            }
          } else {
            console.log(`[vLLM] Configs differ - will re-upload`);
            if (localConfigHash) console.log(`[vLLM] Local config hash: ${localConfigHash.substring(0, 50)}...`);
            if (remoteConfigHash) console.log(`[vLLM] Remote config hash: ${remoteConfigHash.substring(0, 50)}...`);
          }
        } catch (error) {
          console.log(`[vLLM] Error comparing adapters: ${error.message} - will re-upload to be safe`);
        }
      } else {
        console.log(`[vLLM] ${versionName} adapter not found on remote - will upload`);
      }
      
      if (!shouldUpload) {
        console.log(`[vLLM] ✓ Skipping upload of ${versionName} adapter: ${skipReason}`);
        results.steps.push(`✓ ${versionName} adapter already exists (skipped upload)`);
        continue;
      }
      
      results.steps.push(`Uploading ${versionName} adapter...`);
      console.log(`[vLLM] Uploading ${versionName} adapter...`);
      
      try {
        // Create remote directory for this version
        const versionMkdirCommand = `mkdir -p ${remoteVersionDir}`;
        console.log(`[SSH] Executing: ${versionMkdirCommand}`);
        const versionMkdirResult = await ssh.execCommand(versionMkdirCommand);
        console.log(`[SSH] Exit code: ${versionMkdirResult.code}`);
        if (versionMkdirResult.stderr) {
          console.log(`[SSH] stderr: ${versionMkdirResult.stderr.trim()}`);
        }
        
        // Upload adapter directory using NodeSSH
        console.log(`[SSH] Starting file upload from ${version.path} to ${remoteAdapterPath}`);
        let fileCount = 0;
        let totalSize = 0;
        let uploadedSize = 0;
        
        // Calculate total size first
        const getAllFiles = (dir) => {
          const files = [];
          const entries = fs.readdirSync(dir, { withFileTypes: true });
          for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
              files.push(...getAllFiles(fullPath));
            } else if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
              files.push(fullPath);
            }
          }
          return files;
        };
        
        const allFiles = getAllFiles(version.path);
        for (const file of allFiles) {
          try {
            const stats = fs.statSync(file);
            totalSize += stats.size;
          } catch (e) {
            // Ignore errors
          }
        }
        
        const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
        console.log(`[SSH] Total size to upload: ${totalSizeMB} MB (${allFiles.length} file(s))`);
        console.log(`[SSH] Files to upload: ${allFiles.map(f => path.basename(f)).join(', ')}`);
        results.steps.push(`Uploading ${allFiles.length} file(s), ${totalSizeMB} MB total...`);
        console.log(`[SSH] Starting upload progress monitoring...`);
        
        // Retry logic for upload
        let uploadSuccess = false;
        let uploadAttempts = 0;
        const maxUploadAttempts = 3;
        
        while (!uploadSuccess && uploadAttempts < maxUploadAttempts) {
          uploadAttempts++;
          
          if (uploadAttempts > 1) {
            console.log(`[SSH] Retrying upload (attempt ${uploadAttempts}/${maxUploadAttempts})...`);
            results.steps.push(`Retrying upload (attempt ${uploadAttempts}/${maxUploadAttempts})...`);
            
            // Always reconnect on retry to ensure fresh connection
            console.log(`[SSH] Reconnecting for retry...`);
            try {
              if (ssh.isConnected) {
                await ssh.dispose();
              }
            } catch (e) {
              // Ignore dispose errors
            }
            
            // Reconnect with fresh connection
            try {
              await ssh.connect(connectOptions);
              console.log(`[SSH] ✓ Reconnected`);
            } catch (reconnectError) {
              console.error(`[SSH] ✗ Reconnection failed: ${reconnectError.message}`);
              if (uploadAttempts < maxUploadAttempts) {
                await new Promise(resolve => setTimeout(resolve, 5000)); // Wait longer before retry
                continue;
              } else {
                throw new Error(`Failed to reconnect after ${maxUploadAttempts} attempts: ${reconnectError.message}`);
              }
            }
          }
          
          // Verify connection is healthy before upload
          try {
            const healthCheck = await ssh.execCommand('echo "health_check"');
            if (healthCheck.code !== 0) {
              throw new Error('Health check failed');
            }
            console.log(`[SSH] ✓ Connection health check passed`);
          } catch (healthError) {
            console.error(`[SSH] ✗ Connection health check failed: ${healthError.message}`);
            if (uploadAttempts < maxUploadAttempts) {
              await new Promise(resolve => setTimeout(resolve, 3000));
              continue;
            } else {
              throw new Error(`Connection unhealthy after ${maxUploadAttempts} attempts`);
            }
          }
          
          // Reset counters for this attempt
          fileCount = 0;
          uploadedSize = 0;
          
          try {
            console.log(`[SSH] Starting putDirectory upload...`);
            console.log(`[SSH] Local path: ${version.path}`);
            console.log(`[SSH] Remote path: ${remoteVersionDir}/adapter`);
            
            // Wrap putDirectory in a promise with timeout to catch hanging uploads
            const uploadPromise = ssh.putDirectory(version.path, `${remoteVersionDir}/adapter`, {
          recursive: true,
              concurrency: 1, // Use concurrency of 1 to avoid overwhelming connection and reduce EPIPE errors
          validate: (itemPath) => {
            const baseName = path.basename(itemPath);
            // Skip common unwanted files
            return !baseName.startsWith('.') && baseName !== 'node_modules';
          },
          tick: (localPath, remotePath, error) => {
            if (error) {
                  console.error(`[SSH] ✗ Upload error for ${localPath}: ${error.message}`);
                  // Don't throw here - let the outer catch handle it
            } else {
              fileCount++;
                  console.log(`[SSH] [TICK] File ${fileCount}/${allFiles.length} processed: ${path.basename(localPath)}`);
                  try {
                    const stats = fs.statSync(localPath);
                    uploadedSize += stats.size;
                    const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
                    const progressMB = (uploadedSize / (1024 * 1024)).toFixed(2);
                    const progressPercent = totalSize > 0 ? ((uploadedSize / totalSize) * 100).toFixed(1) : '0';
                    
                    const fileName = path.basename(localPath);
                    const isLargeFile = stats.size > 1024 * 1024;
                    
                    // Always log progress for EVERY file upload
                    if (isLargeFile) {
                      // For large files, log with detailed progress
                      console.log(`[SSH] ✓ Uploaded ${fileName} (${fileSizeMB} MB) - ${fileCount}/${allFiles.length} files, ${progressMB} MB / ${totalSizeMB} MB (${progressPercent}%)`);
                      
                      // Update or add progress step
                      const progressStep = `Uploading ${fileName} (${fileSizeMB} MB)... ${progressPercent}%`;
                      const lastStepIndex = results.steps.length - 1;
                      if (lastStepIndex >= 0 && results.steps[lastStepIndex].includes('Uploading') && results.steps[lastStepIndex].includes('%')) {
                        results.steps[lastStepIndex] = progressStep;
                      } else {
                        results.steps.push(progressStep);
                      }
                    } else {
                      // For smaller files, log every file
                      console.log(`[SSH] ✓ Uploaded ${fileName} - ${fileCount}/${allFiles.length} files (${progressMB} MB / ${totalSizeMB} MB, ${progressPercent}%)`);
                      
                      // Update progress step
                      const progressStep = `Uploading... ${fileCount}/${allFiles.length} files (${progressPercent}%)`;
                      const lastStepIndex = results.steps.length - 1;
                      if (lastStepIndex >= 0 && results.steps[lastStepIndex].includes('Uploading') && results.steps[lastStepIndex].includes('%')) {
                        results.steps[lastStepIndex] = progressStep;
                      } else {
                        results.steps.push(progressStep);
                      }
                    }
                  } catch (e) {
                    // If we can't get file stats, still log progress
                    console.log(`[SSH] ✓ Uploaded ${fileCount}/${allFiles.length} file(s)... (${path.basename(localPath)})`);
                    
                    // Update progress step
                    const progressStep = `Uploading... ${fileCount}/${allFiles.length} files`;
                    const lastStepIndex = results.steps.length - 1;
                    if (lastStepIndex >= 0 && results.steps[lastStepIndex].includes('Uploading') && !results.steps[lastStepIndex].includes(path.basename(localPath))) {
                      results.steps[lastStepIndex] = progressStep;
                    } else if (!results.steps[lastStepIndex] || !results.steps[lastStepIndex].includes('Uploading')) {
                      results.steps.push(progressStep);
                    }
              }
            }
          }
        });
        
            // Add timeout to upload (30 minutes max for large uploads)
            let uploadTimeout = null;
            let timeoutRejected = false;
            const timeoutPromise = new Promise((_, reject) => {
              uploadTimeout = setTimeout(() => {
                timeoutRejected = true;
                console.error(`[SSH] Upload timeout after 30 minutes`);
                reject(new Error('Upload timeout: Upload took longer than 30 minutes. This may indicate a connection issue.'));
              }, 30 * 60 * 1000);
            });
            
            // Race between upload and timeout
            console.log(`[SSH] Starting upload of ${allFiles.length} file(s)...`);
            console.log(`[SSH] Files to upload: ${allFiles.map(f => path.basename(f)).join(', ')}`);
            
            // Set up periodic progress logging (every 3 seconds) as backup
            const progressInterval = setInterval(() => {
              if (fileCount > 0) {
                const currentProgressMB = (uploadedSize / (1024 * 1024)).toFixed(2);
                const currentProgressPercent = totalSize > 0 ? ((uploadedSize / totalSize) * 100).toFixed(1) : '0';
                console.log(`[SSH] [Progress Check] ${fileCount}/${allFiles.length} files uploaded, ${currentProgressMB} MB / ${totalSizeMB} MB (${currentProgressPercent}%)`);
              } else {
                console.log(`[SSH] [Progress Check] Upload in progress, waiting for first file...`);
              }
            }, 3000);
            
            await Promise.race([uploadPromise, timeoutPromise]);
            clearInterval(progressInterval);
            
            if (uploadTimeout) {
              clearTimeout(uploadTimeout);
            }
            
            uploadSuccess = true;
            const finalSizeMB = (uploadedSize / (1024 * 1024)).toFixed(2);
            console.log(`[SSH] ✓ Upload complete: ${fileCount} file(s) uploaded, ${finalSizeMB} MB total`);
            results.steps.push(`✓ ${versionName} adapter uploaded (${fileCount} files, ${finalSizeMB} MB)`);
        console.log(`[vLLM] ✓ ${versionName} adapter uploaded`);
            
          } catch (uploadError) {
            clearTimeout(uploadTimeout); // Clear timeout if error occurs
            const errorMsg = uploadError.message || String(uploadError);
            const errorStack = uploadError.stack || '';
            console.error(`[SSH] Upload failed (attempt ${uploadAttempts}/${maxUploadAttempts}): ${errorMsg}`);
            console.error(`[SSH] Error stack: ${errorStack}`);
            
            // Check if it's a connection error that we can retry
            const isConnectionError = errorMsg.includes('EPIPE') || 
                                     errorMsg.includes('ECONNRESET') || 
                                     errorMsg.includes('connection') || 
                                     errorMsg.includes('broken pipe') ||
                                     errorMsg.includes('timeout') ||
                                     errorMsg.includes('ETIMEDOUT') ||
                                     errorMsg.includes('ENOTFOUND') ||
                                     errorMsg.includes('ECONNREFUSED');
            
            if (isConnectionError) {
              if (uploadAttempts < maxUploadAttempts) {
                console.log(`[SSH] Connection error detected (${errorMsg}), will retry in 3 seconds...`);
                results.steps.push(`⚠ Connection error, retrying (${uploadAttempts}/${maxUploadAttempts})...`);
                // Wait a bit before retrying
                await new Promise(resolve => setTimeout(resolve, 3000));
                // Reset counters for retry
                fileCount = 0;
                uploadedSize = 0;
                continue; // Retry
              } else {
                const finalError = new Error(`Upload failed after ${maxUploadAttempts} attempts: ${errorMsg}`);
                finalError.stack = errorStack;
                throw finalError;
              }
            } else {
              // Non-connection error, don't retry but provide better error message
              const enhancedError = new Error(`Upload error: ${errorMsg}`);
              enhancedError.stack = errorStack;
              throw enhancedError;
            }
          }
        }
        
        if (!uploadSuccess) {
          throw new Error(`Failed to upload ${versionName} adapter after ${maxUploadAttempts} attempts`);
        }
      } catch (error) {
        const errorMsg = error.message || String(error);
        const errorStack = error.stack || '';
        console.error(`[SSH] ✗ Error uploading ${versionName}: ${errorMsg}`);
        console.error(`[SSH] Error stack: ${errorStack}`);
        results.errors.push(`Error uploading ${versionName}: ${errorMsg}`);
        // Don't continue to next version if upload fails - this is critical
        throw error;
      }
    }

    // Create a script to prepare vLLM with all versions
    results.steps.push('Creating vLLM preparation script...');
    
    const versionList = versions.map(v => `V${v.version}`).join(', ');
    const versionEchoLines = versions.map(v => 
      `echo "  - ${remoteModelDir}/V${v.version}/adapter"`
    ).join('\n');
    const loraModules = versions.map((v, i) => 
      `adapter${i+1}=${remoteModelDir}/V${v.version}/adapter`
    ).join(' ');
    
    const prepareScript = `#!/bin/bash
# Prepare vLLM with model: ${profileName}
# Base model: ${baseModel}
# Versions: ${versionList}

echo "Model preparation complete for ${profileName}"
echo "Base model: ${baseModel}"
echo "Available versions: ${versionList}"
echo ""
echo "Model directory: ${remoteModelDir}"
echo "Version adapter paths:"
${versionEchoLines}
echo ""
echo "To use with vLLM, you can:"
echo "1. Load base model: ${baseModel}"
echo "2. Apply adapters using the paths above"
echo ""
echo "Example vLLM command (adjust as needed):"
echo "vllm serve ${baseModel} --enable-lora --lora-modules ${loraModules}"
`;

    // Write script to remote using NodeSSH putFile
    const scriptPath = `${remoteModelDir}/prepare_vllm.sh`;
    const tempScriptPath = path.join(os.tmpdir(), `prepare_vllm_${Date.now()}.sh`);
    
    // Write script to temporary local file
    fs.writeFileSync(tempScriptPath, prepareScript, 'utf8');
    
    console.log(`[SSH] Uploading script: ${scriptPath}`);
    await ssh.putFile(tempScriptPath, scriptPath);
    
    // Clean up temporary file
    fs.unlinkSync(tempScriptPath);
    
    console.log(`[SSH] ✓ Script uploaded`);
    
    const chmodCommand = `chmod +x ${scriptPath}`;
    console.log(`[SSH] Executing: ${chmodCommand}`);
    const chmodResult = await ssh.execCommand(chmodCommand);
    console.log(`[SSH] Exit code: ${chmodResult.code}`);
    if (chmodResult.stderr) {
      console.log(`[SSH] stderr: ${chmodResult.stderr.trim()}`);
    }
    
    results.steps.push('✓ Preparation script created');
    console.log(`[vLLM] ✓ Preparation script created`);
    
    // Also create a summary file
    const summary = {
      profileName,
      baseModel,
      modelDir: remoteModelDir,
      versions: versions.map(v => ({
        version: v.version,
        adapterPath: `${remoteModelDir}/V${v.version}/adapter`
      }))
    };
    
    // Write JSON file using NodeSSH putFile
    const summaryPath = `${remoteModelDir}/model_info.json`;
    const tempSummaryPath = path.join(os.tmpdir(), `model_info_${Date.now()}.json`);
    
    // Write JSON to temporary local file
    fs.writeFileSync(tempSummaryPath, JSON.stringify(summary, null, 2), 'utf8');
    
    console.log(`[SSH] Uploading model info: ${summaryPath}`);
    await ssh.putFile(tempSummaryPath, summaryPath);
    
    // Clean up temporary file
    fs.unlinkSync(tempSummaryPath);
    
    console.log(`[SSH] ✓ Model info uploaded`);
    
    results.steps.push('✓ Model info file created');
    console.log(`[vLLM] ✓ Model info file created`);
    
    // Start or restart vLLM with the correct base model and adapters
    results.steps.push('Starting/restarting vLLM with base model and adapters...');
    console.log('[vLLM] Starting/restarting vLLM...');
    
    // Build LoRA modules string (reuse from earlier if available, otherwise build it)
    const loraModulesForStart = versions.map((v, i) => 
      `adapter${i+1}=${remoteModelDir}/V${v.version}/adapter`
    ).join(' ');
    
    // Build vLLM command
    const vllmPort = vllmUrl ? new URL(vllmUrl).port || '8000' : '8000';
    let envVars = '';
    if (hfToken) {
      // Escape the token for shell
      const escapedToken = hfToken.replace(/'/g, "'\\''");
      envVars = `HF_TOKEN='${escapedToken}' `;
    }
    
    let vllmCommandToRun = '';
    if (versions.length > 0) {
      vllmCommandToRun = `${envVars}${vllmCommand} serve ${actualBaseModel} --port ${vllmPort} --enable-lora --lora-modules ${loraModulesForStart}`;
    } else {
      vllmCommandToRun = `${envVars}${vllmCommand} serve ${actualBaseModel} --port ${vllmPort}`;
    }
    
    console.log(`[vLLM] Command to run: ${vllmCommandToRun.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
    results.steps.push(`Starting vLLM with base model: ${actualBaseModel}`);
    results.steps.push(`vLLM will download model from HuggingFace if not already cached`);
    if (hfToken) {
      results.steps.push(`✓ Using HuggingFace token for authentication`);
    }
    if (versions.length > 0) {
      results.steps.push(`✓ Will load ${versions.length} LoRA adapter(s): ${versions.map(v => `V${v.version}`).join(', ')}`);
    }
    
    // Check if vLLM is already running
    const vllmProcessCheck = await ssh.execCommand('pgrep -f vllm || echo "not_running"');
    const isVLLMRunning = vllmProcessCheck.stdout.trim() !== 'not_running';
    
    if (isVLLMRunning) {
      console.log('[vLLM] vLLM is already running, checking if restart is needed...');
      results.steps.push('vLLM is running, checking configuration...');
      
      // Check if supervisor is managing vLLM
      const supervisorCheck = await ssh.execCommand('supervisorctl status vllm 2>/dev/null || echo "not_managed"');
      const isSupervisorManaged = !supervisorCheck.stdout.includes('not_managed');
      
      if (isSupervisorManaged) {
        console.log('[vLLM] vLLM is managed by supervisor, stopping...');
        results.steps.push('Stopping vLLM via supervisor...');
        await ssh.execCommand('supervisorctl stop vllm');
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
        
        // Note: Supervisor config needs to be updated manually or via config file
        // For now, we'll try to restart and let the user know if manual config is needed
        console.log('[vLLM] Restarting vLLM via supervisor...');
        results.steps.push('Restarting vLLM via supervisor...');
        results.steps.push('⚠ Note: If supervisor config needs updating, you may need to update /etc/supervisor/conf.d/vllm.conf manually');
        const restartResult = await ssh.execCommand('supervisorctl restart vllm');
        if (restartResult.code !== 0) {
          results.errors.push('Could not restart vLLM via supervisor. You may need to update supervisor config manually.');
          results.errors.push(`Expected command: ${vllmCommandToRun.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
        } else {
          results.steps.push('✓ vLLM restarted via supervisor');
          console.log('[vLLM] ✓ vLLM restarted via supervisor');
        }
      } else {
        // vLLM is running but not managed by supervisor - kill and restart
        console.log('[vLLM] vLLM is running but not managed by supervisor, stopping...');
        results.steps.push('Stopping existing vLLM process...');
        await ssh.execCommand('pkill -f vllm');
        await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds
        
        // Start vLLM in background
        console.log('[vLLM] Starting vLLM in background...');
        results.steps.push('Starting vLLM in background...');
        const startCommand = `nohup bash -c '${vllmCommandToRun} > /tmp/vllm.log 2>&1' &`;
        const startResult = await ssh.execCommand(startCommand);
        if (startResult.code === 0) {
          results.steps.push('✓ vLLM started in background');
          console.log('[vLLM] ✓ vLLM started in background');
        } else {
          results.errors.push(`Failed to start vLLM: ${startResult.stderr || 'Unknown error'}`);
          console.error(`[vLLM] ✗ Failed to start: ${startResult.stderr}`);
        }
      }
    } else {
      // vLLM is not running - start it
      console.log('[vLLM] vLLM is not running, starting...');
      results.steps.push('Starting vLLM...');
      
      // Check if supervisor is configured
      const supervisorCheck = await ssh.execCommand('supervisorctl status vllm 2>/dev/null || echo "not_managed"');
      const isSupervisorManaged = !supervisorCheck.stdout.includes('not_managed');
      
      if (isSupervisorManaged) {
        // Start via supervisor
        console.log('[vLLM] Starting vLLM via supervisor...');
        results.steps.push('Starting vLLM via supervisor...');
        const startResult = await ssh.execCommand('supervisorctl start vllm');
        if (startResult.code === 0) {
          results.steps.push('✓ vLLM started via supervisor');
          console.log('[vLLM] ✓ vLLM started via supervisor');
        } else {
          results.errors.push('Could not start vLLM via supervisor. Check supervisor config.');
          results.errors.push(`Expected command: ${vllmCommandToRun.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
        }
      } else {
        // Start directly in background
        console.log('[vLLM] Starting vLLM directly in background...');
        results.steps.push('Starting vLLM in background...');
        const startCommand = `nohup bash -c '${vllmCommandToRun} > /tmp/vllm.log 2>&1' &`;
        const startResult = await ssh.execCommand(startCommand);
        if (startResult.code === 0) {
          results.steps.push('✓ vLLM started in background');
          console.log('[vLLM] ✓ vLLM started in background');
          console.log('[vLLM] Logs will be written to /tmp/vllm.log');
        } else {
          results.errors.push(`Failed to start vLLM: ${startResult.stderr || 'Unknown error'}`);
          console.error(`[vLLM] ✗ Failed to start: ${startResult.stderr}`);
        }
      }
    }
    
    // Wait a moment for vLLM to start
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Verify vLLM is running
    const verifyCheck = await ssh.execCommand('pgrep -f vllm || echo "not_running"');
    if (verifyCheck.stdout.trim() === 'not_running') {
      results.errors.push('vLLM process not found after startup attempt');
      console.error('[vLLM] ✗ vLLM process not found after startup');
    } else {
      results.steps.push('✓ vLLM process is running');
      console.log('[vLLM] ✓ vLLM process is running');
    }
    
    // Monitor for model download/loading if vLLM is running and URL provided
    if (vllmUrl) {
      results.steps.push('Monitoring model download/loading progress...');
      console.log('[vLLM] Starting model download/loading monitoring...');
      
      // Progress callback to update steps
      const progressCallback = (progress) => {
        const message = progress.progress 
          ? `${progress.message} (${progress.progress}%)`
          : progress.message;
        results.steps.push(`[${progress.type.toUpperCase()}] ${message}`);
        console.log(`[vLLM] ${message}`);
      };
      
      // Monitor logs for download/loading progress (up to 10 minutes)
      const logMonitorResult = await monitorVLLMLogs(ssh, 600000, progressCallback);
      
      if (logMonitorResult.ready) {
        results.steps.push('✓ Model download/loading completed (from logs)');
        console.log('[vLLM] ✓ Model ready according to logs');
      } else if (logMonitorResult.downloadDetected || logMonitorResult.loadingDetected) {
        results.steps.push('⚠ Model download/loading detected but not yet complete');
        console.log('[vLLM] ⚠ Model still downloading/loading');
      } else {
        results.steps.push('ℹ No download/loading activity detected in logs (model may already be loaded)');
        console.log('[vLLM] ℹ No download activity detected');
      }
      
      // Now poll HTTP endpoint to confirm readiness (up to 5 minutes)
      results.steps.push('Checking if model is ready via HTTP API...');
      console.log('[vLLM] Polling HTTP endpoint for model readiness...');
      
      const httpPollResult = await pollModelReadiness(
        vllmUrl, 
        baseModel, 
        300000, // 5 minutes max
        10000   // Check every 10 seconds
      );
      
      if (httpPollResult.ready) {
        results.steps.push(`✓ Model is ready! Available models: ${httpPollResult.models.join(', ')}`);
        console.log(`[vLLM] ✓ Model ready! Available: ${httpPollResult.models.join(', ')}`);
      } else {
        results.steps.push('⚠ Model may still be loading. You can try chatting, but responses may be delayed.');
        console.log('[vLLM] ⚠ Model readiness not confirmed via HTTP');
      }
    }
    
    results.success = results.errors.length === 0;

    // Store SSH config, model info, and vLLM config for chat interface
    if (results.success) {
      storedSSHConfig = {
        host: host,
        port: port || 22,
        username: username
      };
      storedModelInfo = {
        profileName,
        baseModel,
        modelDir: remoteModelDir,
        versions: versions.map(v => ({
          version: v.version,
          adapterPath: `${remoteModelDir}/V${v.version}/adapter`
        }))
      };
      storedVLLMUrl = vllmUrl || `http://${host}:8000`;
    }

    ssh.dispose();
    console.log('[SSH] Connection closed');
    if (results.success) {
      console.log('[vLLM] ✓ Preparation completed successfully\n');
    } else {
      console.log(`[vLLM] ✗ Preparation completed with ${results.errors.length} error(s)\n`);
    }
    return results;

  } catch (error) {
    const errorMsg = error.message || String(error);
    const errorStack = error.stack || '';
    console.error(`[vLLM] ✗ Fatal error: ${errorMsg}`);
    console.error(`[vLLM] Error stack: ${errorStack}`);
    if (ssh && ssh.isConnected) {
      try {
      ssh.dispose();
      console.log('[SSH] Connection closed after error');
      } catch (disposeError) {
        console.error(`[SSH] Error disposing connection: ${disposeError.message}`);
    }
    }
    results.errors.push(`Fatal error: ${errorMsg}`);
    results.success = false;
    console.log('[vLLM] ✗ Preparation failed\n');
    return results;
  }
});

// Get stored model info for chat interface
ipcMain.handle('get-model-info', async () => {
  return storedModelInfo;
});

// Test vLLM URL connectivity
ipcMain.handle('test-vllm-url', async (event, { vllmUrl, sshHost, sshPort, sshUsername = 'root' }) => {
  if (!vllmUrl) {
    return {
      success: false,
      message: 'No URL provided'
    };
  }
  
  // Add http:// if protocol is missing
  if (!vllmUrl.match(/^https?:\/\//i)) {
    vllmUrl = 'http://' + vllmUrl;
    console.log(`[TEST] Added http:// prefix. New URL: ${vllmUrl}`);
  }
  
  // Token will be retrieved from server if needed (not loaded from config)
  console.log(`[TEST] Testing vLLM URL: ${vllmUrl}`);
  
  // First, try without authentication
  try {
    // Use minimal headers - browsers send Accept and User-Agent by default
    const headers = {
      'Accept': 'application/json',
      'User-Agent': 'Electron-vLLM-Client/1.0'
    };
    
    // Ensure URL doesn't have trailing slash, then add /v1/models
    const testUrl = vllmUrl.replace(/\/$/, '') + '/v1/models';
    console.log(`[TEST] Testing URL: ${testUrl}`);
    console.log(`[TEST] Original URL: ${vllmUrl}`);
    console.log(`[TEST] Headers being sent:`, JSON.stringify(headers, null, 2));
    console.log(`[TEST] Attempting connection without authentication...`);
    
    const response = await makeHttpRequest(testUrl, {
      method: 'GET',
      headers: headers
    });
    
    console.log(`[TEST] Response received - Status: ${response.statusCode}`);
    console.log(`[TEST] Response data length: ${response.data ? response.data.length : 0}`);
    if (response.data) {
      console.log(`[TEST] Full response data: ${response.data}`);
    }
    
    if (response.statusCode === 200) {
      const modelsData = JSON.parse(response.data);
      const availableModels = modelsData.data || [];
      const modelNames = availableModels.map(m => m.id || m.name || m);
      
      console.log(`[TEST] ✓ vLLM is accessible without authentication. Available models: ${modelNames.join(', ')}`);
      
      return {
        success: true,
        message: 'vLLM is accessible',
        models: modelNames,
        statusCode: response.statusCode,
        requiresAuth: false
      };
    } else if (response.statusCode === 404) {
      // 404 means the endpoint doesn't exist, but vLLM is responding
      // This could mean vLLM is still starting or the endpoint path is different
      console.log(`[TEST] Got 404 - endpoint not found, but vLLM is responding`);
      return {
        success: true,
        message: 'vLLM is accessible (endpoint may not be ready yet)',
        details: 'vLLM responded but /v1/models endpoint returned 404. vLLM may still be starting.',
        statusCode: 404,
        requiresAuth: false
      };
    } else if (response.statusCode === 401) {
      // Got 401 - server is using Basic Auth (Vast.ai pattern)
      // First try using saved OPEN_BUTTON_TOKEN if available
      if (storedOpenButtonToken) {
        console.log(`[TEST] Using saved OPEN_BUTTON_TOKEN for Basic Auth`);
        const basicAuthValue = Buffer.from(`vastai:${storedOpenButtonToken}`).toString('base64');
        const authHeaders = {
          ...headers,
          'Authorization': `Basic ${basicAuthValue}`
        };
        
        const authResponse = await makeHttpRequest(testUrl, {
          method: 'GET',
          headers: authHeaders
        });
        
        if (authResponse.statusCode === 200) {
          const modelsData = JSON.parse(authResponse.data);
          const availableModels = modelsData.data || [];
          const modelNames = availableModels.map(m => m.id || m.name || m);
          
          console.log(`[TEST] ✓ vLLM is accessible with saved OPEN_BUTTON_TOKEN. Available models: ${modelNames.join(', ')}`);
          
          return {
            success: true,
            message: 'vLLM is accessible (using saved Vast.ai Basic Auth)',
            models: modelNames,
            statusCode: authResponse.statusCode,
            requiresAuth: false
          };
        } else if (authResponse.statusCode === 404) {
          return {
            success: true,
            message: 'vLLM is accessible (using saved Vast.ai Basic Auth, endpoint may not be ready yet)',
            details: 'Basic Auth credentials used, but /v1/models returned 404. vLLM may still be starting.',
            statusCode: 404,
            requiresAuth: false
          };
        } else {
          console.log(`[TEST] Saved token didn't work (status ${authResponse.statusCode}), will retrieve fresh token`);
        }
      }
      
      // If saved token didn't work or doesn't exist, retrieve from instance
      console.log(`[TEST] Got 401 - server requires Basic Auth`);
      console.log(`[TEST] Attempting to retrieve OPEN_BUTTON_TOKEN from instance...`);
      
      if (sshHost && sshPort) {
        try {
          const ssh = new NodeSSH();
          const sshKeyPath = findSSHKey();
          const connectOptions = {
            host: sshHost,
            port: sshPort || 22,
            username: sshUsername,
            readyTimeout: 30000,
            keepaliveInterval: 10000,
            keepaliveCountMax: 10,
          };
          
          if (sshKeyPath) {
            try {
              connectOptions.privateKey = fs.readFileSync(sshKeyPath, 'utf8');
            } catch (error) {
              console.error('[SSH] Error reading SSH key file:', error);
            }
          }
          
          await ssh.connect(connectOptions);
          console.log('[SSH] ✓ Connection established for retrieving OPEN_BUTTON_TOKEN');
          
          // Get OPEN_BUTTON_TOKEN from environment variables
          let openButtonToken = null;
          try {
            // Try to get it from environment
            const envResult = await ssh.execCommand('echo $OPEN_BUTTON_TOKEN');
            if (envResult.stdout && envResult.stdout.trim()) {
              openButtonToken = envResult.stdout.trim();
              console.log(`[TEST] Found OPEN_BUTTON_TOKEN from environment`);
            }
          } catch (error) {
            console.log(`[TEST] Could not get OPEN_BUTTON_TOKEN from environment: ${error.message}`);
          }
          
          // If not found in environment, try to get it from process environment
          if (!openButtonToken) {
            try {
              const procResult = await ssh.execCommand('cat /proc/1/environ 2>/dev/null | tr "\\0" "\\n" | grep OPEN_BUTTON_TOKEN | cut -d= -f2');
              if (procResult.stdout && procResult.stdout.trim()) {
                openButtonToken = procResult.stdout.trim();
                console.log(`[TEST] Found OPEN_BUTTON_TOKEN from process environment`);
              }
            } catch (error) {
              console.log(`[TEST] Could not get OPEN_BUTTON_TOKEN from process environment: ${error.message}`);
            }
          }
          
          // If still not found, try checking supervisor/systemd environment
          if (!openButtonToken) {
            try {
              const systemdResult = await ssh.execCommand('systemctl show-environment 2>/dev/null | grep OPEN_BUTTON_TOKEN | cut -d= -f2');
              if (systemdResult.stdout && systemdResult.stdout.trim()) {
                openButtonToken = systemdResult.stdout.trim();
                console.log(`[TEST] Found OPEN_BUTTON_TOKEN from systemd environment`);
              }
            } catch (error) {
              console.log(`[TEST] Could not get OPEN_BUTTON_TOKEN from systemd: ${error.message}`);
            }
          }
          
          await ssh.dispose();
          
          if (openButtonToken) {
            // Store the token for use in chat requests (not saved to config - retrieved during setup)
            storedOpenButtonToken = openButtonToken;
            
            // Use Vast.ai standard Basic Auth: username "vastai", password is OPEN_BUTTON_TOKEN
            console.log(`[TEST] Using Vast.ai Basic Auth pattern: vastai:${openButtonToken.substring(0, 8)}...`);
            const basicAuthValue = Buffer.from(`vastai:${openButtonToken}`).toString('base64');
            const authHeaders = {
              ...headers,
              'Authorization': `Basic ${basicAuthValue}`
            };
            
            // Test again with Basic Auth
            const authResponse = await makeHttpRequest(testUrl, {
              method: 'GET',
              headers: authHeaders
            });
            
            if (authResponse.statusCode === 200) {
              const modelsData = JSON.parse(authResponse.data);
              const availableModels = modelsData.data || [];
              const modelNames = availableModels.map(m => m.id || m.name || m);
              
              console.log(`[TEST] ✓ vLLM is accessible with Vast.ai Basic Auth. Available models: ${modelNames.join(', ')}`);
              
              return {
                success: true,
                message: 'vLLM is accessible (using Vast.ai Basic Auth)',
                models: modelNames,
                statusCode: authResponse.statusCode,
                requiresAuth: false
              };
            } else if (authResponse.statusCode === 404) {
              return {
                success: true,
                message: 'vLLM is accessible (using Vast.ai Basic Auth, endpoint may not be ready yet)',
                details: 'Basic Auth credentials retrieved, but /v1/models returned 404. vLLM may still be starting.',
                statusCode: 404,
                requiresAuth: false
              };
            } else {
              console.log(`[TEST] Basic Auth with OPEN_BUTTON_TOKEN returned status ${authResponse.statusCode}`);
            }
          } else {
            console.log(`[TEST] Could not find OPEN_BUTTON_TOKEN on instance`);
          }
        } catch (sshError) {
          console.error(`[TEST] Failed to retrieve OPEN_BUTTON_TOKEN via SSH: ${sshError.message}`);
        }
      }
      
      // If we couldn't get the token or it still doesn't work, return error
      return {
        success: false,
        message: 'vLLM requires authentication',
        details: 'vLLM returned 401 (Basic Auth required). Could not retrieve OPEN_BUTTON_TOKEN from instance. Please check SSH connection.',
        statusCode: 401,
        requiresAuth: true
      };
    } else if (response.statusCode === 404) {
      // 404 means the endpoint doesn't exist, but vLLM is responding
      // If the response is {"detail":"Not Found"}, vLLM is accessible but endpoint may not be ready
      console.log(`[TEST] Got 404 - endpoint not found, but vLLM is responding`);
      try {
        const errorData = JSON.parse(response.data || '{}');
        if (errorData.detail === 'Not Found') {
          // vLLM is responding, just the endpoint isn't ready yet
          return {
            success: true,
            message: 'vLLM is accessible',
            details: 'vLLM is responding but /v1/models endpoint is not ready yet. This is normal if vLLM is still starting.',
            statusCode: 404,
            requiresAuth: false
          };
        }
      } catch (e) {
        // If we can't parse the response, still treat 404 as accessible
      }
      // Even if we can't parse, if we got a response (not a connection error), vLLM is accessible
      return {
        success: true,
        message: 'vLLM is accessible',
        details: 'vLLM responded but /v1/models returned 404. vLLM may still be starting.',
        statusCode: 404,
        requiresAuth: false
      };
    } else {
      return {
        success: false,
        message: `HTTP ${response.statusCode}`,
        details: 'vLLM responded but with an error status code',
        statusCode: response.statusCode
      };
    }
  } catch (error) {
    console.error(`[TEST] ✗ vLLM URL test failed: ${error.message}`);
    console.error(`[TEST] Error type: ${error.constructor.name}`);
    console.error(`[TEST] Full error:`, error);
    
    // makeHttpRequest now returns responses for all status codes, so if we get here,
    // it's a connection error, not an authentication error
    // Only treat as 401 if the error explicitly says "HTTP 401 Unauthorized"
    const isExplicit401 = error.message.includes('HTTP 401') && error.message.includes('Unauthorized');
    
    if (isExplicit401) {
      // If we have a token, it might be invalid
      if (vllmToken) {
        return {
          success: false,
          message: 'Authentication failed',
          details: 'vLLM requires authentication but the provided token is invalid.',
          statusCode: 401
        };
      } else {
        return {
          success: false,
          message: 'Authentication required',
          details: 'vLLM requires an API token. Please enter your token.',
          statusCode: 401
        };
      }
    }
    
    // All other errors are connection/network errors, not authentication errors
    let errorMessage = 'Connection failed';
    let details = error.message;
    
    if (error.message.includes('ECONNREFUSED') || error.message.includes('connection refused')) {
      errorMessage = 'Cannot connect to vLLM';
      details = 'Make sure vLLM is running and the URL/port is correct';
    } else if (error.message.includes('timeout') || error.message.includes('Request timeout')) {
      errorMessage = 'Connection timeout';
      details = 'vLLM did not respond in time. It may still be starting.';
    } else if (error.message.includes('ENOTFOUND') || error.message.includes('getaddrinfo')) {
      errorMessage = 'Invalid hostname';
      details = 'Could not resolve the hostname. Check the URL.';
    } else {
      // For any other error, show the actual error message
      errorMessage = 'Connection error';
      details = error.message;
    }
    
    return {
      success: false,
      message: errorMessage,
      details: details,
      error: error.message
    };
  }
});

// Helper function to monitor vLLM logs for download/loading progress
async function monitorVLLMLogs(ssh, maxWaitTime = 600000, progressCallback = null) {
  const startTime = Date.now();
  const checkInterval = 5000; // Check every 5 seconds
  let lastLogPosition = 0;
  let downloadDetected = false;
  let loadingDetected = false;
  let modelReady = false;
  
  console.log('[MONITOR] Starting vLLM log monitoring...');
  
  while (Date.now() - startTime < maxWaitTime) {
    try {
      // Try to get logs from supervisor first, then fall back to /tmp/vllm.log
      const logCommand = 'supervisorctl tail -100 vllm 2>/dev/null | tail -50 || tail -50 /tmp/vllm.log 2>/dev/null || echo ""';
      const logResult = await ssh.execCommand(logCommand);
      const logs = logResult.stdout || '';
      
      // Look for download indicators
      const downloadPatterns = [
        /Downloading.*?(\d+%)/gi,
        /downloading.*?(\d+%)/gi,
        /Download.*?(\d+%)/gi,
        /Downloading.*model/gi,
        /Fetching.*from.*huggingface/gi
      ];
      
      // Look for loading indicators
      const loadingPatterns = [
        /Loading.*model/gi,
        /Loading.*weights/gi,
        /Initializing.*model/gi,
        /Model.*loaded/gi,
        /Ready.*to.*serve/gi
      ];
      
      // Look for completion indicators
      const readyPatterns = [
        /Uvicorn running on/gi,
        /Application startup complete/gi,
        /Model.*ready/gi,
        /Ready.*to.*process.*requests/gi,
        /Serving.*on.*port/gi
      ];
      
      // Check for download progress
      for (const pattern of downloadPatterns) {
        const matches = logs.match(pattern);
        if (matches) {
          downloadDetected = true;
          const progressMatch = logs.match(/(\d+)%/);
          const progress = progressMatch ? progressMatch[1] : null;
          const message = progress ? `Downloading model: ${progress}%` : 'Downloading model from HuggingFace...';
          console.log(`[MONITOR] ${message}`);
          if (progressCallback) {
            progressCallback({ type: 'download', message, progress });
          }
        }
      }
      
      // Check for loading progress
      for (const pattern of loadingPatterns) {
        if (pattern.test(logs)) {
          loadingDetected = true;
          const message = 'Loading model into memory...';
          console.log(`[MONITOR] ${message}`);
          if (progressCallback) {
            progressCallback({ type: 'loading', message });
          }
        }
      }
      
      // Check for readiness
      for (const pattern of readyPatterns) {
        if (pattern.test(logs)) {
          modelReady = true;
          const message = 'Model loaded and ready!';
          console.log(`[MONITOR] ${message}`);
          if (progressCallback) {
            progressCallback({ type: 'ready', message });
          }
          return { success: true, ready: true };
        }
      }
      
      // If we've seen download/loading but no ready signal, continue waiting
      if (downloadDetected || loadingDetected) {
        // Continue monitoring
      }
      
    } catch (error) {
      console.warn(`[MONITOR] Error checking logs: ${error.message}`);
    }
    
    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }
  
  // Timeout reached
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  console.warn(`[MONITOR] Timeout after ${elapsed}s. Model may still be downloading/loading.`);
  return { 
    success: false, 
    ready: false, 
    timeout: true,
    downloadDetected,
    loadingDetected
  };
}

// Helper function to poll HTTP endpoint for model readiness
async function pollModelReadiness(vllmUrl, expectedModel, maxWaitTime = 300000, checkInterval = 10000) {
  const startTime = Date.now();
  const healthHeaders = { 'Content-Type': 'application/json' };
  
  console.log(`[POLL] Polling ${vllmUrl}/v1/models for model readiness...`);
  
  while (Date.now() - startTime < maxWaitTime) {
    try {
      const response = await makeHttpRequest(`${vllmUrl}/v1/models`, {
        method: 'GET',
        headers: healthHeaders
      });
      
      if (response.statusCode === 200) {
        const modelsData = JSON.parse(response.data);
        const availableModels = modelsData.data || [];
        const modelNames = availableModels.map(m => m.id || m.name || m);
        
        console.log(`[POLL] Available models: ${modelNames.join(', ')}`);
        
        // Check if expected model is available
        if (expectedModel) {
          const modelFound = modelNames.some(name => 
            name.toLowerCase().includes(expectedModel.toLowerCase()) ||
            expectedModel.toLowerCase().includes(name.toLowerCase())
          );
          
          if (modelFound) {
            console.log(`[POLL] ✓ Model ${expectedModel} is ready!`);
            return { success: true, ready: true, models: modelNames };
          }
        } else if (modelNames.length > 0) {
          // If no specific model expected, any model being available is good
          console.log(`[POLL] ✓ vLLM is ready with ${modelNames.length} model(s)!`);
          return { success: true, ready: true, models: modelNames };
        }
      }
    } catch (error) {
      // If it's a connection error, model is still loading
      if (error.message.includes('connection') || error.message.includes('ECONNREFUSED')) {
        console.log(`[POLL] vLLM not yet accessible (still loading)...`);
      } else {
        console.warn(`[POLL] Error checking model readiness: ${error.message}`);
      }
    }
    
    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }
  
  // Timeout
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  console.warn(`[POLL] Timeout after ${elapsed}s. Model may still be loading.`);
  return { success: false, ready: false, timeout: true };
}

// Helper function to make HTTP request
function makeHttpRequest(url, options) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const isHttps = urlObj.protocol === 'https:';
    const httpModule = isHttps ? https : http;
    
    const requestOptions = {
      hostname: urlObj.hostname,
      port: urlObj.port || (isHttps ? 443 : 80),
      path: urlObj.pathname + urlObj.search,
      method: options.method || 'GET',
      headers: options.headers || {}
    };
    
    const req = httpModule.request(requestOptions, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve({ statusCode: res.statusCode, data: data });
        } else {
          // Return response object even for error status codes, don't reject
          // This allows the caller to handle 401 and other status codes
          resolve({ statusCode: res.statusCode, data: data });
        }
      });
    });
    
    req.on('error', (error) => {
      reject(error);
    });
    
    req.setTimeout(10000, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
    
    if (options.body) {
      req.write(options.body);
    }
    
    req.end();
  });
}

// Send chat message to vLLM
ipcMain.handle('send-chat-message', async (event, { message, version, prependedText, useSummary, conversationSummary, conversationHistory }) => {
  if (!storedModelInfo || !storedVLLMUrl) {
    throw new Error('vLLM not prepared. Please prepare vLLM first.');
  }

  try {
    
    // Build messages array for OpenAI-compatible API
    const messages = [];
    
    // Build the current user message with structured sections
    let userMessage = '';
    const sections = [];
    
    // CONTEXT section (only if prepended text exists)
    if (prependedText && prependedText.trim()) {
      sections.push(`### CONTEXT ###\n\nSome context for you to know:\n${prependedText}`);
    }
    
    // SUMMARY section (only if summary exists and is enabled)
    if (useSummary && conversationSummary && conversationSummary.trim()) {
      sections.push(`### SUMMARY ###\n\nWe've been talking about:\n${conversationSummary}`);
    }
    
    // PROMPT section (always present)
    sections.push(`### PROMPT ###\n\n${message}`);
    
    // Combine all sections
    userMessage = sections.join('\n\n');
    
    // Add conversation history (excluding the last user message if it exists, as we're replacing it)
    if (conversationHistory && conversationHistory.length > 0) {
      // Filter out the last user message if it exists, as we're building a new one
      const historyToAdd = [...conversationHistory];
      if (historyToAdd.length > 0 && historyToAdd[historyToAdd.length - 1].role === 'user') {
        historyToAdd.pop(); // Remove last user message
      }
      messages.push(...historyToAdd);
    }
    
    // Add current message with prepended text and summary
    messages.push({
      role: 'user',
      content: userMessage
    });
    
    // Determine model name
    let modelName = storedModelInfo.baseModel;
    if (version && version !== 'base') {
      // For versioned models, we might need to use a specific model name
      // This depends on how vLLM is configured with LoRA adapters
      // For now, we'll use the base model and let vLLM handle adapter selection
      modelName = storedModelInfo.baseModel;
    }
    
    // Create payload for vLLM
    const payload = JSON.stringify({
      model: modelName,
      messages: messages,
      max_tokens: 2048,
      temperature: 0.7,
      stream: false
    });
    
    console.log('\n[CHAT] ========================================');
    console.log('[CHAT] Sending message to vLLM...');
    console.log(`[CHAT] Model: ${modelName}`);
    console.log(`[CHAT] Version: ${version || 'base'}`);
    console.log('[CHAT] Full prompt being sent (including context and summary):');
    console.log('----------------------------------------');
    console.log(userMessage);
    console.log('----------------------------------------');
    console.log(`[CHAT] Full user message length: ${userMessage.length} chars`);
    console.log('[CHAT] ========================================\n');
    
    // Use the stored vLLM URL (user-provided, not localhost)
    const vllmApiUrl = storedVLLMUrl;
    console.log(`[CHAT] Using vLLM URL: ${vllmApiUrl}`);
    
    // Check if vLLM is accessible (from local machine, not via SSH)
    console.log(`[CHAT] Checking if vLLM is accessible at ${vllmApiUrl}...`);
    
    // Build headers for health check
    const healthHeaders = { 'Content-Type': 'application/json' };
    
    // Use stored OPEN_BUTTON_TOKEN for Basic Auth if available
    if (storedOpenButtonToken) {
      const basicAuthValue = Buffer.from(`vastai:${storedOpenButtonToken}`).toString('base64');
      healthHeaders['Authorization'] = `Basic ${basicAuthValue}`;
      console.log(`[CHAT] Using Vast.ai Basic Auth for health check`);
    }
    
    let healthCode = null;
    try {
      const healthResponse = await makeHttpRequest(`${vllmApiUrl}/v1/models`, {
        method: 'GET',
        headers: healthHeaders
      });
      healthCode = healthResponse.statusCode.toString();
      console.log(`[CHAT] vLLM health check HTTP code: ${healthCode}`);
    } catch (error) {
      const statusMatch = error.message.match(/HTTP (\d+)/);
      if (statusMatch) {
        healthCode = statusMatch[1];
      } else {
        healthCode = '000';
      }
      console.log(`[CHAT] vLLM health check failed: ${error.message}`);
    }
    
    if (healthCode === '000' || healthCode === '' || healthCode.includes('connection')) {
      throw new Error(`vLLM is not accessible at ${vllmApiUrl}. Make sure vLLM is running and the URL is correct.`);
    }
    
    if (healthCode === '401') {
      throw new Error(`vLLM requires authentication (HTTP 401). The app should have automatically disabled Basic Auth. Please check the vLLM URL test.`);
    }
    
    if (healthCode !== '200') {
      console.warn(`[CHAT] ⚠ vLLM health check returned HTTP ${healthCode} - may still be loading`);
    }
    
    // Make HTTP request directly (not via SSH)
    const chatApiUrl = `${vllmApiUrl}/v1/chat/completions`;
    console.log(`[CHAT] Sending HTTP request to: ${chatApiUrl}`);
    
    // Build headers for chat request
    const chatHeaders = { 'Content-Type': 'application/json' };
    
    // Use stored OPEN_BUTTON_TOKEN for Basic Auth if available
    if (storedOpenButtonToken) {
      const basicAuthValue = Buffer.from(`vastai:${storedOpenButtonToken}`).toString('base64');
      chatHeaders['Authorization'] = `Basic ${basicAuthValue}`;
      console.log(`[CHAT] Using Vast.ai Basic Auth for chat request`);
    }
    
    const response = await makeHttpRequest(chatApiUrl, {
      method: 'POST',
      headers: chatHeaders,
      body: payload
    });
    
    console.log(`[CHAT] HTTP status: ${response.statusCode}`);
    console.log(`[CHAT] Response length: ${response.data ? response.data.length : 0} chars`);
    
    if (!response.data || response.data.trim() === '') {
      throw new Error('vLLM returned an empty response');
    }
    
    // Parse response
    let responseData;
    try {
      console.log(`[CHAT] Parsing response (first 500 chars): ${response.data.substring(0, 500)}...`);
      responseData = JSON.parse(response.data);
      console.log(`[CHAT] ✓ Successfully parsed JSON response`);
    } catch (error) {
      console.error(`[CHAT] ✗ Failed to parse response as JSON`);
      console.error(`[CHAT] Error: ${error.message}`);
      console.error(`[CHAT] Raw response (first 1000 chars):`);
      console.error(response.data.substring(0, 1000));
      throw new Error(`Failed to parse vLLM response: ${error.message}. Raw response: ${response.data.substring(0, 200)}`);
    }
    
    if (!responseData.choices || responseData.choices.length === 0) {
      throw new Error('vLLM returned no choices in response');
    }
    
    const responseText = responseData.choices[0].message.content;
    const usage = responseData.usage || {};
    
    console.log(`[CHAT] Response received:`);
    console.log(`[CHAT] ${responseText}`);
    console.log(`[CHAT] Tokens: ${usage.total_tokens || 'unknown'} (prompt: ${usage.prompt_tokens || 'unknown'}, completion: ${usage.completion_tokens || 'unknown'})`);
    console.log('[CHAT] ✓ Message exchange complete\n');
    
    return {
      success: true,
      response: responseText,
      usage: usage,
      userMessage: message, // Original user message (without context/summary)
      assistantMessage: responseText // Full assistant response
    };
    
  } catch (error) {
    console.error(`[CHAT] Fatal error: ${error.message}`);
    throw error;
  }
});


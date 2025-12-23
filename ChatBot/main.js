const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { NodeSSH } = require('node-ssh');
const http = require('http');
const https = require('https');
const { URL } = require('url');
const { spawn, exec } = require('child_process');
const net = require('net');
const { promisify } = require('util');
const execAsync = promisify(exec);

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

// Helper function to prepare SSH connection options with auto-accept host keys
// This automatically answers "yes" to SSH host key verification prompts
// Only used in ChatBot companion app - not applied elsewhere in the codebase
function prepareSSHConnectionOptions(baseOptions) {
  const options = { ...baseOptions };
  
  // Add callbacks to automatically accept host keys (answers "yes" to prompts)
  // This handles the "Are you sure you want to continue connecting (yes/no/[fingerprint])?" prompt
  // The hostVerifier callback receives (keyHash, callback) where callback takes a boolean
  options.callbacks = {
    hostVerifier: (keyHash, callback) => {
      // Automatically accept any host key
      console.log('[SSH] Auto-accepting host key (fingerprint: ' + keyHash + ')');
      // Call the callback with true to accept the host key
      if (typeof callback === 'function') {
        callback(true);
      }
    }
  };
  
  // Enable SSH keepalive to prevent silent disconnects during long uploads
  // This sends keepalive packets every 10 seconds, and disconnects after 10 failed attempts
  options.keepaliveInterval = 10000; // Send keepalive every 10 seconds
  options.keepaliveCountMax = 10; // Disconnect after 10 failed keepalive attempts
  
  return options;
}

let mainWindow;

// Store SSH config and model info after successful preparation
let storedSSHConfig = null;
let storedModelInfo = null;
let storedVLLMUrl = null;
let storedOpenButtonToken = null; // Vast.ai OPEN_BUTTON_TOKEN for Basic Auth

// SSH tunnel for port forwarding
let sshTunnelProcess = null;
let sshTunnelLocalPort = null;
let sshTunnelConfig = null; // { host, port, username, sshKeyPath, remotePort }

// Helper function to find an available local port
function findAvailablePort(startPort = 8889) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(startPort, () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        // Try next port
        findAvailablePort(startPort + 1).then(resolve).catch(reject);
      } else {
        reject(err);
      }
    });
  });
}


// INVARIANT: All FastAPI access MUST go through the SSH tunnel.
// localhost:8888 must mean the same thing for the entire app lifecycle.
// Do not add pre-tunnel health checks or alternate local ports.

// Helper function to create a gzip archive of a directory
async function createGzipArchive(sourceDir, outputFile) {
  return new Promise((resolve, reject) => {
    console.log(`[GZIP] Creating gzip archive...`);
    console.log(`[GZIP] Source directory: ${sourceDir}`);
    console.log(`[GZIP] Output file: ${outputFile}`);
    
    // Use tar with gzip to create archive
    // -czf: create, gzip, file
    // -C: change to directory before processing
    // --exclude: exclude unwanted files
    const tarArgs = [
      '-czf',
      outputFile,
      '-C',
      path.dirname(sourceDir),
      '--exclude=node_modules',
      '--exclude=.git',
      path.basename(sourceDir)
    ];
    
    console.log(`[GZIP] Running: tar ${tarArgs.join(' ')}`);
    
    const tarProcess = spawn('tar', tarArgs, {
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    let stdout = '';
    let stderr = '';
    
    tarProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    tarProcess.stderr.on('data', (data) => {
      const output = data.toString();
      stderr += output;
      // tar sends progress to stderr
      console.log(`[GZIP] ${output.trim()}`);
    });
    
    tarProcess.on('close', (code) => {
      if (code === 0) {
        const stats = fs.statSync(outputFile);
        const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
        console.log(`[GZIP] ✓ Archive created successfully: ${sizeMB} MB`);
        resolve(outputFile);
      } else {
        const error = new Error(`tar failed with exit code ${code}`);
        error.code = code;
        error.stdout = stdout;
        error.stderr = stderr;
        console.error(`[GZIP] ✗ Archive creation failed: ${error.message}`);
        if (stderr) {
          console.error(`[GZIP] stderr: ${stderr}`);
        }
        reject(error);
      }
    });
    
    tarProcess.on('error', (error) => {
      console.error(`[GZIP] ✗ Failed to start tar: ${error.message}`);
      reject(error);
    });
  });
}

// Helper function to upload gzipped adapter and extract it remotely
async function uploadAdapterWithRsync(localPath, remotePath, host, sshPort, username, sshKeyPath, onProgress) {
  const tempArchivePath = path.join(os.tmpdir(), `adapter_${Date.now()}_${Math.random().toString(36).substring(7)}.tar.gz`);
  const remoteArchivePath = `${remotePath}.tar.gz`;
  let archiveCreated = false;
  
  try {
    // Step 1: Create gzip archive locally
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] Step 1: Creating gzip archive`);
    console.log(`[UPLOAD] ========================================`);
    await createGzipArchive(localPath, tempArchivePath);
    archiveCreated = true;
    
    const archiveStats = fs.statSync(tempArchivePath);
    const archiveSizeMB = (archiveStats.size / (1024 * 1024)).toFixed(2);
    console.log(`[UPLOAD] Archive size: ${archiveSizeMB} MB`);
    
    // Step 2: Upload the gzip file using rsync
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] Step 2: Uploading gzip archive with rsync`);
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] Local archive: ${tempArchivePath}`);
    console.log(`[UPLOAD] Remote archive: ${username}@${host}:${remoteArchivePath}`);
    
    await new Promise((resolve, reject) => {
      const sshCommand = `ssh -p ${sshPort} -i "${sshKeyPath}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR`;
      const rsyncArgs = [
        '-avz',
        '--progress',
        '--partial',
        '--partial-dir=.rsync-partial',
        '-e', sshCommand,
        tempArchivePath,
        `${username}@${host}:${remoteArchivePath}`
      ];
      
      console.log(`[UPLOAD] Running: rsync ${rsyncArgs.join(' ')}`);
      
      const rsyncProcess = spawn('rsync', rsyncArgs, {
        stdio: ['ignore', 'pipe', 'pipe']
      });
      
      let stdout = '';
      let stderr = '';
      let lastProgressUpdate = Date.now();
      
      rsyncProcess.stdout.on('data', (data) => {
        const output = data.toString();
        stdout += output;
        
        // Parse rsync progress output
        // Format: "filename\n  1,234,567  50%  123.45kB/s    0:00:10"
        const lines = output.split('\n');
        for (const line of lines) {
          if (line.trim()) {
            // Check if it's a progress line (contains %)
            if (line.includes('%')) {
              const now = Date.now();
              // Log progress every 2 seconds
              if (now - lastProgressUpdate > 2000) {
                const progressInfo = line.trim();
                console.log(`[UPLOAD] [Progress] ${progressInfo}`);
                if (onProgress) {
                  onProgress(progressInfo);
                }
                lastProgressUpdate = now;
              }
            } else if (!line.startsWith(' ') && line.trim()) {
              // File name line
              console.log(`[UPLOAD] ${line.trim()}`);
            }
          }
        }
      });
      
      rsyncProcess.stderr.on('data', (data) => {
        const output = data.toString();
        stderr += output;
        // rsync sends some info to stderr (like connection info)
        if (!output.includes('Warning: Permanently added')) {
          console.log(`[UPLOAD] ${output.trim()}`);
        }
      });
      
      rsyncProcess.on('close', (code) => {
        if (code === 0) {
          console.log(`[UPLOAD] ✓ Upload completed successfully`);
          resolve({ stdout, stderr, code });
        } else {
          const error = new Error(`rsync failed with exit code ${code}`);
          error.code = code;
          error.stdout = stdout;
          error.stderr = stderr;
          console.error(`[UPLOAD] ✗ Upload failed: ${error.message}`);
          if (stderr) {
            console.error(`[UPLOAD] stderr: ${stderr}`);
          }
          reject(error);
        }
      });
      
      rsyncProcess.on('error', (error) => {
        console.error(`[UPLOAD] ✗ Failed to start rsync: ${error.message}`);
        reject(error);
      });
    });
    
    // Step 3: Extract the archive on remote server
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] Step 3: Extracting archive on remote server`);
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] Remote archive: ${remoteArchivePath}`);
    console.log(`[UPLOAD] Remote destination: ${remotePath}`);
    
    // Create SSH connection for extraction
    const ssh = new NodeSSH();
    const connectOptions = {
      host: host,
      port: sshPort,
      username: username,
      readyTimeout: 10000,
    };
    
    if (sshKeyPath) {
      try {
        connectOptions.privateKey = fs.readFileSync(sshKeyPath, 'utf8');
      } catch (error) {
        console.warn(`[UPLOAD] Could not read SSH key: ${error.message}`);
      }
    }
    
    const preparedOptions = prepareSSHConnectionOptions(connectOptions);
    await ssh.connect(preparedOptions);
    console.log(`[UPLOAD] ✓ SSH connection established for extraction`);
    
    try {
      // Create parent directory if it doesn't exist
      const parentDir = path.dirname(remotePath).replace(/\\/g, '/');
      console.log(`[UPLOAD] Creating parent directory: ${parentDir}`);
      const mkdirResult = await ssh.execCommand(`mkdir -p "${parentDir}"`);
      if (mkdirResult.code !== 0) {
        throw new Error(`Failed to create parent directory: ${mkdirResult.stderr}`);
      }
      console.log(`[UPLOAD] ✓ Parent directory created`);
      
      // Remove existing destination if it exists
      console.log(`[UPLOAD] Removing existing destination (if any): ${remotePath}`);
      await ssh.execCommand(`rm -rf "${remotePath}"`);
      console.log(`[UPLOAD] ✓ Cleaned up existing destination`);
      
      // Extract the archive
      console.log(`[UPLOAD] Extracting archive...`);
      const extractCmd = `cd "${parentDir}" && tar -xzf "${remoteArchivePath}"`;
      console.log(`[UPLOAD] Running: ${extractCmd}`);
      const extractResult = await ssh.execCommand(extractCmd);
      
      if (extractResult.code !== 0) {
        throw new Error(`Extraction failed: ${extractResult.stderr || 'Unknown error'}`);
      }
      console.log(`[UPLOAD] ✓ Archive extracted successfully`);
      
      // Verify extraction
      console.log(`[UPLOAD] Verifying extracted files...`);
      const verifyCmd = `test -d "${remotePath}" && echo "EXISTS" || echo "NOT_FOUND"`;
      const verifyResult = await ssh.execCommand(verifyCmd);
      if (!verifyResult.stdout.includes('EXISTS')) {
        throw new Error(`Extraction verification failed: directory not found at ${remotePath}`);
      }
      console.log(`[UPLOAD] ✓ Extraction verified: ${remotePath} exists`);
      
      // Clean up remote archive
      console.log(`[UPLOAD] Cleaning up remote archive...`);
      await ssh.execCommand(`rm -f "${remoteArchivePath}"`);
      console.log(`[UPLOAD] ✓ Remote archive cleaned up`);
      
    } finally {
      ssh.dispose();
    }
    
    // Step 4: Clean up local archive
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] Step 4: Cleaning up local archive`);
    console.log(`[UPLOAD] ========================================`);
    if (fs.existsSync(tempArchivePath)) {
      fs.unlinkSync(tempArchivePath);
      console.log(`[UPLOAD] ✓ Local archive cleaned up`);
    }
    
    console.log(`[UPLOAD] ========================================`);
    console.log(`[UPLOAD] ✓ All steps completed successfully`);
    console.log(`[UPLOAD] ========================================`);
    
  } catch (error) {
    // Clean up on error
    if (archiveCreated && fs.existsSync(tempArchivePath)) {
      try {
        fs.unlinkSync(tempArchivePath);
        console.log(`[UPLOAD] Cleaned up local archive after error`);
      } catch (cleanupError) {
        console.warn(`[UPLOAD] Could not clean up local archive: ${cleanupError.message}`);
      }
    }
    throw error;
  }
}

// Establish SSH tunnel for port forwarding
async function establishSSHTunnel(host, sshPort, username, sshKeyPath, remotePort = 8888) {
  // Clean up existing tunnel if any
  await cleanupSSHTunnel();
  
  try {
    // INVARIANT: Always bind to port 8888 locally to match remote port
    // Do not use dynamic port selection - localhost:8888 must be consistent
    const localPort = 8888;
    console.log(`[SSH TUNNEL] Binding to local port ${localPort} (hard-bound, no dynamic selection)`);
    
    // Build SSH command for port forwarding: ssh -L 127.0.0.1:8888:127.0.0.1:8888
    const sshArgs = [
      '-N', // No remote command execution
      '-L', `127.0.0.1:${localPort}:127.0.0.1:${remotePort}`, // Always 8888 locally, remote port from user config
      '-o', 'ExitOnForwardFailure=no', // Don't exit if remote port not ready yet - server starts later
      '-o', 'StrictHostKeyChecking=no', // Auto-accept host keys
      '-o', 'UserKnownHostsFile=/dev/null', // Don't save host keys
      '-o', 'LogLevel=ERROR', // Reduce noise
      '-o', 'ServerAliveInterval=10', // Keepalive
      '-o', 'ServerAliveCountMax=3',
      `${username}@${host}`,
      '-p', sshPort.toString()
    ];
    
    // Add SSH key if provided
    if (sshKeyPath && fs.existsSync(sshKeyPath)) {
      sshArgs.splice(-2, 0, '-i', sshKeyPath);
    }
    
    console.log(`[SSH TUNNEL] Establishing tunnel: localhost:${localPort} -> ${host}:${sshPort} -> localhost:${remotePort}`);
    console.log(`[SSH TUNNEL] SSH command: ssh ${sshArgs.join(' ')}`);
    
    // Spawn SSH process
    const sshProcess = spawn('ssh', sshArgs, {
      stdio: ['ignore', 'pipe', 'pipe'] // Ignore stdin, capture stdout/stderr for debugging
    });
    
    // Store tunnel info
    sshTunnelProcess = sshProcess;
    sshTunnelLocalPort = localPort;
    sshTunnelConfig = { host, port: sshPort, username, sshKeyPath, remotePort }; // remotePort is user-supplied internal port
    
    // Handle process events
    sshProcess.stdout.on('data', (data) => {
      console.log(`[SSH TUNNEL] stdout: ${data.toString().trim()}`);
    });
    
    sshProcess.stderr.on('data', (data) => {
      const stderr = data.toString().trim();
      // Ignore common SSH warnings that don't indicate failure
      if (!stderr.includes('Warning: Permanently added') && !stderr.includes('deprecated')) {
        console.log(`[SSH TUNNEL] stderr: ${stderr}`);
      }
    });
    
    sshProcess.on('error', (error) => {
      console.error(`[SSH TUNNEL] Process error: ${error.message}`);
      cleanupSSHTunnel();
    });
    
    sshProcess.on('exit', (code, signal) => {
      console.log(`[SSH TUNNEL] Process exited with code ${code}, signal ${signal}`);
      if (code !== 0 && code !== null) {
        console.error(`[SSH TUNNEL] Tunnel process exited unexpectedly`);
        // Don't immediately cleanup - preserve config for potential reconnection
        // Only clear the process reference
        sshTunnelProcess = null;
      } else {
        // Normal exit (code 0) - full cleanup
        cleanupSSHTunnel();
      }
    });
    
    // Give SSH time to bind the local port; trust the process if still alive
    await new Promise(resolve => setTimeout(resolve, 1500));

    if (sshProcess.killed || sshProcess.exitCode !== null) {
      throw new Error('SSH tunnel process exited during startup');
    }

    console.log(`[SSH TUNNEL] ✓ Tunnel assumed established on localhost:${localPort}`);
    return localPort;
    
  } catch (error) {
    console.error(`[SSH TUNNEL] Failed to establish tunnel: ${error.message}`);
    cleanupSSHTunnel();
    throw error;
  }
}

// Clean up SSH tunnel
async function cleanupSSHTunnel() {
  if (sshTunnelProcess) {
    console.log(`[SSH TUNNEL] Cleaning up tunnel on localhost:${sshTunnelLocalPort}`);
    try {
      sshTunnelProcess.kill('SIGTERM');
      // Wait a bit for graceful shutdown
      await new Promise(resolve => setTimeout(resolve, 500));
      if (sshTunnelProcess && !sshTunnelProcess.killed) {
        sshTunnelProcess.kill('SIGKILL');
      }
    } catch (error) {
      console.error(`[SSH TUNNEL] Error cleaning up tunnel: ${error.message}`);
    }
    sshTunnelProcess = null;
    sshTunnelLocalPort = null;
    sshTunnelConfig = null;
  }
}

// Check if tunnel is alive and reconnect if needed
async function ensureSSHTunnel() {
  // Only check/reconnect if we're using localhost (tunneled connection)
  if (!sshTunnelConfig) {
    return; // No tunnel configured, using direct connection
  }
  
  // Check if tunnel process is still alive
  const isAlive = sshTunnelProcess && 
                  !sshTunnelProcess.killed && 
                  sshTunnelProcess.exitCode === null;
  
  if (!isAlive) {
    console.warn(`[SSH TUNNEL] Tunnel process is not alive, attempting to reconnect...`);
    try {
      const { host, port, username, sshKeyPath, remotePort } = sshTunnelConfig;
      await establishSSHTunnel(host, port, username, sshKeyPath, remotePort);
      console.log(`[SSH TUNNEL] ✓ Tunnel re-established successfully`);
      // Give tunnel a moment to bind
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (error) {
      console.error(`[SSH TUNNEL] ✗ Failed to re-establish tunnel: ${error.message}`);
      throw new Error(`SSH tunnel connection lost and could not be re-established: ${error.message}`);
    }
  }
}

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

// Cleanup SSH tunnel on app quit
app.on('before-quit', async () => {
  console.log('[APP] Cleaning up SSH tunnel before quit...');
  await cleanupSSHTunnel();
});

// Also cleanup on process exit
process.on('exit', () => {
  if (sshTunnelProcess) {
    console.log('[APP] Force cleaning up SSH tunnel on process exit...');
    if (sshTunnelProcess && !sshTunnelProcess.killed) {
      sshTunnelProcess.kill('SIGKILL');
    }
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
        const versionDir = path.join(profileDir, entry.name);
        
        // Check for adapter in the standard location: V{version}/weights/adapter
        const weightsPath = path.join(versionDir, 'weights', 'adapter');
        
        // Also check alternative locations for backward compatibility
        const altPaths = [
          weightsPath,  // Standard: V{version}/weights/adapter
          path.join(versionDir, 'adapter'),  // Alternative: V{version}/adapter
          path.join(versionDir, 'model', 'adapter'),  // Alternative: V{version}/model/adapter
        ];
        
        let foundPath = null;
        let hasEssentialFiles = false;
        
        for (const checkPath of altPaths) {
          if (fs.existsSync(checkPath)) {
            // Verify essential files exist
            const configFile = path.join(checkPath, 'adapter_config.json');
            const safetensorsFile = path.join(checkPath, 'adapter_model.safetensors');
            const binFile = path.join(checkPath, 'adapter_model.bin');
            
            const hasConfig = fs.existsSync(configFile);
            const hasWeights = fs.existsSync(safetensorsFile) || fs.existsSync(binFile);
            
            if (hasConfig && hasWeights) {
              foundPath = checkPath;
              hasEssentialFiles = true;
              console.log(`[APP] ✓ Found version ${versionNum} with essential files at: ${checkPath}`);
              break;
            } else if (fs.existsSync(checkPath) && fs.readdirSync(checkPath).length > 0) {
              // Directory exists but missing essential files
              console.log(`[APP] ⚠ Version ${versionNum} directory exists at ${checkPath} but missing essential files (config: ${hasConfig}, weights: ${hasWeights})`);
            }
          }
        }
        
        if (foundPath && hasEssentialFiles) {
          versions.push({
            version: versionNum,
            path: foundPath,
            exists: true
          });
        } else {
          console.log(`[APP] Version ${versionNum} directory exists but no valid adapter weights found in any expected location`);
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

// NOTE: behavior_packs.json is intentionally treated as passive configuration data.
// main.js is responsible ONLY for loading and returning behavior packs to the renderer.
// All behavior selection, trigger evaluation, exemplar injection, voice resets,
// and drift correction MUST occur in the renderer at prompt-assembly time.
// Do NOT add behavioral logic here or attempt to enforce exemplars in main.js.
// Load behavior packs configuration for a specific profile
ipcMain.handle('load-behavior-packs', async (event, profileName) => {
  try {
    if (!profileName) {
      console.log('[BEHAVIOR] No profile name provided, using defaults');
      return {
        success: true,
        data: {
          behavior_version: "1.0",
          default_mode: "coaching",
          exemplars: {}
        },
        isEmpty: true
      };
    }
    
    const behaviorPacksPath = path.join(MODELS_DIR, profileName, 'behavior_packs.json');
    if (fs.existsSync(behaviorPacksPath)) {
      const behaviorPacks = JSON.parse(fs.readFileSync(behaviorPacksPath, 'utf8'));
      console.log(`[BEHAVIOR] behavior_packs.json loaded for profile "${profileName}". Renderer is responsible for applying exemplars.`);
      
      // Check if pack is empty - handle both old and new schema formats
      let isEmpty = true;
      
      // New schema: check triggers with exemplars arrays
      if (behaviorPacks.triggers && typeof behaviorPacks.triggers === 'object') {
        const hasContent = Object.values(behaviorPacks.triggers).some(trigger => {
          if (trigger.exemplars && Array.isArray(trigger.exemplars)) {
            return trigger.exemplars.some(ex => {
              if (typeof ex === 'string') return ex.trim().length > 0;
              if (ex && typeof ex === 'object' && ex.text) return ex.text.trim().length > 0;
              return false;
            });
          }
          return false;
        });
        isEmpty = !hasContent;
      }
      // Old schema: check exemplars object directly
      else if (behaviorPacks.exemplars && typeof behaviorPacks.exemplars === 'object') {
        const hasContent = Object.values(behaviorPacks.exemplars).some(ex => {
          if (typeof ex === 'string') return ex.trim().length > 0;
          if (ex && typeof ex === 'object') {
            if (Array.isArray(ex.text)) {
              return ex.text.some(t => typeof t === 'string' && t.trim().length > 0);
            }
            return ex.text && typeof ex.text === 'string' && ex.text.trim().length > 0;
          }
          return false;
        });
        isEmpty = !hasContent;
      }
      
      return {
        success: true,
        data: behaviorPacks,
        isEmpty: isEmpty
      };
    } else {
      console.log(`[BEHAVIOR] No behavior_packs.json file found for profile ${profileName}, using defaults`);
      return {
        success: true,
        data: {
          behavior_version: "1.0",
          default_mode: "coaching",
          exemplars: {}
        },
        isEmpty: true
      };
    }
  } catch (error) {
    console.error('[BEHAVIOR] Error loading behavior packs:', error.message);
    return {
      success: false,
      error: error.message,
      data: null,
      isEmpty: true
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
      externalPort: config.externalPort !== undefined && config.externalPort !== null ? config.externalPort : null,
      internalPort: config.internalPort !== undefined && config.internalPort !== null ? config.internalPort : 8080,
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
    
    // Prepare connection options with auto-accept host keys
    const finalConnectOptions = prepareSSHConnectionOptions(connectOptions);
    await ssh.connect(finalConnectOptions);
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
    
    // Prepare connection options with auto-accept host keys
    const finalConnectOptions = prepareSSHConnectionOptions(connectOptions);
    await ssh.connect(finalConnectOptions);
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

// Initialize Inference Environment - comprehensive setup
ipcMain.handle('initialize-inference-environment', async (event, { host, port, username = 'root', profileName, baseModel, versions, inferenceUrl, vllmUrl, externalPort, internalPort }) => {
  // Determine ports: use provided values or defaults
  const internalPortValue = internalPort || 8080; // Default internal port for server
  const externalPortValue = externalPort || null; // External port for API calls
  
  // Support both inferenceUrl and vllmUrl for backward compatibility
  let serverUrl = inferenceUrl || vllmUrl;
  if (serverUrl && !serverUrl.match(/^https?:\/\//i)) {
    serverUrl = 'http://' + serverUrl;
  }
  
  // If external port is provided, construct URL from it
  if (externalPortValue && !serverUrl) {
    serverUrl = `http://${host}:${externalPortValue}`;
  }
  
  inferenceUrl = serverUrl || (externalPortValue ? `http://${host}:${externalPortValue}` : `http://${host}:8080`);
  
  const ssh = new NodeSSH();
  // Increase max listeners to prevent memory leak warnings during uploads
  if (ssh.connection && ssh.connection.setMaxListeners) {
    ssh.connection.setMaxListeners(20);
  }
  const results = {
    success: false,
    steps: [],
    errors: [],
    inferenceUrl: null
  };
  
  let pythonCmd = 'python3';
  let hfToken = null;
  let actualBaseModel = baseModel;
  let openButtonToken = null;
  
  // Expand short model names to full HuggingFace IDs (e.g., "gemma3" -> "google/gemma-3-4b-it")
  const MODEL_MAPPING = {
    "gemma3": "google/gemma-3-4b-it",
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma-3": "google/gemma-3-4b-it",
    "gemma-3:4b": "google/gemma-3-4b-it",
    "gemma2": "google/gemma-2-2b-it",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "gemma2:27b": "google/gemma-2-27b-it",
    // Default Gemma to instruction-tuned variants for chat. Base (non -it) models will often look like "gibberish"
    // when driven with chat-style prompts.
    "gemma": "google/gemma-7b-it",
    "gemma:7b": "google/gemma-7b-it",
    "gemma:2b": "google/gemma-2b-it",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
  };
  
  // Expand model name if it's a short name
  if (baseModel && !baseModel.includes('/')) {
    const normalized = baseModel.toLowerCase();
    if (MODEL_MAPPING[normalized]) {
      actualBaseModel = MODEL_MAPPING[normalized];
      console.log(`[INIT] Expanded model name: ${baseModel} -> ${actualBaseModel}`);
    }
  }

  // Helpful warning: Gemma base models (non -it) are not instruction tuned and often produce low-quality chat output.
  if (typeof actualBaseModel === 'string' &&
      actualBaseModel.startsWith('google/gemma') &&
      !actualBaseModel.includes('-it')) {
    console.warn(`[INIT] ⚠ Selected Gemma model is not instruction-tuned (-it): ${actualBaseModel}`);
    console.warn(`[INIT] ⚠ For chat, prefer e.g. google/gemma-2-2b-it, google/gemma-7b-it, or google/gemma-3-4b-it`);
  }
  
  console.log('\n[INIT] ========================================');
  console.log('[INIT] Initializing Inference Environment');
  console.log('[INIT] ========================================');
  console.log(`[INIT] Host: ${host}`);
  console.log(`[INIT] Port: ${port || 22}`);
  console.log(`[INIT] Username: ${username}`);
  if (profileName) {
    console.log(`[INIT] Profile: ${profileName}`);
    console.log(`[INIT] Base Model: ${baseModel}`);
    console.log(`[INIT] Versions: ${versions ? versions.map(v => `V${v.version}`).join(', ') : 'none'}`);
  }
  console.log('');
  
  results.steps.push('Starting inference environment initialization...');
  
  // Get SSH key path
  const sshKeyPath = findSSHKey();
  const sshPort = port || 22;
  console.log(`[INIT] SSH connection port: ${sshPort} (provided: ${port || 'default'})`);
  const connectOptions = {
    host: host,
    port: sshPort,
    username: username,
    readyTimeout: 10000,
  };
  
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
    // Prepare connection options with auto-accept host keys
    const finalConnectOptions = prepareSSHConnectionOptions(connectOptions);
    await ssh.connect(finalConnectOptions);
    // Increase max listeners to prevent memory leak warnings during file operations
    if (ssh.connection && typeof ssh.connection.setMaxListeners === 'function') {
      ssh.connection.setMaxListeners(20);
    }
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
    
    // STEP 1: Test SSH Connection
    if (result.code !== 0) {
      const errorMsg = `SSH connection test failed: ${result.stderr || 'Unknown error'}`;
      console.error(`[INIT] ✗ ${errorMsg}`);
      results.errors.push(errorMsg);
      results.steps.push(`✗ ${errorMsg}`);
      ssh.dispose();
      return results;
    }
    
    console.log('[INIT] ✓ Step 1: SSH connection successful');
    results.steps.push('✓ Step 1: SSH connection successful');
    
    // STEP 1.5: Establish SSH tunnel immediately after SSH connection
    // INVARIANT: All FastAPI access MUST go through the SSH tunnel.
    // localhost:8888 must mean the same thing for the entire app lifecycle.
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Step 1.5: Establishing SSH Tunnel (Tunnel First)');
    console.log('[INIT] ========================================');
    results.steps.push('Step 1.5: Establishing SSH tunnel...');
    
    // Only establish tunnel if no external port mapping exists
    if (!externalPortValue) {
      try {
        const sshKeyPath = findSSHKey();
        const remoteInferencePort = internalPortValue; // Use user-supplied internal port
        
        console.log(`[INIT] [TUNNEL] Setting up tunnel: localhost:8888 -> ${host}:${remoteInferencePort}`);
        await establishSSHTunnel(
          host,
          port || 22,
          username || 'root',
          sshKeyPath,
          remoteInferencePort
        );
        
        console.log(`[INIT] ✓ SSH tunnel established: localhost:8888 -> ${host}:${remoteInferencePort}`);
        results.steps.push(`✓ SSH tunnel established on localhost:8888`);
        
        // Hard-bind inference URL - use 127.0.0.1 to force IPv4 (tunnel is bound to IPv4)
        results.inferenceUrl = 'http://127.0.0.1:8888';
        console.log(`[INIT] Inference URL hard-bound to: http://127.0.0.1:8888`);
        
      } catch (tunnelError) {
        console.error(`[INIT] ✗ Failed to establish SSH tunnel: ${tunnelError.message}`);
        console.error(`[INIT] Tunnel error details:`, tunnelError);
        results.errors.push(`SSH tunnel failed: ${tunnelError.message}`);
        results.steps.push(`✗ SSH tunnel failed: ${tunnelError.message}`);
        ssh.dispose();
        return results;
      }
    } else {
      // External port mapping exists - use it directly
      const externalUrl = `http://${host}:${externalPortValue}`;
      console.log(`[INIT] [EXTERNAL PORT] External port mapping detected: ${externalUrl}`);
      console.log(`[INIT] [EXTERNAL PORT] Using external port directly (no SSH tunnel needed)`);
      results.inferenceUrl = externalUrl;
      results.steps.push(`✓ Using external port mapping ${externalPortValue} (no SSH tunnel needed)`);
    }
    
    // STEP 2: Check and Install FastAPI Dependencies
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Step 2: Ensuring FastAPI dependencies are installed');
    console.log('[INIT] ========================================');
    results.steps.push('Step 2: Checking FastAPI dependencies...');
    
    try {
      // Check which Python to use
      const pythonCheck = await ssh.execCommand('which python3 || which python || echo "not_found"');
      if (pythonCheck.stdout.includes('not_found')) {
        const errorMsg = 'Python not found on remote system';
        console.error(`[INIT] ✗ ${errorMsg}`);
        results.errors.push(errorMsg);
        results.steps.push(`✗ ${errorMsg}`);
        ssh.dispose();
        return results;
      }
      
      if (pythonCheck.stdout.includes('python')) {
        pythonCmd = pythonCheck.stdout.trim();
      }
      console.log(`[INIT] Using Python: ${pythonCmd}`);
      results.steps.push(`Using Python: ${pythonCmd}`);
      
      // Required packages for PEFT inference
      const requiredPackages = {
        'transformers': 'transformers',
        'peft': 'peft',
        'torch': 'torch',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'accelerate': 'accelerate',
        'huggingface-hub': 'huggingface_hub'
      };
      
      // Check each package
      const missingPackages = [];
      const installedPackages = [];
      
      console.log(`[INIT] Checking ${Object.keys(requiredPackages).length} required packages...`);
      for (const [pkgName, importName] of Object.entries(requiredPackages)) {
        const checkCmd = `${pythonCmd} -c "import ${importName}; print('installed')" 2>&1`;
        const checkResult = await ssh.execCommand(checkCmd);
        
        if (checkResult.code === 0 && checkResult.stdout.includes('installed')) {
          installedPackages.push(pkgName);
          console.log(`[INIT] ✓ ${pkgName} is installed`);
        } else {
          missingPackages.push(pkgName);
          console.log(`[INIT] ✗ ${pkgName} is missing`);
        }
      }
      
      console.log(`[INIT] Package status: ${installedPackages.length} installed, ${missingPackages.length} missing`);
      
      // Install missing packages if any
      if (missingPackages.length > 0) {
        console.log(`[INIT] Installing missing packages: ${missingPackages.join(', ')}...`);
        results.steps.push(`Installing ${missingPackages.length} missing package(s)...`);
        
        const pipCheck = await ssh.execCommand('which pip3 || which pip || echo "not_found"');
        let pipCmd = 'pip3';
        if (pipCheck.stdout.includes('not_found')) {
          const errorMsg = 'pip not found - cannot install packages';
          console.error(`[INIT] ✗ ${errorMsg}`);
          results.errors.push(errorMsg);
          results.steps.push(`✗ ${errorMsg}`);
          ssh.dispose();
          return results;
        }
        if (pipCheck.stdout.includes('pip')) {
          pipCmd = pipCheck.stdout.trim();
        }
        
        const installCmd = `${pipCmd} install --quiet ${missingPackages.join(' ')}`;
        console.log(`[INIT] Running: ${installCmd}`);
        const installResult = await ssh.execCommand(installCmd);
        
        if (installResult.code !== 0) {
          const errorMsg = `Failed to install packages: ${installResult.stderr || installResult.stdout || 'Unknown error'}`;
          console.error(`[INIT] ✗ ${errorMsg.substring(0, 500)}`);
          results.errors.push(errorMsg.substring(0, 300));
          results.steps.push(`✗ Package installation failed`);
          ssh.dispose();
          return results;
        }
        
        // Verify installation
        for (const pkg of missingPackages) {
          const importName = requiredPackages[pkg];
          const verifyCmd = `${pythonCmd} -c "import ${importName}; print('installed')" 2>&1`;
          const verifyResult = await ssh.execCommand(verifyCmd);
          if (verifyResult.code !== 0 || !verifyResult.stdout.includes('installed')) {
            const errorMsg = `Package ${pkg} installation verification failed`;
            console.error(`[INIT] ✗ ${errorMsg}`);
            results.errors.push(errorMsg);
            results.steps.push(`✗ ${errorMsg}`);
            ssh.dispose();
            return results;
          }
        }
        
        console.log(`[INIT] ✓ Successfully installed ${missingPackages.length} package(s)`);
        results.steps.push(`✓ Installed ${missingPackages.length} missing package(s)`);
      } else {
        console.log(`[INIT] ✓ All required packages are installed`);
        results.steps.push(`✓ All required packages are installed`);
      }
      
      
      // Check CUDA
      const cudaCheck = await ssh.execCommand(`${pythonCmd} -c "import torch; print('cuda_available' if torch.cuda.is_available() else 'cpu_only')" 2>&1`);
      if (cudaCheck.code === 0) {
        if (cudaCheck.stdout.includes('cuda_available')) {
          const gpuInfo = await ssh.execCommand(`${pythonCmd} -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1`);
          const gpuName = gpuInfo.stdout.trim();
          const cudaVersion = await ssh.execCommand(`${pythonCmd} -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1`);
          const cudaVer = cudaVersion.stdout.trim();
          console.log(`[INIT] ✓ CUDA available - GPU: ${gpuName}, CUDA: ${cudaVer}`);
          results.steps.push(`✓ CUDA available - GPU: ${gpuName}`);
        } else {
          console.log(`[INIT] ⚠ CUDA not available - will use CPU (slower)`);
          results.steps.push(`⚠ CUDA not available - will use CPU`);
        }
      }
      
      console.log('[INIT] ✓ FastAPI dependencies ready');
      results.steps.push('✓ FastAPI dependencies ready');
      
    } catch (depError) {
      const errorMsg = `Dependency check/installation failed: ${depError.message}`;
      console.error(`[INIT] ✗ ${errorMsg}`);
      console.error(`[INIT] Error details:`, depError);
      results.errors.push(errorMsg);
      results.steps.push(`✗ ${errorMsg}`);
      ssh.dispose();
      return results;
    }
    
    console.log('[INIT] ✓ Step 2: Complete (dependencies)');
    results.steps.push('✓ Step 2: Complete');
    
    // If no profile/model info provided, we're done (just SSH + dependencies)
    if (!profileName || !baseModel) {
      const serverUrl = inferenceUrl || `http://${host}:8080`;
      results.inferenceUrl = serverUrl;
      results.success = true;
      results.steps.push(`✓ Initialization complete (SSH + dependencies only)`);
      console.log('[INIT] ✓ Initialization complete (no model/profile specified)');
      ssh.dispose();
      return results;
    }
    
    // STEP 3: Ensure Base Model is Cached (BEFORE starting server)
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Step 3: Ensuring base model is cached');
    console.log('[INIT] ========================================');
    results.steps.push('Step 3: Checking base model cache...');
    
    try {
      // Read HF token
      if (fs.existsSync(APP_CONFIG_FILE)) {
        const appConfig = JSON.parse(fs.readFileSync(APP_CONFIG_FILE, 'utf8'));
        if (appConfig.hf_token) {
          hfToken = appConfig.hf_token;
          console.log(`[INIT] ✓ Found HF token`);
        }
      }
      
      // Determine actual base model from adapter config if available
      if (versions && versions.length > 0 && versions[0].exists) {
        try {
          const firstAdapterConfigPath = path.join(versions[0].path, 'adapter_config.json');
          if (fs.existsSync(firstAdapterConfigPath)) {
            const adapterConfig = JSON.parse(fs.readFileSync(firstAdapterConfigPath, 'utf8'));
            if (adapterConfig.base_model_name_or_path && adapterConfig.base_model_name_or_path.includes('/')) {
              actualBaseModel = adapterConfig.base_model_name_or_path;
              console.log(`[INIT] Using base model from adapter config: ${actualBaseModel}`);
            }
          }
        } catch (e) {
          console.log(`[INIT] Using provided base model: ${baseModel}`);
        }
      }
      
      // Check if model is cached
      const cacheCheckCmd = `${pythonCmd} -c "
import os
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
model_cache_path = os.path.join(cache_dir, 'models--${actualBaseModel.replace(/\//g, '--')}')
if os.path.exists(model_cache_path):
    files = [f for f in os.listdir(model_cache_path) if not f.startswith('.')]
    if files:
        print('CACHED')
    else:
        print('NOT_CACHED')
else:
    print('NOT_CACHED')
" 2>&1`;
      
      const cacheCheck = await ssh.execCommand(cacheCheckCmd);
      const isCached = cacheCheck.stdout.includes('CACHED');
      
      if (isCached) {
        console.log(`[INIT] ✓ Base model '${actualBaseModel}' is already cached`);
        results.steps.push(`✓ Base model '${actualBaseModel}' is cached`);
      } else {
        console.log(`[INIT] Base model not cached, downloading...`);
        results.steps.push(`Downloading base model '${actualBaseModel}' (this may take several minutes)...`);
        
        const escapedToken = hfToken ? hfToken.replace(/'/g, "'\\''") : '';
        const downloadCmd = `${pythonCmd} -c "
import os
import sys
from huggingface_hub import snapshot_download
try:
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        print('Using HF_TOKEN for authentication...', file=sys.stderr)
    print('Starting model download...', file=sys.stderr)
    snapshot_download(
        '${actualBaseModel}',
        token=hf_token if hf_token else None,
        local_files_only=False,
        resume_download=True
    )
    print('DOWNLOAD_COMPLETE')
except Exception as e:
    print(f'DOWNLOAD_ERROR: {str(e)}', file=sys.stderr)
    sys.exit(1)
" 2>&1`;
        
        const downloadCommandWithEnv = hfToken
          ? `HF_TOKEN='${escapedToken}' ${downloadCmd}`
          : downloadCmd;
        
        console.log(`[INIT] Starting model download (this may take 5-10 minutes)...`);
        const downloadResult = await ssh.execCommand(downloadCommandWithEnv);
        
        if (downloadResult.code !== 0 || !downloadResult.stdout.includes('DOWNLOAD_COMPLETE')) {
          const errorMsg = `Model download failed: ${downloadResult.stderr || downloadResult.stdout || 'Unknown error'}`;
          console.error(`[INIT] ✗ ${errorMsg.substring(0, 500)}`);
          results.errors.push(errorMsg.substring(0, 300));
          results.steps.push(`✗ Model download failed`);
          ssh.dispose();
          return results;
        }
        
        console.log(`[INIT] ✓ Base model downloaded successfully`);
        results.steps.push(`✓ Base model downloaded successfully`);
      }
      
      console.log('[INIT] ✓ Step 3: Base model ready');
      results.steps.push('✓ Step 3: Base model ready');
      
    } catch (modelError) {
      const errorMsg = `Base model check/download failed: ${modelError.message}`;
      console.error(`[INIT] ✗ ${errorMsg}`);
      results.errors.push(errorMsg);
      results.steps.push(`✗ ${errorMsg}`);
      ssh.dispose();
      return results;
    }
    
    // STEP 4: Upload LoRA Adapters (BEFORE starting server)
    if (versions && versions.length > 0) {
      console.log('\n[INIT] ========================================');
      console.log('[INIT] Step 4: Uploading LoRA adapter versions');
      console.log('[INIT] ========================================');
      results.steps.push(`Step 4: Uploading ${versions.length} LoRA version(s)...`);
      
      const remoteModelDir = `/workspace/models/${profileName}`;
      
      // Create model directory
      const mkdirResult = await ssh.execCommand(`mkdir -p ${remoteModelDir}`);
      if (mkdirResult.code !== 0) {
        const errorMsg = `Failed to create model directory: ${mkdirResult.stderr || 'Unknown error'}`;
        console.error(`[INIT] ✗ ${errorMsg}`);
        results.errors.push(errorMsg);
        results.steps.push(`✗ ${errorMsg}`);
        ssh.dispose();
        return results;
      }
      console.log(`[INIT] ✓ Created directory: ${remoteModelDir}`);
      
      // Upload each version
      for (const version of versions) {
        if (!version.exists) {
          console.log(`[INIT] ⚠ Skipping V${version.version} - not found locally`);
          continue;
        }
        
        const versionName = `V${version.version}`;
        const remoteVersionDir = `${remoteModelDir}/${versionName}`;
        const remoteAdapterPath = `${remoteVersionDir}/adapter`;
        
        console.log(`[INIT] ========================================`);
        console.log(`[INIT] Starting upload for ${versionName}`);
        console.log(`[INIT] Local path: ${version.path}`);
        console.log(`[INIT] Remote path: ${remoteAdapterPath}`);
        console.log(`[INIT] SSH connection: ${ssh.isConnected ? 'connected' : 'DISCONNECTED'}`);
        console.log(`[INIT] ========================================`);
        results.steps.push(`Uploading ${versionName} adapter...`);
        
        try {
          // Verify local path exists
          if (!fs.existsSync(version.path)) {
            throw new Error(`Local adapter path does not exist: ${version.path}`);
          }
          console.log(`[INIT] ✓ Local path verified: ${version.path}`);
          
          // Verify essential adapter files exist before upload
          const essentialFiles = [
            'adapter_config.json',
            'adapter_model.safetensors',
            'adapter_model.bin'
          ];
          
          const foundEssentialFiles = essentialFiles.filter(fileName => {
            const filePath = path.join(version.path, fileName);
            return fs.existsSync(filePath);
          });
          
          if (foundEssentialFiles.length === 0) {
            throw new Error(`No essential adapter files found in ${version.path}. Expected at least one of: ${essentialFiles.join(', ')}`);
          }
          
          console.log(`[INIT] ✓ Found essential files: ${foundEssentialFiles.join(', ')}`);
          
          // Verify SSH connection before proceeding
          if (!ssh.isConnected) {
            throw new Error('SSH connection lost before upload. Please reconnect.');
          }
          
          // Create remote directory
          console.log(`[INIT] Creating remote directory: ${remoteVersionDir}`);
          const mkdirResult = await ssh.execCommand(`mkdir -p ${remoteVersionDir}`);
          if (mkdirResult.code !== 0) {
            throw new Error(`Failed to create remote directory: ${mkdirResult.stderr || 'Unknown error'}`);
          }
          console.log(`[INIT] ✓ Remote directory created`);
          
          // Calculate total size and file count before upload
          const getAllFiles = (dir) => {
            const files = [];
            try {
              const entries = fs.readdirSync(dir, { withFileTypes: true });
              for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                if (entry.isDirectory()) {
                  files.push(...getAllFiles(fullPath));
                } else if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
                  files.push(fullPath);
                }
              }
            } catch (error) {
              console.error(`[INIT] Error reading directory ${dir}:`, error.message);
            }
            return files;
          };
          
          const allFiles = getAllFiles(version.path);
          
          if (allFiles.length === 0) {
            throw new Error(`No files found in adapter directory: ${version.path}`);
          }
          let totalSize = 0;
          const localFileSizes = new Map(); // Map of relative path -> size
          
          for (const file of allFiles) {
            try {
              const stats = fs.statSync(file);
              totalSize += stats.size;
              const relativePath = path.relative(version.path, file).replace(/\\/g, '/');
              localFileSizes.set(relativePath, stats.size);
            } catch (e) {
              // Ignore errors
            }
          }
          
          const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
          console.log(`[INIT] ${versionName} adapter: ${allFiles.length} file(s), ${totalSizeMB} MB total`);
          
          // Check if files already exist on server with matching sizes
          console.log(`[INIT] [CHECK] Checking if ${versionName} adapter already exists on server...`);
          let allFilesMatch = true;
          let filesToUpload = [];
          
          for (const localFile of allFiles) {
            const relativePath = path.relative(version.path, localFile).replace(/\\/g, '/');
            const remoteFile = path.join(remoteAdapterPath, relativePath).replace(/\\/g, '/');
            const localSize = localFileSizes.get(relativePath);
            
            try {
              // Check if file exists and get its size
              const sizeCheckCmd = `stat -c %s "${remoteFile}" 2>/dev/null || echo "NOT_FOUND"`;
              const sizeCheckResult = await ssh.execCommand(sizeCheckCmd);
              
              if (sizeCheckResult.stdout.trim() === 'NOT_FOUND' || sizeCheckResult.code !== 0) {
                console.log(`[INIT] [CHECK] File not found on server: ${relativePath}`);
                allFilesMatch = false;
                filesToUpload.push(localFile);
              } else {
                const remoteSize = parseInt(sizeCheckResult.stdout.trim());
                if (isNaN(remoteSize) || remoteSize !== localSize) {
                  console.log(`[INIT] [CHECK] Size mismatch for ${relativePath}: local=${localSize}, remote=${remoteSize}`);
                  allFilesMatch = false;
                  filesToUpload.push(localFile);
                } else {
                  console.log(`[INIT] [CHECK] ✓ File matches: ${relativePath} (${localSize} bytes)`);
                }
              }
            } catch (checkError) {
              console.log(`[INIT] [CHECK] Error checking file ${relativePath}: ${checkError.message}`);
              allFilesMatch = false;
              filesToUpload.push(localFile);
            }
          }
          
          if (allFilesMatch && filesToUpload.length === 0) {
            console.log(`[INIT] ✓ ${versionName} adapter already exists on server with matching file sizes - skipping upload`);
            console.log(`[INIT] [CHECK] Summary: All ${allFiles.length} files match, no upload needed`);
            results.steps.push(`✓ ${versionName} adapter already exists - skipped upload`);
            continue; // Skip to next version
          }
          
          // Log summary of file check results
          const existingCount = allFiles.length - filesToUpload.length;
          console.log(`[INIT] [CHECK] Summary: ${existingCount}/${allFiles.length} files already exist, ${filesToUpload.length} files need upload`);
          
          const filesToUploadCount = filesToUpload.length;
          const filesToUploadSize = filesToUpload.reduce((sum, file) => {
            try {
              const stats = fs.statSync(file);
              return sum + stats.size;
            } catch (e) {
              return sum;
            }
          }, 0);
          const filesToUploadSizeMB = (filesToUploadSize / (1024 * 1024)).toFixed(2);
          
          if (filesToUploadCount < allFiles.length) {
            console.log(`[INIT] ${versionName} adapter: ${filesToUploadCount}/${allFiles.length} files need upload (${filesToUploadSizeMB} MB)`);
            results.steps.push(`Uploading ${versionName} adapter (${filesToUploadCount}/${allFiles.length} files, ${filesToUploadSizeMB} MB)...`);
          } else {
          console.log(`[INIT] Starting upload of ${versionName} adapter...`);
          results.steps.push(`Uploading ${versionName} adapter (${allFiles.length} files, ${totalSizeMB} MB)...`);
          }
          
          // Upload using rsync (handles incremental transfers, progress, and checksums automatically)
          const uploadSshKeyPath = findSSHKey();
          if (!uploadSshKeyPath) {
            throw new Error('SSH key not found - cannot use rsync');
          }
          
          const uploadSshPort = port || 22;
          let uploadSuccess = false;
          let uploadAttempts = 0;
          const maxUploadAttempts = 3;
          
          while (!uploadSuccess && uploadAttempts < maxUploadAttempts) {
            uploadAttempts++;
            
            if (uploadAttempts > 1) {
              console.log(`[INIT] Retrying ${versionName} upload with rsync (attempt ${uploadAttempts}/${maxUploadAttempts})...`);
              results.steps.push(`Retrying ${versionName} upload (attempt ${uploadAttempts}/${maxUploadAttempts})...`);
              await new Promise(resolve => setTimeout(resolve, 3000)); // Wait before retry
            }
            
            try {
              // Progress callback for rsync
              let lastProgressStep = '';
              const progressCallback = (progressLine) => {
                // Log progress to console
                console.log(`[INIT] [${versionName} Upload Progress] ${progressLine}`);
                
                // Update progress step
                const progressStep = `Uploading ${versionName} adapter... ${progressLine}`;
                if (progressStep !== lastProgressStep) {
                  const lastStepIndex = results.steps.length - 1;
                  if (lastStepIndex >= 0 && results.steps[lastStepIndex].includes('Uploading') && results.steps[lastStepIndex].includes(versionName)) {
                    results.steps[lastStepIndex] = progressStep;
                  } else {
                    results.steps.push(progressStep);
                  }
                  lastProgressStep = progressStep;
                }
              };
              
              // Use rsync to upload (rsync handles incremental transfers automatically)
              await uploadAdapterWithRsync(
                version.path,
                remoteAdapterPath,
                host,
                uploadSshPort,
                username,
                uploadSshKeyPath,
                progressCallback
              );
              
              uploadSuccess = true;
              console.log(`[INIT] ✓ ${versionName} adapter uploaded successfully with rsync`);
              results.steps.push(`✓ ${versionName} adapter uploaded (${allFiles.length} files, ${totalSizeMB} MB)`);
              
            } catch (uploadError) {
              const errorMsg = uploadError.message || String(uploadError);
              const errorName = uploadError.name || 'UnknownError';
              
              console.error(`[INIT] ========================================`);
              console.error(`[INIT] rsync upload failed (attempt ${uploadAttempts}/${maxUploadAttempts})`);
              console.error(`[INIT] Error name: ${errorName}`);
              console.error(`[INIT] Error message: ${errorMsg}`);
              console.error(`[INIT] ========================================`);
              
              // Check if it's a connection error that we can retry
              const errorMsgLower = errorMsg.toLowerCase();
              const isConnectionError = errorMsgLower.includes('connection') || 
                                       errorMsgLower.includes('timeout') ||
                                       errorMsgLower.includes('refused') ||
                                       errorMsgLower.includes('network') ||
                                       errorMsgLower.includes('host') ||
                                       errorMsgLower.includes('no route');
              
              if (isConnectionError && uploadAttempts < maxUploadAttempts) {
                console.log(`[INIT] Connection error detected, will retry after delay...`);
                results.steps.push(`⚠ Connection error on attempt ${uploadAttempts}, retrying...`);
                continue;
              } else {
                // Not a retryable error or max attempts reached
                const finalErrorMsg = `Failed to upload ${versionName} adapter after ${uploadAttempts} attempt(s): ${errorName} - ${errorMsg}`;
                console.error(`[INIT] ✗ ${finalErrorMsg}`);
                results.errors.push(finalErrorMsg);
                results.steps.push(`✗ ${finalErrorMsg}`);
                throw new Error(finalErrorMsg);
              }
            }
          }
          
        } catch (uploadError) {
          const errorMsg = `Failed to upload ${versionName} adapter: ${uploadError.message}`;
          console.error(`[INIT] ✗ ${errorMsg}`);
          results.errors.push(errorMsg);
          results.steps.push(`✗ ${errorMsg}`);
          ssh.dispose();
          return results;
        }
      }
      
      console.log('[INIT] ✓ Step 4: LoRA adapters uploaded');
      results.steps.push('✓ Step 4: LoRA adapters uploaded');
    }
    
    // STEP 5: Start Inference Server (AFTER base model and adapters are ready)
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Step 5: Starting Inference Server');
    console.log('[INIT] ========================================');
    results.steps.push('Step 5: Starting inference server...');
    
    try {
      // Ensure utils directory exists
      console.log(`[INIT] [SERVER PREP] Creating /workspace/utils directory...`);
      const mkdirResult = await ssh.execCommand('mkdir -p /workspace/utils');
      console.log(`[INIT] [SERVER PREP] mkdir exit code: ${mkdirResult.code}`);
      if (mkdirResult.stderr) {
        console.log(`[INIT] [SERVER PREP] mkdir stderr: ${mkdirResult.stderr}`);
      }
      
      // Upload inference_server.py
      const localInferenceServerPath = path.join(__dirname, '..', 'utils', 'inference_server.py');
      console.log(`[INIT] [SERVER PREP] Local inference server path: ${localInferenceServerPath}`);
      console.log(`[INIT] [SERVER PREP] File exists: ${fs.existsSync(localInferenceServerPath)}`);
      
      if (fs.existsSync(localInferenceServerPath)) {
        console.log(`[INIT] [SERVER PREP] Uploading inference_server.py to /workspace/utils/inference_server.py...`);
        await ssh.putFile(localInferenceServerPath, '/workspace/utils/inference_server.py');
        console.log(`[INIT] ✓ Inference server script uploaded`);
        results.steps.push('✓ Inference server script uploaded');
      } else {
        const errorMsg = 'inference_server.py not found locally';
        console.error(`[INIT] ✗ ${errorMsg}`);
        console.error(`[INIT] [SERVER PREP] Expected path: ${localInferenceServerPath}`);
        results.errors.push(errorMsg);
        results.steps.push(`✗ ${errorMsg}`);
        ssh.dispose();
        return results;
      }
      
      // Build FastAPI command - use internal port for server, external port for API calls
      const inferencePort = internalPortValue.toString();
      console.log(`[INIT] [SERVER PREP] Inference port: ${inferencePort}`);
      console.log(`[INIT] [SERVER PREP] Base model: ${actualBaseModel}`);
      console.log(`[INIT] [SERVER PREP] Profile name: ${profileName}`);
      
      const envVarsList = [];
      if (hfToken) {
        envVarsList.push(`HF_TOKEN='${hfToken.replace(/'/g, "'\\''")}'`);
        console.log(`[INIT] [SERVER PREP] HF_TOKEN will be set`);
      }
      envVarsList.push(`BASE_MODEL='${actualBaseModel}'`);
      envVarsList.push(`ADAPTER_BASE_PATH='/workspace/models'`);
      envVarsList.push(`PROFILE_NAME='${profileName}'`);
      envVarsList.push(`INFERENCE_PORT='${inferencePort}'`);
      envVarsList.push(`MAX_NEW_TOKENS_CAP='1536'`);
      const envString = envVarsList.join(' ');
      
      // Start server with nohup (no supervisor dependency)
      console.log(`[INIT] [SERVER PREP] Starting inference server with nohup...`);
      results.steps.push('Starting inference server in background (nohup)...');
      
      // Explicitly kill any existing inference server processes before starting new one
      console.log(`[INIT] [SERVER PREP] Checking for existing inference server processes...`);
      results.steps.push('Stopping any existing inference server processes...');
      
      // Kill processes matching the uvicorn inference server pattern
      const killResult = await ssh.execCommand('pkill -f "uvicorn utils.inference_server:app" || true');
      console.log(`[INIT] [SERVER PREP] pkill exit code: ${killResult.code}`);
      if (killResult.stdout) {
        console.log(`[INIT] [SERVER PREP] pkill stdout: ${killResult.stdout}`);
      }
      if (killResult.stderr) {
        console.log(`[INIT] [SERVER PREP] pkill stderr: ${killResult.stderr}`);
      }
      
      // Also kill any processes using the inference port (in case of orphaned processes)
      const portKillResult = await ssh.execCommand(`fuser -k ${inferencePort}/tcp 2>/dev/null || true`);
      console.log(`[INIT] [SERVER PREP] Port kill exit code: ${portKillResult.code}`);
      
      // Wait for processes to fully terminate
      console.log(`[INIT] [SERVER PREP] Waiting 3 seconds for processes to terminate...`);
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Verify no inference server processes are still running
      const verifyKill = await ssh.execCommand('pgrep -f "uvicorn utils.inference_server:app" || echo "none"');
      const remainingPids = verifyKill.stdout.trim();
      if (remainingPids !== 'none' && remainingPids.length > 0) {
        console.warn(`[INIT] [SERVER PREP] Warning: Some processes may still be running: ${remainingPids}`);
        // Force kill with SIGKILL
        await ssh.execCommand(`kill -9 ${remainingPids.split('\n').join(' ')} 2>/dev/null || true`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      } else {
        console.log(`[INIT] [SERVER PREP] ✓ All existing inference server processes terminated`);
      }
      
      // Create log directory
      await ssh.execCommand('mkdir -p /workspace/logs');
      
      // Start server with nohup - use env to set environment variables
      // envVarsList already contains strings like "HF_TOKEN='value'", "BASE_MODEL='value'", etc.
      const envVarsForNohup = envVarsList.join(' ');
      // Use < /dev/null to fully detach the process and ensure SSH command returns immediately
      const nohupCommand = `cd /workspace && nohup env ${envVarsForNohup} ${pythonCmd} -m uvicorn utils.inference_server:app --host 0.0.0.0 --port ${inferencePort} > /workspace/logs/inference-stdout.log 2> /workspace/logs/inference-stderr.log < /dev/null & echo "nohup_started"`;
      console.log(`[INIT] [SERVER PREP] Starting server with: ${nohupCommand}`);
      console.log(`[INIT] [SERVER PREP] Executing nohup command...`);
      try {
        const nohupResult = await Promise.race([
          ssh.execCommand(nohupCommand),
          new Promise((_, reject) => setTimeout(() => reject(new Error('nohup command timeout')), 10000))
        ]);
        console.log(`[INIT] [SERVER PREP] nohup command completed`);
        console.log(`[INIT] [SERVER PREP] nohup exit code: ${nohupResult.code}`);
        console.log(`[INIT] [SERVER PREP] nohup stdout: ${nohupResult.stdout ? nohupResult.stdout.substring(0, 200) : 'empty'}`);
        if (nohupResult.stderr) {
          console.log(`[INIT] [SERVER PREP] nohup stderr: ${nohupResult.stderr}`);
        }
      } catch (nohupError) {
        console.error(`[INIT] [SERVER PREP] Error executing nohup command: ${nohupError.message}`);
        console.error(`[INIT] [SERVER PREP] Error details:`, nohupError);
        // Continue anyway - the process might have started
        console.log(`[INIT] [SERVER PREP] Continuing despite error - process may have started in background`);
      }
      
      // Wait a moment for server to start
      console.log(`[INIT] [SERVER PREP] Waiting 5 seconds for server to start...`);
      await new Promise(resolve => setTimeout(resolve, 5000));
      console.log(`[INIT] [SERVER PREP] Wait completed, proceeding to process check...`);
      
      // Verify server is running by checking process
      const processCheck = await ssh.execCommand('pgrep -f "uvicorn utils.inference_server:app" || echo "not_running"');
      console.log(`[INIT] [SERVER PREP] Process check: ${processCheck.stdout.trim()}`);
      
      if (processCheck.stdout.trim() === 'not_running') {
        // Check logs for errors
        const errorLogCheck = await ssh.execCommand('tail -50 /workspace/logs/inference-stderr.log 2>/dev/null || echo no_logs');
        const errorMsg = 'Inference server process not running. Check logs: /workspace/logs/inference-stderr.log';
        console.error(`[INIT] ✗ ${errorMsg}`);
        if (errorLogCheck.stdout.trim() !== 'no_logs') {
          console.error(`[INIT] [SERVER PREP] Error logs: ${errorLogCheck.stdout.substring(0, 500)}`);
        }
        results.errors.push(errorMsg);
        results.steps.push(`✗ ${errorMsg}`);
        ssh.dispose();
        return results;
      }
      
      console.log(`[INIT] ✓ Inference server is running (PID: ${processCheck.stdout.trim()})`);
      results.steps.push(`✓ Inference server is running (background process)`);
      
      // Check server health via SSH (using localhost on remote server)
      // This verifies the server is running before we establish the SSH tunnel
      console.log(`[INIT] [SERVER PREP] Checking server health via SSH (localhost on remote)...`);
      const localHealthUrl = `http://localhost:${inferencePort}/health`;
      console.log(`[INIT] [SERVER PREP] Testing via SSH: ${localHealthUrl}`);
      
      // Wait a bit more for server to fully initialize (FastAPI needs time to load models)
      console.log(`[INIT] [SERVER PREP] Waiting 3 seconds for server to fully initialize...`);
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Retry health check every 3 seconds for up to 60 seconds (20 attempts)
      const maxRetries = 20;
      const retryInterval = 3000; // 3 seconds
      const maxRetryTime = 60000; // 60 seconds
      let healthCheckPassed = false;
      let attemptCount = 0;
      const startTime = Date.now();
      
      console.log(`[INIT] [HEALTH CHECK] Starting health check retry loop (max ${maxRetries} attempts, ${maxRetryTime/1000}s total)...`);
      results.steps.push(`Checking server health (will retry every 3s for up to 60s)...`);
      
      while (!healthCheckPassed && attemptCount < maxRetries && (Date.now() - startTime) < maxRetryTime) {
        attemptCount++;
        const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
        console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}/${maxRetries} (${elapsedSeconds}s elapsed)...`);
        
        try {
          // Check health endpoint over SSH (internal, no authentication needed)
          const healthCheck = await ssh.execCommand(`curl -s ${localHealthUrl} 2>&1 | head -100 || echo "curl_failed"`);
          console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: exit code: ${healthCheck.code}`);
          console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: stdout length: ${healthCheck.stdout ? healthCheck.stdout.length : 0} chars`);
          if (healthCheck.stderr) {
            console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: stderr: ${healthCheck.stderr.substring(0, 200)}`);
          }
          
          if (healthCheck.stdout.includes('curl_failed') || !healthCheck.stdout.trim()) {
            console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: FAILED - curl failed or empty response`);
            
            // Check if process is still running
            const processStillRunning = await ssh.execCommand('pgrep -f "uvicorn utils.inference_server:app" || echo "not_running"');
            if (processStillRunning.stdout.trim() === 'not_running') {
              console.error(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Process is NOT running! Server may have crashed.`);
              // Get error logs
              const errorLogCheck = await ssh.execCommand('tail -100 /workspace/logs/inference-stderr.log 2>/dev/null || echo no_logs');
              if (errorLogCheck.stdout.trim() !== 'no_logs') {
                console.error(`[INIT] [HEALTH CHECK] Server error logs (last 100 lines):`);
                console.error(errorLogCheck.stdout);
              }
              const stdoutLogCheck = await ssh.execCommand('tail -50 /workspace/logs/inference-stdout.log 2>/dev/null || echo no_logs');
              if (stdoutLogCheck.stdout.trim() !== 'no_logs') {
                console.log(`[INIT] [HEALTH CHECK] Server stdout logs (last 50 lines):`);
                console.log(stdoutLogCheck.stdout);
              }
            } else {
              console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Process is still running (PID: ${processStillRunning.stdout.trim()}), but health endpoint not responding`);
              // Check if port is listening
              const portCheck = await ssh.execCommand(`netstat -tlnp 2>/dev/null | grep :${inferencePort} || ss -tlnp 2>/dev/null | grep :${inferencePort} || echo "port_not_listening"`);
              if (portCheck.stdout.includes('port_not_listening')) {
                console.error(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Port ${inferencePort} is NOT listening! Server may not have bound to port.`);
              } else {
                console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Port ${inferencePort} appears to be listening`);
              }
              
              // Show recent error logs even if process is running (might be startup errors)
              if (attemptCount % 3 === 0) { // Every 3rd attempt, check logs
                const errorLogCheck = await ssh.execCommand('tail -50 /workspace/logs/inference-stderr.log 2>/dev/null || echo no_logs');
                if (errorLogCheck.stdout.trim() !== 'no_logs' && errorLogCheck.stdout.trim().length > 0) {
                  console.log(`[INIT] [HEALTH CHECK] Recent error logs (last 50 lines):`);
                  console.log(errorLogCheck.stdout.substring(0, 1000));
                }
              }
            }
            
            if (attemptCount < maxRetries && (Date.now() - startTime) < maxRetryTime) {
              console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Retrying in 3 seconds...`);
              await new Promise(resolve => setTimeout(resolve, retryInterval));
              continue;
            }
          } else {
          try {
              const healthData = JSON.parse(healthCheck.stdout);
              console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Parsed health data:`, JSON.stringify(healthData, null, 2));
            if (healthData.status === 'healthy' || healthData.status === 'ok') {
                console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: ✓ PASSED - server is responding correctly`);
                console.log(`[INIT] ✓ Server health check passed after ${attemptCount} attempt(s) (${elapsedSeconds}s)`);
                results.steps.push(`✓ Server health check passed after ${attemptCount} attempt(s) (${elapsedSeconds}s)`);
                // Server is responding - set flag to auto-launch chat interface
                results.healthCheckPassed = true;
                healthCheckPassed = true;
                break;
            } else {
                console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: FAILED - status is not 'healthy': ${healthData.status || 'unknown'}`);
                if (attemptCount < maxRetries && (Date.now() - startTime) < maxRetryTime) {
                  console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Retrying in 3 seconds...`);
                  await new Promise(resolve => setTimeout(resolve, retryInterval));
                  continue;
                }
            }
          } catch (e) {
              console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: FAILED - could not parse health response: ${e.message}`);
              console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Raw response: ${healthCheck.stdout ? healthCheck.stdout.substring(0, 200) : 'no data'}`);
              if (attemptCount < maxRetries && (Date.now() - startTime) < maxRetryTime) {
                console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Retrying in 3 seconds...`);
                await new Promise(resolve => setTimeout(resolve, retryInterval));
                continue;
              }
            }
        }
      } catch (healthError) {
          console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: ERROR - ${healthError.message}`);
          console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Error details:`, healthError);
          if (attemptCount < maxRetries && (Date.now() - startTime) < maxRetryTime) {
            console.log(`[INIT] [HEALTH CHECK] Attempt ${attemptCount}: Retrying in 3 seconds...`);
            await new Promise(resolve => setTimeout(resolve, retryInterval));
            continue;
          }
        }
      }
      
      // If health check failed after all retries, abort with error
      if (!healthCheckPassed) {
        const finalElapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
        
        // Final diagnostic: check process status and logs
        console.error(`[INIT] [HEALTH CHECK] Final diagnostic check...`);
        const finalProcessCheck = await ssh.execCommand('pgrep -f "uvicorn utils.inference_server:app" || echo "not_running"');
        console.error(`[INIT] [HEALTH CHECK] Final process check: ${finalProcessCheck.stdout.trim()}`);
        
        // Get comprehensive error logs
        const finalErrorLogs = await ssh.execCommand('tail -200 /workspace/logs/inference-stderr.log 2>/dev/null || echo no_logs');
        if (finalErrorLogs.stdout.trim() !== 'no_logs' && finalErrorLogs.stdout.trim().length > 0) {
          console.error(`[INIT] [HEALTH CHECK] Final error logs (last 200 lines):`);
          console.error(finalErrorLogs.stdout);
        }
        
        const finalStdoutLogs = await ssh.execCommand('tail -100 /workspace/logs/inference-stdout.log 2>/dev/null || echo no_logs');
        if (finalStdoutLogs.stdout.trim() !== 'no_logs' && finalStdoutLogs.stdout.trim().length > 0) {
          console.log(`[INIT] [HEALTH CHECK] Final stdout logs (last 100 lines):`);
          console.log(finalStdoutLogs.stdout);
        }
        
        // Check port binding
        const finalPortCheck = await ssh.execCommand(`netstat -tlnp 2>/dev/null | grep :${inferencePort} || ss -tlnp 2>/dev/null | grep :${inferencePort} || echo "port_not_listening"`);
        console.error(`[INIT] [HEALTH CHECK] Port ${inferencePort} status: ${finalPortCheck.stdout.trim()}`);
        
        // Try a verbose curl to see what's happening
        const verboseCurl = await ssh.execCommand(`curl -v http://localhost:${inferencePort}/health 2>&1 | head -50 || echo "curl_failed"`);
        console.error(`[INIT] [HEALTH CHECK] Verbose curl output:`);
        console.error(verboseCurl.stdout);
        
        const errorMsg = `Server health check failed after ${attemptCount} attempt(s) over ${finalElapsedSeconds} seconds. The inference server did not respond correctly. Check logs above for details.`;
        console.error(`[INIT] ✗ ${errorMsg}`);
        console.error(`[INIT] [HEALTH CHECK] All ${attemptCount} attempts failed. Aborting initialization.`);
        results.errors.push(errorMsg);
        results.steps.push(`✗ ${errorMsg}`);
        ssh.dispose();
        return results;
      }
      
      console.log('[INIT] ✓ Step 5: Inference server started');
      results.steps.push('✓ Step 5: Inference server started');
      
    } catch (serverPrepError) {
      const errorMsg = `Server startup failed: ${serverPrepError.message}`;
      console.error(`[INIT] ✗ ${errorMsg}`);
      console.error(`[INIT] Server startup error details:`, serverPrepError);
      results.errors.push(errorMsg);
      results.steps.push(`✗ ${errorMsg}`);
      ssh.dispose();
      return results;
    }
    
    // STEP 6: Health check through tunnel (if tunnel exists) or external URL
    // INVARIANT: All health checks MUST go through the tunnel (localhost:8888) or external URL
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Step 6: Health Check Through Tunnel');
    console.log('[INIT] ========================================');
    results.steps.push('Step 6: Checking health endpoint through tunnel...');
    
    try {
      // Verify tunnel is still active if we're using one
      if (!externalPortValue && sshTunnelProcess) {
        // Check if tunnel process is still running
        if (sshTunnelProcess.killed || sshTunnelProcess.exitCode !== null) {
          console.error(`[INIT] ✗ SSH tunnel process has exited (exit code: ${sshTunnelProcess.exitCode})`);
          results.errors.push('SSH tunnel process died before health check');
          results.steps.push('✗ SSH tunnel process died - cannot perform health check');
          throw new Error('SSH tunnel process is not running');
        }
        console.log(`[INIT] [HEALTH] Tunnel process is active (PID check passed)`);
      }
      
      // Use the hard-bound inference URL (127.0.0.1:8888 if tunnel, or external URL)
      // Use 127.0.0.1 instead of localhost to force IPv4 (tunnel is bound to IPv4)
      const healthUrl = results.inferenceUrl || 'http://127.0.0.1:8888';
      console.log(`[INIT] [HEALTH] Checking health endpoint: ${healthUrl}/health`);
      
      // Wait a bit for server to fully initialize (if we just started it)
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Check health endpoint through tunnel (localhost:8888) or external URL
      const healthResponse = await makeHttpRequest(`${healthUrl}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'Electron-Inference-Client/1.0'
        },
        timeout: 10000 // 10 second timeout for health check
      });
      
      console.log(`[INIT] [HEALTH] Health check response status: ${healthResponse.statusCode}`);
      
      if (healthResponse.statusCode === 200) {
        try {
          const healthData = JSON.parse(healthResponse.data);
          console.log(`[INIT] [HEALTH] Parsed health data:`, JSON.stringify(healthData, null, 2));
          if (healthData.status === 'healthy' || healthData.status === 'ok') {
            console.log(`[INIT] ✓ Health check passed through tunnel`);
            results.steps.push('✓ Health check passed through tunnel');
            results.healthCheckPassed = true;
          } else {
            console.log(`[INIT] ⚠ Health check - status is not 'healthy': ${healthData.status || 'unknown'}`);
            results.steps.push(`⚠ Health check - status: ${healthData.status || 'unknown'}`);
          }
        } catch (e) {
          console.log(`[INIT] [HEALTH] Could not parse health response: ${e.message}`);
          results.steps.push('⚠ Health check - response format unexpected');
        }
      } else {
        console.log(`[INIT] ⚠ Health check returned ${healthResponse.statusCode} (server may still be starting)`);
        results.steps.push(`⚠ Health check returned ${healthResponse.statusCode} (server may still be starting)`);
      }
      
      console.log('[INIT] ✓ Step 6: Health check complete');
      results.steps.push('✓ Step 6: Health check complete');
      
    } catch (healthError) {
      console.warn(`[INIT] ⚠ Health check warning: ${healthError.message}`);
      console.warn(`[INIT] Health check error details:`, healthError);
      results.steps.push(`⚠ Health check: ${healthError.message}`);
    }
    
    // STEP 7: Finalize - inference URL already set in Step 1.5
    // INVARIANT: inferenceUrl is hard-bound to http://localhost:8888 (if tunnel) or external URL
    // Never override this value
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Step 7: Finalizing Configuration');
    console.log('[INIT] ========================================');
    results.steps.push('Step 7: Finalizing configuration...');
    
    // Ensure inference URL is set (should already be set in Step 1.5)
    if (!results.inferenceUrl) {
      // Fallback only if somehow not set (should not happen)
      if (externalPortValue) {
        results.inferenceUrl = `http://${host}:${externalPortValue}`;
            } else {
        results.inferenceUrl = 'http://127.0.0.1:8888';
      }
    }
    
    results.success = true;
    console.log(`[INIT] ✓ Final inference URL: ${results.inferenceUrl}`);
    results.steps.push(`✓ Final inference URL: ${results.inferenceUrl}`);
    
    // Store SSH config, model info, and inference URL for chat interface
    storedSSHConfig = {
      host: host,
      port: port || 22,
      username: username || 'root'
    };
    
    // Store model info for chat interface (only if profileName is provided)
    if (profileName) {
      const remoteModelDir = `/workspace/models/${profileName}`;
      storedModelInfo = {
        profileName: profileName,
        baseModel: actualBaseModel, // Use the actual base model (expanded from short name)
        modelDir: remoteModelDir,
        versions: versions ? versions.map(v => ({
          version: v.version,
          adapterPath: `${remoteModelDir}/V${v.version}/adapter`
        })) : []
      };
      console.log(`[INIT] Stored model info with baseModel: ${actualBaseModel}`);
    } else {
      // If no profile, still store basic info for chat interface
      storedModelInfo = {
        profileName: null,
        baseModel: actualBaseModel || baseModel,
        modelDir: null,
        versions: []
      };
      console.log(`[INIT] Stored basic model info (no profile): ${actualBaseModel || baseModel}`);
    }
    // Hard-bind stored URL - never override (invariant: localhost:8888 if tunnel, or external URL)
    storedVLLMUrl = results.inferenceUrl;
    
    console.log(`[INIT] Stored inference URL: ${storedVLLMUrl}`);
    
    console.log('\n[INIT] ========================================');
    console.log('[INIT] Initialization Complete!');
    console.log('[INIT] ========================================');
    console.log(`[INIT] Inference Server URL: ${storedVLLMUrl}`);
    console.log(`[INIT]   Health: ${storedVLLMUrl}/health`);
    console.log(`[INIT]   Models: ${storedVLLMUrl}/models`);
    console.log(`[INIT]   Chat: ${storedVLLMUrl}/chat`);
    console.log('[INIT] ========================================\n');
    
    results.steps.push(`✓ Initialization complete!`);
    results.steps.push(`✓ Inference Server URL: ${storedVLLMUrl}`);
    
    ssh.dispose();
    return results;
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

// Prepare FastAPI inference server with model and versions
ipcMain.handle('prepare-inference-server', async (event, { host, port, username = 'root', profileName, baseModel, versions, inferenceUrl, vllmUrl }) => {
  // Support both inferenceUrl and vllmUrl for backward compatibility
  let serverUrl = inferenceUrl || vllmUrl;
  
  // Add http:// if protocol is missing
  if (serverUrl && !serverUrl.match(/^https?:\/\//i)) {
    serverUrl = 'http://' + serverUrl;
    console.log(`[INFERENCE] Added http:// prefix. New URL: ${serverUrl}`);
  }
  
  // Use serverUrl as inferenceUrl for the rest of the function
  inferenceUrl = serverUrl;
  
  const ssh = new NodeSSH();
  const results = {
    success: false,
    steps: [],
    errors: [],
    inferenceUrl: null  // Will be set after dependency check
  };
  let modelReady = false; // Track model readiness across all checks
  let storedVLLMPid = null; // Track PID to detect process restarts
  let openButtonToken = null; // OPEN_BUTTON_TOKEN for Basic Auth

  try {
    // Connect
    results.steps.push('Connecting via SSH...');
    console.log('\n[INFERENCE] ========================================');
    console.log('[INFERENCE] Starting FastAPI inference server preparation...');
    console.log('[INFERENCE] ========================================');
    console.log(`[INFERENCE] Profile: ${profileName}`);
    console.log(`[INFERENCE] Base Model: ${baseModel}`);
    console.log(`[INFERENCE] Versions: ${versions.map(v => `V${v.version}`).join(', ')}`);
    console.log('');
    
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
    // Prepare connection options with auto-accept host keys
    const finalConnectOptions = prepareSSHConnectionOptions(connectOptions);
    await ssh.connect(finalConnectOptions);
    results.steps.push('✓ SSH connection established');
    console.log('[SSH] ✓ Connection established');
    
    // Check and install PEFT inference dependencies
    console.log('\n[INFERENCE] ========================================');
    console.log('[INFERENCE] Starting PEFT dependency check...');
    console.log('[INFERENCE] ========================================');
    results.steps.push('Checking PEFT inference dependencies...');
    console.log('[INFERENCE] Checking required Python packages for PEFT inference...');
    console.log('');
    
    try {
      // Required packages for PEFT inference
      // Format: { pipPackageName: importName }
      const requiredPackages = {
        'transformers': 'transformers',
        'peft': 'peft',
        'torch': 'torch',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'accelerate': 'accelerate',
        'huggingface-hub': 'huggingface_hub'  // pip package name is huggingface-hub, but import is huggingface_hub
      };
      
      // Check which Python to use
      let pythonCmd = 'python3';
      const pythonCheck = await ssh.execCommand('which python3 || which python || echo "not_found"');
      if (pythonCheck.stdout.includes('not_found')) {
        results.errors.push('Python not found on remote system');
        throw new Error('Python not found');
      }
      if (pythonCheck.stdout.includes('python')) {
        pythonCmd = pythonCheck.stdout.trim();
      }
      
      console.log(`[INFERENCE] Using Python: ${pythonCmd}`);
      
      // Check each package
      const missingPackages = [];
      const installedPackages = [];
      
      for (const [pkgName, importName] of Object.entries(requiredPackages)) {
        const checkCmd = `${pythonCmd} -c "import ${importName}; print('installed')" 2>&1`;
        const checkResult = await ssh.execCommand(checkCmd);
        
        if (checkResult.code === 0 && checkResult.stdout.includes('installed')) {
          installedPackages.push(pkgName);
          console.log(`[INFERENCE] ✓ ${pkgName} is installed`);
        } else {
          missingPackages.push(pkgName);
          console.log(`[INFERENCE] ✗ ${pkgName} is missing`);
        }
      }
      
      // Log initial status
      console.log(`[INFERENCE] Dependency check results:`);
      console.log(`[INFERENCE]   Installed: ${installedPackages.length} package(s) - ${installedPackages.join(', ')}`);
      console.log(`[INFERENCE]   Missing: ${missingPackages.length} package(s) - ${missingPackages.join(', ')}`);
      results.steps.push(`Found ${installedPackages.length} installed, ${missingPackages.length} missing`);
      
      if (missingPackages.length > 0) {
        results.steps.push(`Installing missing packages: ${missingPackages.join(', ')}...`);
        console.log(`[INFERENCE] Installing missing packages: ${missingPackages.join(', ')}`);
        
        // Check if pip is available
        const pipCheck = await ssh.execCommand('which pip3 || which pip || echo "not_found"');
        let pipCmd = 'pip3';
        if (pipCheck.stdout.includes('not_found')) {
          results.errors.push('pip not found - cannot install packages');
          throw new Error('pip not found');
        }
        if (pipCheck.stdout.includes('pip')) {
          pipCmd = pipCheck.stdout.trim();
        }
        
        console.log(`[INFERENCE] Using pip: ${pipCmd}`);
        
        // Install missing packages
        const installCmd = `${pipCmd} install --quiet ${missingPackages.join(' ')}`;
        console.log(`[INFERENCE] Running: ${installCmd}`);
        const installResult = await ssh.execCommand(installCmd);
        
        if (installResult.code === 0) {
          console.log(`[INFERENCE] ✓ Successfully installed ${missingPackages.length} package(s)`);
          results.steps.push(`✓ Installed ${missingPackages.length} missing package(s)`);
          
          // Verify installation
          const verifiedPackages = [];
          const failedPackages = [];
          for (const pkg of missingPackages) {
            const importName = requiredPackages[pkg] || pkg.replace(/-/g, '_');
            const verifyCmd = `${pythonCmd} -c "import ${importName}; print('installed')" 2>&1`;
            const verifyResult = await ssh.execCommand(verifyCmd);
            if (verifyResult.code === 0 && verifyResult.stdout.includes('installed')) {
              verifiedPackages.push(pkg);
              console.log(`[INFERENCE] ✓ Verified ${pkg} installation`);
            } else {
              failedPackages.push(pkg);
              console.warn(`[INFERENCE] ⚠ ${pkg} installation may have failed`);
              results.errors.push(`Package ${pkg} installation verification failed`);
            }
          }
          
          // Log verification results
          if (verifiedPackages.length > 0) {
            console.log(`[INFERENCE] Verified installations: ${verifiedPackages.join(', ')}`);
            results.steps.push(`✓ Verified ${verifiedPackages.length} package installation(s)`);
          }
          if (failedPackages.length > 0) {
            console.warn(`[INFERENCE] ⚠ Failed verifications: ${failedPackages.join(', ')}`);
            results.steps.push(`⚠ ${failedPackages.length} package(s) verification failed`);
          }
        } else {
          const errorMsg = installResult.stderr || installResult.stdout || 'Unknown error';
          console.error(`[INFERENCE] ✗ Failed to install packages: ${errorMsg.substring(0, 500)}`);
          results.errors.push(`Failed to install required packages: ${missingPackages.join(', ')}`);
          results.errors.push(`Installation error: ${errorMsg.substring(0, 300)}`);
          throw new Error(`Package installation failed: ${errorMsg.substring(0, 200)}`);
        }
      } else {
        console.log(`[INFERENCE] ✓ All required packages are installed`);
        results.steps.push(`✓ All required packages are installed`);
      }
      
      // Get version information for key packages
      console.log(`[INFERENCE] Package versions:`);
      const versionChecks = {
        'transformers': 'transformers',
        'peft': 'peft',
        'torch': 'torch',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn'
      };
      
      for (const [pkg, importName] of Object.entries(versionChecks)) {
        try {
          const versionCmd = `${pythonCmd} -c "import ${importName}; print(${importName}.__version__)" 2>&1`;
          const versionResult = await ssh.execCommand(versionCmd);
          if (versionResult.code === 0 && versionResult.stdout.trim()) {
            const version = versionResult.stdout.trim();
            console.log(`[INFERENCE]   ${pkg}: ${version}`);
          }
        } catch (e) {
          // Ignore version check errors
        }
      }
      
      // Check for CUDA availability (optional but recommended for GPU inference)
      console.log(`[INFERENCE] Checking CUDA availability...`);
      const cudaCheck = await ssh.execCommand(`${pythonCmd} -c "import torch; print('cuda_available' if torch.cuda.is_available() else 'cpu_only')" 2>&1`);
      if (cudaCheck.code === 0) {
        if (cudaCheck.stdout.includes('cuda_available')) {
          const gpuInfo = await ssh.execCommand(`${pythonCmd} -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1`);
          const gpuName = gpuInfo.stdout.trim();
          const cudaVersion = await ssh.execCommand(`${pythonCmd} -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1`);
          const cudaVer = cudaVersion.stdout.trim();
          console.log(`[INFERENCE] ✓ CUDA available - GPU: ${gpuName}, CUDA: ${cudaVer}`);
          results.steps.push(`✓ CUDA available - GPU: ${gpuName}, CUDA: ${cudaVer}`);
        } else {
          console.warn(`[INFERENCE] ⚠ CUDA not available - will use CPU (slower)`);
          results.steps.push(`⚠ CUDA not available - will use CPU`);
        }
      }
      
      // Final summary
      const totalPackages = Object.keys(requiredPackages).length;
      const allPackagesInstalled = installedPackages.length === totalPackages || (installedPackages.length + missingPackages.length === totalPackages && missingPackages.length === 0);
      console.log(`[INFERENCE] ========================================`);
      console.log(`[INFERENCE] Dependency check summary:`);
      console.log(`[INFERENCE]   Python: ${pythonCmd}`);
      console.log(`[INFERENCE]   Total packages checked: ${totalPackages}`);
      console.log(`[INFERENCE]   Already installed: ${installedPackages.length}`);
      if (missingPackages.length > 0) {
        console.log(`[INFERENCE]   Newly installed: ${missingPackages.length}`);
      }
      console.log(`[INFERENCE]   Status: ${allPackagesInstalled ? '✓ All packages ready' : '⚠ Some packages may be missing'}`);
      console.log(`[INFERENCE] ========================================`);
      results.steps.push(`✓ Dependency check complete - ${totalPackages} packages ready`);
      
      // Surface the inference server URL immediately after dependency check completes
      console.log('');
      console.log(`[INFERENCE] ========================================`);
      console.log(`[INFERENCE] PEFT Inference Server Connection URL:`);
      console.log(`[INFERENCE] ========================================`);
      
      // Determine the inference server URL
      let serverUrl = inferenceUrl;
      if (!serverUrl) {
        // Construct from host and port
        const inferencePort = '8000'; // Default port
        serverUrl = `http://${host}:${inferencePort}`;
        console.log(`[INFERENCE] Constructed URL from host: ${serverUrl}`);
      } else {
        console.log(`[INFERENCE] Using provided URL: ${serverUrl}`);
      }
      
      // Parse URL to extract IP and port
      let serverIP = host;
      let serverPort = '8000';
      try {
        const urlObj = new URL(serverUrl);
        serverIP = urlObj.hostname;
        serverPort = urlObj.port || '8000';
      } catch (e) {
        // If URL parsing fails, try to extract from the string
        const urlMatch = serverUrl.match(/https?:\/\/([^:]+):?(\d+)?/);
        if (urlMatch) {
          serverIP = urlMatch[1];
          serverPort = urlMatch[2] || '8000';
        }
      }
      
      // Display URL prominently
      console.log(`[INFERENCE]   Server IP: ${serverIP}`);
      console.log(`[INFERENCE]   Server Port: ${serverPort}`);
      console.log(`[INFERENCE]   Full URL: ${serverUrl}`);
      console.log(`[INFERENCE]   Health Check: ${serverUrl}/health`);
      console.log(`[INFERENCE]   Models Endpoint: ${serverUrl}/models`);
      console.log(`[INFERENCE]   Chat Endpoint: ${serverUrl}/chat`);
      console.log(`[INFERENCE] ========================================`);
      console.log('');
      
      // Add to results steps for UI display
      results.steps.push(`✓ PEFT Inference Server URL: ${serverUrl}`);
      results.steps.push(`  → Health: ${serverUrl}/health`);
      results.steps.push(`  → Chat: ${serverUrl}/chat`);
      
      // Store in results object for UI access
      results.inferenceUrl = serverUrl;
      console.log(`[INFERENCE] ✓ URL stored in results: ${results.inferenceUrl}`);
      console.log(`[INFERENCE] ✓ URL will be displayed in UI steps list`);
      
    } catch (depError) {
      console.error(`[INFERENCE] ✗ Dependency check failed: ${depError.message}`);
      results.errors.push(`Dependency check failed: ${depError.message}`);
      // Don't throw - continue with preparation, but user will see the error
      results.steps.push(`⚠ Dependency check failed - inference server may not work correctly`);
      
      // Still display URL even if dependency check failed
      console.log('');
      console.log(`[INFERENCE] ========================================`);
      console.log(`[INFERENCE] PEFT Inference Server Connection URL:`);
      console.log(`[INFERENCE] ========================================`);
      
      let serverUrl = inferenceUrl;
      if (!serverUrl) {
        const inferencePort = '8000';
        serverUrl = `http://${host}:${inferencePort}`;
      }
      
      console.log(`[INFERENCE]   Full URL: ${serverUrl}`);
      console.log(`[INFERENCE]   Health Check: ${serverUrl}/health`);
      console.log(`[INFERENCE]   Chat Endpoint: ${serverUrl}/chat`);
      console.log(`[INFERENCE] ========================================`);
      console.log('');
      
      results.steps.push(`✓ PEFT Inference Server URL: ${serverUrl}`);
      results.inferenceUrl = serverUrl;
    }
    
    // Also ensure URL is set outside try-catch as fallback
    if (!results.inferenceUrl) {
      let serverUrl = inferenceUrl;
      if (!serverUrl) {
        serverUrl = `http://${host}:8000`;
      }
      results.inferenceUrl = serverUrl;
      console.log(`[INFERENCE] Fallback: Setting inference URL to ${serverUrl}`);
    }
    
    // Parse URL to extract IP and port
    let serverIP = host;
    let serverPort = '8000';
    try {
      const urlObj = new URL(serverUrl);
      serverIP = urlObj.hostname;
      serverPort = urlObj.port || '8000';
    } catch (e) {
      // If URL parsing fails, try to extract from the string
      const urlMatch = serverUrl.match(/https?:\/\/([^:]+):?(\d+)?/);
      if (urlMatch) {
        serverIP = urlMatch[1];
        serverPort = urlMatch[2] || '8000';
      }
    }
    
    // Display URL prominently
    console.log(`[INFERENCE]   Server IP: ${serverIP}`);
    console.log(`[INFERENCE]   Server Port: ${serverPort}`);
    console.log(`[INFERENCE]   Full URL: ${serverUrl}`);
    console.log(`[INFERENCE]   Health Check: ${serverUrl}/health`);
    console.log(`[INFERENCE]   Models Endpoint: ${serverUrl}/models`);
    console.log(`[INFERENCE]   Chat Endpoint: ${serverUrl}/chat`);
    console.log(`[INFERENCE] ========================================`);
    console.log('');
    
    // Add to results steps for UI display (always add, even if duplicate - UI can handle it)
    results.steps.push(`✓ PEFT Inference Server URL: ${serverUrl}`);
    results.steps.push(`  → Health: ${serverUrl}/health`);
    results.steps.push(`  → Chat: ${serverUrl}/chat`);
    
    // Store in results object for UI access
    results.inferenceUrl = serverUrl;
    console.log(`[INFERENCE] URL stored in results: ${results.inferenceUrl}`);
    
    // Retrieve OPEN_BUTTON_TOKEN for Basic Auth (Vast.ai pattern)
    console.log('[vLLM] Retrieving OPEN_BUTTON_TOKEN for Basic Auth...');
    try {
      // Try multiple methods to get OPEN_BUTTON_TOKEN
      const tokenMethods = [
        { name: 'environment', cmd: 'echo $OPEN_BUTTON_TOKEN' },
        { name: 'process environment', cmd: 'cat /proc/1/environ 2>/dev/null | tr "\\0" "\\n" | grep OPEN_BUTTON_TOKEN | cut -d= -f2' },
        { name: 'systemd', cmd: 'systemctl show-environment 2>/dev/null | grep OPEN_BUTTON_TOKEN | cut -d= -f2' },
        { name: 'supervisor', cmd: 'supervisorctl -c /etc/supervisor/supervisord.conf environment 2>/dev/null | grep OPEN_BUTTON_TOKEN | cut -d= -f2 || echo ""' },
        { name: 'running processes', cmd: 'ps e -o command 2>/dev/null | grep -o "OPEN_BUTTON_TOKEN=[^ ]*" | head -1 | cut -d= -f2 || echo ""' },
        { name: '/etc/environment', cmd: 'grep OPEN_BUTTON_TOKEN /etc/environment 2>/dev/null | cut -d= -f2 || echo ""' }
      ];
      
      for (const method of tokenMethods) {
        try {
          const tokenResult = await ssh.execCommand(method.cmd);
          const token = tokenResult.stdout.trim();
          if (token && token.length > 0) {
            openButtonToken = token;
            storedOpenButtonToken = token; // Also store globally for chat requests
            console.log(`[vLLM] ✓ Found OPEN_BUTTON_TOKEN from ${method.name}`);
            results.steps.push(`✓ Retrieved OPEN_BUTTON_TOKEN for Basic Auth`);
            break;
          }
        } catch (e) {
          // Try next method
          continue;
        }
      }
      
      if (!openButtonToken) {
        console.warn('[vLLM] ⚠ Could not retrieve OPEN_BUTTON_TOKEN - Basic Auth may not work');
        results.steps.push('⚠ Could not retrieve OPEN_BUTTON_TOKEN - API checks may fail if Basic Auth is required');
      }
    } catch (tokenError) {
      console.warn(`[vLLM] ⚠ Error retrieving OPEN_BUTTON_TOKEN: ${tokenError.message}`);
      results.steps.push('⚠ Error retrieving OPEN_BUTTON_TOKEN');
    }

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
    // Try to find which Python has vllm installed
    let vllmCommand = 'python3 -m vllm'; // Default
    let pythonPath = 'python3'; // Default
    
    // FIRST: Check running vLLM processes to see what they're using (most reliable)
    if (vllmCheck.stdout.trim() !== 'not_running') {
      const runningPids = vllmCheck.stdout.trim().split('\n').filter(p => p.trim());
      for (const runningPid of runningPids) {
        const processCmd = await ssh.execCommand(`ps -p ${runningPid} -o args= 2>/dev/null || echo ""`);
        if (processCmd.stdout) {
          const cmdLine = processCmd.stdout.trim();
          // Extract Python path from command line - look for full paths or python commands
          const pythonMatch = cmdLine.match(/(\/[^\s]+\/python[0-9.]*[^\s]*|python[0-9.]*)\s+-m\s+vllm/);
          if (pythonMatch) {
            pythonPath = pythonMatch[1];
            // If it's not a full path, try to resolve it
            if (!pythonPath.startsWith('/')) {
              const whichPython = await ssh.execCommand(`which ${pythonPath} 2>/dev/null || echo "${pythonPath}"`);
              pythonPath = whichPython.stdout.trim() || pythonPath;
            }
            vllmCommand = `${pythonPath} -m vllm`;
            console.log(`[vLLM] Found vLLM from running process (PID ${runningPid}): ${pythonPath}`);
            break;
          }
        }
      }
    }
    
    // If not found from running process, try to find vllm executable directly
    if (vllmCommand === 'python3 -m vllm') {
      const whichVllm = await ssh.execCommand('which vllm 2>/dev/null || echo ""');
      if (whichVllm.stdout.trim()) {
        vllmCommand = 'vllm';
        console.log(`[vLLM] Found vLLM executable: ${whichVllm.stdout.trim()}`);
      } else {
        // Try to find which Python has vllm module by trying to import it
        const pythonPaths = [];
        
        // Check common locations
        const commonPaths = [
          '/usr/bin/python3',
          '/usr/local/bin/python3',
          '/opt/conda/bin/python3',
          '/opt/conda/bin/python',
          'python3',
          'python',
          'python3.10',
          'python3.11',
          'python3.12'
        ];
        
        // Also check for conda/venv Python
        const condaCheck = await ssh.execCommand('which python3 2>/dev/null || echo ""');
        if (condaCheck.stdout.trim()) {
          pythonPaths.push(condaCheck.stdout.trim());
        }
        
        // Add common paths
        pythonPaths.push(...commonPaths);
        
        // Try each Python to see if it can import vllm
        for (const pyPath of pythonPaths) {
          // Try importing vllm - this is more reliable than --version
          const testCmd = `${pyPath} -c "import vllm; print('vllm found')" 2>&1`;
          const testResult = await ssh.execCommand(testCmd);
          if (testResult.code === 0 && testResult.stdout.includes('vllm found')) {
            // Get full path if it's not already
            let fullPath = pyPath;
            if (!fullPath.startsWith('/')) {
              const whichPython = await ssh.execCommand(`which ${pyPath} 2>/dev/null || echo "${pyPath}"`);
              fullPath = whichPython.stdout.trim() || pyPath;
            }
            pythonPath = fullPath;
            vllmCommand = `${fullPath} -m vllm`;
            console.log(`[vLLM] Found vLLM with Python: ${fullPath}`);
            break;
          }
        }
      }
    }
    
    console.log(`[vLLM] Using vLLM command: ${vllmCommand}`);
    
    // Verify the detected command actually works (unless we found it from a running process)
    if (vllmCheck.stdout.trim() === 'not_running' && vllmCommand === 'python3 -m vllm') {
      // We're using the default - verify it works
      const verifyCmd = vllmCommand.includes('/') ? 
        `${vllmCommand.split(' ')[0]} -c "import vllm; print('ok')" 2>&1` :
        `python3 -c "import vllm; print('ok')" 2>&1`;
      const verifyResult = await ssh.execCommand(verifyCmd);
      if (verifyResult.code !== 0 || !verifyResult.stdout.includes('ok')) {
        console.error(`[vLLM] ✗ Warning: Default Python (python3) cannot import vllm`);
        console.error(`[vLLM] ✗ This will likely cause supervisor to fail. Trying to find alternative...`);
        results.steps.push('⚠ Warning: Default Python does not have vllm installed');
        
        // Try to find any Python that can import vllm
        const findPythonCmd = `for py in $(which -a python3 python 2>/dev/null | head -10); do $py -c "import vllm" 2>/dev/null && echo "$py" && break; done`;
        const foundPython = await ssh.execCommand(findPythonCmd);
        if (foundPython.stdout.trim()) {
          const workingPython = foundPython.stdout.trim();
          vllmCommand = `${workingPython} -m vllm`;
          console.log(`[vLLM] ✓ Found working Python: ${workingPython}`);
          results.steps.push(`✓ Found working Python: ${workingPython}`);
        } else {
          results.errors.push('Could not find a Python installation with vllm module. Please install vllm or ensure it is accessible.');
          console.error(`[vLLM] ✗ Could not find any Python with vllm installed`);
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
      // Only log success if supervisor explicitly reports RUNNING status
      results.steps.push('✓ vLLM is running (supervisor reports RUNNING)');
      console.log('[vLLM] ✓ vLLM is running (supervisor reports RUNNING)');
    } else {
      results.steps.push(`⚠ vLLM supervisor status: ${supervisorStatus}`);
      console.log(`[vLLM] ⚠ vLLM supervisor status: ${supervisorStatus}`);
    }

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
    
    // Helper function to normalize model names from adapter_config.json
    // Axolotl may store local paths or Ollama tags - we need HuggingFace repo IDs
    const normalizeModelName = (modelName) => {
      if (!modelName || typeof modelName !== 'string') {
        return null;
      }
      
      // If it's already in org/model format, use it
      if (modelName.includes('/') && !modelName.startsWith('/') && !modelName.includes('\\')) {
        return modelName;
      }
      
      // Map known Ollama tags to HuggingFace repo IDs
      const ollamaToHF = {
        'gemma3:4b': 'google/gemma-3-4b-it',
        'gemma3': 'google/gemma-3-4b-it',
        'gemma2:2b': 'google/gemma-2-2b-it',
        'gemma2:9b': 'google/gemma-2-9b-it',
        'gemma2:27b': 'google/gemma-2-27b-it',
        'gemma:7b': 'google/gemma-7b-it',
        'gemma:2b': 'google/gemma-2b-it'
      };
      
      // Check if it's a known Ollama tag
      const normalized = ollamaToHF[modelName.toLowerCase()];
      if (normalized) {
        console.log(`[vLLM] Normalized Ollama tag '${modelName}' to HuggingFace ID: ${normalized}`);
        return normalized;
      }
      
      // If it's a local path, we can't use it - return null to fall back to provided model
      if (modelName.startsWith('/') || modelName.includes('\\') || modelName.includes(':')) {
        console.warn(`[vLLM] ⚠ Model name appears to be a local path or invalid format: ${modelName}`);
        return null;
      }
      
      return null;
    };
    
    // Try to read base model from first adapter's adapter_config.json
    if (versions.length > 0 && versions[0].exists) {
      try {
        const firstAdapterConfigPath = path.join(versions[0].path, 'adapter_config.json');
        if (fs.existsSync(firstAdapterConfigPath)) {
          const adapterConfig = JSON.parse(fs.readFileSync(firstAdapterConfigPath, 'utf8'));
          if (adapterConfig.base_model_name_or_path) {
            const normalizedModel = normalizeModelName(adapterConfig.base_model_name_or_path);
            if (normalizedModel) {
              actualBaseModel = normalizedModel;
              console.log(`[vLLM] ✓ Found and normalized base model in adapter config: ${actualBaseModel}`);
            results.steps.push(`✓ Base model determined: ${actualBaseModel}`);
            } else {
              console.log(`[vLLM] ⚠ Base model in adapter config is not a valid HuggingFace ID: ${adapterConfig.base_model_name_or_path}`);
              console.log(`[vLLM] ⚠ Using provided base model instead: ${baseModel}`);
              results.steps.push(`⚠ Adapter config has invalid model format, using provided: ${baseModel}`);
            }
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
        // Verify essential adapter files exist before upload
        const essentialFiles = [
          'adapter_config.json',
          'adapter_model.safetensors',
          'adapter_model.bin'
        ];
        
        const foundEssentialFiles = essentialFiles.filter(fileName => {
          const filePath = path.join(version.path, fileName);
          return fs.existsSync(filePath);
        });
        
        if (foundEssentialFiles.length === 0) {
          throw new Error(`No essential adapter files found in ${version.path}. Expected at least one of: ${essentialFiles.join(', ')}`);
        }
        
        console.log(`[vLLM] ✓ Found essential files: ${foundEssentialFiles.join(', ')}`);
        
        // Create remote directory for this version
        const versionMkdirCommand = `mkdir -p ${remoteVersionDir}`;
        console.log(`[SSH] Executing: ${versionMkdirCommand}`);
        const versionMkdirResult = await ssh.execCommand(versionMkdirCommand);
        console.log(`[SSH] Exit code: ${versionMkdirResult.code}`);
        if (versionMkdirResult.stderr) {
          console.log(`[SSH] stderr: ${versionMkdirResult.stderr.trim()}`);
        }
        
        // Upload adapter directory using rsync
        console.log(`[vLLM] Starting rsync upload from ${version.path} to ${remoteAdapterPath}`);
        
        // Calculate total size for display
        const getAllFiles = (dir) => {
          const files = [];
          try {
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const entry of entries) {
              const fullPath = path.join(dir, entry.name);
              if (entry.isDirectory()) {
                files.push(...getAllFiles(fullPath));
              } else if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
                files.push(fullPath);
              }
            }
          } catch (error) {
            console.error(`[vLLM] Error reading directory ${dir}:`, error.message);
          }
          return files;
        };
        
        const allFiles = getAllFiles(version.path);
        
        if (allFiles.length === 0) {
          throw new Error(`No files found in adapter directory: ${version.path}`);
        }
        
        let totalSize = 0;
        for (const file of allFiles) {
          try {
            const stats = fs.statSync(file);
            totalSize += stats.size;
          } catch (e) {
            // Ignore errors
          }
        }
        
        const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
        console.log(`[vLLM] Total size to upload: ${totalSizeMB} MB (${allFiles.length} file(s))`);
        results.steps.push(`Uploading ${versionName} adapter (${allFiles.length} files, ${totalSizeMB} MB)...`);
        
        // Get SSH key path
        const uploadSshKeyPath = findSSHKey();
        if (!uploadSshKeyPath) {
          throw new Error('SSH key not found - cannot use rsync');
        }
        
        // Retry logic for upload
        let uploadSuccess = false;
        let uploadAttempts = 0;
        const maxUploadAttempts = 3;
        
        while (!uploadSuccess && uploadAttempts < maxUploadAttempts) {
          uploadAttempts++;
          
          if (uploadAttempts > 1) {
            console.log(`[vLLM] Retrying ${versionName} upload with rsync (attempt ${uploadAttempts}/${maxUploadAttempts})...`);
            results.steps.push(`Retrying ${versionName} upload (attempt ${uploadAttempts}/${maxUploadAttempts})...`);
            await new Promise(resolve => setTimeout(resolve, 3000)); // Wait before retry
          }
          
          try {
            // Progress callback for rsync
            let lastProgressStep = '';
            const progressCallback = (progressLine) => {
              // Log progress to console
              console.log(`[vLLM] [${versionName} Upload Progress] ${progressLine}`);
                      
                      // Update progress step
              const progressStep = `Uploading ${versionName} adapter... ${progressLine}`;
              if (progressStep !== lastProgressStep) {
                      const lastStepIndex = results.steps.length - 1;
                if (lastStepIndex >= 0 && results.steps[lastStepIndex].includes('Uploading') && results.steps[lastStepIndex].includes(versionName)) {
                        results.steps[lastStepIndex] = progressStep;
                      } else {
                        results.steps.push(progressStep);
                      }
                lastProgressStep = progressStep;
              }
            };
        
            // Use rsync to upload
            await uploadAdapterWithRsync(
              version.path,
              remoteAdapterPath,
              host,
              port || 22,
              username,
              uploadSshKeyPath,
              progressCallback
            );
            
            uploadSuccess = true;
            console.log(`[vLLM] ✓ ${versionName} adapter uploaded successfully with rsync`);
            results.steps.push(`✓ ${versionName} adapter uploaded (${allFiles.length} files, ${totalSizeMB} MB)`);
            
          } catch (uploadError) {
            const errorMsg = uploadError.message || String(uploadError);
            const errorStack = uploadError.stack || '';
            console.error(`[vLLM] rsync upload failed (attempt ${uploadAttempts}/${maxUploadAttempts}): ${errorMsg}`);
            console.error(`[vLLM] Error stack: ${errorStack}`);
            
            // Check if it's a connection error that we can retry
            const isConnectionError = errorMsg.includes('connection') || 
                                     errorMsg.includes('timeout') ||
                                     errorMsg.includes('refused') ||
                                     errorMsg.includes('network') ||
                                     errorMsg.includes('host') ||
                                     errorMsg.includes('no route');
            
            if (isConnectionError && uploadAttempts < maxUploadAttempts) {
              console.log(`[vLLM] Connection error detected, will retry...`);
                results.steps.push(`⚠ Connection error, retrying (${uploadAttempts}/${maxUploadAttempts})...`);
                continue; // Retry
              } else {
              // Not a retryable error or max attempts reached
                const finalError = new Error(`Upload failed after ${maxUploadAttempts} attempts: ${errorMsg}`);
                finalError.stack = errorStack;
                throw finalError;
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
    
    // CRITICAL: Pre-download the model before starting vLLM
    // This prevents vLLM from exiting too quickly due to model download issues
    results.steps.push('Pre-downloading base model to ensure it\'s cached...');
    console.log('[vLLM] Pre-downloading base model to ensure it\'s cached...');
    console.log(`[vLLM] Model: ${actualBaseModel}`);
    
    try {
      // Use the same Python that has vllm installed (not hardcoded python3)
      // Extract Python path from vllmCommand (e.g., "/usr/bin/python3" from "/usr/bin/python3 -m vllm")
      const pythonPath = vllmCommand.split(' ')[0]; // Get just the Python executable path
      console.log(`[vLLM] Using Python path for pre-download: ${pythonPath}`);
      
      // Check if model is already cached
      const cacheCheckCmd = `${pythonPath} -c "
import os
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
model_cache_path = os.path.join(cache_dir, 'models--${actualBaseModel.replace(/\//g, '--')}')
if os.path.exists(model_cache_path):
    # Also check if it has actual files (not just empty dir)
    files = [f for f in os.listdir(model_cache_path) if not f.startswith('.')]
    if files:
        print('CACHED')
    else:
        print('NOT_CACHED')
else:
    print('NOT_CACHED')
" 2>&1`;
      
      const cacheCheck = await ssh.execCommand(cacheCheckCmd);
      const isCached = cacheCheck.stdout.includes('CACHED');
      
      if (isCached) {
        console.log('[vLLM] ✓ Model is already cached');
        results.steps.push('✓ Model is already cached');
      } else {
        console.log('[vLLM] Model not cached, downloading...');
        results.steps.push('Downloading model from HuggingFace (this may take a few minutes)...');
        
        // Download the model using huggingface_hub
        // Use environment variable for HF_TOKEN to avoid shell escaping issues
        const escapedToken = hfToken ? hfToken.replace(/'/g, "'\\''") : '';
        const downloadCmd = `${pythonPath} -c "
import os
import sys
from huggingface_hub import snapshot_download
try:
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        print('Using HF_TOKEN for authentication...', file=sys.stderr)
    else:
        print('No HF_TOKEN found, downloading without authentication...', file=sys.stderr)
    print('Starting model download...', file=sys.stderr)
    snapshot_download(
        '${actualBaseModel}',
        token=hf_token if hf_token else None,
        local_files_only=False,
        resume_download=True
    )
    print('DOWNLOAD_COMPLETE')
except Exception as e:
    print(f'DOWNLOAD_ERROR: {str(e)}', file=sys.stderr)
    sys.exit(1)
" 2>&1`;
        
        // Set HF_TOKEN in environment for the download command
        const downloadCommandWithEnv = hfToken 
          ? `HF_TOKEN='${escapedToken}' ${downloadCmd}`
          : downloadCmd;
        
        console.log('[vLLM] Starting model download...');
        console.log('[vLLM] This may take several minutes depending on model size...');
        results.steps.push('Downloading model (this may take 5-10 minutes for large models)...');
        const downloadResult = await ssh.execCommand(downloadCommandWithEnv);
        
        if (downloadResult.code === 0 && downloadResult.stdout.includes('DOWNLOAD_COMPLETE')) {
          console.log('[vLLM] ✓ Model downloaded successfully');
          results.steps.push('✓ Model downloaded successfully');
        } else {
          const errorMsg = downloadResult.stderr || downloadResult.stdout || 'Unknown error';
          console.error(`[vLLM] ✗ Model download failed: ${errorMsg.substring(0, 500)}`);
          results.errors.push(`Model download failed: ${errorMsg.substring(0, 300)}`);
          results.errors.push('vLLM will attempt to download the model during startup, but this may cause startup issues.');
          results.steps.push('⚠ Model download failed - vLLM will try to download during startup');
        }
      }
    } catch (downloadError) {
      console.warn(`[vLLM] ⚠ Error checking/downloading model: ${downloadError.message}`);
      results.steps.push('⚠ Could not pre-download model - vLLM will attempt download during startup');
      // Don't fail completely - vLLM might still work
    }
    
    // Start or restart vLLM with the correct base model and adapters
    results.steps.push('Starting/restarting vLLM with base model and adapters...');
    console.log('[vLLM] Starting/restarting vLLM...');
    
    // Clean up any stuck Ray processes that might interfere with vLLM startup
    console.log('[vLLM] Checking for stuck Ray processes...');
    try {
      const rayCheck = await ssh.execCommand('pgrep -f "ray" || echo "no_ray"');
      if (rayCheck.stdout.trim() !== 'no_ray') {
        const rayPids = rayCheck.stdout.trim().split('\n').filter(p => p.trim());
        console.log(`[vLLM] Found ${rayPids.length} Ray process(es), checking if they're stuck...`);
        
        // Check if Ray is actually responding (Ray head should be on port 8265)
        const rayPortCheck = await ssh.execCommand('netstat -tlnp 2>/dev/null | grep :8265 || ss -tlnp 2>/dev/null | grep :8265 || echo "no_ray_port"');
        if (rayPortCheck.stdout.includes('no_ray_port')) {
          console.log('[vLLM] Ray processes found but Ray head not responding - cleaning up...');
          results.steps.push('Cleaning up stuck Ray processes...');
          await ssh.execCommand('pkill -9 -f "ray" || true');
          await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for cleanup
          console.log('[vLLM] ✓ Stuck Ray processes cleaned up');
        } else {
          console.log('[vLLM] Ray appears to be running normally');
        }
      }
    } catch (rayError) {
      console.warn(`[vLLM] ⚠ Error checking Ray processes: ${rayError.message}`);
      // Continue anyway
    }
    
    // CRITICAL: Verify all adapters exist and are valid before starting vLLM
    if (versions.length > 0) {
      console.log('[vLLM] Verifying adapter paths and configurations before starting...');
      results.steps.push('Verifying adapter paths and configurations...');
      
      const adapterVerificationErrors = [];
      const verifiedAdapterPaths = [];
      
      for (const version of versions) {
        const versionName = `V${version.version}`;
        const remoteAdapterPath = `${remoteModelDir}/${versionName}/adapter`;
        
        console.log(`[vLLM] Verifying ${versionName} adapter at: ${remoteAdapterPath}`);
        
        // Check if adapter directory exists
        const dirCheck = await ssh.execCommand(`test -d ${remoteAdapterPath} && echo "exists" || echo "not_found"`);
        if (dirCheck.stdout.trim() !== 'exists') {
          adapterVerificationErrors.push(`${versionName} adapter directory not found: ${remoteAdapterPath}`);
          console.error(`[vLLM] ✗ ${versionName} adapter directory missing: ${remoteAdapterPath}`);
          continue;
        }
        
        // Check for required files
        const requiredFiles = ['adapter_config.json'];
        const modelFiles = ['adapter_model.safetensors', 'adapter_model.bin'];
        
        for (const requiredFile of requiredFiles) {
          const fileCheck = await ssh.execCommand(`test -f ${remoteAdapterPath}/${requiredFile} && echo "exists" || echo "not_found"`);
          if (fileCheck.stdout.trim() !== 'exists') {
            adapterVerificationErrors.push(`${versionName} missing required file: ${requiredFile}`);
            console.error(`[vLLM] ✗ ${versionName} missing: ${requiredFile}`);
          }
        }
        
        // Check for at least one model file
        let hasModelFile = false;
        for (const modelFile of modelFiles) {
          const modelFileCheck = await ssh.execCommand(`test -f ${remoteAdapterPath}/${modelFile} && echo "exists" || echo "not_found"`);
          if (modelFileCheck.stdout.trim() === 'exists') {
            hasModelFile = true;
            break;
          }
        }
        
        if (!hasModelFile) {
          adapterVerificationErrors.push(`${versionName} missing adapter model file (adapter_model.safetensors or adapter_model.bin)`);
          console.error(`[vLLM] ✗ ${versionName} missing adapter model file`);
        }
        
        // Validate adapter_config.json
        const configCheck = await ssh.execCommand(`cat ${remoteAdapterPath}/adapter_config.json 2>/dev/null || echo "error"`);
        if (configCheck.stdout.trim() === 'error' || !configCheck.stdout.trim()) {
          adapterVerificationErrors.push(`${versionName} adapter_config.json is missing or unreadable`);
          console.error(`[vLLM] ✗ ${versionName} adapter_config.json invalid`);
        } else {
          try {
            const adapterConfig = JSON.parse(configCheck.stdout);
            
            // Verify base model matches
            if (adapterConfig.base_model_name_or_path && adapterConfig.base_model_name_or_path !== actualBaseModel) {
              console.warn(`[vLLM] ⚠ ${versionName} adapter config specifies base model: ${adapterConfig.base_model_name_or_path}, but we're using: ${actualBaseModel}`);
              results.steps.push(`⚠ Warning: ${versionName} adapter was trained for ${adapterConfig.base_model_name_or_path}, but using ${actualBaseModel}`);
            }
            
            // Check adapter type
            if (adapterConfig.peft_type && adapterConfig.peft_type !== 'LORA') {
              console.warn(`[vLLM] ⚠ ${versionName} adapter type is ${adapterConfig.peft_type}, not LORA`);
            }
            
            console.log(`[vLLM] ✓ ${versionName} adapter verified: ${adapterConfig.peft_type || 'LORA'} adapter for ${adapterConfig.base_model_name_or_path || actualBaseModel}`);
          } catch (e) {
            adapterVerificationErrors.push(`${versionName} adapter_config.json is invalid JSON: ${e.message}`);
            console.error(`[vLLM] ✗ ${versionName} adapter_config.json parse error: ${e.message}`);
          }
        }
        
        if (adapterVerificationErrors.length === 0 || !adapterVerificationErrors.some(e => e.startsWith(versionName))) {
          verifiedAdapterPaths.push(remoteAdapterPath);
          console.log(`[vLLM] ✓ ${versionName} adapter verified successfully`);
        }
      }
      
      if (adapterVerificationErrors.length > 0) {
        console.error(`[vLLM] ✗ Adapter verification failed with ${adapterVerificationErrors.length} error(s)`);
        results.errors.push('Adapter verification failed:');
        adapterVerificationErrors.forEach(error => {
          results.errors.push(`  - ${error}`);
        });
        results.errors.push('Please ensure all adapters are uploaded correctly before starting vLLM.');
        // Don't continue if adapters are invalid
        throw new Error(`Adapter verification failed: ${adapterVerificationErrors.join('; ')}`);
      } else {
        console.log(`[vLLM] ✓ All ${versions.length} adapter(s) verified successfully`);
        results.steps.push(`✓ All ${versions.length} adapter(s) verified`);
      }
    }
    
    // Build FastAPI inference server command
    const inferencePort = inferenceUrl ? new URL(inferenceUrl).port || '8000' : '8000';
    let envVars = '';
    if (hfToken) {
      // Escape the token for shell
      const escapedToken = hfToken.replace(/'/g, "'\\''");
      envVars = `HF_TOKEN='${escapedToken}' `;
    }
    
    // Build environment variables for FastAPI server
    const envVarsList = [];
    if (hfToken) {
      envVarsList.push(`HF_TOKEN='${hfToken.replace(/'/g, "'\\''")}'`);
    }
    envVarsList.push(`BASE_MODEL='${actualBaseModel}'`);
    envVarsList.push(`ADAPTER_BASE_PATH='/workspace/models'`);
    envVarsList.push(`PROFILE_NAME='${profileName}'`);
    envVarsList.push(`INFERENCE_PORT='${inferencePort}'`);
    const envString = envVarsList.join(' ');
    
    // Validate model name format (common issues: wrong format, missing slashes, etc.)
    console.log('\n[vLLM] ========================================');
    console.log(`[vLLM] VALIDATING MODEL NAME: ${actualBaseModel}`);
    console.log('[vLLM] ========================================');
    
    if (!actualBaseModel.includes('/')) {
      results.errors.push(`Model name '${actualBaseModel}' appears invalid (should be in format 'org/model-name'). Check adapter config.`);
      console.error(`[vLLM] ✗ Invalid model name format: ${actualBaseModel}`);
      console.error(`[vLLM] ✗ Model name should be in format: 'organization/model-name'`);
    } else {
      const [org, modelName] = actualBaseModel.split('/');
      console.log(`[vLLM] ✓ Model name format looks valid`);
      console.log(`[vLLM]   Organization: ${org}`);
      console.log(`[vLLM]   Model name: ${modelName}`);
      
      // Check for common model name issues
      if (modelName.includes(' ')) {
        console.warn(`[vLLM] ⚠ Model name contains spaces - this may cause issues`);
        results.steps.push(`⚠ Warning: Model name contains spaces: ${actualBaseModel}`);
      }
      
      // Model name validation is generic - no model-specific checks needed
      // vLLM supports all standard HuggingFace models including Mistral, Llama, Gemma, etc.
    }
    
    console.log('[vLLM] ========================================\n');
    
    // Log adapter paths that will be used
    if (versions.length > 0) {
      console.log('[INFERENCE] Adapter paths to be loaded:');
      versions.forEach((v) => {
        const adapterPath = `${remoteModelDir}/V${v.version}/adapter`;
        console.log(`[INFERENCE]   V${v.version}: ${adapterPath}`);
      });
      console.log('');
    }
    
    // Build FastAPI inference server command using uvicorn
    // The server will load adapters from /workspace/models/ProfileName/V*/adapter
    // We need to find Python that has uvicorn and fastapi installed
    let pythonCommand = 'python3'; // Default
    try {
      // Try to find Python with uvicorn
      const uvicornCheck = await ssh.execCommand('python3 -c "import uvicorn, fastapi" 2>&1 || python3 -m pip show uvicorn fastapi 2>&1 | head -1');
      if (uvicornCheck.code === 0 || uvicornCheck.stdout.includes('Name:')) {
        pythonCommand = 'python3';
        console.log('[INFERENCE] ✓ Found Python with uvicorn/fastapi');
    } else {
        // Try python
        const pythonCheck = await ssh.execCommand('python -c "import uvicorn, fastapi" 2>&1');
        if (pythonCheck.code === 0) {
          pythonCommand = 'python';
          console.log('[INFERENCE] ✓ Found python with uvicorn/fastapi');
        }
      }
    } catch (e) {
      console.warn('[INFERENCE] ⚠ Could not verify uvicorn installation, using default python3');
    }
    
    // Build command to run FastAPI server
    // The inference_server.py should be in /workspace/utils/ or we need to upload it
    const inferenceServerPath = '/workspace/utils/inference_server.py';
    const fastApiCommand = `${envString} ${pythonCommand} -m uvicorn utils.inference_server:app --host 0.0.0.0 --port ${inferencePort}`;
    
    console.log(`[INFERENCE] Command to run: ${fastApiCommand.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
    console.log(`[INFERENCE] Full command breakdown:`);
    console.log(`[INFERENCE]   Base model: ${actualBaseModel}`);
    console.log(`[INFERENCE]   Port: ${inferencePort}`);
    console.log(`[INFERENCE]   Profile: ${profileName}`);
    if (versions.length > 0) {
      console.log(`[INFERENCE]   Adapters: ${versions.length} version(s)`);
      versions.forEach((v) => {
        console.log(`[INFERENCE]     - V${v.version}: ${remoteModelDir}/V${v.version}/adapter`);
      });
    }
    console.log('');
    results.steps.push(`Starting FastAPI inference server with base model: ${actualBaseModel}`);
    results.steps.push(`⚠ Note: If model fails to load, verify model name '${actualBaseModel}' exists on HuggingFace`);
    results.steps.push(`Server will download model from HuggingFace if not already cached`);
    if (hfToken) {
      results.steps.push(`✓ Using HuggingFace token for authentication`);
    }
    if (versions.length > 0) {
      results.steps.push(`✓ Will load ${versions.length} adapter version(s): ${versions.map(v => `V${v.version}`).join(', ')}`);
    }
    
    // First check if supervisor is managing the inference server (this is the authoritative source)
    // We check supervisor FIRST because it's the authoritative source, even if pgrep finds a process
    const supervisorCheck = await ssh.execCommand('supervisorctl status inference 2>&1');
    const supervisorOutput = supervisorCheck.stdout.trim();
    
    // Check if supervisor knows about inference (it will return status even if STOPPED/FATAL/EXITED)
    // If it doesn't know about it, it will return an error like "inference: ERROR (no such process)"
    const isSupervisorManaged = !supervisorOutput.includes('ERROR (no such process)') && 
                                 !supervisorOutput.includes('no such process') &&
                                 supervisorOutput.length > 0 &&
                                 (supervisorOutput.includes('RUNNING') || 
                                  supervisorOutput.includes('STOPPED') || 
                                  supervisorOutput.includes('STOPPING') ||
                                  supervisorOutput.includes('FATAL') ||
                                  supervisorOutput.includes('EXITED') ||
                                  supervisorOutput.includes('STARTING'));
    
    // Also check if there's a process running (might be from previous direct start)
    const inferenceProcessCheck = await ssh.execCommand('pgrep -f "uvicorn.*inference_server" | head -1 || echo "not_running"');
    const isInferenceProcessRunning = inferenceProcessCheck.stdout.trim() !== 'not_running';
    
    console.log(`[INFERENCE] Supervisor check output: ${supervisorOutput}`);
    console.log(`[INFERENCE] Is supervisor managing: ${isSupervisorManaged}`);
      
      if (isSupervisorManaged) {
      // Supervisor is managing vLLM - check its status
      const supervisorStatus = supervisorOutput;
      const isStopped = supervisorStatus.includes('STOPPED') || supervisorStatus.includes('STOPPING');
      const isRunning = supervisorStatus.includes('RUNNING');
      const isFatal = supervisorStatus.includes('FATAL');
      const isExited = supervisorStatus.includes('EXITED');
      
      console.log(`[vLLM] Supervisor status: ${supervisorStatus}`);
      
      if (isStopped || isFatal || isExited) {
        // vLLM is stopped or failed - we need to start it
        console.log(`[vLLM] vLLM is ${isStopped ? 'STOPPED' : isFatal ? 'FATAL' : 'EXITED'}, will start it...`);
        results.steps.push(`vLLM is ${isStopped ? 'stopped' : isFatal ? 'failed' : 'exited'}, starting...`);
        
        // Check logs to see why it stopped
        if (isFatal || isExited) {
          const errorLogs = await ssh.execCommand('supervisorctl tail -50 vllm 2>/dev/null | tail -20 || echo ""');
          if (errorLogs.stdout && errorLogs.stdout.trim()) {
            console.log(`[vLLM] Recent error logs:`);
            console.log(errorLogs.stdout.substring(0, 500));
            // Check for common issues
            if (errorLogs.stdout.includes('port') && errorLogs.stdout.includes('already in use')) {
              results.errors.push('Port is already in use. Another vLLM instance may be running. Try: pkill -f vllm');
            } else if (errorLogs.stdout.includes('CUDA') || errorLogs.stdout.includes('GPU')) {
              results.errors.push('GPU/CUDA error detected. Check GPU availability and drivers.');
            } else if (errorLogs.stdout.includes('OOM') || errorLogs.stdout.includes('out of memory')) {
              results.errors.push('Out of memory error. Model may be too large for available RAM.');
            } else if (errorLogs.stdout.includes('model') && errorLogs.stdout.includes('not found')) {
              results.errors.push('Model not found. Check if the model path is correct.');
            }
          }
        }
        
        // Ensure inference_server.py is uploaded to the server
        results.steps.push('Uploading inference server script...');
        console.log('[INFERENCE] Uploading inference_server.py to /workspace/utils/...');
        try {
          // Ensure utils directory exists
          await ssh.execCommand('mkdir -p /workspace/utils');
          
          // Upload inference_server.py
          const localInferenceServerPath = path.join(__dirname, '..', 'utils', 'inference_server.py');
          if (fs.existsSync(localInferenceServerPath)) {
            await ssh.putFile(localInferenceServerPath, '/workspace/utils/inference_server.py');
            console.log('[INFERENCE] ✓ Inference server script uploaded');
            results.steps.push('✓ Inference server script uploaded');
          } else {
            console.warn('[INFERENCE] ⚠ inference_server.py not found locally, assuming it exists on server');
            results.steps.push('⚠ Inference server script not found locally');
          }
        } catch (uploadError) {
          console.error(`[INFERENCE] ✗ Failed to upload inference server script: ${uploadError.message}`);
          results.errors.push(`Failed to upload inference server script: ${uploadError.message}`);
        }
        
        // Check supervisor config before starting
        const supervisorConfigCheck = await ssh.execCommand('cat /etc/supervisor/conf.d/inference.conf 2>/dev/null || echo "not_found"');
        if (!supervisorConfigCheck.stdout.includes('not_found')) {
          const currentConfig = supervisorConfigCheck.stdout;
          console.log(`[INFERENCE] Current supervisor config:`);
          console.log(`[INFERENCE] ${currentConfig.substring(0, 500)}...`);
          
          // Check if config matches what we want to run
          const configPortMatch = currentConfig.match(/--port\s+(\d+)/);
          const configProfileMatch = currentConfig.match(/PROFILE_NAME=['"]([^'"]+)['"]/);
          
          const expectedPort = new URL(inferenceUrl || `http://localhost:${inferencePort}`).port || inferencePort;
          const configPort = configPortMatch ? configPortMatch[1] : null;
          const configProfile = configProfileMatch ? configProfileMatch[1] : null;
          
          // Check for critical supervisor parameters
          const hasStartsecs = currentConfig.includes('startsecs=');
          const hasStartretries = currentConfig.includes('startretries=');
          
          const needsUpdate = (configPort && configPort !== expectedPort) ||
                             (configProfile && configProfile !== profileName) ||
                             !hasStartsecs ||
                             !hasStartretries;
          
          if (needsUpdate) {
            console.warn(`[INFERENCE] ⚠ Supervisor config doesn't match desired configuration`);
            console.warn(`[INFERENCE]   Current port: ${configPort || 'unknown'}, Desired: ${expectedPort}`);
            console.warn(`[INFERENCE]   Current profile: ${configProfile || 'unknown'}, Desired: ${profileName}`);
            
            results.steps.push('⚠ Supervisor config needs updating to match desired configuration');
            results.steps.push(`⚠ Current config: port=${configPort || 'unknown'}, profile=${configProfile || 'unknown'}`);
            results.steps.push(`⚠ Desired config: port=${expectedPort}, profile=${profileName}`);
            
            // Try to update supervisor config
            console.log('[INFERENCE] Attempting to update supervisor config...');
            results.steps.push('Updating supervisor configuration...');
            
            // Create new supervisor config for FastAPI server
            const supervisorConfigContent = `[program:inference]
command=${fastApiCommand}
directory=/workspace
autostart=true
autorestart=true
startsecs=300
startretries=2
stderr_logfile=/var/log/supervisor/inference-stderr.log
stdout_logfile=/var/log/supervisor/inference-stdout.log
user=root
`;
            
            // Write new config
            const tempConfigPath = path.join(os.tmpdir(), `inference_supervisor_${Date.now()}.conf`);
            fs.writeFileSync(tempConfigPath, supervisorConfigContent, 'utf8');
            
            try {
              // Backup old config
              await ssh.execCommand('cp /etc/supervisor/conf.d/inference.conf /etc/supervisor/conf.d/inference.conf.backup.' + Date.now() + ' 2>/dev/null || true');
              
              // Upload new config
              await ssh.putFile(tempConfigPath, '/etc/supervisor/conf.d/inference.conf');
              
              // Reload supervisor
              await ssh.execCommand('supervisorctl reread');
              await ssh.execCommand('supervisorctl update');
              
              console.log('[INFERENCE] ✓ Supervisor config updated');
              results.steps.push('✓ Supervisor configuration updated');
              
              // Clean up temp file
              fs.unlinkSync(tempConfigPath);
            } catch (updateError) {
              console.error(`[INFERENCE] ✗ Failed to update supervisor config: ${updateError.message}`);
              results.errors.push(`Failed to update supervisor config: ${updateError.message}. You may need to update /etc/supervisor/conf.d/inference.conf manually.`);
              results.errors.push(`Expected command: ${fastApiCommand.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
              fs.unlinkSync(tempConfigPath);
            }
          } else {
            // Even if command matches, check if critical parameters are missing
            if (!hasStartsecs || !hasStartretries) {
              console.warn(`[INFERENCE] ⚠ Supervisor config missing startsecs/startretries - this causes "exited too quickly" errors`);
              console.warn(`[INFERENCE] ⚠ Forcing config update to add missing parameters...`);
              results.steps.push('⚠ Supervisor config missing startsecs/startretries - updating...');
              
              // Force update to add missing parameters
              const supervisorConfigContent = `[program:inference]
command=${fastApiCommand}
directory=/workspace
autostart=true
autorestart=true
startsecs=300
startretries=2
stderr_logfile=/var/log/supervisor/inference-stderr.log
stdout_logfile=/var/log/supervisor/inference-stdout.log
user=root
`;
              
              const tempConfigPath = path.join(os.tmpdir(), `inference_supervisor_${Date.now()}.conf`);
              fs.writeFileSync(tempConfigPath, supervisorConfigContent, 'utf8');
              
              try {
                await ssh.execCommand('cp /etc/supervisor/conf.d/inference.conf /etc/supervisor/conf.d/inference.conf.backup.' + Date.now() + ' 2>/dev/null || true');
                await ssh.putFile(tempConfigPath, '/etc/supervisor/conf.d/inference.conf');
                await ssh.execCommand('supervisorctl reread');
                await ssh.execCommand('supervisorctl update');
                console.log('[INFERENCE] ✓ Supervisor config updated with startsecs/startretries');
                results.steps.push('✓ Supervisor configuration updated');
                fs.unlinkSync(tempConfigPath);
              } catch (updateError) {
                console.error(`[INFERENCE] ✗ Failed to update supervisor config: ${updateError.message}`);
                results.errors.push(`Failed to update supervisor config: ${updateError.message}`);
                fs.unlinkSync(tempConfigPath);
              }
            } else {
              console.log('[INFERENCE] ✓ Supervisor config matches desired configuration');
            results.steps.push('✓ Supervisor configuration is correct');
          }
        }
        } else {
          // No existing config - create new one
          console.log('[INFERENCE] No existing supervisor config, creating new one...');
          results.steps.push('Creating new supervisor configuration...');
          
          const supervisorConfigContent = `[program:inference]
command=${fastApiCommand}
directory=/workspace
autostart=true
autorestart=true
startsecs=300
startretries=2
stderr_logfile=/var/log/supervisor/inference-stderr.log
stdout_logfile=/var/log/supervisor/inference-stdout.log
user=root
`;
          
          const tempConfigPath = path.join(os.tmpdir(), `inference_supervisor_${Date.now()}.conf`);
          fs.writeFileSync(tempConfigPath, supervisorConfigContent, 'utf8');
          
          try {
            await ssh.putFile(tempConfigPath, '/etc/supervisor/conf.d/inference.conf');
            await ssh.execCommand('supervisorctl reread');
            await ssh.execCommand('supervisorctl update');
            console.log('[INFERENCE] ✓ Supervisor config created');
            results.steps.push('✓ Supervisor configuration created');
            fs.unlinkSync(tempConfigPath);
          } catch (createError) {
            console.error(`[INFERENCE] ✗ Failed to create supervisor config: ${createError.message}`);
            results.errors.push(`Failed to create supervisor config: ${createError.message}`);
            fs.unlinkSync(tempConfigPath);
          }
        }
        
        // Start inference server
        console.log('[INFERENCE] Starting inference server via supervisor...');
        const startResult = await ssh.execCommand('supervisorctl start inference');
        if (startResult.code === 0) {
          results.steps.push('✓ Inference server started via supervisor');
          console.log('[INFERENCE] ✓ Inference server started via supervisor');
          // Wait for server to start (it may need time to load models)
          console.log('[INFERENCE] Waiting for inference server to initialize (this may take 30-60 seconds)...');
          await new Promise(resolve => setTimeout(resolve, 5000));
          
          // Check if it actually started (might have failed immediately)
          const statusCheck = await ssh.execCommand('supervisorctl status inference 2>&1');
          if (statusCheck.stdout.includes('FATAL') || statusCheck.stdout.includes('spawn error') || statusCheck.stdout.includes('Exited too quickly')) {
            console.error(`[INFERENCE] ✗ Inference server failed to start: ${statusCheck.stdout.trim()}`);
            results.errors.push(`Inference server failed to start: ${statusCheck.stdout.trim()}`);
            
            // Get detailed error from supervisor logs - try multiple sources
            console.log('[INFERENCE] Fetching detailed error logs...');
            const errorLogSources = [
              'supervisorctl tail -200 inference 2>&1 | tail -50',
              'tail -100 /var/log/supervisor/inference-stderr.log 2>&1',
              'tail -100 /var/log/supervisor/inference-stdout.log 2>&1',
              'cat /var/log/supervisor/supervisord.log 2>&1 | grep -i inference | tail -20'
            ];
            
            let errorLogs = { stdout: '', stderr: '' };
            for (const cmd of errorLogSources) {
              try {
                const logCheck = await ssh.execCommand(cmd);
                if (logCheck.stdout && logCheck.stdout.trim() && logCheck.stdout.length > 50) {
                  errorLogs = logCheck;
                  console.log(`[INFERENCE] Retrieved logs from: ${cmd.split(' ')[0]}`);
                  break;
                }
              } catch (e) {
                continue;
              }
            }
            
            if (errorLogs.stdout && errorLogs.stdout.trim()) {
              console.error(`[INFERENCE] Supervisor error logs:`);
              console.error(errorLogs.stdout.substring(0, 1000));
              results.errors.push(`Error details: ${errorLogs.stdout.substring(0, 500)}`);
              
              // Check for common issues
              if (errorLogs.stdout.includes('No module named') || errorLogs.stdout.includes('ModuleNotFoundError')) {
                const missingModule = errorLogs.stdout.match(/No module named ['"]([^'"]+)['"]/);
                if (missingModule) {
                  results.errors.push(`Python cannot find module: ${missingModule[1]}`);
                  results.errors.push(`Try: pip install ${missingModule[1]}`);
                } else {
                  results.errors.push('Python module import error. Check if required packages are installed.');
                }
              } else if (errorLogs.stdout.includes('Permission denied') || errorLogs.stdout.includes('EACCES')) {
                results.errors.push('Permission denied. Check file permissions and user settings in supervisor config.');
              } else if (errorLogs.stdout.includes('No such file') || errorLogs.stdout.includes('ENOENT')) {
                results.errors.push('Command or file not found. Check if the Python path or inference_server.py path is correct.');
              } else if (errorLogs.stdout.includes('spawn error') || errorLogs.stdout.includes('Exited too quickly')) {
                results.errors.push('Inference server spawn error. Common causes:');
                results.errors.push('  - Python path in supervisor config is incorrect');
                results.errors.push('  - Missing dependencies (uvicorn, fastapi, transformers, peft)');
                results.errors.push('  - inference_server.py not found at /workspace/utils/inference_server.py');
                results.errors.push('  - Supervisor config missing startsecs parameter (should be 300)');
                results.errors.push(`Check: supervisorctl tail -200 inference`);
              } else if (errorLogs.stdout.includes('Exited too quickly') || errorLogs.stdout.includes('FATAL')) {
                results.errors.push('Inference server exited too quickly. Common causes:');
                results.errors.push('  - Model not downloaded/cached (check if pre-download completed)');
                results.errors.push('  - GPU not available or CUDA errors');
                results.errors.push('  - Invalid model or adapter configuration');
                results.errors.push('  - Missing dependencies');
                results.errors.push('  - Supervisor config missing startsecs=300 parameter');
                results.errors.push('Check full logs: supervisorctl tail -200 inference');
              }
            } else {
              // No logs available - try to get stderr directly
              const stderrCheck = await ssh.execCommand('cat /var/log/supervisor/inference-stderr.log 2>&1 | tail -50 || echo ""');
              if (stderrCheck.stdout && stderrCheck.stdout.trim()) {
                console.error(`[INFERENCE] Stderr log:`);
                console.error(stderrCheck.stdout.substring(0, 500));
                results.errors.push(`Stderr: ${stderrCheck.stdout.substring(0, 300)}`);
              } else {
                results.errors.push('No error logs found. Check supervisor logs manually: supervisorctl tail -200 inference');
              }
            }
          } else if (statusCheck.stdout.includes('STARTING')) {
            console.log('[INFERENCE] Inference server is starting (status: STARTING)');
            results.steps.push('Inference server is starting, waiting for initialization...');
          } else if (statusCheck.stdout.includes('RUNNING')) {
            console.log('[INFERENCE] ✓ Inference server is running (supervisor reports RUNNING)');
            results.steps.push('✓ Inference server is running (supervisor reports RUNNING)');
        } else {
            console.log(`[INFERENCE] ⚠ Inference server status: ${statusCheck.stdout.trim()}`);
            results.steps.push(`⚠ Inference server status: ${statusCheck.stdout.trim()}`);
          }
        } else {
          results.errors.push('Could not start inference server via supervisor. Check supervisor logs.');
          results.errors.push(`Start command failed: ${startResult.stderr || startResult.stdout}`);
          console.error(`[INFERENCE] ✗ Failed to start: ${startResult.stderr || startResult.stdout}`);
          
          // Get detailed error from supervisor logs
          const errorLogs = await ssh.execCommand('supervisorctl tail -50 inference 2>&1 | tail -20 || tail -20 /var/log/supervisor/inference-stderr.log 2>&1 || echo ""');
          if (errorLogs.stdout && errorLogs.stdout.trim()) {
            console.error(`[INFERENCE] Supervisor error logs:`);
            console.error(errorLogs.stdout.substring(0, 500));
            results.errors.push(`Error details: ${errorLogs.stdout.substring(0, 300)}`);
          }
        }
      } else if (isRunning) {
        // vLLM is running - check if restart is needed
        console.log('[vLLM] vLLM is managed by supervisor and is running, checking configuration...');
        results.steps.push('vLLM is running, checking configuration...');
        
        // Check current supervisor config
        const supervisorConfigCheck = await ssh.execCommand('cat /etc/supervisor/conf.d/vllm.conf 2>/dev/null || echo "not_found"');
        if (!supervisorConfigCheck.stdout.includes('not_found')) {
          const currentConfig = supervisorConfigCheck.stdout;
          console.log(`[vLLM] Current supervisor config:`);
          console.log(`[vLLM] ${currentConfig.substring(0, 500)}...`);
          
          // Check if config matches what we want to run
          const configModelMatch = currentConfig.match(/serve\s+([^\s]+)/);
          const configPortMatch = currentConfig.match(/--port\s+(\d+)/);
          const configLoraMatch = currentConfig.match(/--enable-lora/);
          
          // Check if command has HF_TOKEN in the command line (wrong format - should be in environment)
          const commandLineMatch = currentConfig.match(/command=(.+)/);
          const hasHFTokenInCommand = commandLineMatch && commandLineMatch[1].trim().startsWith("HF_TOKEN=");
          
          const expectedPort = new URL(vllmUrl).port || '8000';
          const configModel = configModelMatch ? configModelMatch[1] : null;
          const configPort = configPortMatch ? configPortMatch[1] : null;
          
          // Check for critical supervisor parameters (these prevent "exited too quickly" errors)
          const hasStartsecs = currentConfig.includes('startsecs=');
          const hasStartretries = currentConfig.includes('startretries=');
          
          const needsUpdate = (configModel && configModel !== actualBaseModel) || 
                             (configPort && configPort !== expectedPort) ||
                             (versions.length > 0 && !configLoraMatch) ||
                             hasHFTokenInCommand || // Force update if command format is wrong
                             !hasStartsecs || // CRITICAL: Missing startsecs causes "exited too quickly"
                             !hasStartretries; // CRITICAL: Missing startretries causes premature failures
          
          if (needsUpdate) {
            console.warn(`[vLLM] ⚠ Supervisor config doesn't match desired configuration`);
            console.warn(`[vLLM]   Current model: ${configModel || 'unknown'}, Desired: ${actualBaseModel}`);
            console.warn(`[vLLM]   Current port: ${configPort || 'unknown'}, Desired: ${expectedPort}`);
            console.warn(`[vLLM]   LoRA enabled: ${configLoraMatch ? 'yes' : 'no'}, Desired: ${versions.length > 0 ? 'yes' : 'no'}`);
            if (hasHFTokenInCommand) {
              console.warn(`[vLLM]   ⚠ Command format is wrong: HF_TOKEN is in command line instead of environment`);
            }
            
            results.steps.push('⚠ Supervisor config needs updating to match desired configuration');
            results.steps.push(`⚠ Current config: model=${configModel || 'unknown'}, port=${configPort || 'unknown'}`);
            results.steps.push(`⚠ Desired config: model=${actualBaseModel}, port=${expectedPort}`);
            if (hasHFTokenInCommand) {
              results.steps.push(`⚠ Command format error: HF_TOKEN should be in environment, not command line`);
            }
            
            // Try to update supervisor config
            console.log('[vLLM] Attempting to update supervisor config...');
            results.steps.push('Updating supervisor configuration...');
            
            // Build command using the detected vllmCommand (which has the correct Python path)
            // This ensures we use the Python that actually has vllm installed
            let commandOnly;
            if (versions.length > 0) {
              commandOnly = `${vllmCommand} serve ${actualBaseModel} --port ${vllmPort} --trust-remote-code --enable-lora --lora-modules ${loraModulesForStart}`;
            } else {
              commandOnly = `${vllmCommand} serve ${actualBaseModel} --port ${vllmPort} --trust-remote-code`;
            }
            
            console.log(`[vLLM] Using detected vLLM command for supervisor: ${commandOnly}`);
            
            // Create new supervisor config
            // Note: startsecs=300 (5 minutes) gives vLLM time to download model if not cached
            // Model downloads can take 5-10 minutes for large models, so we give it time
            // If model is already cached, vLLM should start much faster
            const supervisorConfigContent = `[program:vllm]
command=${commandOnly}
directory=/workspace
autostart=true
autorestart=true
startsecs=300
startretries=2
stderr_logfile=/var/log/supervisor/vllm-stderr.log
stdout_logfile=/var/log/supervisor/vllm-stdout.log
user=root
environment=HF_TOKEN="${hfToken || ''}"
`;
            
            // Write new config
            const tempConfigPath = path.join(os.tmpdir(), `vllm_supervisor_${Date.now()}.conf`);
            fs.writeFileSync(tempConfigPath, supervisorConfigContent, 'utf8');
            
            try {
              // Backup old config
              await ssh.execCommand('cp /etc/supervisor/conf.d/vllm.conf /etc/supervisor/conf.d/vllm.conf.backup.' + Date.now() + ' 2>/dev/null || true');
              
              // Upload new config
              await ssh.putFile(tempConfigPath, '/etc/supervisor/conf.d/vllm.conf');
              
              // Reload supervisor
              await ssh.execCommand('supervisorctl reread');
              await ssh.execCommand('supervisorctl update');
              
              console.log('[vLLM] ✓ Supervisor config updated');
              results.steps.push('✓ Supervisor configuration updated');
              
              // Clean up temp file
              fs.unlinkSync(tempConfigPath);
            } catch (updateError) {
              console.error(`[vLLM] ✗ Failed to update supervisor config: ${updateError.message}`);
              results.errors.push(`Failed to update supervisor config: ${updateError.message}. You may need to update /etc/supervisor/conf.d/vllm.conf manually.`);
              results.errors.push(`Expected command: ${vllmCommandToRun.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
              fs.unlinkSync(tempConfigPath);
            }
          } else {
            console.log('[vLLM] ✓ Supervisor config matches desired configuration');
            results.steps.push('✓ Supervisor configuration is correct');
          }
        }
        
        console.log('[vLLM] vLLM is managed by supervisor, stopping...');
        results.steps.push('Stopping vLLM via supervisor...');
        await ssh.execCommand('supervisorctl stop vllm');
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
        
        console.log('[vLLM] Restarting vLLM via supervisor...');
        results.steps.push('Restarting vLLM via supervisor...');
        const restartResult = await ssh.execCommand('supervisorctl restart vllm');
        if (restartResult.code !== 0) {
          results.errors.push('Could not restart vLLM via supervisor. Check supervisor logs.');
          results.errors.push(`Expected command: ${vllmCommandToRun.replace(hfToken || '', 'HF_TOKEN_HIDDEN')}`);
        } else {
          results.steps.push('✓ vLLM restarted via supervisor');
          console.log('[vLLM] ✓ vLLM restarted via supervisor');
          
          // Wait a moment for the new process to start
          console.log('[vLLM] Waiting for new vLLM process to start...');
          await new Promise(resolve => setTimeout(resolve, 3000));
          
          // Verify the new process is running and get its PID
          const newProcessCheck = await ssh.execCommand('pgrep -f "python.*vllm.*serve" | head -1 || echo "not_running"');
          const newPid = newProcessCheck.stdout.trim();
          if (newPid !== 'not_running') {
            console.log(`[vLLM] ✓ New vLLM process started with PID: ${newPid}`);
            results.steps.push(`✓ New vLLM process running (PID: ${newPid})`);
          } else {
            console.warn(`[vLLM] ⚠ New vLLM process not found after restart`);
            results.steps.push(`⚠ New vLLM process not found - may still be starting`);
          }
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
    
    // Wait a moment for inference server to start
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Verify inference server is running and get the actual PID
    const verifyCheck = await ssh.execCommand('pgrep -f "uvicorn.*inference_server" | head -1 || echo "not_running"');
    const inferencePid = verifyCheck.stdout.trim();
    if (inferencePid === 'not_running') {
      results.errors.push('Inference server process not found after startup attempt');
      console.error('[INFERENCE] ✗ Inference server process not found after startup');
    } else {
      results.steps.push(`✓ Inference server process is running (PID: ${inferencePid})`);
      console.log(`[INFERENCE] ✓ Inference server process is running (PID: ${inferencePid})`);
      
      // Store the PID for later log filtering
      storedVLLMPid = inferencePid;
      
      // Check what command the process is actually running
      const processCmdCheck = await ssh.execCommand(`ps -p ${inferencePid} -o args= 2>/dev/null || echo ""`);
      if (processCmdCheck.stdout) {
        const actualCmd = processCmdCheck.stdout.trim();
        console.log(`[INFERENCE] Process command: ${actualCmd.substring(0, 300)}...`);
        
        // Verify it's running on the correct port
        const cmdPortMatch = actualCmd.match(/--port\s+(\d+)/);
        if (cmdPortMatch) {
          const cmdPort = cmdPortMatch[1];
          const expectedPort = new URL(inferenceUrl || `http://localhost:${inferencePort}`).port || inferencePort;
          if (cmdPort === expectedPort) {
            console.log(`[INFERENCE] ✓ Process is running on correct port: ${expectedPort}`);
          } else {
            console.error(`[INFERENCE] ✗ Process is running on different port: ${cmdPort} (expected: ${expectedPort})`);
            results.errors.push(`Inference server process is running on port ${cmdPort} instead of ${expectedPort}. Supervisor config may not have updated correctly.`);
          }
        }
      }
    }
    
    // Monitor for model download/loading if vLLM is running and URL provided
    if (vllmUrl) {
      // First, do an immediate status check via SSH to see current state
      results.steps.push('Checking vLLM status immediately...');
      console.log('[vLLM] Checking vLLM status immediately via SSH...');
      
      try {
        // Check if vLLM process is running
        const immediateProcessCheck = await ssh.execCommand('pgrep -f vllm || echo "not_running"');
        const isRunning = immediateProcessCheck.stdout.trim() !== 'not_running';
        
        if (isRunning) {
          console.log('[vLLM] ✓ vLLM process is running');
          
          // Check if inference server API is responding from the server
          const localUrl = `http://localhost:${new URL(inferenceUrl || `http://localhost:${inferencePort}`).port || inferencePort}`;
          // Use Basic Auth if we have the token
          const curlAuth = openButtonToken 
            ? `-u vastai:${openButtonToken}`
            : '';
          // Check /health endpoint first (simpler, faster)
          const immediateHealthCheck = await ssh.execCommand(`curl -s ${curlAuth} ${localUrl}/health 2>&1 | head -50 || echo "curl_failed"`);
          
          if (!immediateHealthCheck.stdout.includes('curl_failed') && immediateHealthCheck.stdout.trim()) {
            try {
              const healthData = JSON.parse(immediateHealthCheck.stdout);
              if (healthData.status === 'healthy') {
                // Server is healthy, check /models for available models
                const modelsCheck = await ssh.execCommand(`curl -s ${curlAuth} ${localUrl}/models 2>&1 | head -100 || echo "curl_failed"`);
                if (!modelsCheck.stdout.includes('curl_failed')) {
                  const modelsData = JSON.parse(modelsCheck.stdout);
                  const modelNames = modelsData.map(m => m.name || m);
              if (modelNames.length > 0) {
                    results.steps.push(`✓ Inference server is ready! Available models: ${modelNames.join(', ')}`);
                    console.log(`[INFERENCE] ✓ Inference server is ready! Available: ${modelNames.join(', ')}`);
                    // Skip long monitoring if server is already ready
                    results.steps.push('Skipping log monitoring - server is ready');
                    console.log('[INFERENCE] Skipping log monitoring - server is ready');
                    modelReady = true;
              } else {
                    results.steps.push('⚠ Inference server is running but no models loaded yet (may still be loading)');
                    console.log('[INFERENCE] ⚠ Inference server is running but no models loaded yet');
                // Continue with monitoring
                results.steps.push('Starting log monitoring to track loading progress...');
                    console.log('[INFERENCE] Starting log monitoring to track loading progress...');
                  }
                }
              }
            } catch (e) {
              console.log('[INFERENCE] Could not parse immediate API response, will monitor logs');
              results.steps.push('Inference server is running, monitoring logs for progress...');
            }
          } else {
            console.log('[INFERENCE] Inference server API not yet responding, will monitor logs');
            results.steps.push('Inference server is running but API not yet responding, monitoring logs...');
          }
        } else {
          results.steps.push('⚠ vLLM process not found - may have failed to start');
          console.log('[vLLM] ⚠ vLLM process not found');
        }
      } catch (sshError) {
        console.warn(`[vLLM] Error in immediate status check: ${sshError.message}`);
        results.steps.push('Could not check immediate status, will monitor logs...');
      }
      
      // Check if we should skip monitoring (server already ready)
      const shouldSkipMonitoring = results.steps.some(step => step.includes('already ready')) || modelReady;
      
      if (!shouldSkipMonitoring) {
        // Hybrid approach: Poll API for readiness + monitor logs for loading progress
        results.steps.push('Monitoring model loading progress...');
        console.log('[INFERENCE] Monitoring model loading (checking logs for progress + polling API for readiness)...');
        
        const localUrl = `http://localhost:${new URL(inferenceUrl || `http://localhost:${inferencePort}`).port || inferencePort}`;
        const maxPollTime = 300000; // 5 minutes max
        const pollInterval = 2000; // Check API every 2 seconds
        const logCheckInterval = 5000; // Check logs every 5 seconds
        const statusUpdateInterval = 5000; // Show status updates every 5 seconds
        const diagnosticCheckInterval = 30000; // Run diagnostic checks every 30 seconds
        const startPollTime = Date.now();
        // modelReady is declared at function scope, reset it here for this monitoring session
        modelReady = false;
        let lastStatus = '';
        let lastLogCheck = 0;
        let lastStatusUpdate = 0;
        let lastDiagnosticCheck = 0;
        let downloadDetected = false;
        let loadingDetected = false;
        let lastDownloadProgress = '';
        let apiResponding = false;
        let portListening = false;
        let supervisorStatus = '';
        let pidChangeCount = 0;
        let lastPidChangeTime = 0;
        // storedVLLMPid is declared at function scope, reset tracking here
        pidChangeCount = 0;
        lastPidChangeTime = 0;
        
        console.log('[vLLM] Starting monitoring loop...');
        
        while (Date.now() - startPollTime < maxPollTime && !modelReady) {
          const elapsed = Math.floor((Date.now() - startPollTime) / 1000);
          
          // Run diagnostic checks periodically (every 30 seconds)
          if (Date.now() - lastDiagnosticCheck >= diagnosticCheckInterval) {
            lastDiagnosticCheck = Date.now();
            const elapsed = Math.floor((Date.now() - startPollTime) / 1000);
            console.log(`[vLLM] [DIAGNOSTIC] Running diagnostic check at ${elapsed}s...`);
            
            try {
              // Check supervisor status
              const supervisorCheck = await ssh.execCommand('supervisorctl status vllm 2>/dev/null || echo "not_managed"');
              if (!supervisorCheck.stdout.includes('not_managed')) {
                supervisorStatus = supervisorCheck.stdout.trim();
                console.log(`[vLLM] [DIAGNOSTIC] Supervisor status: ${supervisorStatus}`);
                
                if (supervisorStatus.includes('FATAL') || supervisorStatus.includes('EXITED')) {
                  results.errors.push(`vLLM process has exited or failed. Status: ${supervisorStatus}`);
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ vLLM process failed: ${supervisorStatus}`);
                  break; // Exit monitoring loop
                }
              }
              
              // Check if port is listening
              const expectedPort = new URL(vllmUrl).port || '8000';
              const portCheck = await ssh.execCommand(`netstat -tlnp 2>/dev/null | grep :${expectedPort} || ss -tlnp 2>/dev/null | grep :${expectedPort} || echo "port_not_found"`);
              const isPortListening = !portCheck.stdout.includes('port_not_found');
              
              if (isPortListening && !portListening) {
                portListening = true;
                console.log(`[vLLM] [DIAGNOSTIC] ✓ Port ${expectedPort} is now listening`);
                results.steps.push(`[DIAGNOSTIC] Port ${expectedPort} is listening`);
              } else if (!isPortListening && elapsed > 60) {
                console.log(`[vLLM] [DIAGNOSTIC] ⚠ Port ${expectedPort} not listening after ${elapsed}s`);
                results.steps.push(`[DIAGNOSTIC] Port not listening yet (${elapsed}s elapsed)`);
                
                // Check what ports ARE listening (might be on wrong port)
                const allPortsCheck = await ssh.execCommand(`netstat -tlnp 2>/dev/null | grep python || ss -tlnp 2>/dev/null | grep python || echo ""`);
                if (allPortsCheck.stdout && allPortsCheck.stdout.trim()) {
                  console.log(`[vLLM] [DIAGNOSTIC] Python processes listening on ports:`);
                  console.log(`[vLLM] [DIAGNOSTIC] ${allPortsCheck.stdout.trim()}`);
                  results.steps.push(`[DIAGNOSTIC] Found Python processes on other ports - vLLM may be on different port`);
                }
              }
              
              // Check what command supervisor is actually running
              if (elapsed > 30 && !portListening) {
                try {
                  const supervisorConfig = await ssh.execCommand('cat /etc/supervisor/conf.d/vllm.conf 2>/dev/null | grep -A 5 "command=" || echo ""');
                  if (supervisorConfig.stdout) {
                    console.log(`[vLLM] [DIAGNOSTIC] Supervisor vllm.conf command:`);
                    console.log(`[vLLM] [DIAGNOSTIC] ${supervisorConfig.stdout.trim()}`);
                    
                    // Check if the port in supervisor config matches
                    const configPortMatch = supervisorConfig.stdout.match(/--port\s+(\d+)/);
                    if (configPortMatch) {
                      const configPort = configPortMatch[1];
                      if (configPort !== expectedPort) {
                        console.error(`[vLLM] [DIAGNOSTIC] ✗ Port mismatch! Supervisor config uses port ${configPort}, but we're checking port ${expectedPort}`);
                        results.errors.push(`Port mismatch: Supervisor is running vLLM on port ${configPort}, but we're checking port ${expectedPort}. Update vLLM URL or supervisor config.`);
                        break;
                      }
                    }
                    
                    // Check what model supervisor is configured to run
                    const configModelMatch = supervisorConfig.stdout.match(/serve\s+([^\s]+)/);
                    if (configModelMatch) {
                      const configModel = configModelMatch[1];
                      if (configModel !== actualBaseModel) {
                        console.warn(`[vLLM] [DIAGNOSTIC] ⚠ Model mismatch! Supervisor config uses '${configModel}', but we're trying to use '${actualBaseModel}'`);
                        results.errors.push(`Model mismatch: Supervisor is configured to run '${configModel}', but we're trying to use '${actualBaseModel}'. Supervisor config may need to be updated.`);
                        break;
                      }
                    }
                  }
                } catch (e) {
                  // Continue
                }
              }
              
              // Check for common issues in recent logs
              // Try multiple methods to get logs since supervisor might not be able to read them
              // First, get the current PID to filter out old process logs
              const currentPidCheck = await ssh.execCommand('pgrep -f "python.*vllm.*serve" | head -1 || echo ""');
              const currentPid = currentPidCheck.stdout.trim();
              
              if (currentPid && storedVLLMPid && currentPid !== storedVLLMPid) {
                pidChangeCount++;
                const timeSinceLastChange = Date.now() - lastPidChangeTime;
                lastPidChangeTime = Date.now();
                
                console.log(`[vLLM] [DIAGNOSTIC] ⚠ PID changed from ${storedVLLMPid} to ${currentPid} - process may have restarted (change #${pidChangeCount})`);
                storedVLLMPid = currentPid; // Update stored PID
                
                // If PID has changed multiple times quickly, this indicates frequent crashes
                if (pidChangeCount >= 3) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ Process has restarted ${pidChangeCount} times - likely crashing repeatedly`);
                  results.errors.push(`vLLM process has restarted ${pidChangeCount} times, indicating it's crashing repeatedly.`);
                  results.errors.push('Common causes:');
                  results.errors.push('  - GPU out of memory');
                  results.errors.push('  - Invalid model or adapter configuration');
                  results.errors.push('  - Ray/Gloo initialization failures');
                  results.errors.push('Check supervisor logs: supervisorctl tail -200 vllm');
                  break; // Exit monitoring loop
                }
                
                // If PID changed twice within 30 seconds, that's also a problem
                if (pidChangeCount >= 2 && timeSinceLastChange < 30000) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ Process restarted twice within ${Math.floor(timeSinceLastChange/1000)}s - likely crashing`);
                  results.errors.push(`vLLM process restarted twice within ${Math.floor(timeSinceLastChange/1000)} seconds. Process is likely crashing.`);
                  results.errors.push('Check supervisor logs: supervisorctl tail -200 vllm');
                  break; // Exit monitoring loop
                }
              } else if (currentPid && !storedVLLMPid) {
                // First time we see a PID, store it
                storedVLLMPid = currentPid;
                lastPidChangeTime = Date.now();
              }
              
              let recentLogs = '';
              const logCommands = [
                // Try supervisor log files first (most reliable, shows only new process)
                'tail -100 /var/log/supervisor/vllm-stdout.log 2>/dev/null || tail -100 /var/log/supervisor/vllm-stderr.log 2>/dev/null || echo ""',
                // Try supervisor tail (may show old logs)
                'supervisorctl tail -200 vllm 2>&1 | tail -100',
                // Try direct log file
                'tail -100 /tmp/vllm.log 2>&1',
                // Try finding log file
                'find /tmp /var/log /workspace -name "*vllm*.log" -type f 2>/dev/null | head -1 | xargs tail -100 2>&1',
                // Try journalctl if systemd is available
                'journalctl -u vllm -n 50 --no-pager 2>&1 || journalctl --user-unit vllm -n 50 --no-pager 2>&1 || echo ""'
              ];
              
              for (const cmd of logCommands) {
                try {
                  const logCheck = await ssh.execCommand(cmd);
                  const logs = logCheck.stdout || '';
                  
                  // If we got actual log content (not just errors), use it
                  if (logs && 
                      !logs.includes('ERROR (unknown error reading log)') &&
                      !logs.includes('No such file') &&
                      !logs.includes('cannot access') &&
                      logs.length > 50) {
                    // If we have a current PID, try to filter logs to only show entries from that PID
                    if (currentPid) {
                      // Filter logs to only show lines from the current PID
                      // vLLM logs often include PID in format like "(APIServer pid=12345)"
                      const pidFiltered = logs.split('\n').filter(line => {
                        // Include lines that mention the current PID, or lines that don't mention any PID (general logs)
                        return !line.match(/pid=\d+/) || line.includes(`pid=${currentPid}`);
                      }).join('\n');
                      
                      if (pidFiltered.length > 50) {
                        recentLogs = pidFiltered;
                        console.log(`[vLLM] [DIAGNOSTIC] Successfully retrieved logs using: ${cmd.split(' ')[0]} (filtered to PID ${currentPid})`);
      } else {
                        recentLogs = logs; // Use unfiltered if filtered is too short
                        console.log(`[vLLM] [DIAGNOSTIC] Successfully retrieved logs using: ${cmd.split(' ')[0]} (PID filter too aggressive, using all logs)`);
                      }
                    } else {
                      recentLogs = logs;
                      console.log(`[vLLM] [DIAGNOSTIC] Successfully retrieved logs using: ${cmd.split(' ')[0]}`);
                    }
                    break;
                  }
                } catch (e) {
                  // Try next method
                  continue;
                }
              }
              
              // If still no logs, try to get output from the running process directly
              if (!recentLogs || recentLogs.length < 50) {
                try {
                  // Get process info to see what's running
                  const processInfo = await ssh.execCommand('ps aux | grep "[p]ython.*vllm" | head -1');
                  if (processInfo.stdout) {
                    console.log(`[vLLM] [DIAGNOSTIC] vLLM process: ${processInfo.stdout.trim()}`);
                    results.steps.push(`[DIAGNOSTIC] Process running: ${processInfo.stdout.trim().substring(0, 120)}...`);
                  }
                  
                  // Try to find where logs might actually be
                  const logFind = await ssh.execCommand('find /tmp /var/log /workspace /root -name "*vllm*" -type f 2>/dev/null | head -10');
                  if (logFind.stdout) {
                    const logFiles = logFind.stdout.trim().split('\n').filter(f => f);
                    console.log(`[vLLM] [DIAGNOSTIC] Found ${logFiles.length} potential log file(s):`);
                    logFiles.forEach(file => console.log(`[vLLM] [DIAGNOSTIC]   ${file}`));
                    
                    // Try to read from the first found log file
                    if (logFiles.length > 0) {
                      const firstLog = await ssh.execCommand(`tail -50 "${logFiles[0]}" 2>&1`);
                      if (firstLog.stdout && firstLog.stdout.length > 50) {
                        recentLogs = firstLog.stdout;
                        console.log(`[vLLM] [DIAGNOSTIC] Retrieved logs from: ${logFiles[0]}`);
                      }
                    }
                  }
                  
                  // Try supervisor config to see where it's logging
                  const supervisorConfigCheck = await ssh.execCommand('cat /etc/supervisor/conf.d/vllm.conf 2>/dev/null | grep -E "(stdout|stderr|logfile)" || echo ""');
                  if (supervisorConfigCheck.stdout) {
                    console.log(`[vLLM] [DIAGNOSTIC] Supervisor log config: ${supervisorConfigCheck.stdout.trim()}`);
                  }
                  
                  // Try to read from supervisor's stdout/stderr files (common locations)
                  const supervisorLogPaths = [
                    '/var/log/supervisor/vllm-stdout.log',
                    '/var/log/supervisor/vllm-stderr.log',
                    '/var/log/supervisor/supervisord.log',
                    '/tmp/vllm.log',
                    '/workspace/vllm.log'
                  ];
                  
                  for (const logPath of supervisorLogPaths) {
                    const supervisorLogCheck = await ssh.execCommand(`tail -100 "${logPath}" 2>&1`);
                    if (supervisorLogCheck.stdout && 
                        supervisorLogCheck.stdout.length > 50 &&
                        !supervisorLogCheck.stdout.includes('No such file')) {
                      recentLogs = supervisorLogCheck.stdout;
                      console.log(`[vLLM] [DIAGNOSTIC] Retrieved logs from: ${logPath}`);
                      break;
                    }
                  }
                } catch (e) {
                  console.warn(`[vLLM] [DIAGNOSTIC] Error trying to find logs: ${e.message}`);
                }
              }
              
              // If we still have no logs, at least show that we tried
              if (!recentLogs || recentLogs.length < 50) {
                console.warn(`[vLLM] [DIAGNOSTIC] ⚠ Could not retrieve logs from any source. vLLM may be logging elsewhere or logs may not be accessible.`);
                results.steps.push(`[DIAGNOSTIC] ⚠ Logs not accessible - vLLM may still be starting`);
              }
              
              // Check if we're seeing logs from the current process or old process
              if (recentLogs && currentPid) {
                const pidMatches = recentLogs.match(/pid=(\d+)/g);
                if (pidMatches) {
                  const uniquePids = [...new Set(pidMatches.map(m => m.match(/\d+/)[0]))];
                  const currentPidInLogs = uniquePids.includes(currentPid);
                  
                  if (!currentPidInLogs && uniquePids.length > 0) {
                    console.warn(`[vLLM] [DIAGNOSTIC] ⚠ Logs show PIDs [${uniquePids.join(', ')}] but current process is PID ${currentPid}`);
                    console.warn(`[vLLM] [DIAGNOSTIC] ⚠ New process (PID ${currentPid}) may still be starting - waiting for new log entries...`);
                    results.steps.push(`[DIAGNOSTIC] Seeing logs from old process(es) [${uniquePids.join(', ')}], new process (PID ${currentPid}) may still be starting`);
                    
                    // Check if the new process is actually running
                    const newProcessCheck = await ssh.execCommand(`ps -p ${currentPid} -o pid,state,etime,cmd --no-headers 2>/dev/null || echo "not_running"`);
                    if (newProcessCheck.stdout && !newProcessCheck.stdout.includes('not_running')) {
                      console.log(`[vLLM] [DIAGNOSTIC] New process status: ${newProcessCheck.stdout.trim()}`);
                      results.steps.push(`[DIAGNOSTIC] New process (PID ${currentPid}) is running but hasn't produced logs yet`);
                    } else {
                      console.error(`[vLLM] [DIAGNOSTIC] ✗ New process (PID ${currentPid}) not found - may have crashed`);
                      results.errors.push(`New vLLM process (PID ${currentPid}) not found - may have crashed during startup`);
                    }
                  } else if (currentPidInLogs) {
                    console.log(`[vLLM] [DIAGNOSTIC] ✓ Logs show current process PID ${currentPid}`);
                  }
                }
              }
              
              if (recentLogs) {
                // Show recent log activity (last 5-10 lines) for visibility
                const logLines = recentLogs.split('\n').filter(l => l.trim()).slice(-10);
                if (logLines.length > 0) {
                  console.log(`[vLLM] [DIAGNOSTIC] Recent log activity (last ${logLines.length} lines):`);
                  logLines.forEach((line, idx) => {
                    if (line.length > 150) line = line.substring(0, 150) + '...';
                    console.log(`[vLLM] [DIAGNOSTIC]   ${line}`);
                  });
                }
                
                // Check for API readiness in logs (this is the most reliable indicator)
                if (recentLogs.includes('Application startup complete') || 
                    recentLogs.includes('Uvicorn running') || 
                    recentLogs.includes('Serving on')) {
                  console.log(`[vLLM] [DIAGNOSTIC] ✓ API server startup detected in logs!`);
                  results.steps.push(`[DIAGNOSTIC] ✓ API server startup complete (detected in logs)`);
                  
                  // Immediately try to verify via API
                  try {
                    const verifyUrl = `http://localhost:${new URL(vllmUrl).port || '8000'}`;
                    const curlAuth = openButtonToken 
                      ? `-u vastai:${openButtonToken}`
                      : '';
                    const verifyCheck = await ssh.execCommand(`curl -s ${curlAuth} --max-time 5 ${verifyUrl}/v1/models 2>&1 | head -100 || echo "curl_failed"`);
                    
                    if (!verifyCheck.stdout.includes('curl_failed') && verifyCheck.stdout.trim()) {
                      try {
                        const modelsData = JSON.parse(verifyCheck.stdout);
                        const availableModels = modelsData.data || [];
                        const modelNames = availableModels.map(m => m.id || m.name || m);
                        if (modelNames.length > 0) {
                          modelReady = true;
                          results.steps.push(`✓ Model is ready! Available models: ${modelNames.join(', ')}`);
                          console.log(`[vLLM] ✓ Model is ready! Available: ${modelNames.join(', ')}`);
                          break; // Exit monitoring loop
      } else {
                          console.log(`[vLLM] [DIAGNOSTIC] API responding but no models listed yet`);
                        }
                      } catch (e) {
                        console.log(`[vLLM] [DIAGNOSTIC] API responding but response not JSON: ${verifyCheck.stdout.substring(0, 100)}`);
                      }
                    } else {
                      console.log(`[vLLM] [DIAGNOSTIC] API startup detected but curl failed: ${verifyCheck.stdout.substring(0, 100)}`);
                    }
                  } catch (e) {
                    console.warn(`[vLLM] [DIAGNOSTIC] Error verifying API: ${e.message}`);
                  }
                }
                
                // Check for API readiness in logs (this is the most reliable indicator)
                if (recentLogs.includes('Application startup complete') || 
                    recentLogs.includes('Uvicorn running') || 
                    recentLogs.includes('Serving on')) {
                  console.log(`[vLLM] [DIAGNOSTIC] ✓ API server startup detected in logs!`);
                  results.steps.push(`[DIAGNOSTIC] ✓ API server startup complete (detected in logs)`);
                  
                  // Immediately try to verify via API
                  try {
                    const verifyUrl = `http://localhost:${new URL(vllmUrl).port || '8000'}`;
                    console.log(`[vLLM] [DIAGNOSTIC] Verifying API at ${verifyUrl}/v1/models...`);
                    const curlAuth = openButtonToken 
                      ? `-u vastai:${openButtonToken}`
                      : '';
                    const verifyCheck = await ssh.execCommand(`timeout 5 curl -s ${curlAuth} ${verifyUrl}/v1/models 2>&1 | head -200 || echo "curl_failed"`);
                    
                    console.log(`[vLLM] [DIAGNOSTIC] API verification response length: ${verifyCheck.stdout.length}`);
                    console.log(`[vLLM] [DIAGNOSTIC] API verification response (first 300 chars): ${verifyCheck.stdout.substring(0, 300)}`);
                    
                    if (!verifyCheck.stdout.includes('curl_failed') && verifyCheck.stdout.trim()) {
                      try {
                        const modelsData = JSON.parse(verifyCheck.stdout);
                        const availableModels = modelsData.data || [];
                        const modelNames = availableModels.map(m => m.id || m.name || m);
                        if (modelNames.length > 0) {
                          modelReady = true;
                          results.steps.push(`✓ Model is ready! Available models: ${modelNames.join(', ')}`);
                          console.log(`[vLLM] ✓ Model is ready! Available: ${modelNames.join(', ')}`);
                          break; // Exit monitoring loop
                        } else {
                          console.log(`[vLLM] [DIAGNOSTIC] API responding but no models listed yet`);
                          results.steps.push(`[DIAGNOSTIC] API ready but models not yet available`);
                        }
                      } catch (e) {
                        console.log(`[vLLM] [DIAGNOSTIC] API responding but response not JSON: ${e.message}`);
                        console.log(`[vLLM] [DIAGNOSTIC] Raw response: ${verifyCheck.stdout.substring(0, 500)}`);
                      }
                    } else {
                      console.log(`[vLLM] [DIAGNOSTIC] API startup detected but curl failed`);
                      console.log(`[vLLM] [DIAGNOSTIC] Curl stdout: ${verifyCheck.stdout || '(empty)'}`);
                      console.log(`[vLLM] [DIAGNOSTIC] Curl stderr: ${verifyCheck.stderr || '(empty)'}`);
                      console.log(`[vLLM] [DIAGNOSTIC] Curl exit code: ${verifyCheck.code}`);
                      
                      // Try verbose curl to see what's happening
                      const curlAuth = openButtonToken 
                        ? `-u vastai:${openButtonToken}`
                        : '';
                      const verboseCurl = await ssh.execCommand(`curl -v ${curlAuth} --max-time 5 ${verifyUrl}/v1/models 2>&1 | head -40 || echo "curl_failed"`);
                      console.log(`[vLLM] [DIAGNOSTIC] Verbose curl output:`);
                      console.log(`[vLLM] [DIAGNOSTIC] ${verboseCurl.stdout.substring(0, 800)}`);
                      
                      // Check if port is actually listening
                      const portListenCheck = await ssh.execCommand(`netstat -tlnp 2>/dev/null | grep :${new URL(vllmUrl).port || '8000'} || ss -tlnp 2>/dev/null | grep :${new URL(vllmUrl).port || '8000'} || echo "not_listening"`);
                      if (portListenCheck.stdout.includes('not_listening')) {
                        console.error(`[vLLM] [DIAGNOSTIC] ✗ Port ${new URL(vllmUrl).port || '8000'} is NOT listening`);
                        console.error(`[vLLM] [DIAGNOSTIC] ✗ API says it started but port is not bound`);
                        results.errors.push(`Port ${new URL(vllmUrl).port || '8000'} is not listening. API may have started but failed to bind to port, or supervisor is running a different configuration.`);
                      } else {
                        console.log(`[vLLM] [DIAGNOSTIC] Port IS listening: ${portListenCheck.stdout.trim()}`);
                        // Port is listening but curl fails - might be a connection issue
                        results.errors.push(`Port ${new URL(vllmUrl).port || '8000'} is listening but API not responding. Check if vLLM is actually running the correct model.`);
                      }
                    }
                  } catch (e) {
                    console.warn(`[vLLM] [DIAGNOSTIC] Error verifying API: ${e.message}`);
                  }
                }
                
                // Check for model download/loading issues
                if (recentLogs.includes('404') || recentLogs.includes('Not Found') || recentLogs.includes('model not found')) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ Model not found error detected`);
                  results.errors.push(`Model '${actualBaseModel}' not found. Check if the model name is correct or if authentication is needed.`);
                  break; // Exit monitoring loop
                }
                
                // Check for authentication issues
                if (recentLogs.includes('401') || recentLogs.includes('Unauthorized') || recentLogs.includes('authentication')) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ Authentication error detected`);
                  results.errors.push('Authentication failed. Check if HF_TOKEN is valid and has access to the model.');
                  break; // Exit monitoring loop
                }
                
                // Check for Ray waiting issues (including Gloo initialization problems)
                const rayStuckPatterns = [
                  'waiting for ray',
                  'Waiting for Ray',
                  '[Gloo] Rank 0 is connected to 0 peer ranks',
                  'Gloo.*Rank.*connected to 0 peer',
                  'Ray.*not.*starting',
                  'Failed to start Ray'
                ];
                
                const isRayStuck = rayStuckPatterns.some(pattern => {
                  if (pattern.includes('[') || pattern.includes('.*')) {
                    // Use regex for complex patterns
                    const regex = new RegExp(pattern, 'i');
                    return regex.test(recentLogs);
                  }
                  return recentLogs.toLowerCase().includes(pattern.toLowerCase());
                });
                
                if (isRayStuck) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ vLLM appears stuck waiting for Ray/Gloo (common GPU/memory issue)`);
                  results.errors.push('vLLM is stuck waiting for Ray/Gloo initialization. This usually indicates:');
                  results.errors.push('  - GPU not available or not properly configured');
                  results.errors.push('  - Insufficient GPU memory');
                  results.errors.push('  - Ray/Gloo initialization failure');
                  results.errors.push('  - Process restarting too quickly (check supervisor logs)');
                  results.errors.push('Troubleshooting:');
                  results.errors.push('  1. Check GPU: ssh to server and run: nvidia-smi');
                  results.errors.push('  2. Check memory: free -h');
                  results.errors.push('  3. Check supervisor logs: supervisorctl tail -100 vllm');
                  results.errors.push('  4. Try restarting: supervisorctl restart vllm');
                  results.errors.push('  5. Check if Ray ports are in use: netstat -tlnp | grep -E "8265|10001"');
                  break; // Exit monitoring loop
                }
                
                // Check for out of memory errors
                if (recentLogs.includes('out of memory') || recentLogs.includes('OOM') || recentLogs.includes('CUDA out of memory')) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ Out of memory error detected`);
                  results.errors.push('Out of memory error detected. Model may be too large for available GPU memory.');
                  break; // Exit monitoring loop
                }
                
                // Check for LoRA adapter errors
                if (recentLogs.includes('lora') && (recentLogs.includes('error') || recentLogs.includes('Error') || recentLogs.includes('failed'))) {
                  console.error(`[vLLM] [DIAGNOSTIC] ✗ LoRA adapter error detected`);
                  results.errors.push('Error loading LoRA adapters. Check if adapter paths are correct and adapters are compatible with the base model.');
                  break; // Exit monitoring loop
                }
                
                // Check for other errors
                const errorLines = recentLogs.split('\n').filter(l => 
                  (l.includes('Error') || l.includes('Exception') || l.includes('Traceback') || l.includes('Failed')) &&
                  !l.includes('Downloading') && // Exclude download-related messages
                  !l.includes('INFO') && // Exclude INFO level messages that might contain "error" in text
                  !l.toLowerCase().includes('warning') // Exclude warnings
                ).slice(-3);
                
                if (errorLines.length > 0 && elapsed > 60) {
                  console.warn(`[vLLM] [DIAGNOSTIC] ⚠ Errors detected in logs:`);
                  errorLines.forEach(line => {
                    const shortLine = line.length > 200 ? line.substring(0, 200) + '...' : line;
                    console.warn(`[vLLM] [DIAGNOSTIC]   ${shortLine}`);
                  });
                  results.steps.push(`[DIAGNOSTIC] Errors in logs: ${errorLines[errorLines.length - 1].substring(0, 150)}...`);
                }
              } else {
                console.warn(`[vLLM] [DIAGNOSTIC] ⚠ No logs found - vLLM may not be writing to expected log location`);
              }
            } catch (diagError) {
              console.warn(`[vLLM] [DIAGNOSTIC] Error running diagnostics: ${diagError.message}`);
            }
          }
          
          // Check logs for download/loading progress (every 5 seconds)
          if (Date.now() - lastLogCheck >= logCheckInterval) {
            lastLogCheck = Date.now();
            try {
              const logCheck = await ssh.execCommand('supervisorctl tail -50 vllm 2>/dev/null | tail -30 || tail -30 /tmp/vllm.log 2>/dev/null || echo ""');
              const logs = logCheck.stdout || '';
              
              if (logs.trim()) {
                // Look for download progress
                const downloadPatterns = [
                  /Downloading.*?(\d+(?:\.\d+)?%)/gi,
                  /downloading.*?(\d+(?:\.\d+)?%)/gi,
                  /Download.*?(\d+(?:\.\d+)?%)/gi,
                  /(\d+(?:\.\d+)?%).*?downloaded/gi,
                  /Fetching.*from.*huggingface/gi
                ];
                
                for (const pattern of downloadPatterns) {
                  const matches = logs.match(pattern);
                  if (matches) {
                    downloadDetected = true;
                    // Try to extract percentage
                    const percentMatch = logs.match(/(\d+(?:\.\d+)?)%/);
                    const progress = percentMatch ? percentMatch[1] : null;
                    const progressMsg = progress ? `Downloading model: ${progress}%` : 'Downloading model from HuggingFace...';
                    
                    if (progressMsg !== lastDownloadProgress) {
                      lastDownloadProgress = progressMsg;
                      console.log(`[vLLM] ${progressMsg}`);
                      // Update or add progress step
                      const progressStep = `[DOWNLOAD] ${progressMsg}`;
                      const lastStep = results.steps[results.steps.length - 1];
                      if (lastStep && lastStep.includes('[DOWNLOAD]')) {
                        results.steps[results.steps.length - 1] = progressStep;
                      } else {
                        results.steps.push(progressStep);
                      }
                    }
                    break;
                  }
                }
                
                // Look for loading progress
                const loadingPatterns = [
                  /Loading.*model/gi,
                  /Loading.*weights/gi,
                  /Initializing.*model/gi,
                  /Loading.*into.*memory/gi
                ];
                
                for (const pattern of loadingPatterns) {
                  if (pattern.test(logs)) {
                    loadingDetected = true;
                    const loadingMsg = 'Loading model into GPU memory...';
                    const lastStep = results.steps[results.steps.length - 1];
                    if (!lastStep || !lastStep.includes('[LOADING]')) {
                      console.log(`[vLLM] ${loadingMsg}`);
                      results.steps.push(`[LOADING] ${loadingMsg}`);
                    }
                    break;
                  }
                }
                
                // Check for readiness in logs
                if (logs.includes('Uvicorn running') || logs.includes('Application startup complete') || logs.includes('Serving on')) {
                  modelReady = true;
                  results.steps.push('✓ Model ready (detected in logs)');
                  console.log('[vLLM] ✓ Model ready (detected in logs)');
                  break;
                }
              }
            } catch (logError) {
              console.warn(`[vLLM] Error checking logs: ${logError.message}`);
            }
          }
          
              // Poll API endpoint for readiness (every 2 seconds)
          if (!modelReady) {
            try {
              // Use a longer timeout and better error handling
              // Use Basic Auth if we have the token
              const curlAuth = openButtonToken 
                ? `-u vastai:${openButtonToken}`
                : '';
              const pollCheck = await ssh.execCommand(`timeout 5 curl -s ${curlAuth} ${localUrl}/v1/models 2>&1 | head -200 || echo "curl_failed"`);
              
              // Check if curl actually succeeded (not just if output exists)
              const curlOutput = pollCheck.stdout || '';
              const isCurlFailed = curlOutput.includes('curl_failed') || 
                                   curlOutput.includes('Connection refused') ||
                                   curlOutput.includes('timeout') ||
                                   curlOutput.trim().length === 0;
              
              if (!isCurlFailed && curlOutput.trim()) {
                apiResponding = true;
                try {
                  // Try to parse as JSON
                  const modelsData = JSON.parse(curlOutput);
                  const availableModels = modelsData.data || [];
                  const modelNames = availableModels.map(m => m.id || m.name || m);
                  
                  if (modelNames.length > 0) {
                    modelReady = true;
                    results.steps.push(`✓ Model is ready! Available models: ${modelNames.join(', ')}`);
                    console.log(`[vLLM] ✓ Model is ready! Available: ${modelNames.join(', ')}`);
                    break;
                  } else {
                    // API responding but no models yet - show status every 5 seconds
                    if (Date.now() - lastStatusUpdate >= statusUpdateInterval) {
                      lastStatusUpdate = Date.now();
                      const statusMsg = `API responding, waiting for models... (${elapsed}s)`;
                      if (statusMsg !== lastStatus) {
                        lastStatus = statusMsg;
                        console.log(`[vLLM] ${statusMsg}`);
                        console.log(`[vLLM] API response: ${curlOutput.substring(0, 200)}`);
                        // Update step
                        const lastStep = results.steps[results.steps.length - 1];
                        if (lastStep && lastStep.includes('API responding')) {
                          results.steps[results.steps.length - 1] = statusMsg;
                        } else {
                          results.steps.push(statusMsg);
                        }
                      }
                    }
                  }
                } catch (e) {
                  // API responding but not JSON yet - vLLM still starting
                  if (Date.now() - lastStatusUpdate >= statusUpdateInterval) {
                    lastStatusUpdate = Date.now();
                    const statusMsg = `API responding but response not JSON (${elapsed}s)`;
                    if (statusMsg !== lastStatus) {
                      lastStatus = statusMsg;
                      console.log(`[vLLM] ${statusMsg}`);
                      console.log(`[vLLM] Raw response: ${curlOutput.substring(0, 300)}`);
                    }
                  }
                }
              } else {
                apiResponding = false;
                // API not responding yet - show status every 5 seconds
                if (Date.now() - lastStatusUpdate >= statusUpdateInterval) {
                  lastStatusUpdate = Date.now();
                  const statusMsg = `Waiting for API to respond... (${elapsed}s)`;
                  if (statusMsg !== lastStatus) {
                    lastStatus = statusMsg;
                    // Only log if we haven't seen "Application startup complete" in logs
                    if (!recentLogs || !recentLogs.includes('Application startup complete')) {
                      console.log(`[vLLM] ${statusMsg}`);
                    } else {
                      console.log(`[vLLM] ${statusMsg} (but logs show API started - may be port/connection issue)`);
                    }
                    // Update step
                    const lastStep = results.steps[results.steps.length - 1];
                    if (lastStep && (lastStep.includes('Waiting for API') || lastStep.includes('API responding'))) {
                      results.steps[results.steps.length - 1] = statusMsg;
                    } else {
                      results.steps.push(statusMsg);
                    }
                  }
                }
              }
            } catch (pollError) {
              apiResponding = false;
              // Show error status occasionally
              if (Date.now() - lastStatusUpdate >= statusUpdateInterval * 2) {
                lastStatusUpdate = Date.now();
                console.warn(`[vLLM] Error polling API: ${pollError.message}`);
              }
            }
          }
          
          // Wait before next poll
          await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
        
        if (!modelReady) {
          const elapsed = Math.floor((Date.now() - startPollTime) / 1000);
          if (downloadDetected) {
            results.steps.push(`⚠ Model still downloading/loading after ${elapsed}s`);
            console.log(`[vLLM] ⚠ Model still downloading/loading after ${elapsed}s`);
          } else {
            results.steps.push(`⚠ Model not ready after ${elapsed}s. It may still be loading.`);
            console.log(`[vLLM] ⚠ Model not ready after ${elapsed}s`);
          }
        }
      }
      
      // Final comprehensive status check via SSH
      results.steps.push('Performing final vLLM status check...');
      console.log('[vLLM] Performing final vLLM status check via SSH...');
      
      try {
        // Check if vLLM process is still running
        const processCheck = await ssh.execCommand('pgrep -f vllm || echo "not_running"');
        const isRunning = processCheck.stdout.trim() !== 'not_running';
        
        if (isRunning) {
          results.steps.push('✓ vLLM process is running');
          console.log('[vLLM] ✓ vLLM process is running');
          
          // Check supervisor status for more details
          const supervisorStatus = await ssh.execCommand('supervisorctl status vllm 2>/dev/null || echo "not_managed"');
          if (!supervisorStatus.stdout.includes('not_managed')) {
            console.log(`[vLLM] Supervisor status: ${supervisorStatus.stdout.trim()}`);
            if (supervisorStatus.stdout.includes('FATAL') || supervisorStatus.stdout.includes('EXITED')) {
              results.errors.push('vLLM process exited or failed. Check logs: supervisorctl tail vllm');
              console.error('[vLLM] ✗ vLLM process has exited or failed');
            }
          }
          
          // Check recent logs for errors or issues
          console.log('[vLLM] Checking recent logs for errors or issues...');
          const recentLogs = await ssh.execCommand('supervisorctl tail -100 vllm 2>/dev/null | tail -50 || tail -50 /tmp/vllm.log 2>/dev/null || echo ""');
          const logs = recentLogs.stdout || '';
          
          if (logs) {
            // Check for common errors
            if (logs.includes('Error') || logs.includes('Exception') || logs.includes('Traceback')) {
              const errorLines = logs.split('\n').filter(l => l.includes('Error') || l.includes('Exception')).slice(-3);
              if (errorLines.length > 0) {
                results.errors.push(`vLLM errors detected in logs: ${errorLines.join('; ')}`);
                console.error(`[vLLM] ✗ Errors in logs: ${errorLines.join('; ')}`);
              }
            }
            
            // Check if it's stuck waiting for Ray (including Gloo initialization problems)
            const rayStuckPatterns = [
              'waiting for ray',
              'Waiting for Ray',
              '[Gloo] Rank 0 is connected to 0 peer ranks',
              'Gloo.*Rank.*connected to 0 peer',
              'Ray.*not.*starting'
            ];
            
            const isRayStuck = rayStuckPatterns.some(pattern => {
              if (pattern.includes('[') || pattern.includes('.*')) {
                const regex = new RegExp(pattern, 'i');
                return regex.test(logs);
              }
              return logs.toLowerCase().includes(pattern.toLowerCase());
            });
            
            if (isRayStuck) {
              results.errors.push('vLLM appears stuck waiting for Ray/Gloo initialization. This usually indicates GPU issues, insufficient memory, or Ray not starting properly.');
              console.error('[vLLM] ✗ vLLM stuck waiting for Ray/Gloo');
              results.errors.push('Troubleshooting steps:');
              results.errors.push('  1. Check GPU: nvidia-smi');
              results.errors.push('  2. Check supervisor logs: supervisorctl tail -100 vllm');
              results.errors.push('  3. Try restarting: supervisorctl restart vllm');
            }
            
            // Check for download/loading activity
            if (logs.includes('Downloading') || logs.includes('downloading')) {
              results.steps.push('ℹ Download activity detected in logs (model may still be downloading)');
              console.log('[vLLM] ℹ Download activity detected in logs');
            }
            if (logs.includes('Loading') || logs.includes('Initializing')) {
              results.steps.push('ℹ Loading activity detected in logs (model may still be loading into memory)');
              console.log('[vLLM] ℹ Loading activity detected in logs');
            }
          }
          
          // Try to check if vLLM is listening on the expected port
          const portCheck = await ssh.execCommand(`netstat -tlnp 2>/dev/null | grep :${new URL(vllmUrl).port || '8000'} || ss -tlnp 2>/dev/null | grep :${new URL(vllmUrl).port || '8000'} || echo "port_not_found"`);
          if (!portCheck.stdout.includes('port_not_found')) {
            results.steps.push(`✓ vLLM is listening on port ${new URL(vllmUrl).port || '8000'}`);
            console.log(`[vLLM] ✓ vLLM is listening on port ${new URL(vllmUrl).port || '8000'}`);
          } else {
            results.errors.push(`vLLM is not listening on port ${new URL(vllmUrl).port || '8000'}. It may still be starting or may have failed.`);
            console.error(`[vLLM] ✗ vLLM is not listening on port ${new URL(vllmUrl).port || '8000'}`);
          }
          
          // Try to check vLLM status via curl from the server itself
          const localUrl = `http://localhost:${new URL(vllmUrl).port || '8000'}`;
          const curlAuth = openButtonToken 
            ? `-u vastai:${openButtonToken}`
            : '';
          const curlCheck = await ssh.execCommand(`curl -s ${curlAuth} --max-time 5 ${localUrl}/v1/models 2>&1 | head -50 || echo "curl_failed"`);
          
          if (!curlCheck.stdout.includes('curl_failed') && curlCheck.stdout.trim()) {
            try {
              const modelsData = JSON.parse(curlCheck.stdout);
              const availableModels = modelsData.data || [];
              const modelNames = availableModels.map(m => m.id || m.name || m);
              if (modelNames.length > 0) {
                results.steps.push(`✓ Model is ready! Available models: ${modelNames.join(', ')}`);
                console.log(`[vLLM] ✓ Model ready! Available: ${modelNames.join(', ')}`);
                modelReady = true; // Mark as ready for final success check
              } else {
                results.errors.push('vLLM API is responding but no models are listed. Model may still be loading or there may be a configuration issue.');
                console.error('[vLLM] ✗ vLLM API responding but no models listed');
              }
            } catch (e) {
              results.errors.push(`vLLM API is responding but response is invalid: ${e.message}. Check supervisor logs.`);
              console.error(`[vLLM] ✗ vLLM API response invalid: ${e.message}`);
            }
          } else {
            results.errors.push(`vLLM API is not responding after 5 minutes. Check supervisor logs: supervisorctl tail vllm`);
            console.error('[vLLM] ✗ vLLM API not responding');
          }
        } else {
          results.errors.push('vLLM process is not running. Check supervisor logs: supervisorctl tail vllm');
          console.error('[vLLM] ✗ vLLM process is not running');
        }
      } catch (sshError) {
        console.warn(`[vLLM] Error checking vLLM status via SSH: ${sshError.message}`);
        results.steps.push(`⚠ Could not check vLLM status via SSH: ${sshError.message}`);
      }
      
      // Also try HTTP polling from local machine (may fail if URL not accessible)
      results.steps.push('Checking if model is accessible from local machine...');
      console.log('[vLLM] Polling HTTP endpoint for model readiness...');
      
      const httpPollResult = await pollModelReadiness(
        vllmUrl, 
        baseModel, 
        30000, // Reduced to 30 seconds since we already checked via SSH
        5000   // Check every 5 seconds
      );
      
      if (httpPollResult.ready) {
        results.steps.push(`✓ Model is accessible from local machine! Available models: ${httpPollResult.models.join(', ')}`);
        console.log(`[vLLM] ✓ Model accessible from local machine! Available: ${httpPollResult.models.join(', ')}`);
      } else {
        results.steps.push('ℹ Model may not be accessible from local machine (this is OK if using SSH tunnel or Vast.ai instance)');
        console.log('[vLLM] ℹ Model may not be accessible from local machine - check if you need SSH tunnel');
      }
    }
    
    // Only mark as successful if model is actually ready
    // Check if we detected model readiness anywhere
    const modelIsReady = modelReady || results.steps.some(step => 
      step.includes('Model is ready') || 
      step.includes('Model ready') ||
      step.includes('Available models:')
    );
    
    // If we have errors, definitely not successful
    // If no errors but model isn't ready, also not successful (but may still be loading)
    results.success = results.errors.length === 0 && modelIsReady;
    
    if (!results.success && results.errors.length === 0 && !modelIsReady) {
      // Add a warning if no errors but model isn't ready
      results.errors.push('Model preparation completed but model is not yet ready. It may still be loading. Check supervisor logs: supervisorctl tail vllm');
    }

    // Store SSH config, model info, and vLLM config for chat interface
    if (results.success) {
      storedSSHConfig = {
        host: host,
        port: port || 22,
        username: username
      };
      // Use actualBaseModel (from adapter config) instead of baseModel parameter
      // This ensures we use the correct model name that vLLM was started with
      storedModelInfo = {
        profileName,
        baseModel: actualBaseModel, // Use the actual base model from adapter config
        modelDir: remoteModelDir,
        versions: versions.map(v => ({
          version: v.version,
          adapterPath: `${remoteModelDir}/V${v.version}/adapter`
        }))
      };
      // Establish SSH tunnel for inference requests (if not already established)
      try {
        const sshKeyPath = findSSHKey();
        const remoteInferencePort = 8888; // FastAPI inference server port
        
        // Only establish tunnel if we don't already have one for this connection
        if (!sshTunnelProcess || sshTunnelConfig?.host !== host || sshTunnelConfig?.port !== (port || 22)) {
          console.log(`[vLLM] [TUNNEL] Setting up SSH tunnel to remote port ${remoteInferencePort}`);
          const tunnelLocalPort = await establishSSHTunnel(
            host,
            port || 22,
            username || 'root',
            sshKeyPath,
            remoteInferencePort
          );
          
          const tunneledUrl = `http://localhost:${tunnelLocalPort}`;
          storedVLLMUrl = tunneledUrl;
          console.log(`[vLLM] ✓ SSH tunnel established: ${tunneledUrl}`);
        } else {
          // Reuse existing tunnel
          storedVLLMUrl = `http://localhost:${sshTunnelLocalPort}`;
          console.log(`[vLLM] Reusing existing SSH tunnel: ${storedVLLMUrl}`);
        }
      } catch (tunnelError) {
        console.warn(`[vLLM] ⚠ Failed to establish SSH tunnel: ${tunnelError.message}`);
        console.warn(`[vLLM] Falling back to external URL`);
      storedVLLMUrl = vllmUrl || `http://${host}:8000`;
      }
      
      console.log(`[vLLM] Stored model info with baseModel: ${actualBaseModel}`);
      console.log(`[vLLM] Inference URL: ${storedVLLMUrl}`);
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
  
  // Token will be retrieved from server if SSH credentials are provided
  console.log(`[TEST] Testing vLLM URL: ${vllmUrl}`);
  
  // Retrieve OPEN_BUTTON_TOKEN first if SSH credentials are available
  let openButtonToken = storedOpenButtonToken; // Use stored token if available
  if (sshHost && sshPort && !openButtonToken) {
    console.log(`[TEST] Retrieving OPEN_BUTTON_TOKEN from instance via SSH...`);
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
      
      // Prepare connection options with auto-accept host keys
      const finalConnectOptions = prepareSSHConnectionOptions(connectOptions);
      await ssh.connect(finalConnectOptions);
      console.log('[SSH] ✓ Connection established for retrieving OPEN_BUTTON_TOKEN');
      
      // Try multiple methods to get OPEN_BUTTON_TOKEN
      const tokenMethods = [
        { name: 'environment', cmd: 'echo $OPEN_BUTTON_TOKEN' },
        { name: 'process environment', cmd: 'cat /proc/1/environ 2>/dev/null | tr "\\0" "\\n" | grep OPEN_BUTTON_TOKEN | cut -d= -f2' },
        { name: 'systemd', cmd: 'systemctl show-environment 2>/dev/null | grep OPEN_BUTTON_TOKEN | cut -d= -f2' },
        { name: 'supervisor', cmd: 'supervisorctl -c /etc/supervisor/supervisord.conf environment 2>/dev/null | grep OPEN_BUTTON_TOKEN | cut -d= -f2 || echo ""' },
        { name: 'running processes', cmd: 'ps e -o command 2>/dev/null | grep -o "OPEN_BUTTON_TOKEN=[^ ]*" | head -1 | cut -d= -f2 || echo ""' },
        { name: '/etc/environment', cmd: 'grep OPEN_BUTTON_TOKEN /etc/environment 2>/dev/null | cut -d= -f2 || echo ""' }
      ];
      
      for (const method of tokenMethods) {
        try {
          const tokenResult = await ssh.execCommand(method.cmd);
          const token = tokenResult.stdout.trim();
          if (token && token.length > 0) {
            openButtonToken = token;
            storedOpenButtonToken = token; // Store globally
            console.log(`[TEST] ✓ Found OPEN_BUTTON_TOKEN from ${method.name}`);
            break;
          }
        } catch (e) {
          // Try next method
          continue;
        }
      }
      
      await ssh.dispose();
    } catch (sshError) {
      console.error(`[TEST] Failed to retrieve OPEN_BUTTON_TOKEN via SSH: ${sshError.message}`);
      console.error(`[TEST] SSH connection may have failed. Check if SSH is working: ${sshHost}:${sshPort || 22}`);
    }
  }
  
    // Use minimal headers - browsers send Accept and User-Agent by default
    const headers = {
      'Accept': 'application/json',
    'User-Agent': 'Electron-Inference-Client/1.0'
  };
  
  // If we have a token, use Basic Auth from the start
  if (openButtonToken) {
    const basicAuthValue = Buffer.from(`vastai:${openButtonToken}`).toString('base64');
    headers['Authorization'] = `Basic ${basicAuthValue}`;
    console.log(`[TEST] Using Vast.ai Basic Auth: vastai:${openButtonToken.substring(0, 8)}...`);
  }
  
  // Test FastAPI endpoints: /health first (simpler, faster)
  const healthUrl = vllmUrl.replace(/\/$/, '') + '/health';
  console.log(`[TEST] Testing health endpoint: ${healthUrl}`);
    console.log(`[TEST] Original URL: ${vllmUrl}`);
    console.log(`[TEST] Headers being sent:`, JSON.stringify(headers, null, 2));
  console.log(`[TEST] Attempting connection${openButtonToken ? ' with Basic Auth' : ' without authentication'}...`);
    
  try {
    // First check /health endpoint
    const healthResponse = await makeHttpRequest(healthUrl, {
      method: 'GET',
      headers: headers
    });
    
    console.log(`[TEST] Health check response - Status: ${healthResponse.statusCode}`);
    if (healthResponse.data) {
      console.log(`[TEST] Health response: ${healthResponse.data.substring(0, 200)}`);
    }
    
    // If health check passes, try /models endpoint
    if (healthResponse.statusCode === 200) {
      try {
        const healthData = JSON.parse(healthResponse.data);
        if (healthData.status === 'healthy') {
          // Health check passed, now check /models
          const modelsUrl = vllmUrl.replace(/\/$/, '') + '/models';
          console.log(`[TEST] Health check passed, checking models endpoint: ${modelsUrl}`);
          
          const modelsResponse = await makeHttpRequest(modelsUrl, {
            method: 'GET',
            headers: headers
          });
          
          if (modelsResponse.statusCode === 200) {
            const modelsData = JSON.parse(modelsResponse.data);
            const modelNames = modelsData.map(m => m.name || m);
            
            const authUsed = openButtonToken ? ' (using Vast.ai Basic Auth)' : '';
            console.log(`[TEST] ✓ Inference server is accessible${authUsed}. Available models: ${modelNames.join(', ')}`);
      
      return {
        success: true,
              message: `Inference server is accessible${authUsed}`,
        models: modelNames,
              statusCode: modelsResponse.statusCode,
              requiresAuth: !!openButtonToken
            };
          }
        }
      } catch (e) {
        console.log(`[TEST] Could not parse health response: ${e.message}`);
      }
    }
    
    // If we get here, health check didn't return 200 or wasn't parseable
    // Treat 200 on health as success even if models endpoint fails
    if (healthResponse.statusCode === 200) {
      const authUsed = openButtonToken ? ' (using Vast.ai Basic Auth)' : '';
      return {
        success: true,
        message: `Inference server is accessible${authUsed} (health check passed)`,
        statusCode: healthResponse.statusCode,
        requiresAuth: !!openButtonToken
      };
    }
    
    // Fall through to handle other status codes from health check
    const response = healthResponse;
    
    if (response.statusCode === 502 || response.statusCode === 503) {
      // 502/503 - service may be starting up or temporarily unavailable
      // This is acceptable - it means the gateway/proxy is working, just server isn't ready yet
      const authUsed = openButtonToken ? ' (using Vast.ai Basic Auth)' : '';
      console.log(`[TEST] Got ${response.statusCode} - Inference server may still be starting${authUsed}`);
      return {
        success: true,
        message: `Inference server is accessible${authUsed} (service may still be starting)`,
        details: `Server returned ${response.statusCode} (Bad Gateway/Service Unavailable). This usually means the server is still starting up. Wait a moment and try again, or run Prepare to start/restart the server.`,
        statusCode: response.statusCode,
        requiresAuth: !!openButtonToken,
        isStarting: true
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
          
          // Prepare connection options with auto-accept host keys
          const finalConnectOptions = prepareSSHConnectionOptions(connectOptions);
          await ssh.connect(finalConnectOptions);
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
          
          // Additional fallback: try checking supervisor environment
          if (!openButtonToken) {
            try {
              const supervisorResult = await ssh.execCommand('supervisorctl -c /etc/supervisor/supervisord.conf environment 2>/dev/null | grep OPEN_BUTTON_TOKEN | cut -d= -f2 || echo ""');
              if (supervisorResult.stdout && supervisorResult.stdout.trim()) {
                openButtonToken = supervisorResult.stdout.trim();
                console.log(`[TEST] Found OPEN_BUTTON_TOKEN from supervisor environment`);
              }
            } catch (error) {
              console.log(`[TEST] Could not get OPEN_BUTTON_TOKEN from supervisor: ${error.message}`);
            }
          }
          
          // Additional fallback: try checking all running processes
          if (!openButtonToken) {
            try {
              const psResult = await ssh.execCommand('ps e -o command 2>/dev/null | grep -o "OPEN_BUTTON_TOKEN=[^ ]*" | head -1 | cut -d= -f2 || echo ""');
              if (psResult.stdout && psResult.stdout.trim()) {
                openButtonToken = psResult.stdout.trim();
                console.log(`[TEST] Found OPEN_BUTTON_TOKEN from running processes`);
              }
            } catch (error) {
              console.log(`[TEST] Could not get OPEN_BUTTON_TOKEN from processes: ${error.message}`);
            }
          }
          
          // Additional fallback: try checking /etc/environment or /root/.bashrc
          if (!openButtonToken) {
            try {
              const etcEnvResult = await ssh.execCommand('grep OPEN_BUTTON_TOKEN /etc/environment 2>/dev/null | cut -d= -f2 || echo ""');
              if (etcEnvResult.stdout && etcEnvResult.stdout.trim()) {
                openButtonToken = etcEnvResult.stdout.trim();
                console.log(`[TEST] Found OPEN_BUTTON_TOKEN from /etc/environment`);
              }
            } catch (error) {
              console.log(`[TEST] Could not get OPEN_BUTTON_TOKEN from /etc/environment: ${error.message}`);
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
                requiresAuth: false,
                isValid: true // Still valid - credentials work, just endpoint not ready
              };
            } else if (authResponse.statusCode === 502 || authResponse.statusCode === 503) {
              // 502/503 - service may be starting up or temporarily unavailable
              // This is acceptable - it means the gateway/proxy is working, just vLLM isn't ready yet
              console.log(`[TEST] Got ${authResponse.statusCode} - vLLM may still be starting (using Vast.ai Basic Auth)`);
              return {
                success: true,
                message: `vLLM is accessible (using Vast.ai Basic Auth, service may still be starting)`,
                details: `vLLM returned ${authResponse.statusCode} (Bad Gateway/Service Unavailable). This usually means vLLM is still starting up. Wait a moment and try again, or run Prepare to start/restart vLLM.`,
                statusCode: authResponse.statusCode,
                requiresAuth: false,
                isStarting: true
              };
            } else {
              console.log(`[TEST] Basic Auth with OPEN_BUTTON_TOKEN returned status ${authResponse.statusCode}`);
              // If we got a different status code, the credentials might be wrong or there's another issue
              if (authResponse.statusCode === 401) {
                console.error(`[TEST] ✗ Basic Auth credentials rejected (401) - OPEN_BUTTON_TOKEN may be incorrect`);
                return {
                  success: false,
                  message: 'Basic Auth credentials rejected',
                  details: 'OPEN_BUTTON_TOKEN was retrieved but authentication failed (401). The token may be incorrect or expired.',
                  statusCode: 401,
                  requiresAuth: true
                };
              }
            }
          } else {
            console.log(`[TEST] Could not find OPEN_BUTTON_TOKEN on instance`);
            console.log(`[TEST] Tried: environment variables, /proc/1/environ, systemd, supervisor, running processes, /etc/environment`);
            console.log(`[TEST] This is normal for non-Vast.ai instances. If this is a Vast.ai instance, the token may not be set.`);
          }
        } catch (sshError) {
          console.error(`[TEST] Failed to retrieve OPEN_BUTTON_TOKEN via SSH: ${sshError.message}`);
          console.error(`[TEST] SSH connection may have failed. Check if SSH is working: ${sshHost}:${sshPort || 22}`);
          console.error(`[TEST] SSH error details:`, sshError);
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
  console.log('[MONITOR] Will check logs every 5 seconds for up to', Math.floor(maxWaitTime / 1000), 'seconds');
  
  let checkCount = 0;
  while (Date.now() - startTime < maxWaitTime) {
    try {
      checkCount++;
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      
      // Log progress every 30 seconds (every 6 checks)
      if (checkCount % 6 === 0) {
        console.log(`[MONITOR] Still monitoring... (${elapsed}s elapsed, checking logs...)`);
      }
      
      // Try to get logs from supervisor first, then fall back to /tmp/vllm.log
      const logCommand = 'supervisorctl tail -100 vllm 2>/dev/null | tail -50 || tail -50 /tmp/vllm.log 2>/dev/null || echo ""';
      const logResult = await ssh.execCommand(logCommand);
      const logs = logResult.stdout || '';
      
      // Log if we got logs but they're empty (helps debug)
      if (checkCount === 1 && !logs.trim()) {
        console.log('[MONITOR] Warning: No logs found. vLLM may still be starting or logs are in a different location.');
      }
      
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
        // Continue monitoring - log periodic updates
        if (checkCount % 6 === 0) {
          console.log(`[MONITOR] Model activity detected (download/loading), continuing to monitor... (${elapsed}s elapsed)`);
        }
      } else {
        // Log status updates more frequently to show monitoring is active
        if (checkCount % 6 === 0) {
          // Every 30 seconds, log that we're still waiting for activity
          console.log(`[MONITOR] Waiting for model activity... (${elapsed}s elapsed, checking logs every 5s)`);
          console.log(`[MONITOR] Note: Model may already be cached or loading silently. Monitoring will continue...`);
        }
      }
      
    } catch (error) {
      console.warn(`[MONITOR] Error checking logs: ${error.message}`);
      // Still continue monitoring even if one check fails
    }
    
    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }
  
  // Timeout reached
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  console.warn(`[MONITOR] Timeout after ${elapsed}s. Model may still be downloading/loading.`);
  console.warn(`[MONITOR] Download detected: ${downloadDetected}, Loading detected: ${loadingDetected}`);
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
    
    // Fix 3: Increase timeout for chat requests (60 seconds for LLM responses)
    // Default timeout is 10 seconds for other requests
    const timeout = options.timeout || 10000;
    
    const requestOptions = {
      hostname: urlObj.hostname,
      port: urlObj.port || (isHttps ? 443 : 80),
      path: urlObj.pathname + urlObj.search,
      method: options.method || 'GET',
      headers: options.headers || {}
    };
    
    // Force IPv4 for localhost connections (tunnel is bound to 127.0.0.1, not ::1)
    if (urlObj.hostname === 'localhost' || urlObj.hostname === '127.0.0.1') {
      requestOptions.family = 4; // Force IPv4
    }
    
    // For HTTPS, disable strict SSL certificate validation to handle self-signed certs
    // (Browsers allow this with a warning, so we should too)
    if (isHttps) {
      requestOptions.rejectUnauthorized = false;
    }
    
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
      // Provide more detailed error message including the URL and error code
      const errorDetails = {
        message: error.message,
        code: error.code,
        url: url,
        hostname: urlObj.hostname,
        port: urlObj.port || (isHttps ? 443 : 80)
      };
      
      // Create a more helpful error message
      let errorMsg = `Connection failed to ${url}`;
      if (error.code === 'ECONNREFUSED') {
        errorMsg = `Connection refused to ${urlObj.hostname}:${urlObj.port || (isHttps ? 443 : 80)}. `;
        errorMsg += `If this works in your browser, possible causes: `;
        errorMsg += `(1) Browser is using a proxy/VPN that Node.js isn't configured to use, `;
        errorMsg += `(2) Browser is using HTTPS while app is trying HTTP (or vice versa), `;
        errorMsg += `(3) Firewall is blocking Node.js but allowing browsers, `;
        errorMsg += `(4) Server may need a moment to fully start. `;
        errorMsg += `The chat interface will still work if the URL is correct in your browser.`;
      } else if (error.code === 'ENOTFOUND') {
        errorMsg = `Hostname not found: ${urlObj.hostname}. Check if the URL is correct.`;
      } else if (error.code === 'ETIMEDOUT' || error.code === 'ECONNRESET') {
        errorMsg = `Connection timeout/reset to ${url}. The server may be slow to respond or unreachable.`;
      } else {
        errorMsg = `Connection error to ${url}: ${error.message} (code: ${error.code})`;
      }
      
      const enhancedError = new Error(errorMsg);
      enhancedError.originalError = error;
      enhancedError.details = errorDetails;
      reject(enhancedError);
    });
    
    req.setTimeout(timeout, () => {
      req.destroy();
      const timeoutError = new Error(`Request timeout after ${timeout}ms to ${url}`);
      timeoutError.code = 'ETIMEDOUT';
      reject(timeoutError);
    });
    
    if (options.body) {
      req.write(options.body);
    }
    
    req.end();
  });
}

// Send chat message to FastAPI inference server
ipcMain.handle('send-chat-message', async (event, { message, version, prependedText, useSummary, conversationSummary, conversationHistory, temperature, max_tokens, repetition_penalty }) => {
  if (!storedModelInfo || !storedVLLMUrl) {
    throw new Error('Inference server not prepared. Please prepare the inference server first.');
  }

  try {
    // Validate that message is the user's actual input (not PDF content or other large text)
    const userMessage = message ? message.trim() : '';
    if (!userMessage) {
      throw new Error('Message cannot be empty');
    }
    
    // Log the actual user message being sent
    console.log('[CHAT] User message received (first 200 chars):', userMessage.substring(0, 200));
    console.log('[CHAT] User message total length:', userMessage.length);
    
    // Build messages array matching FastAPI /chat endpoint contract
    // Format: single system message (optional) followed by alternating user/assistant messages
    const messages = [];
    
    // Add system message from user input (always send, even if empty - server will handle default)
    // Server requires roles to alternate: system (optional) -> user -> assistant -> user -> assistant...
    if (prependedText && prependedText.trim().length > 0) {
      messages.push({
        role: 'system',
        content: prependedText.trim()
      });
      console.log('[CHAT] System message added (length:', prependedText.trim().length, 'chars)');
    }
    
    // Add conversation summary as separate system message if enabled
    if (useSummary && conversationSummary && conversationSummary.trim().length > 0) {
      messages.push({
        role: 'system',
        content: conversationSummary.trim()
      });
      console.log('[CHAT] Conversation summary added as system message (length:', conversationSummary.trim().length, 'chars)');
    }
    
    // Add conversation history (excluding the last user message if it exists, as we're replacing it)
    // History should already be in alternating user/assistant format
    // Fix: Handle expansions that create consecutive assistant messages
    if (conversationHistory && conversationHistory.length > 0) {
      // Filter out the last user message if it exists, as we're building a new one
      let historyToAdd = [...conversationHistory];
      if (historyToAdd.length > 0 && historyToAdd[historyToAdd.length - 1].role === 'user') {
        historyToAdd.pop(); // Remove last user message
      }
      
      // Ensure history starts with a user message (after system message)
      // If history starts with assistant, we have a problem - skip it
      if (historyToAdd.length > 0 && historyToAdd[0].role === 'assistant') {
        console.warn('[CHAT] Warning: Conversation history starts with assistant message, removing it');
        historyToAdd.shift(); // Remove first assistant message if history starts with it
      }
      
      // Fix: Normalize conversation history to remove consecutive assistant messages
      // When expansions exist, they create consecutive assistant messages - keep only the last one
      const normalizedHistory = [];
      for (let i = 0; i < historyToAdd.length; i++) {
        const msg = historyToAdd[i];
        const lastNormalized = normalizedHistory[normalizedHistory.length - 1];
        
        // Skip consecutive assistant messages - keep only the last one (expansion if it exists)
        if (msg.role === 'assistant' && lastNormalized && lastNormalized.role === 'assistant') {
          // Replace the previous assistant with this one (expansion takes precedence)
          normalizedHistory[normalizedHistory.length - 1] = msg;
          console.log('[CHAT] Normalized: Replaced previous assistant message with expansion');
        } else {
          normalizedHistory.push(msg);
        }
      }
      
      messages.push(...normalizedHistory);
      console.log('[CHAT] Added', normalizedHistory.length, 'messages from conversation history (normalized from', historyToAdd.length, 'original)');
    }
    
    // REQUIRED user message - this is ONLY the user's typed message, nothing else
    // This must come after system message(s) and must be a user message
    messages.push({
      role: 'user',
      content: userMessage // Clean user message, no PDF content, no section headers
    });
    console.log('[CHAT] User message added (length:', userMessage.length, 'chars)');
    
    // Validate message sequence: system (optional) -> user -> assistant -> user -> assistant...
    // Check that after system messages, we have alternating user/assistant
    let lastRole = null;
    let systemCount = 0;
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role === 'system') {
        systemCount++;
        if (i > 0) {
          console.warn('[CHAT] Warning: System message found after position 0');
        }
      } else if (msg.role === 'user' || msg.role === 'assistant') {
        if (lastRole === msg.role) {
          console.error('[CHAT] ERROR: Duplicate role found:', msg.role, 'at position', i);
          throw new Error(`Invalid message sequence: duplicate ${msg.role} role at position ${i}`);
        }
        lastRole = msg.role;
      }
    }
    
    if (systemCount > 1) {
      console.warn('[CHAT] Warning: Multiple system messages found, but we combined them');
    }
    
    // Determine model version for FastAPI (base, v1, v2, etc.)
    // FastAPI inference server uses simple version names that correspond to adapter directories
    // Server supports per-request model switching - just pass the selected version
    // UI sends "V1", "V2", etc. - convert to lowercase "v1", "v2" for server
    const modelVersion = version && version !== 'base' && version !== '' 
      ? version.toLowerCase()  // Convert "V1" -> "v1", "V2" -> "v2", etc.
      : 'base';
    
    // Create payload matching FastAPI /chat endpoint contract
    const payload = {
      model: modelVersion, // "base", "v1", "v2", etc. - server supports per-request model switching
      messages: messages
    };
    
    // Add optional generation parameters if provided
    if (temperature !== undefined && temperature !== null) {
      payload.temperature = temperature;
    }
    if (max_tokens !== undefined && max_tokens !== null) {
      payload.max_new_tokens = max_tokens;
    }
    if (repetition_penalty !== undefined && repetition_penalty !== null) {
      payload.repetition_penalty = repetition_penalty;
    }
    
    console.log('\n[CHAT] ========================================');
    console.log('[CHAT] Sending message to FastAPI inference server...');
    console.log(`[CHAT] Selected version from UI: "${version}"`);
    console.log(`[CHAT] Model version for request: ${modelVersion} (server supports per-request switching)`);
    console.log(`[CHAT] Base model: ${storedModelInfo.baseModel}`);
    console.log('[CHAT] Messages array being sent:');
    console.log('----------------------------------------');
    console.log(JSON.stringify(payload, null, 2));
    console.log('----------------------------------------');
    console.log(`[CHAT] Total messages: ${messages.length}`);
    console.log(`[CHAT] User message length: ${message.trim().length} chars`);
    console.log('[CHAT] ========================================\n');
    
    // Use the stored inference server URL (user-provided, not localhost)
    const inferenceApiUrl = storedVLLMUrl; // Reusing the same stored URL variable
    console.log(`[CHAT] Using inference server URL: ${inferenceApiUrl}`);
    
    // Ensure SSH tunnel is alive before health check (if using tunneled connection)
    if (inferenceApiUrl.includes('127.0.0.1') || inferenceApiUrl.includes('localhost')) {
      try {
        await ensureSSHTunnel();
      } catch (error) {
        console.warn(`[CHAT] Tunnel check failed during health check: ${error.message}`);
        // Continue anyway - health check will fail if tunnel is truly needed
      }
    }
    
    // Check if inference server is accessible (from local machine, not via SSH)
    console.log(`[CHAT] Checking if inference server is accessible at ${inferenceApiUrl}...`);
    
    // Build headers for health check
    const healthHeaders = { 'Content-Type': 'application/json' };
    
    // Use stored OPEN_BUTTON_TOKEN for Basic Auth if available
    if (storedOpenButtonToken) {
      const basicAuthValue = Buffer.from(`vastai:${storedOpenButtonToken}`).toString('base64');
      healthHeaders['Authorization'] = `Basic ${basicAuthValue}`;
      console.log(`[CHAT] Using Vast.ai Basic Auth for health check`);
    }
    
    let healthCode = null;
    let availableModels = [];
    try {
      // Check /health endpoint first with increased timeout (30 seconds for slow servers)
      const healthResponse = await makeHttpRequest(`${inferenceApiUrl}/health`, {
        method: 'GET',
        headers: healthHeaders,
        timeout: 30000 // 30 seconds for health check (server may be slow to respond)
      });
      healthCode = healthResponse.statusCode.toString();
      console.log(`[CHAT] Inference server health check HTTP code: ${healthCode}`);
      
      // If successful, get the actual available model names from /models
      if (healthCode === '200' && healthResponse.data) {
        try {
          const healthData = JSON.parse(healthResponse.data);
          if (healthData.available_models) {
            availableModels = healthData.available_models;
            console.log(`[CHAT] Available models from inference server: ${availableModels.join(', ')}`);
          }
        } catch (e) {
          console.warn(`[CHAT] Could not parse health response: ${e.message}`);
        }
        
        // Also try /models endpoint for more details
        try {
          const modelsResponse = await makeHttpRequest(`${inferenceApiUrl}/models`, {
            method: 'GET',
            headers: healthHeaders,
            timeout: 30000
          });
          if (modelsResponse.statusCode === 200 && modelsResponse.data) {
            const modelsData = JSON.parse(modelsResponse.data);
            availableModels = modelsData.map(m => m.name || m);
            console.log(`[CHAT] Available models from /models endpoint: ${availableModels.join(', ')}`);
          }
        } catch (e) {
          // Ignore - health check is enough
        }
      }
    } catch (error) {
      const statusMatch = error.message.match(/HTTP (\d+)/);
      if (statusMatch) {
        healthCode = statusMatch[1];
      } else {
        healthCode = '000';
      }
      console.warn(`[CHAT] Inference server health check failed: ${error.message}`);
      console.warn(`[CHAT] ⚠ Health check failed, but proceeding with chat request anyway (server may be slow to respond)`);
    }
    
    // Make health check non-blocking - don't throw errors, just warn
    // The actual chat request will fail if the server is truly unreachable
    if (healthCode === '000' || healthCode === '' || healthCode.includes('connection')) {
      console.warn(`[CHAT] ⚠ Health check indicates server may not be accessible, but proceeding with request anyway`);
      console.warn(`[CHAT] ⚠ If the request fails, check that the server is running at ${inferenceApiUrl}`);
    }
    
    if (healthCode === '401') {
      console.warn(`[CHAT] ⚠ Health check returned 401 (authentication required), but proceeding with request`);
      console.warn(`[CHAT] ⚠ If the request fails, check Basic Auth configuration`);
    }
    
    if (healthCode && healthCode !== '200' && healthCode !== '000' && healthCode !== '') {
      console.warn(`[CHAT] ⚠ Inference server health check returned HTTP ${healthCode} - may still be loading, proceeding anyway`);
    }
    
    // Ensure SSH tunnel is alive before making request (if using tunneled connection)
    if (inferenceApiUrl.includes('127.0.0.1') || inferenceApiUrl.includes('localhost')) {
      try {
        await ensureSSHTunnel();
      } catch (error) {
        console.error(`[CHAT] Tunnel check failed: ${error.message}`);
        // Continue anyway - the request will fail if tunnel is truly needed
      }
    }
    
    // Make HTTP request directly (not via SSH) to FastAPI /chat endpoint
    const chatApiUrl = `${inferenceApiUrl}/chat`;
    console.log(`[CHAT] Sending HTTP request to: ${chatApiUrl}`);
    
    // Build headers for chat request
    const chatHeaders = { 'Content-Type': 'application/json' };
    
    // Use stored OPEN_BUTTON_TOKEN for Basic Auth if available
    if (storedOpenButtonToken) {
      const basicAuthValue = Buffer.from(`vastai:${storedOpenButtonToken}`).toString('base64');
      chatHeaders['Authorization'] = `Basic ${basicAuthValue}`;
      console.log(`[CHAT] Using Vast.ai Basic Auth for chat request`);
    }
    
    // Build request body as JSON string
    const requestBody = JSON.stringify(payload);
    
    // Tier 1 Fix: Raise renderer HTTP timeout to 120 seconds for longer expansions
    // Wrap in try-catch to handle connection failures and attempt tunnel reconnection
    let response;
    try {
      response = await makeHttpRequest(chatApiUrl, {
        method: 'POST',
        headers: {
          ...chatHeaders,
          'Accept': 'application/json'
        },
        body: requestBody,
        timeout: 120000 // 120 seconds for chat requests (allows longer expansions up to 900 tokens)
      });
    } catch (error) {
      // If connection refused and we're using localhost, try reconnecting tunnel once
      if ((error.code === 'ECONNREFUSED' || error.originalError?.code === 'ECONNREFUSED') &&
          (inferenceApiUrl.includes('127.0.0.1') || inferenceApiUrl.includes('localhost'))) {
        console.warn(`[CHAT] Connection refused, attempting tunnel reconnection...`);
        try {
          await ensureSSHTunnel();
          // Retry the request after reconnection
          await new Promise(resolve => setTimeout(resolve, 1500)); // Give tunnel time to bind
          response = await makeHttpRequest(chatApiUrl, {
            method: 'POST',
            headers: {
              ...chatHeaders,
              'Accept': 'application/json'
            },
            body: requestBody,
            timeout: 120000
          });
          console.log(`[CHAT] ✓ Request succeeded after tunnel reconnection`);
        } catch (retryError) {
          console.error(`[CHAT] ✗ Request failed even after tunnel reconnection: ${retryError.message}`);
          throw error; // Throw original error
        }
      } else {
        throw error; // Not a tunnel issue, throw original error
      }
    }
    
    console.log(`[CHAT] HTTP status: ${response.statusCode}`);
    console.log(`[CHAT] Response length: ${response.data ? response.data.length : 0} chars`);
    
    if (!response.data || response.data.trim() === '') {
      throw new Error('Inference server returned an empty response');
    }
    
    // Parse FastAPI response with robust error handling
    let parsed;
    try {
      console.log(`[CHAT] Parsing response (first 500 chars): ${response.data.substring(0, 500)}...`);
      parsed = JSON.parse(response.data);
      console.log(`[CHAT] ✓ Successfully parsed JSON response`);
    } catch (error) {
      console.error(`[CHAT] ✗ Failed to parse response as JSON`);
      console.error(`[CHAT] Error: ${error.message}`);
      console.error(`[CHAT] Raw response (first 1000 chars):`);
      console.error(response.data.substring(0, 1000));
      throw new Error('Invalid JSON returned from inference server');
    }
    
    // Check for error details in response
    if (parsed.detail) {
      throw new Error(`Inference error: ${parsed.detail}`);
    }
    
    // FastAPI returns { response, model_used, tokens_generated, inference_time }
    // Check for property existence, not truthiness (empty string is allowed)
    if (!("response" in parsed)) {
      throw new Error('Inference server response missing response field');
    }
    
    // Empty string is allowed
    const responseText = parsed.response ?? "";
    const usage = {
      total_tokens: parsed.tokens_generated || 0,
      completion_tokens: parsed.tokens_generated || 0,
      prompt_tokens: 0 // FastAPI doesn't provide this
    };
    
    console.log(`[CHAT] Response received:`);
    console.log(`[CHAT] ${responseText}`);
    console.log(`[CHAT] Model used: ${parsed.model_used || modelVersion}`);
    console.log(`[CHAT] Tokens generated: ${parsed.tokens_generated || 'unknown'}`);
    console.log(`[CHAT] Inference time: ${parsed.inference_time ? parsed.inference_time.toFixed(2) + 's' : 'unknown'}`);
    console.log('[CHAT] ✓ Message exchange complete\n');
    
    return {
      success: true,
      response: responseText,
      usage: usage,
      userMessage: message, // Original user message (clean, without context/summary)
      assistantMessage: responseText // Full assistant response
    };
    
  } catch (error) {
    console.error(`[CHAT] Fatal error: ${error.message}`);
    throw error;
  }
});


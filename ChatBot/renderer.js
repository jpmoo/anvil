let selectedProfile = null;
let profileVersions = [];
let sshConfig = { host: '', port: 22 };
let sshTested = false;
let modelInfo = null;
let conversationHistory = [];
let conversationSummary = '';
let configLoaded = false; // Track if config has been loaded to prevent overwriting user input

// Fix 3: Normalize conversation history before sending to enforce strict role alternation
function normalizeConversation(messages) {
  const cleaned = [];
  for (const msg of messages) {
    if (cleaned.length === 0) {
      // First message must be user or system
      if (msg.role === "user" || msg.role === "system") {
        cleaned.push(msg);
      }
      continue;
    }
    const last = cleaned[cleaned.length - 1];
    // Only add if role alternates (no duplicate consecutive roles)
    if (msg.role !== last.role) {
      cleaned.push(msg);
    }
  }
  return cleaned;
}

// Save configuration to file
async function saveConfig() {
  try {
    // Read directly from input fields to ensure we save current values, not stale sshConfig
    const sshHostInput = document.getElementById('sshHost');
    const sshPortInput = document.getElementById('sshPort');
    const prependedTextInput = document.getElementById('prependedText');
    
    // Get SSH values from input fields
    let sshHost = sshHostInput ? sshHostInput.value.trim() : '';
    let sshPort = 22; // Default
    
    console.log('[CONFIG] Reading port from input field:', {
      sshPortInputExists: !!sshPortInput,
      rawValue: sshPortInput ? sshPortInput.value : 'N/A',
      trimmedValue: sshPortInput ? sshPortInput.value.trim() : 'N/A',
      isEmpty: sshPortInput ? !sshPortInput.value.trim() : true
    });
    
    if (sshPortInput && sshPortInput.value.trim()) {
      const portValue = parseInt(sshPortInput.value.trim());
      console.log('[CONFIG] Parsed port value:', portValue, 'isNaN:', isNaN(portValue));
      if (!isNaN(portValue)) {
        sshPort = portValue;
        console.log('[CONFIG] Using parsed port:', sshPort);
      } else {
        console.warn('[CONFIG] Port value is NaN, using default 22');
      }
    } else {
      console.warn('[CONFIG] Port input is empty or missing, using default 22');
    }
    
    // Process SSH host (strip port if present)
    if (sshHost.includes(':')) {
      const parts = sshHost.split(':');
      sshHost = parts[0].trim();
      if (parts.length > 1 && parts[1].trim()) {
        const extractedPort = parseInt(parts[1].trim());
        if (!isNaN(extractedPort)) {
          sshPort = extractedPort;
        }
      }
    }
    
    // Get profile and version selections
    const profileSelect = document.getElementById('profileSelect');
    const versionSelect = document.getElementById('versionSelect');
    
    // Get internal port (for FastAPI server inside container)
    const internalPortInput = document.getElementById('internalPort');
    let internalPort = internalPortInput ? parseInt(internalPortInput.value.trim()) : 8888;
    
    if (isNaN(internalPort)) internalPort = 8888;
    
    const config = {
      sshHost: sshHost,
      sshPort: sshPort,
      internalPort: internalPort,
      prependedText: prependedTextInput ? prependedTextInput.value.trim() : '',
      conversationHistory: conversationHistory || [], // Save the conversation history
      selectedProfile: profileSelect ? profileSelect.value : '', // Save selected profile
      selectedVersion: versionSelect ? versionSelect.value : 'base' // Save selected version (default to 'base')
      // Token is not saved - it's retrieved automatically during setup
    };
    
    console.log('[CONFIG] Full config object being saved:', JSON.stringify(config, null, 2));
    console.log('[CONFIG] Port value type:', typeof config.sshPort, 'value:', config.sshPort);
    const result = await window.electronAPI.saveConfig(config);
    if (result && result.success) {
      console.log('[CONFIG] ✓ Config saved successfully');
    } else {
      console.error('[CONFIG] ✗ Failed to save config:', result?.error || 'Unknown error');
    }
  } catch (error) {
    console.error('Error saving config:', error);
  }
}

// Load configuration from file
async function loadConfig() {
  try {
    const result = await window.electronAPI.loadConfig();
    if (result.success && result.config) {
      const config = result.config;
      
      // Load SSH config (only if fields are empty, to avoid overwriting user input)
      const sshHostInput = document.getElementById('sshHost');
      const sshPortInput = document.getElementById('sshPort');
      
      // Only load if the fields are empty (user hasn't entered anything yet)
      if (config.sshHost && sshHostInput && !sshHostInput.value.trim()) {
        sshHostInput.value = config.sshHost;
      }
      
      // Load port - check if config has a port value (including 0, which is falsy but valid)
      // Also check if the field has the default "22" value (from HTML) - if so, load from config
      if (sshPortInput) {
        const currentValue = sshPortInput.value.trim();
        const isDefaultOrEmpty = !currentValue || currentValue === '22';
        
        console.log('[CONFIG] Loading port - current value:', currentValue, 'isDefaultOrEmpty:', isDefaultOrEmpty, 'saved port:', config.sshPort);
        
        if (isDefaultOrEmpty) {
          if (config.sshPort !== undefined && config.sshPort !== null && config.sshPort !== '') {
            sshPortInput.value = config.sshPort.toString();
            console.log('[CONFIG] Loaded port from config:', config.sshPort);
          } else {
            // Keep default 22 if no saved config
            console.log('[CONFIG] No saved port, keeping default 22');
          }
        } else {
          console.log('[CONFIG] Port field already has non-default value, not overwriting');
        }
      }
      
      // Process the loaded values through handleSSHConfigChange to ensure proper formatting
      // This will strip any port from host, update sshConfig, and reset sshTested
      // Pass skipSave=true to avoid saving during load (which could cause issues)
      if ((sshHostInput && sshHostInput.value.trim()) || (sshPortInput && sshPortInput.value.trim())) {
        // Use a small delay to ensure DOM is ready
        setTimeout(() => {
          handleSSHConfigChange(true); // skipSave=true during load
        }, 100);
      }
      
      // Mark config as loaded
      configLoaded = true;
      
      // Load Inference Server URL (only if field is empty, to avoid overwriting user input)
      // Reset Inference Server URL test flags since we're loading from saved config
      // Don't test the URL automatically on startup - user should test when ready
      
      // Load port forwarding values
      const internalPortInput = document.getElementById('internalPort');
      if (internalPortInput && config.internalPort !== undefined && config.internalPort !== null) {
        if (!internalPortInput.value.trim() || internalPortInput.value === '8888') {
          internalPortInput.value = config.internalPort.toString();
        }
      }
      
      // Load system message
      if (config.prependedText !== undefined) {
        const prependedTextInput = document.getElementById('prependedText');
        if (prependedTextInput) {
          prependedTextInput.value = config.prependedText;
        }
      }
      
      // Load conversation history
      if (config.conversationHistory !== undefined && Array.isArray(config.conversationHistory)) {
        conversationHistory = config.conversationHistory;
        console.log('[CONFIG] Loaded conversation history:', conversationHistory.length, 'messages');
        
        // Restore chat messages from history
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages && conversationHistory.length > 0) {
          chatMessages.innerHTML = ''; // Clear existing messages
          conversationHistory.forEach(msg => {
            addMessage(msg.role, msg.content, false); // Add to UI but don't add to history again
          });
        }
      }
      
      // Load selected profile (will be restored after profiles are loaded)
      if (config.selectedProfile) {
        // Store for later restoration after profiles load
        window._pendingProfileSelection = config.selectedProfile;
        console.log('[CONFIG] Will restore profile selection:', config.selectedProfile);
      }
      
      // Load selected version (will be restored after chat interface is shown)
      if (config.selectedVersion) {
        // Store for later restoration after version select is populated
        window._pendingVersionSelection = config.selectedVersion;
        console.log('[CONFIG] Will restore version selection:', config.selectedVersion);
      }
      
      // Token is not loaded from config - it's retrieved automatically during setup
      
      console.log('[CONFIG] Configuration loaded from file');
    }
  } catch (error) {
    console.error('Error loading config:', error);
  }
}

// Load profiles on startup
window.addEventListener('DOMContentLoaded', async () => {
  // Load saved configuration first
  await loadConfig();
  
  // Set up event listeners BEFORE loading profiles (so profile restoration works)
  const profileSelect = document.getElementById('profileSelect');
  if (profileSelect) {
    profileSelect.addEventListener('change', handleProfileChange);
    console.log('[INIT] Profile select event listener set up');
  }
  
  // Initialize button states (all disabled until profile is selected)
  updateButtonStates();
  
  await loadProfiles();
  await checkVastApiKey();
  await checkSSHKey();
  
  const sshHostInput = document.getElementById('sshHost');
  const sshPortInput = document.getElementById('sshPort');
  
  if (sshHostInput) {
    console.log('[INIT] Setting up sshHost input listener');
    sshHostInput.addEventListener('input', () => {
      console.log('[EVENT] sshHost input event fired');
      handleSSHConfigChange();
    });
  } else {
    console.error('[INIT] sshHost input not found!');
  }
  
  if (sshPortInput) {
    console.log('[INIT] Setting up sshPort input listener');
    sshPortInput.addEventListener('input', () => {
      console.log('[EVENT] sshPort input event fired, value:', sshPortInput.value);
      handleSSHConfigChange();
    });
  } else {
    console.error('[INIT] sshPort input not found!');
  }
  
  document.getElementById('testSSHBtn').addEventListener('click', initializeInferenceEnvironment);
  
  // Set up internal port listener
  const internalPortInput = document.getElementById('internalPort');
  if (internalPortInput) {
    internalPortInput.addEventListener('input', saveConfig);
  }
  
  // Set up Inference Server URL listeners after functions are defined (see end of file)
  setTimeout(setupVLLMUrlListeners, 0);
  document.getElementById('sendBtn').addEventListener('click', sendMessage);
  
  // Set up listeners for system message
  const prependedTextInput = document.getElementById('prependedText');
  const clearChatBtn = document.getElementById('clearChatBtn');
  
  // Set up listener for version selection
  const versionSelect = document.getElementById('versionSelect');
  if (versionSelect) {
    versionSelect.addEventListener('change', () => {
      console.log('[CONFIG] Version selection changed, saving config...');
      saveConfig();
    });
  }
  
  if (prependedTextInput) {
    prependedTextInput.addEventListener('input', () => {
      saveConfig();
    });
  }
  
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      conversationHistory = [];
      const chatMessages = document.getElementById('chatMessages');
      if (chatMessages) {
        chatMessages.innerHTML = '';
      }
      console.log('[CHAT] Conversation history cleared');
      
      // Save config to persist the cleared history
      saveConfig();
      // Optionally add a message indicating the conversation was reset
      if (chatMessages) {
        const resetMsg = document.createElement('div');
        resetMsg.className = 'message assistant';
        resetMsg.textContent = 'Conversation history cleared. Starting fresh conversation.';
        chatMessages.appendChild(resetMsg);
      }
    });
  }
  document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  // Final button state update after all initialization
  setTimeout(() => {
    console.log('[INIT] Final button state update after initialization');
    updateButtonStates();
  }, 500);
});

async function loadProfiles() {
  console.log('[UI] ========================================');
  console.log('[UI] loadProfiles() called');
  console.log('[UI] ========================================');
  try {
    console.log('[UI] Checking if window.electronAPI exists:', typeof window.electronAPI);
    console.log('[UI] Checking if getProfiles exists:', typeof window.electronAPI?.getProfiles);
    
    if (!window.electronAPI || !window.electronAPI.getProfiles) {
      console.error('[UI] ✗ window.electronAPI.getProfiles is not available!');
      const select = document.getElementById('profileSelect');
      if (select) {
        select.innerHTML = '<option value="">Error: IPC not available</option>';
      }
      return;
    }
    
    console.log('[UI] Calling window.electronAPI.getProfiles()...');
    const profiles = await window.electronAPI.getProfiles();
    console.log('[UI] Received profiles:', profiles);
    console.log('[UI] Profiles type:', typeof profiles);
    console.log('[UI] Profiles is array?', Array.isArray(profiles));
    console.log('[UI] Profiles length:', profiles?.length);
    
    const select = document.getElementById('profileSelect');
    if (!select) {
      console.error('[UI] ✗ Profile select element not found!');
      return;
    }
    
    select.innerHTML = '<option value="">Select a profile...</option>';
    
    if (!profiles || !Array.isArray(profiles) || profiles.length === 0) {
      console.warn('[UI] No profiles found or invalid response');
      select.innerHTML = '<option value="">No profiles found</option>';
      showStatus('profileInfo', 'No profiles found. Make sure you have created at least one model profile in the parent Anvil app.', 'error');
      return;
    }
    
    console.log(`[UI] Adding ${profiles.length} profile(s) to dropdown`);
    profiles.forEach((profile, index) => {
      console.log(`[UI] Adding profile ${index + 1}:`, profile);
      const option = document.createElement('option');
      option.value = profile.name;
      option.textContent = `${profile.name} (${profile.baseModel})`;
      select.appendChild(option);
    });
    
    // Restore saved profile selection if available
    if (window._pendingProfileSelection) {
      const savedProfile = window._pendingProfileSelection;
      console.log('[CONFIG] Restoring saved profile selection:', savedProfile);
      if (select.querySelector(`option[value="${savedProfile}"]`)) {
        select.value = savedProfile;
        // Trigger change event to load versions
        select.dispatchEvent(new Event('change'));
      } else {
        console.warn('[CONFIG] Saved profile not found in available profiles:', savedProfile);
      }
      delete window._pendingProfileSelection;
    }
    
    console.log('[UI] ✓ Profiles loaded successfully');
    console.log('[UI] ========================================');
  } catch (error) {
    console.error('[UI] ✗ Error loading profiles:', error);
    console.error('[UI] Error stack:', error.stack);
    console.log('[UI] ========================================');
    const select = document.getElementById('profileSelect');
    if (select) {
      select.innerHTML = '<option value="">Error loading profiles</option>';
    }
    showStatus('profileInfo', `Error loading profiles: ${error.message}. Check the terminal for details.`, 'error');
  }
}

async function checkVastApiKey() {
  try {
    const apiKey = await window.electronAPI.getVastApiKey();
    if (!apiKey) {
      showStatus('profileInfo', '⚠️ Vast.ai API key not found in parent app config', 'error');
    }
  } catch (error) {
    console.error('Error checking API key:', error);
  }
}

async function checkSSHKey() {
  try {
    const sshKeyPath = await window.electronAPI.getSSHKeyPath();
    const sshConfigSection = document.querySelector('.section:nth-of-type(2)');
    if (sshConfigSection && sshKeyPath) {
      // Add SSH key info below the section title
      const keyInfo = document.createElement('div');
      keyInfo.id = 'sshKeyInfo';
      keyInfo.style.cssText = 'font-size: 12px; color: #666; margin-bottom: 10px; font-family: monospace;';
      keyInfo.textContent = `Using SSH key: ${sshKeyPath}`;
      const sectionTitle = sshConfigSection.querySelector('.section-title');
      if (sectionTitle && !document.getElementById('sshKeyInfo')) {
        sectionTitle.parentNode.insertBefore(keyInfo, sectionTitle.nextSibling);
      }
    } else if (!sshKeyPath) {
      const sshConfigSection = document.querySelector('.section:nth-of-type(2)');
      if (sshConfigSection) {
        const keyInfo = document.createElement('div');
        keyInfo.id = 'sshKeyInfo';
        keyInfo.style.cssText = 'font-size: 12px; color: #d32f2f; margin-bottom: 10px;';
        keyInfo.textContent = '⚠️ No SSH key found. Please configure SSH key in parent app config or ensure ~/.ssh/id_rsa exists.';
        const sectionTitle = sshConfigSection.querySelector('.section-title');
        if (sectionTitle && !document.getElementById('sshKeyInfo')) {
          sectionTitle.parentNode.insertBefore(keyInfo, sectionTitle.nextSibling);
        }
      }
    }
  } catch (error) {
    console.error('Error checking SSH key:', error);
  }
}

async function handleProfileChange(event) {
  const profileName = event.target.value;
  
  console.log('[PROFILE] Profile selection changed:', profileName);
  
  // Save config when profile changes
  if (profileName) {
    saveConfig();
  }
  
  if (!profileName) {
    selectedProfile = null;
    profileVersions = [];
    document.getElementById('profileInfo').textContent = '';
    document.getElementById('versionsSection').style.display = 'none';
    updateButtonStates();
    return;
  }
  
  try {
    // Get profile info from select option
    const option = event.target.options[event.target.selectedIndex];
    const baseModel = option.textContent.match(/\(([^)]+)\)/)?.[1] || 'unknown';
    
    selectedProfile = {
      name: profileName,
      baseModel: baseModel
    };
    
    console.log('[PROFILE] Selected profile:', selectedProfile);
    document.getElementById('profileInfo').textContent = `Base Model: ${baseModel}`;
    
    // Load versions
    console.log('[PROFILE] Loading versions for profile:', profileName);
    profileVersions = await window.electronAPI.getProfileVersions(profileName);
    console.log('[PROFILE] Loaded versions:', profileVersions);
    console.log('[PROFILE] Version count:', profileVersions.length);
    
    displayVersions();
    updateButtonStates();
  } catch (error) {
    console.error('[PROFILE] Error loading profile versions:', error);
    showStatus('profileInfo', `Error loading versions: ${error.message}`, 'error');
    // Reset versions on error
    profileVersions = [];
    displayVersions();
    updateButtonStates();
  }
}

function displayVersions() {
  const versionsList = document.getElementById('versionsList');
  const versionsSection = document.getElementById('versionsSection');
  
  console.log('[VERSIONS] Displaying versions, count:', profileVersions.length);
  
  if (profileVersions.length === 0) {
    versionsList.innerHTML = '<div class="version-item">No versions found for this profile</div>';
    versionsSection.style.display = 'block';
    console.log('[VERSIONS] No versions found - Prepare button will be disabled');
    return;
  }
  
  versionsList.innerHTML = profileVersions.map(v => 
    `<div class="version-item">✓ Version ${v.version} (adapter found)</div>`
  ).join('');
  
  versionsSection.style.display = 'block';
  console.log('[VERSIONS] Displayed', profileVersions.length, 'version(s)');
}

// Update Inference Server URL from external port and SSH host
// updateInferenceUrlFromPorts() removed - no longer needed with SSH tunnel approach
// All inference requests now go through SSH tunnel to localhost:<forwarded_port>

function handleSSHConfigChange(skipSave = false) {
  console.log('[SSH] handleSSHConfigChange called, skipSave:', skipSave);
  const sshHostInput = document.getElementById('sshHost');
  const sshPortInput = document.getElementById('sshPort');
  
  if (!sshHostInput) {
    console.error('[SSH] sshHostInput not found!');
    return;
  }
  
  let hostValue = sshHostInput.value.trim();
  console.log('[SSH] Current host value:', hostValue);
  console.log('[SSH] Current port value:', sshPortInput ? sshPortInput.value : 'N/A');
  
  // If host contains a colon, extract just the IP address (user might have pasted URL with port)
  if (hostValue.includes(':')) {
    const parts = hostValue.split(':');
    hostValue = parts[0].trim();
    // If we extracted a port from the host field, update the port field if it's empty
    if (parts.length > 1 && parts[1].trim()) {
      const extractedPort = parseInt(parts[1].trim());
      if (!isNaN(extractedPort) && sshPortInput && (!sshPortInput.value || sshPortInput.value.trim() === '')) {
        sshPortInput.value = extractedPort;
        console.log(`[SSH] Extracted port from host field: ${extractedPort}`);
      }
    }
    // Update the host input to remove the port
    if (sshHostInput.value !== hostValue) {
      sshHostInput.value = hostValue;
    }
    console.log(`[SSH] Extracted IP from host field: ${hostValue} (removed port)`);
  }
  
  sshConfig.host = hostValue;
  sshConfig.port = parseInt(sshPortInput ? sshPortInput.value : '22') || 22;
  console.log('[SSH] Updated sshConfig:', { host: sshConfig.host, port: sshConfig.port });
  sshTested = false;
  updateButtonStates();
  
  // Save config when it changes (unless we're loading from saved config)
  if (!skipSave) {
    console.log('[SSH] Calling saveConfig()...');
    saveConfig();
  } else {
    console.log('[SSH] Skipping save (loading from config)');
  }
  
  // Clear SSH status
  const statusDiv = document.getElementById('sshStatus');
  if (statusDiv) {
    statusDiv.innerHTML = '';
  }
}

async function initializeInferenceEnvironment() {
  const btn = document.getElementById('testSSHBtn');
  const statusDiv = document.getElementById('sshStatus');
  const sshHostInput = document.getElementById('sshHost');
  const sshPortInput = document.getElementById('sshPort');
  const vllmUrlInput = document.getElementById('vllmUrl');
  
  // Read directly from input fields to ensure we use current values, not stale sshConfig
  let hostValue = sshHostInput ? sshHostInput.value.trim() : '';
  let portValue = 22; // Default
  
  // Get port from input field, respecting saved config
  if (sshPortInput && sshPortInput.value.trim()) {
    const parsedPort = parseInt(sshPortInput.value.trim());
    if (!isNaN(parsedPort) && parsedPort > 0) {
      portValue = parsedPort;
      console.log(`[INIT] Using SSH port from input: ${portValue}`);
    }
  } else {
    // If input is empty, try to use saved config or sshConfig
    if (sshConfig.port && sshConfig.port > 0) {
      portValue = sshConfig.port;
      console.log(`[INIT] Using SSH port from sshConfig: ${portValue}`);
    }
  }
  
  // Process host value (strip port if present)
  if (hostValue.includes(':')) {
    const parts = hostValue.split(':');
    hostValue = parts[0].trim();
    if (parts.length > 1 && parts[1].trim()) {
      const extractedPort = parseInt(parts[1].trim());
      if (!isNaN(extractedPort) && extractedPort > 0) {
        portValue = extractedPort;
        console.log(`[INIT] Using SSH port extracted from host: ${portValue}`);
      }
    }
  }
  
  console.log(`[INIT] Final SSH port value: ${portValue}`);
  
  if (!hostValue) {
    showStatus('sshStatus', 'Please enter SSH host/IP', 'error');
    return;
  }
  
  // Update sshConfig with current values
  sshConfig.host = hostValue;
  sshConfig.port = portValue;
  
  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span> Initializing...';
  statusDiv.innerHTML = '';
  
  // Get profile info if available
  let profileName = null;
  let baseModel = null;
  let versions = [];
  
  if (selectedProfile) {
    profileName = selectedProfile.name;
    baseModel = selectedProfile.baseModel;
    versions = profileVersions || [];
  }
  
  // Get internal port (for FastAPI server inside container)
  const internalPortInput = document.getElementById('internalPort');
  let internalPort = internalPortInput ? parseInt(internalPortInput.value.trim()) : 8888;
  
  if (isNaN(internalPort)) internalPort = 8888;
  
  try {
    const result = await window.electronAPI.initializeInferenceEnvironment({
      host: hostValue,
      port: portValue,
      username: 'root',
      profileName: profileName,
      baseModel: baseModel,
      versions: versions,
      inferenceUrl: undefined, // No longer needed - using SSH tunnel
      vllmUrl: undefined, // No longer needed - using SSH tunnel
      externalPort: null, // No longer needed - using SSH tunnel
      internalPort: internalPort
    });
    
    // Display steps and errors
    if (result.steps && result.steps.length > 0) {
      let statusHTML = '<div class="steps-container">';
      result.steps.forEach(step => {
        statusHTML += `<div class="step">${step}</div>`;
      });
      statusHTML += '</div>';
      statusDiv.innerHTML = statusHTML;
    }
    
    if (result.errors && result.errors.length > 0) {
      let errorHTML = '<div class="errors-container" style="margin-top: 10px; color: #ff4444;">';
      result.errors.forEach(error => {
        errorHTML += `<div class="error">✗ ${error}</div>`;
      });
      errorHTML += '</div>';
      statusDiv.innerHTML += errorHTML;
    }
    
    if (result.success) {
      showStatus('sshStatus', '✓ Initialization complete!', 'success');
      sshTested = true;
      // Ensure sshConfig is set after successful initialization
      sshConfig.host = hostValue;
      sshConfig.port = portValue;
      
      // Update inference URL if provided
      if (result.inferenceUrl) {
        if (vllmUrlInput) {
          vllmUrlInput.value = result.inferenceUrl;
        }
        vllmUrl = result.inferenceUrl;
        vllmUrlTested = true;
        vllmUrlValid = true;
      }
      
      console.log('[INIT] Initialization successful, config updated:', sshConfig);
      console.log('[INIT] Health check passed:', result.healthCheckPassed);
      
      // Automatically launch the chat interface if initialization succeeded
      // Even if health check didn't pass, the server might still be starting
      if (result.success) {
        console.log('[INIT] Initialization successful - automatically launching chat interface...');
        // Small delay to ensure UI is updated
        setTimeout(async () => {
          try {
            await showChatInterface();
            console.log('[INIT] ✓ Chat interface launched automatically');
          } catch (error) {
            console.error('[INIT] Error launching chat interface:', error);
            console.error('[INIT] Error details:', error);
            // Show error in UI
            showStatus('sshStatus', 'Chat interface error: ' + error.message, 'error');
          }
        }, 500);
      }
    } else {
      // Show error message - errors are already displayed above if any
      let errorMessage = '✗ Initialization failed';
      if (result.errors && result.errors.length > 0) {
        // Errors are already shown above, just add a note to check terminal
        errorMessage += ' - Check terminal for details';
      } else {
        // No specific errors returned, definitely check terminal
        errorMessage += ' - Check terminal for details';
      }
      showStatus('sshStatus', errorMessage, 'error');
      sshTested = false;
    }
  } catch (error) {
    showStatus('sshStatus', '✗ Error: ' + error.message, 'error');
    sshTested = false;
  } finally {
    btn.disabled = !selectedProfile; // Re-enable only if profile is selected
    btn.textContent = 'Initialize Inference Environment';
    updateButtonStates();
  }
}


// Debounce helper function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Create debounced test function
let debouncedTestVLLMUrl = null;

// Test Inference Server URL connectivity
let vllmUrlTested = false;
let vllmUrlValid = false;

// Initialize debounced function after testVLLMUrl is defined
async function testVLLMUrl() {
  const vllmUrlInput = document.getElementById('vllmUrl');
  const statusDiv = document.getElementById('vllmUrlStatus');
  
  // Read directly from input field to ensure we use current value
  const vllmUrl = vllmUrlInput ? vllmUrlInput.value.trim() : '';
  
  if (!vllmUrl) {
    statusDiv.innerHTML = '<span style="color: #999;">Enter an Inference Server URL to test</span>';
    vllmUrlTested = false;
    vllmUrlValid = false;
    updateButtonStates();
    return;
  }
  
  // Get current SSH config from input fields (not from potentially stale sshConfig)
  const sshHostInput = document.getElementById('sshHost');
  const sshPortInput = document.getElementById('sshPort');
  let sshHost = sshHostInput ? sshHostInput.value.trim() : '';
  let sshPort = parseInt(sshPortInput ? sshPortInput.value : '22') || 22;
  
  // Process SSH host (strip port if present)
  if (sshHost.includes(':')) {
    const parts = sshHost.split(':');
    sshHost = parts[0].trim();
    if (parts.length > 1 && parts[1].trim()) {
      const extractedPort = parseInt(parts[1].trim());
      if (!isNaN(extractedPort)) {
        sshPort = extractedPort;
      }
    }
  }
  
  statusDiv.innerHTML = '<span style="color: #666;">Testing connection...</span>';
  
  try {
    const result = await window.electronAPI.testVLLMUrl({
      vllmUrl: vllmUrl,
      sshHost: sshHost,
      sshPort: sshPort,
      sshUsername: 'root'
    });
    
    if (result.success) {
      // Check if it's a "starting up" status (502/503) - show as warning but still valid
      if (result.isStarting && (result.statusCode === 502 || result.statusCode === 503)) {
        statusDiv.innerHTML = `<span style="color: #ff9800;">⚠ ${result.message}</span>`;
        if (result.details) {
          statusDiv.innerHTML += `<br><small style="color: #666;">${result.details}</small>`;
        }
        // Show supervisor status if available
        if (result.supervisorStatus) {
          const statusColor = result.supervisorStatus.includes('STOPPED') ? '#f44336' : 
                             result.supervisorStatus.includes('RUNNING') ? '#4caf50' : '#666';
          statusDiv.innerHTML += `<br><small style="color: ${statusColor}; font-weight: bold;">Supervisor: ${result.supervisorStatus}</small>`;
        }
        vllmUrlTested = true;
        vllmUrlValid = true; // Still valid - credentials work, service just starting
      } else {
        statusDiv.innerHTML = `<span style="color: #4caf50;">✓ ${result.message}</span>`;
        if (result.models && result.models.length > 0) {
          statusDiv.innerHTML += `<br><small style="color: #666;">Available models: ${result.models.join(', ')}</small>`;
        }
        vllmUrlTested = true;
        vllmUrlValid = true;
      }
    } else {
      statusDiv.innerHTML = `<span style="color: #f44336;">✗ ${result.message}</span>`;
      if (result.details) {
        statusDiv.innerHTML += `<br><small style="color: #666;">${result.details}</small>`;
      }
      vllmUrlTested = true;
      // Check if it's a "starting up" status even in error case
      if (result.statusCode && [502, 503].includes(result.statusCode) && result.isStarting) {
        vllmUrlValid = true; // Credentials work, service just starting
      } else {
        vllmUrlValid = false;
      }
    }
  } catch (error) {
    statusDiv.innerHTML = `<span style="color: #f44336;">✗ Error: ${error.message}</span>`;
    vllmUrlTested = true;
    vllmUrlValid = false;
  } finally {
    updateButtonStates();
  }
}

// Set up Inference Server URL event listeners (called after functions are defined)
function setupVLLMUrlListeners() {
  const vllmUrlInput = document.getElementById('vllmUrl');
  
  if (vllmUrlInput && typeof testVLLMUrl === 'function') {
    debouncedTestVLLMUrl = debounce(testVLLMUrl, 1000);
    vllmUrlInput.addEventListener('input', () => {
      // Save config when URL changes
      saveConfig();
      if (debouncedTestVLLMUrl) {
        debouncedTestVLLMUrl();
      }
    });
  }
  
  // Token is now retrieved automatically, no need to save it
}


async function prepareVLLM() {
  if (!selectedProfile || !sshConfig.host || !sshTested || !vllmUrlTested || !vllmUrlValid) {
    return;
  }
  
  const btn = document.getElementById('prepareBtn');
  const statusDiv = document.getElementById('prepareStatus');
  const stepsList = document.getElementById('stepsList');
  const errorsList = document.getElementById('errorsList');
  
  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span> Preparing...';
  statusDiv.innerHTML = '';
  stepsList.style.display = 'none';
  errorsList.style.display = 'none';
  
  try {
    // Read directly from input fields to ensure we use current values
    const sshHostInput = document.getElementById('sshHost');
    const sshPortInput = document.getElementById('sshPort');
    
    const vllmUrl = vllmUrlInput ? vllmUrlInput.value.trim() : '';
    
    if (!vllmUrl) {
      showStatus('prepareStatus', '✗ Please enter Inference Server HTTP API URL', 'error');
      return;
    }
    
    // Get current SSH values from input fields
    let sshHost = sshHostInput ? sshHostInput.value.trim() : '';
    let sshPort = parseInt(sshPortInput ? sshPortInput.value : '22') || 22;
    
    // Process SSH host (strip port if present)
    if (sshHost.includes(':')) {
      const parts = sshHost.split(':');
      sshHost = parts[0].trim();
      if (parts.length > 1 && parts[1].trim()) {
        const extractedPort = parseInt(parts[1].trim());
        if (!isNaN(extractedPort)) {
          sshPort = extractedPort;
        }
      }
    }
    
    // Update sshConfig with current values
    sshConfig.host = sshHost;
    sshConfig.port = sshPort;
    
    const result = await window.electronAPI.prepareVLLM({
      host: sshHost,
      port: sshPort,
      username: 'root',
      profileName: selectedProfile.name,
      baseModel: selectedProfile.baseModel,
      versions: profileVersions,
      vllmUrl: vllmUrl,
    });
    
    // Display steps
    if (result.steps && result.steps.length > 0) {
      stepsList.innerHTML = result.steps.map(step => 
        `<div class="step-item">${step}</div>`
      ).join('');
      stepsList.style.display = 'block';
    }
    
    // Display errors
    if (result.errors && result.errors.length > 0) {
      errorsList.innerHTML = result.errors.map(error => 
        `<div class="error-item">✗ ${error}</div>`
      ).join('');
      errorsList.style.display = 'block';
    }
    
    if (result.success) {
      showStatus('prepareStatus', '✓ vLLM preparation completed successfully!', 'success');
      // Show chat interface
      await showChatInterface();
    } else {
      showStatus('prepareStatus', '✗ vLLM preparation completed with errors. See details below.', 'error');
    }
  } catch (error) {
    showStatus('prepareStatus', '✗ Error: ' + error.message, 'error');
  } finally {
    updateButtonStates(); // This will set the correct disabled state
    btn.textContent = 'Prepare Inference Server with Selected Model';
  }
}

// Update conversation summary incrementally using LLM
async function updateIncrementalSummary(newUserMessage, newAssistantResponse, currentSummary) {
  try {
    console.log('\n[SUMMARY] ========================================');
    console.log('[SUMMARY] Updating conversation summary incrementally...');
    
    // Build the summary update prompt in a clear, structured way
    let summaryPrompt = '';
    
    if (currentSummary && currentSummary.trim()) {
      summaryPrompt = `Previous conversation summary:\n\n${currentSummary}\n\n`;
    }
    
    // Get profile name for summary prompt
    const profileName = (modelInfo && modelInfo.profileName) 
      ? modelInfo.profileName 
      : (selectedProfile && selectedProfile.name) 
        ? selectedProfile.name 
        : 'Assistant';
    summaryPrompt += `New exchange:\n\nUser: ${newUserMessage}\n\n${profileName}: ${newAssistantResponse}\n\n`;
    summaryPrompt += `Please provide an updated summary that:\n`;
    summaryPrompt += `- Incorporates the new exchange above\n`;
    
    if (currentSummary && currentSummary.trim()) {
      summaryPrompt += `- Preserves important context from the previous summary\n`;
    }
    
    summaryPrompt += `- Is concise (2-3 sentences)\n`;
    summaryPrompt += `- Focuses on main topics, decisions, and key information\n\n`;
    summaryPrompt += `Updated summary:`;
    
    console.log('[SUMMARY] Summary update prompt:');
    console.log('----------------------------------------');
    console.log(summaryPrompt);
    console.log('----------------------------------------');
    
    // Send summary generation request to inference server
    // The sendChatMessage function will format it properly with system/user messages
    const summaryResult = await window.electronAPI.sendChatMessage({
      message: summaryPrompt,
      version: 'base', // Use base model for summary generation
      prependedText: '', // No prepended text for summary
      useSummary: false, // Don't include summary in summary generation
      conversationSummary: '', // No summary context for summary generation
      conversationHistory: [] // Empty history for summary generation
    });
    
    if (summaryResult.success && summaryResult.response) {
      conversationSummary = summaryResult.response.trim();
      console.log('[SUMMARY] Updated summary:');
      console.log('----------------------------------------');
      console.log(conversationSummary);
      console.log('----------------------------------------');
      console.log('[SUMMARY] ========================================\n');
      
      // Save config to persist the updated summary
      saveConfig();
    } else {
      console.error('[SUMMARY] Failed to generate summary:', summaryResult.error);
      // Fallback: create a simple summary from recent messages
      const recentMessages = conversationHistory.slice(-4); // Last 4 messages (2 exchanges)
      const profileName = (modelInfo && modelInfo.profileName) 
        ? modelInfo.profileName 
        : (selectedProfile && selectedProfile.name) 
          ? selectedProfile.name 
          : 'Assistant';
      conversationSummary = recentMessages
        .map(m => {
          const roleLabel = m.role === 'user' ? 'User' : profileName;
          const content = m.content.length > 150 ? m.content.substring(0, 150) + '...' : m.content;
          return `${roleLabel}: ${content}`;
        })
        .join('\n\n');
      console.log('[SUMMARY] Using fallback summary method');
      
      // Save config to persist the fallback summary
      saveConfig();
    }
  } catch (error) {
    console.error('[SUMMARY] Error updating summary:', error);
      // Fallback: create a simple summary from recent messages
      const recentMessages = conversationHistory.slice(-4);
      const profileName = (modelInfo && modelInfo.profileName) 
        ? modelInfo.profileName 
        : (selectedProfile && selectedProfile.name) 
          ? selectedProfile.name 
          : 'Assistant';
      conversationSummary = recentMessages
        .map(m => {
          const roleLabel = m.role === 'user' ? 'User' : profileName;
          const content = m.content.length > 150 ? m.content.substring(0, 150) + '...' : m.content;
          return `${roleLabel}: ${content}`;
        })
        .join('\n\n');
    
    // Save config to persist the error fallback summary
    saveConfig();
  }
}

async function showChatInterface() {
  try {
    console.log('[CHAT] Attempting to show chat interface...');
    // Get model info
    modelInfo = await window.electronAPI.getModelInfo();
    console.log('[CHAT] Model info received:', modelInfo ? 'present' : 'null');
    if (!modelInfo) {
      console.error('[CHAT] No model info available - cannot show chat interface');
      showStatus('sshStatus', 'Error: Model info not available. Please check terminal logs.', 'error');
      return;
    }
    console.log('[CHAT] Model info:', JSON.stringify(modelInfo, null, 2));

    // Populate version selector
    const versionSelect = document.getElementById('versionSelect');
    versionSelect.innerHTML = '<option value="base">Base Model</option>';
    
    if (modelInfo.versions && modelInfo.versions.length > 0) {
      modelInfo.versions.forEach(v => {
        const option = document.createElement('option');
        option.value = `V${v.version}`;
        option.textContent = `Version ${v.version}`;
        versionSelect.appendChild(option);
      });
    }
    
    // Restore saved version selection if available
    if (window._pendingVersionSelection) {
      const savedVersion = window._pendingVersionSelection;
      console.log('[CONFIG] Restoring saved version selection:', savedVersion);
      if (versionSelect.querySelector(`option[value="${savedVersion}"]`)) {
        versionSelect.value = savedVersion;
      } else {
        console.warn('[CONFIG] Saved version not found, using default "base"');
        versionSelect.value = 'base';
      }
      delete window._pendingVersionSelection;
    }

    // Show chat interface
    document.getElementById('chatInterface').style.display = 'block';
    
    // Scroll to chat interface
    document.getElementById('chatInterface').scrollIntoView({ behavior: 'smooth' });
    
    // Add welcome message
    addMessage('assistant', 'vLLM is ready! You can now start chatting. Select a model version and type your message.');
  } catch (error) {
    console.error('Error showing chat interface:', error);
    showStatus('prepareStatus', 'Error loading chat interface: ' + error.message, 'error');
  }
}

function addMessage(role, content, addToHistory = true) {
  const messagesDiv = document.getElementById('chatMessages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;
  
  const header = document.createElement('div');
  header.className = 'message-header';
  
  // Use profile name for assistant messages, fallback to "Assistant" if no profile
  if (role === 'user') {
    header.textContent = 'You';
  } else {
    // Try to get profile name from modelInfo first, then selectedProfile
    const profileName = (modelInfo && modelInfo.profileName) 
      ? modelInfo.profileName 
      : (selectedProfile && selectedProfile.name) 
        ? selectedProfile.name 
        : 'Assistant';
    header.textContent = profileName;
  }
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;
  
  messageDiv.appendChild(header);
  messageDiv.appendChild(contentDiv);
  messagesDiv.appendChild(messageDiv);
  
  // Scroll to bottom
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  
  // Add to conversation history if requested
  if (addToHistory) {
    conversationHistory.push({
      role: role,
      content: content
    });
  }
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  // Get the user's actual typed message - this is what will be sent as the user message
  const message = input.value.trim();
  
  if (!message) {
    return;
  }

  if (!modelInfo) {
    alert('Model not prepared. Please prepare the inference server first.');
    return;
  }

  const sendBtn = document.getElementById('sendBtn');
  const versionSelect = document.getElementById('versionSelect');
  const systemMessage = document.getElementById('prependedText').value.trim();

  // Log what we're sending to verify it's correct
  console.log('[SEND] User message (exactly as typed):', message);
  console.log('[SEND] Message length:', message.length);
  console.log('[SEND] System message length:', systemMessage.length);
  if (systemMessage.length > 1000) {
    console.warn('[SEND] WARNING: System message is very long (' + systemMessage.length + ' chars). Make sure this is intentional.');
  }

  // Disable input while sending
  input.disabled = true;
  sendBtn.disabled = true;
  sendBtn.textContent = 'Sending...';

  // Add user message to chat (display only, not in history yet)
  addMessage('user', message, false);
  
  // Clear input immediately after capturing the message
  input.value = '';

  try {
    // Fix 3: Normalize conversation history before sending to enforce strict role alternation
    const normalizedHistory = normalizeConversation(conversationHistory);
    
    // Send ONLY the user's typed message - system message and summary are handled separately as system messages
    // Conversation history is always sent (always enabled)
    const result = await window.electronAPI.sendChatMessage({
      message: message, // This is the user's actual typed message, nothing else
      version: versionSelect.value,
      prependedText: systemMessage, // System message from user input
      useSummary: false, // Always false - history is always enabled, no summary feature
      conversationSummary: '', // Not used - history is always enabled
      conversationHistory: normalizedHistory // Always send conversation history
    });

    if (result.success) {
      // Note: System messages are added per-request, not stored in history
      // Only user and assistant messages are stored in conversation history
      
      // Add user message to history (clean, without system message formatting)
      conversationHistory.push({
        role: 'user',
        content: message.trim()
      });
      
      // Save config to persist conversation history
      saveConfig();
      
      // Fix 1: Only append assistant messages with real content (non-empty)
      const assistantText = result.response?.trim() ?? "";
      if (assistantText.length > 0) {
        // Add assistant response to UI
        addMessage('assistant', assistantText);
        
        // Add assistant response to history (only if non-empty)
        conversationHistory.push({
          role: 'assistant',
          content: assistantText
        });
        
        // Save config to persist conversation history
        saveConfig();
      } else {
        // Empty response - show in UI but don't add to history
        addMessage('assistant', '(Empty response)');
        console.warn('[CHAT] Received empty assistant response - not adding to history');
      }
    } else {
      // Fix 2: Never append assistant messages on error - errors are out-of-band events
      const errorMsg = result.error || 'Unknown error';
      console.error('[CHAT] Chat error:', errorMsg);
      addMessage('assistant', 'Error: ' + errorMsg);
      // Do NOT mutate conversation history here - errors are not part of the conversation
    }
  } catch (error) {
    // Fix 2: Never append assistant messages on error - errors are out-of-band events
    console.error('[CHAT] Error sending message:', error);
    addMessage('assistant', 'Error: ' + error.message);
    // Do NOT mutate conversation history here - errors are not part of the conversation
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';
    input.focus();
  }
}

// Update all button states based on profile selection
function updateButtonStates() {
  const hasProfile = !!selectedProfile;
  
  // Update Initialize Inference Environment button
  const testSSHBtn = document.getElementById('testSSHBtn');
  if (testSSHBtn) {
    testSSHBtn.disabled = !hasProfile;
    if (!hasProfile) {
      testSSHBtn.title = 'Please select a model profile first';
    } else {
      testSSHBtn.title = '';
    }
  }
  
  // Update Send button
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) {
    sendBtn.disabled = !hasProfile;
    if (!hasProfile) {
      sendBtn.title = 'Please select a model profile first';
    } else {
      sendBtn.title = '';
    }
  }
  
}


function showStatus(elementId, message, type) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  element.className = `status ${type}`;
  element.textContent = message;
}


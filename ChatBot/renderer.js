let selectedProfile = null;
let profileVersions = [];
let sshConfig = { host: '', port: 22 };
let sshTested = false;
let modelInfo = null;
let conversationHistory = [];
let conversationSummary = '';
let configLoaded = false; // Track if config has been loaded to prevent overwriting user input

// Save configuration to file
async function saveConfig() {
  try {
    // Read directly from input fields to ensure we save current values, not stale sshConfig
    const sshHostInput = document.getElementById('sshHost');
    const sshPortInput = document.getElementById('sshPort');
    const vllmUrlInput = document.getElementById('vllmUrl');
    const prependedTextInput = document.getElementById('prependedText');
    const useSummaryCheckbox = document.getElementById('useSummaryCheckbox');
    
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
    
    const config = {
      sshHost: sshHost,
      sshPort: sshPort,
      vllmUrl: vllmUrlInput ? vllmUrlInput.value.trim() : '',
      prependedText: prependedTextInput ? prependedTextInput.value.trim() : '',
      useSummary: useSummaryCheckbox ? useSummaryCheckbox.checked : false,
      conversationSummary: conversationSummary || '', // Save the conversation summary
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
      
      // Load vLLM URL (only if field is empty, to avoid overwriting user input)
      // Reset vLLM URL test flags since we're loading from saved config
      // Don't test the URL automatically on startup - user should test when ready
      vllmUrlTested = false;
      vllmUrlValid = false;
      
      if (config.vllmUrl) {
        const vllmUrlInput = document.getElementById('vllmUrl');
        if (vllmUrlInput && !vllmUrlInput.value.trim()) {
          vllmUrlInput.value = config.vllmUrl;
          // Don't test automatically - user will test when they're ready
        }
      }
      
      // Load prepended text
      if (config.prependedText !== undefined) {
        const prependedTextInput = document.getElementById('prependedText');
        if (prependedTextInput) {
          prependedTextInput.value = config.prependedText;
        }
      }
      
      // Load use summary option
      if (config.useSummary !== undefined) {
        const useSummaryCheckbox = document.getElementById('useSummaryCheckbox');
        if (useSummaryCheckbox) {
          useSummaryCheckbox.checked = config.useSummary;
        }
      }
      
      // Load conversation summary
      if (config.conversationSummary !== undefined && config.conversationSummary) {
        conversationSummary = config.conversationSummary;
        console.log('[CONFIG] Loaded conversation summary (length:', conversationSummary.length, 'chars)');
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
  
  await loadProfiles();
  await checkVastApiKey();
  await checkSSHKey();
  
  // Set up event listeners
  document.getElementById('profileSelect').addEventListener('change', handleProfileChange);
  
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
  
  document.getElementById('testSSHBtn').addEventListener('click', testSSHConnection);
  document.getElementById('prepareBtn').addEventListener('click', prepareVLLM);
  
  // Set up vLLM URL listeners after functions are defined (see end of file)
  setTimeout(setupVLLMUrlListeners, 0);
  document.getElementById('sendBtn').addEventListener('click', sendMessage);
  
  // Set up listeners for prepended text and summary checkbox
  const prependedTextInput = document.getElementById('prependedText');
  const useSummaryCheckbox = document.getElementById('useSummaryCheckbox');
  const clearSummaryBtn = document.getElementById('clearSummaryBtn');
  
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
  if (useSummaryCheckbox) {
    useSummaryCheckbox.addEventListener('change', () => {
      saveConfig();
    });
  }
  if (clearSummaryBtn) {
    clearSummaryBtn.addEventListener('click', () => {
      conversationSummary = '';
      conversationHistory = [];
      const chatMessages = document.getElementById('chatMessages');
      if (chatMessages) {
        chatMessages.innerHTML = '';
      }
      console.log('[CHAT] Summary and conversation history cleared');
      
      // Save config to persist the cleared summary
      saveConfig();
      // Optionally add a message indicating the conversation was reset
      if (chatMessages) {
        const resetMsg = document.createElement('div');
        resetMsg.className = 'message assistant-message';
        resetMsg.textContent = 'Conversation summary and history cleared. Starting fresh conversation.';
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
  
  // Save config when profile changes
  if (profileName) {
    saveConfig();
  }
  
  if (!profileName) {
    selectedProfile = null;
    profileVersions = [];
    document.getElementById('profileInfo').textContent = '';
    document.getElementById('versionsSection').style.display = 'none';
    updatePrepareButton();
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
    
    document.getElementById('profileInfo').textContent = `Base Model: ${baseModel}`;
    
    // Load versions
    profileVersions = await window.electronAPI.getProfileVersions(profileName);
    displayVersions();
    updatePrepareButton();
  } catch (error) {
    console.error('Error loading profile versions:', error);
    showStatus('profileInfo', `Error loading versions: ${error.message}`, 'error');
  }
}

function displayVersions() {
  const versionsList = document.getElementById('versionsList');
  const versionsSection = document.getElementById('versionsSection');
  
  if (profileVersions.length === 0) {
    versionsList.innerHTML = '<div class="version-item">No versions found for this profile</div>';
    versionsSection.style.display = 'block';
    return;
  }
  
  versionsList.innerHTML = profileVersions.map(v => 
    `<div class="version-item">✓ Version ${v.version} (adapter found)</div>`
  ).join('');
  
  versionsSection.style.display = 'block';
}

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
  updatePrepareButton();
  
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

async function testSSHConnection() {
  const btn = document.getElementById('testSSHBtn');
  const statusDiv = document.getElementById('sshStatus');
  const sshHostInput = document.getElementById('sshHost');
  const sshPortInput = document.getElementById('sshPort');
  
  // Read directly from input fields to ensure we use current values, not stale sshConfig
  let hostValue = sshHostInput ? sshHostInput.value.trim() : '';
  let portValue = parseInt(sshPortInput ? sshPortInput.value : '22') || 22;
  
  // Process host value (strip port if present)
  if (hostValue.includes(':')) {
    const parts = hostValue.split(':');
    hostValue = parts[0].trim();
    if (parts.length > 1 && parts[1].trim()) {
      const extractedPort = parseInt(parts[1].trim());
      if (!isNaN(extractedPort)) {
        portValue = extractedPort;
      }
    }
  }
  
  if (!hostValue) {
    showStatus('sshStatus', 'Please enter SSH host/IP', 'error');
    return;
  }
  
  // Update sshConfig with current values
  sshConfig.host = hostValue;
  sshConfig.port = portValue;
  
  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span> Testing...';
  statusDiv.innerHTML = '';
  
  try {
    const result = await window.electronAPI.testSSH({
      host: hostValue,
      port: portValue,
      username: 'root'
    });
    
    if (result.success) {
      showStatus('sshStatus', '✓ ' + result.message, 'success');
      sshTested = true;
    } else {
      showStatus('sshStatus', '✗ ' + result.message, 'error');
      sshTested = false;
    }
  } catch (error) {
    showStatus('sshStatus', '✗ Error: ' + error.message, 'error');
    sshTested = false;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Test SSH Connection';
    updatePrepareButton();
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

// Test vLLM URL connectivity
let vllmUrlTested = false;
let vllmUrlValid = false;

// Initialize debounced function after testVLLMUrl is defined
async function testVLLMUrl() {
  const vllmUrlInput = document.getElementById('vllmUrl');
  const statusDiv = document.getElementById('vllmUrlStatus');
  
  // Read directly from input field to ensure we use current value
  const vllmUrl = vllmUrlInput ? vllmUrlInput.value.trim() : '';
  
  if (!vllmUrl) {
    statusDiv.innerHTML = '<span style="color: #999;">Enter a vLLM URL to test</span>';
    vllmUrlTested = false;
    vllmUrlValid = false;
    updatePrepareButton();
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
      statusDiv.innerHTML = `<span style="color: #4caf50;">✓ ${result.message}</span>`;
      if (result.models && result.models.length > 0) {
        statusDiv.innerHTML += `<br><small style="color: #666;">Available models: ${result.models.join(', ')}</small>`;
      }
      vllmUrlTested = true;
      vllmUrlValid = true;
    } else {
      statusDiv.innerHTML = `<span style="color: #f44336;">✗ ${result.message}</span>`;
      if (result.details) {
        statusDiv.innerHTML += `<br><small style="color: #666;">${result.details}</small>`;
      }
      vllmUrlTested = true;
      vllmUrlValid = false;
    }
  } catch (error) {
    statusDiv.innerHTML = `<span style="color: #f44336;">✗ Error: ${error.message}</span>`;
    vllmUrlTested = true;
    vllmUrlValid = false;
  } finally {
    updatePrepareButton();
  }
}

// Set up vLLM URL event listeners (called after functions are defined)
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
    const vllmUrlInput = document.getElementById('vllmUrl');
    const sshHostInput = document.getElementById('sshHost');
    const sshPortInput = document.getElementById('sshPort');
    
    const vllmUrl = vllmUrlInput ? vllmUrlInput.value.trim() : '';
    
    if (!vllmUrl) {
      showStatus('prepareStatus', '✗ Please enter vLLM HTTP API URL', 'error');
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
    btn.disabled = false;
    btn.textContent = 'Prepare vLLM with Selected Model';
  }
}

// Update conversation summary incrementally using LLM
async function updateIncrementalSummary(newUserMessage, newAssistantResponse, currentSummary) {
  try {
    console.log('\n[SUMMARY] ========================================');
    console.log('[SUMMARY] Updating conversation summary incrementally...');
    
    // Build the summary update prompt
    let summaryPrompt = '';
    if (currentSummary && currentSummary.trim()) {
      summaryPrompt = `Given the previous conversation summary:\n\n${currentSummary}\n\n`;
    } else {
      summaryPrompt = 'Starting a new conversation summary.\n\n';
    }
    
    summaryPrompt += `And this new exchange:\n\nUser: ${newUserMessage}\n\nAssistant: ${newAssistantResponse}\n\n`;
    summaryPrompt += `Please provide an updated summary that incorporates this new information`;
    
    if (currentSummary && currentSummary.trim()) {
      summaryPrompt += ` while preserving important context from the previous summary`;
    }
    
    summaryPrompt += `. Keep it concise (2-3 sentences). Focus on the main topics, decisions, and key information discussed.`;
    
    console.log('[SUMMARY] Summary update prompt:');
    console.log('----------------------------------------');
    console.log(summaryPrompt);
    console.log('----------------------------------------');
    
    // Send summary generation request to vLLM
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
      conversationSummary = recentMessages
        .map(m => {
          const roleLabel = m.role === 'user' ? 'User' : 'Assistant';
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
    conversationSummary = recentMessages
      .map(m => {
        const roleLabel = m.role === 'user' ? 'User' : 'Assistant';
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
    // Get model info
    modelInfo = await window.electronAPI.getModelInfo();
    if (!modelInfo) {
      console.error('No model info available');
      return;
    }

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
  header.textContent = role === 'user' ? 'You' : 'Assistant';
  
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
  const message = input.value.trim();
  
  if (!message) {
    return;
  }

  if (!modelInfo) {
    alert('Model not prepared. Please prepare vLLM first.');
    return;
  }

  const sendBtn = document.getElementById('sendBtn');
  const versionSelect = document.getElementById('versionSelect');
  const prependedText = document.getElementById('prependedText').value.trim();
  const useSummary = document.getElementById('useSummaryCheckbox').checked;

  // Disable input while sending
  input.disabled = true;
  sendBtn.disabled = true;
  sendBtn.textContent = 'Sending...';

  // Add user message to chat (display only, not in history yet)
  addMessage('user', message, false);
  
  // Clear input
  input.value = '';

  try {
    const result = await window.electronAPI.sendChatMessage({
      message: message,
      version: versionSelect.value,
      prependedText: prependedText,
      useSummary: useSummary,
      conversationSummary: conversationSummary,
      conversationHistory: conversationHistory // Send full history
    });

    if (result.success) {
      // Add user message to history (matching what backend sent: structured format)
      // The backend structures it as CONTEXT/SUMMARY/PROMPT, so we'll match that format
      let userMessageContent = '';
      const sections = [];
      
      if (prependedText && prependedText.trim()) {
        sections.push(`### CONTEXT ###\n\nSome context for you to know:\n${prependedText}`);
      }
      
      if (useSummary && conversationSummary && conversationSummary.trim()) {
        sections.push(`### SUMMARY ###\n\nWe've been talking about:\n${conversationSummary}`);
      }
      
      sections.push(`### PROMPT ###\n\n${message}`);
      userMessageContent = sections.join('\n\n');
      
      conversationHistory.push({
        role: 'user',
        content: userMessageContent
      });
      
      // Add assistant response
      addMessage('assistant', result.response);
      
      // Add assistant response to history
      conversationHistory.push({
        role: 'assistant',
        content: result.response
      });
      
      // Update conversation summary if enabled (using LLM-based incremental approach)
      if (useSummary) {
        await updateIncrementalSummary(message, result.response, conversationSummary);
      }
    } else {
      addMessage('assistant', 'Error: ' + (result.error || 'Unknown error'));
    }
  } catch (error) {
    console.error('Error sending message:', error);
    addMessage('assistant', 'Error: ' + error.message);
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';
    input.focus();
  }
}

function updatePrepareButton() {
  const btn = document.getElementById('prepareBtn');
  const canPrepare = selectedProfile && 
                     sshConfig.host && 
                     sshTested && 
                     vllmUrlTested && 
                     vllmUrlValid &&
                     profileVersions.length > 0;
  
  btn.disabled = !canPrepare;
  
  if (!selectedProfile) {
    btn.title = 'Please select a profile first';
  } else if (!sshConfig.host || !sshTested) {
    btn.title = 'Please test SSH connection first';
  } else if (!vllmUrlTested || !vllmUrlValid) {
    btn.title = 'Please test vLLM URL first';
  } else if (profileVersions.length === 0) {
    btn.title = 'No versions available for this profile';
  } else {
    btn.title = '';
  }
  
  btn.disabled = !canPrepare;
}

function showStatus(elementId, message, type) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  element.className = `status ${type}`;
  element.textContent = message;
}


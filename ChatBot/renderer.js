let selectedProfile = null;
let profileVersions = [];
let sshConfig = { host: '', port: 22 };
let sshTested = false;
let modelInfo = null;
let conversationHistory = [];
let conversationSummary = '';
let configLoaded = false; // Track if config has been loaded to prevent overwriting user input

let behaviorPacks = null; // Loaded behavior packs configuration

// Helper to normalize new behavior pack schema into expected fields
function normalizeBehaviorPack(pack) {
  if (!pack) return pack;

  // Map generation defaults
  pack.generation_settings = {
    temperature: pack.generation_defaults?.verbosity === 'high' ? 0.7 : 0.5,
    max_tokens: pack.generation_defaults?.soft_max_tokens || 800,
    min_length_chars: pack.generation_defaults?.min_paragraphs
      ? pack.generation_defaults.min_paragraphs * 200
      : undefined
  };

  // Map opening / style rules
  pack.style_rules = {
    forbidden_opening_patterns: pack.opening_rules?.disallowed_openings || [],
    preferred_openings: pack.opening_rules?.preferred_openings || [],
    forbidden_terms: pack.profile_intent?.avoid || []
  };

  // Flatten triggers → exemplars
  pack.exemplars = {};
  if (pack.triggers) {
    Object.entries(pack.triggers).forEach(([key, trigger]) => {
      if (trigger.exemplars) {
        pack.exemplars[key] = {
          when: trigger.when ? Object.keys(trigger.when) : [],
          text: trigger.exemplars
        };
      }
    });
  }

  return pack;
}

// === Behavior-pack–driven style helpers ===

// Build a system message from the loaded behavior pack, if present
function buildSystemFromBehaviorPack() {
  if (!behaviorPacks) return '';
  const parts = [];
  if (behaviorPacks.profile_intent?.identity) parts.push(behaviorPacks.profile_intent.identity);
  if (behaviorPacks.profile_intent?.stance) parts.push(behaviorPacks.profile_intent.stance);
  if (Array.isArray(behaviorPacks.profile_intent?.avoid) && behaviorPacks.profile_intent.avoid.length) {
    parts.push(`Avoid: ${behaviorPacks.profile_intent.avoid.join(', ')}`);
  }
  if (behaviorPacks.tone_guidance?.voice) parts.push(`Voice: ${behaviorPacks.tone_guidance.voice}`);
  if (behaviorPacks.tone_guidance?.opening_style) parts.push(`Opening style: ${behaviorPacks.tone_guidance.opening_style}`);
  if (Array.isArray(behaviorPacks.tone_guidance?.preferred_openings) && behaviorPacks.tone_guidance.preferred_openings.length) {
    parts.push(`Preferred openings: ${behaviorPacks.tone_guidance.preferred_openings.join(' | ')}`);
  }
  return parts.join('\n');
}
function getStyleRules() {
  return behaviorPacks?.style_rules || {};
}

function violatesOpeningRules(text) {
  if (!text) return false;
  const rules = getStyleRules();
  const patterns = rules.forbidden_opening_patterns || [];
  const opening = text.trim().slice(0, 200).toLowerCase();
  return patterns.some(p => {
    try {
      return new RegExp(p, 'i').test(opening);
    } catch {
      return false;
    }
  });
}

function containsForbiddenTerms(text) {
  if (!text) return false;
  const rules = getStyleRules();
  const terms = rules.forbidden_terms || [];
  return terms.some(term =>
    new RegExp(`\\b${term}\\b`, 'i').test(text)
  );
}
let pendingExemplarSystemText = null; // One-shot exemplar appended to the next request as system text
let exemplarRotationIndex = {}; // Track rotation index for each trigger
let usedTriggers = new Set();
let isSending = false; // Prevent accidental double-send / concurrent sendMessage calls

// Load behavior packs configuration for a specific profile
async function loadBehaviorPacks(profileName) {
  try {
    const result = await window.electronAPI.loadBehaviorPacks(profileName);
    if (result && result.success && result.data) {
      behaviorPacks = normalizeBehaviorPack(result.data);
      const exemplarCount = Object.keys(behaviorPacks.exemplars || {}).length;
      console.log('[BEHAVIOR] Behavior packs loaded:', exemplarCount, 'exemplar(s)');

      // Check if pack is empty and show warning
      if (result.isEmpty) {
        console.warn('[BEHAVIOR] Behavior pack is empty (no exemplars with content)');
        showBehaviorPackWarning();
      } else {
        hideBehaviorPackWarning();
      }

      return behaviorPacks;
    } else {
      console.warn('[BEHAVIOR] Failed to load behavior packs:', result?.error || 'Unknown error');
      showBehaviorPackWarning();
      return null;
    }
  } catch (error) {
    console.error('[BEHAVIOR] Error loading behavior packs:', error);
    showBehaviorPackWarning();
    return null;
  }
}

// Show warning if behavior pack is empty
function showBehaviorPackWarning() {
  // Check if warning element already exists
  let warningDiv = document.getElementById('behaviorPackWarning');
  if (!warningDiv) {
    // Create warning element
    warningDiv = document.createElement('div');
    warningDiv.id = 'behaviorPackWarning';
    warningDiv.style.cssText = 'background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px; margin: 15px 0; color: #856404;';
    warningDiv.innerHTML = '<strong>⚠ Warning:</strong> Behavior pack is empty. No exemplars are configured for this profile. Edit <code>behavior_packs.json</code> in the profile folder to add exemplars.';
    
    // Insert at the top of chat interface, after the section title
    const chatInterface = document.getElementById('chatInterface');
    if (chatInterface) {
      // Find the section-title element and insert after it
      const sectionTitle = chatInterface.querySelector('.section-title');
      if (sectionTitle && sectionTitle.nextSibling) {
        chatInterface.insertBefore(warningDiv, sectionTitle.nextSibling);
      } else if (sectionTitle) {
        sectionTitle.parentNode.insertBefore(warningDiv, sectionTitle.nextSibling);
      } else {
        // Fallback: insert at the beginning
        chatInterface.insertBefore(warningDiv, chatInterface.firstChild);
      }
    }
  }
  warningDiv.style.display = 'block';
}

// Hide warning if behavior pack has content
function hideBehaviorPackWarning() {
  const warningDiv = document.getElementById('behaviorPackWarning');
  if (warningDiv) {
    warningDiv.style.display = 'none';
  }
}

// Get exemplar text for a given trigger condition
// Supports two formats:
//  1) explicit mapping: exemplars[trigger] = { text: ... }
//  2) when-based matching: exemplars[key] = { when: [..triggers..], text: ... }
// Rotates across all matching exemplar texts per trigger.
function getExemplarForTrigger(trigger) {
  if (!behaviorPacks || !behaviorPacks.exemplars) return null;

  const exemplarsObj = behaviorPacks.exemplars;
  const matches = [];

  for (const [key, entry] of Object.entries(exemplarsObj)) {
    if (!entry) continue;

    // Match rule A: explicit key equals trigger
    const keyMatch = key === trigger;

    // Match rule B: `when` includes trigger
    const whenList = Array.isArray(entry.when) ? entry.when : (typeof entry.when === 'string' ? [entry.when] : []);
    const whenMatch = whenList.includes(trigger);

    if (!keyMatch && !whenMatch) continue;

    if (!entry.text) continue;
    const texts = Array.isArray(entry.text) ? entry.text : [entry.text];
    for (const t of texts) {
      if (typeof t === 'string' && t.trim()) matches.push(t.trim());
    }
  }

  if (matches.length === 0) return null;

  const index = exemplarRotationIndex[trigger] ?? 0;
  const selected = matches[index % matches.length];
  exemplarRotationIndex[trigger] = index + 1;
  return selected;
}

// Inject exemplar as one-shot system text (not shown in UI, not saved to history)
// Allows per-turn triggers to fire repeatedly; one-shots only once per session.
function injectExemplar(trigger) {
  const exemplarText = getExemplarForTrigger(trigger);
  if (!exemplarText) return false;

  if (containsForbiddenTerms(exemplarText)) {
    console.warn('[BEHAVIOR] Exemplar blocked due to forbidden terms (behavior pack rules)');
    return false;
  }

  console.log(`[BEHAVIOR] Queuing exemplar for trigger: ${trigger}`);
  pendingExemplarSystemText = exemplarText;
  console.log('[BEHAVIOR] Applied exemplar:', trigger);
  return true;
}

// (Behavior-pack–driven forbidden checks and opening rules now used. See helpers above.)

// Detect lack of concrete, practical practice in assistant responses
function lacksConcretePractice(text) {
  if (!text) return true;
  return !/(student work|common assessment|lesson plan|exit ticket|protocol|agenda|artifact|look at|bring|examine|decide|try next|instructional move|next meeting)/i.test(text);
}

// Detect if a response is too brief to be useful (behavior-pack–driven)
function isTooBrief(text) {
  if (!text) return true;
  const minChars = behaviorPacks?.generation_settings?.min_length_chars;
  if (!minChars) return false;
  return text.trim().length < minChars;
}

// Build a rewrite instruction that forces plain-language, example-driven coaching
function buildRewriteInstruction(userMessage, badAssistantText) {
  return (
    `Rewrite your last response so it follows the system instructions.\n\n` +
    `Hard rules:\n` +
    `- Do NOT use labels or headings like "Current step", "Diagnosis", "Next step", "Reflection gap", "framework", "stage", or "diagram".\n` +
    `- Speak as a thoughtful human colleague. Use concrete, practical examples (routines, artifacts, questions teams examine, decisions teams make).\n` +
    `- Ground it in the leader's situation and explain your reasoning plainly.\n` +
    `- Do NOT end with questions unless they truly help decide the next move.\n\n` +
    `User message:\n${userMessage}\n\n` +
    `Your previous (non-compliant) draft:\n${badAssistantText}\n\n` +
    `Return ONLY the revised answer.`
  );
}

// Wrap system content to reduce the model mistaking it for user content
function wrapSystemBlocks(userSystem, exemplar) {
  const blocks = [];
  if (exemplar && exemplar.trim()) {
    blocks.push(`<SYSTEM_EXEMPLAR>\n${exemplar.trim()}\n</SYSTEM_EXEMPLAR>`);
  }
  if (userSystem && userSystem.trim()) {
    blocks.push(`<SYSTEM_PREPEND>\n${userSystem.trim()}\n</SYSTEM_PREPEND>`);
  }
  return blocks.join('\n\n');
}

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
  // Behavior packs will be loaded when profile is selected
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
    clearChatBtn.addEventListener('click', async () => {
      conversationHistory = [];
      const chatMessages = document.getElementById('chatMessages');
      if (chatMessages) {
        chatMessages.innerHTML = '';
      }
      console.log('[CHAT] Conversation history cleared');

      // Save config to persist the cleared history
      saveConfig();
      
      // Clear one-shot exemplar before voice_reset injection (one-shot semantics)
      pendingExemplarSystemText = null;
      usedTriggers.clear();
      
      // Load behavior packs for the current profile and inject exemplar for voice_reset if available
      const profileName = modelInfo?.profileName || null;
      await loadBehaviorPacks(profileName);
      if (chatMessages) {
        // Try to inject exemplar for voice_reset, fallback to default message
        if (!injectExemplar('voice_reset')) {
          addMessage('assistant', 'Hello! I\'m glad to be talking with you today!');
        }
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
      conversationHistory: [], // Empty history for summary generation
      temperature: behaviorPacks?.generation_settings?.temperature,
      max_tokens: behaviorPacks?.generation_settings?.max_tokens
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
    
    // Load behavior packs for the current profile and inject exemplar if available
    const profileName = modelInfo?.profileName || null;
    await loadBehaviorPacks(profileName);
    
    // Clear one-shot exemplar before conversation_start injection (one-shot semantics)
    pendingExemplarSystemText = null;
    
    // Try to inject exemplar for conversation_start, fallback to default message
    if (!injectExemplar('conversation_start')) {
      addMessage('assistant', 'Hello! I\'m glad to be talking with you today!');
    }
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
  const userText = input.value.trim();
  if (isSending) {
    console.warn('[SEND] sendMessage() called while a send is already in progress. Ignoring duplicate call.');
    return;
  }

  if (!userText) {
    return;
  }

  if (!modelInfo) {
    alert('Model not prepared. Please prepare the inference server first.');
    return;
  }

  // --- Behavior trigger detection: asked_for_next_steps ---
  function detectAskedForNextSteps(text) {
    const patterns = [
      /what should i do/i,
      /what would you suggest/i,
      /next step/i,
      /how do i move/i,
      /how do we move/i,
      /what comes next/i,
      /what can i try/i
    ];
    return patterns.some(p => p.test(text));
  }

  if (behaviorPacks?.triggers?.asked_for_next_steps) {
    if (detectAskedForNextSteps(userText)) {
      queueTrigger("asked_for_next_steps");
      console.warn("[BEHAVIOR] Triggered asked_for_next_steps from user intent");
    }
  }

  const sendBtn = document.getElementById('sendBtn');
  const versionSelect = document.getElementById('versionSelect');

  // Behavior-pack system intent takes precedence. User prepend text is legacy fallback only.
  const userProvidedSystem = document.getElementById('prependedText').value.trim();
  const behaviorSystem = buildSystemFromBehaviorPack();
  const systemToUse = behaviorSystem || userProvidedSystem;

  const effectiveSystemMessage = wrapSystemBlocks(systemToUse, pendingExemplarSystemText);

  // Log what we're sending to verify it's correct
  console.log('[SEND] User message (exactly as typed):', userText);
  console.log('[SEND] Message length:', userText.length);
  console.log('[SEND] System message length:', effectiveSystemMessage.length);
  console.log('[SEND] One-shot exemplar appended?', !!(pendingExemplarSystemText && pendingExemplarSystemText.trim()));
  if (effectiveSystemMessage.length > 1000) {
    console.warn('[SEND] WARNING: System message is very long (' + effectiveSystemMessage.length + ' chars). Make sure this is intentional.');
  }

  // Disable input while sending
  input.disabled = true;
  sendBtn.disabled = true;
  sendBtn.textContent = 'Sending...';
  isSending = true;

  // Fix possible duplicate user-message rendering
  if (conversationHistory.length > 0) {
    const last = conversationHistory[conversationHistory.length - 1];
    if (last.role === 'user' && last.content === userText) {
      console.warn('[SEND] Duplicate user message detected; skipping UI append.');
    } else {
      addMessage('user', userText, false);
    }
  } else {
    addMessage('user', userText, false);
  }

  // Clear input immediately after capturing the message
  input.value = '';

  // --- Behavior-driven generation settings ---
  const gen = behaviorPacks?.generation_defaults || {};
  let requestedMaxTokens = 512; // safe fallback
  if (gen.soft_max_tokens) {
    requestedMaxTokens = gen.soft_max_tokens;
  }
  let verbosityMultiplier = 1;
  if (gen.verbosity === "high") verbosityMultiplier = 1.5;
  if (gen.verbosity === "low") verbosityMultiplier = 0.7;
  requestedMaxTokens = Math.floor(requestedMaxTokens * verbosityMultiplier);

  try {
    // Send ONLY the user's typed message - system message and summary are handled separately as system messages
    // Conversation history is always sent (always enabled)
    const requestBody = {
      message: userText,
      version: versionSelect.value,
      prependedText: effectiveSystemMessage,
      useSummary: false,
      conversationSummary: '',
      conversationHistory: conversationHistory,
      temperature: behaviorPacks?.generation_settings?.temperature,
      max_tokens: requestedMaxTokens
    };
    const result = await window.electronAPI.sendChatMessage(requestBody);

    if (result.success) {
      // Note: System messages are added per-request, not stored in history
      // Only user and assistant messages are stored in conversation history

      // Save config to persist conversation history
      // Add user message to history (clean, without system message formatting)
      conversationHistory.push({
        role: 'user',
        content: userText
      });
      saveConfig();

      // Only append assistant messages with real content (non-empty)
      let assistantText = result.response?.trim() ?? "";

      // --- Behavior enforcement: examples & reasoning ---
      function lacksConcreteExamples(text) {
        return !/(student work|exit ticket|lesson|team meeting|artifact|assessment|instructional|classroom)/i.test(text);
      }
      function lacksReasoning(text) {
        return !/(because|so that|this helps|which allows|as a result)/i.test(text);
      }
      const genDefaults = behaviorPacks?.generation_defaults || {};
      if (genDefaults.require_examples && lacksConcreteExamples(assistantText)) {
        queueTrigger("expand_into_practice");
        console.warn("[BEHAVIOR] Response lacked concrete examples; queuing expand_into_practice");
      }
      if (genDefaults.explain_reasoning && lacksReasoning(assistantText)) {
        queueTrigger("expand_into_practice");
        console.warn("[BEHAVIOR] Response lacked reasoning; queuing expand_into_practice");
      }
      // --- Behavior enforcement: min_paragraphs ---
      if (genDefaults.min_paragraphs) {
        const paragraphCount = assistantText.split(/\n\s*\n/).length;
        if (paragraphCount < genDefaults.min_paragraphs) {
          queueTrigger("expand_into_practice");
          console.warn("[BEHAVIOR] Response below min_paragraphs; queuing expand_into_practice");
        }
      }

      // 3) Prevent any diagnostic response from reaching UI/history:
      // Only call addMessage('assistant', ...) after all rewrites and checks.
      let handled = false;

      if (assistantText.length === 0) {
        // Empty response - show in UI but don't add to history
        addMessage('assistant', '(Empty response)', false);
        console.warn('[CHAT] Received empty assistant response - not adding to history');
        handled = true;
      }

      // 2) Change diagnostic-opening handling to force a FULL rewrite (not just opening rewrite)
      if (!handled && violatesOpeningRules(assistantText)) {
        injectExemplar('diagnostic_opening');
        const diagnosticSystem = wrapSystemBlocks(systemToUse, pendingExemplarSystemText);
        const fullRewritePrompt =
`Rewrite the response below as a thoughtful coaching reply.
- Do NOT diagnose, label, or classify the leader or their situation.
- Open by reflecting what the leader is noticing or experiencing.
- Use concrete examples of teacher team practice (routines, artifacts, questions, decisions).
- Explain why those moves would help practice change.
- Avoid headings, stages, or shorthand.

Original response:
${assistantText}

Return ONLY the revised response.
`;
        try {
          const reframed = await window.electronAPI.sendChatMessage({
            message: fullRewritePrompt,
            version: versionSelect.value,
            prependedText: diagnosticSystem,
            useSummary: false,
            conversationSummary: '',
            conversationHistory: conversationHistory,
            temperature: behaviorPacks?.generation_settings?.temperature,
            max_tokens: requestedMaxTokens
          });
          if (reframed.success && reframed.response?.trim()) {
            assistantText = reframed.response.trim();
            // Only accept if it does NOT contain forbidden terms
            if (!containsForbiddenTerms(assistantText)) {
              handled = false; // allow next checks (for expansion etc.)
            } else {
              // Still contains forbidden terms; do NOT show in UI/history
              handled = true;
            }
          } else {
            // Rewrite failed, do not show in UI/history
            handled = true;
          }
        } catch (e) {
          console.error('[BEHAVIOR] Full rewrite for diagnostic opening failed:', e);
          handled = true;
        }
      }

      // 2b/3) Post-generation enforcement: if the model falls back to forbidden terms, do a single rewrite retry
      if (!handled && containsForbiddenTerms(assistantText)) {
        console.warn('[BEHAVIOR] Assistant output contained forbidden terms. Attempting one rewrite retry.');
        injectExemplar('response_violation');
        const correctiveSystem = wrapSystemBlocks(systemToUse, pendingExemplarSystemText);
        const rewritePrompt = buildRewriteInstruction(userText, assistantText);
        try {
          const retry = await window.electronAPI.sendChatMessage({
            message: rewritePrompt,
            version: versionSelect.value,
            prependedText: correctiveSystem,
            useSummary: false,
            conversationSummary: '',
            conversationHistory: conversationHistory,
            temperature: behaviorPacks?.generation_settings?.temperature,
            max_tokens: requestedMaxTokens
          });
          if (retry.success && (retry.response?.trim() ?? '').length > 0) {
            const rewritten = retry.response.trim();
            if (!containsForbiddenTerms(rewritten)) {
              assistantText = rewritten;
              handled = false; // allow next checks (for expansion etc.)
            } else {
              // Still contains forbidden terms; do NOT show in UI/history
              handled = true;
            }
          } else {
            // Rewrite failed, do not show in UI/history
            handled = true;
          }
        } catch (e) {
          console.error('[BEHAVIOR] Error during rewrite retry:', e);
          handled = true;
        }
      }

      // 2c) Escalate rewrite if response is syntactically compliant but lacks concrete practice or is too brief
      if (!handled && !containsForbiddenTerms(assistantText) && (lacksConcretePractice(assistantText) || isTooBrief(assistantText))) {
        console.warn('[BEHAVIOR] Assistant output lacked concrete practice or was too brief. Attempting expansion rewrite.');
        injectExemplar('expand_into_practice');
        const expansionSystem = wrapSystemBlocks(systemToUse, pendingExemplarSystemText);
        const expansionPrompt =
          `Rewrite the response below as a helpful coaching answer (not a label/diagnosis). ` +
          `Include: (1) a brief reflection of what the leader is noticing, (2) 2–4 actionable next steps, and (3) at least one concrete example (routine/protocol/artifact/question set) a team could use next meeting. ` +
          `Explain why those moves would change practice.\n\n` +
          `Original response:\n${assistantText}\n\n` +
          `Return ONLY the expanded response.`;
        try {
          const expanded = await window.electronAPI.sendChatMessage({
            message: expansionPrompt,
            version: versionSelect.value,
            prependedText: expansionSystem,
            useSummary: false,
            conversationSummary: '',
            conversationHistory: conversationHistory,
            temperature: behaviorPacks?.generation_settings?.temperature,
            max_tokens: requestedMaxTokens
          });
          if (expanded.success && (expanded.response?.trim() ?? '').length > 0) {
            const expandedText = expanded.response.trim();
            if (!containsForbiddenTerms(expandedText)) {
              assistantText = expandedText;
              handled = false;
            } else {
              handled = true;
            }
          } else {
            handled = true;
          }
        } catch (e) {
          console.error('[BEHAVIOR] Error during expansion rewrite:', e);
          handled = true;
        }
      }

      // 3) Prevent any diagnostic response from reaching UI/history:
      // Only append assistant messages if not handled (i.e., assistantText is compliant and all rewrites succeeded)
      if (!handled && assistantText && assistantText.trim().length > 0 && !containsForbiddenTerms(assistantText)) {
        addMessage('assistant', assistantText.trim());
        saveConfig();
      } else if (!handled) {
        // If ALL rewrites fail, display a single fallback assistant message and DO NOT save it to history.
        addMessage('assistant', "Let’s slow this down and look closely at what you’re noticing in practice.", false);
        // Do not call saveConfig here
      }
    } else {
      // Never append assistant messages on error - errors are out-of-band events
      const errorMsg = result.error || 'Unknown error';
      console.error('[CHAT] Chat error:', errorMsg);
      addMessage('assistant', 'Error: ' + errorMsg, false);
      // Do NOT mutate conversation history here - errors are not part of the conversation
    }
    
    // Clear one-shot exemplar after request attempt (one-shot semantics)
    pendingExemplarSystemText = null;
  } catch (error) {
    // Never append assistant messages on error - errors are out-of-band events
    console.error('[CHAT] Error sending message:', error);
    addMessage('assistant', 'Error: ' + error.message, false);
    // Do NOT mutate conversation history here - errors are not part of the conversation
    
    // Clear one-shot exemplar on error (one-shot semantics)
    pendingExemplarSystemText = null;
  } finally {
    isSending = false;
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


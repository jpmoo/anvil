let selectedProfile = null;
let profileVersions = [];
let sshConfig = { host: '', port: 22 };
let sshTested = false;
let modelInfo = null;
let conversationHistory = [];
let conversationSummary = '';
// Memory-safe working memory: layered artifact (recent turns + rolling summary + pinned working-notes)
let conversationWorkingMemory = {
  summary: '', // 2–4 sentences, rolling
  key_threads: [] // 5–9 short bullets (strings)
};
// Shared Understanding: concise rolling summary of agreed context (not a conversation log, but a contract of what is understood)
let sharedUnderstanding = []; // Array of short bullet points (strings) representing established facts/patterns
let configLoaded = false; // Track if config has been loaded to prevent overwriting user input
let kickoffIssued = false; // True if we generated an opening greeting for a brand-new chat

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
  // Supports both legacy `when` mapping and simple trigger-keyed exemplars.
  pack.exemplars = {};
  if (pack.triggers) {
    Object.entries(pack.triggers).forEach(([key, trigger]) => {
      if (!trigger) return;
      if (trigger.exemplars) {
        // If trigger.when exists, preserve it; otherwise default to [key]
        const whenList = trigger.when
          ? Object.keys(trigger.when)
          : [key];

        pack.exemplars[key] = {
          when: whenList,
          text: trigger.exemplars
        };
      }
    });
  }

  return pack;
}

// === Behavior-pack–driven style helpers ===

// Track the highest phase reached (persists across turns and page reloads)
let highestPhaseReached = 'reflection';

// Infer conversation phase based on assistant turn count, but only progress forward
// Flow: reflection → sensemaking → structuring → stay there
function inferConversationPhase(conversationHistory, behaviorPack) {
  if (!behaviorPack?.conversation_phase) {
    return "reflection";
  }

  const assistantTurns = conversationHistory.filter(
    m => m.role === "assistant"
  ).length;

  // Determine target phase based on turn count
  let targetPhase = "reflection";
  if (assistantTurns === 0) {
    targetPhase = "reflection";
  } else if (assistantTurns === 1) {
    targetPhase = "sensemaking";
  } else {
    targetPhase = "structuring";
  }

  // Only progress forward - never go backwards
  // Once we reach structuring, stay there
  const phaseOrder = { 'reflection': 1, 'sensemaking': 2, 'structuring': 3 };
  const targetOrder = phaseOrder[targetPhase] || 1;
  const currentOrder = phaseOrder[highestPhaseReached] || 1;

  if (targetOrder > currentOrder) {
    // Progress forward
    const oldPhase = highestPhaseReached;
    highestPhaseReached = targetPhase;
    console.log(`[PHASE] Progressed to ${targetPhase} (from ${oldPhase})`);
    // Save the updated phase
    saveConfig();
  } else if (targetOrder < currentOrder) {
    // Don't go backwards - use the highest phase reached
    console.log(`[PHASE] Staying at ${highestPhaseReached} (target was ${targetPhase}, but we don't go backwards)`);
  }

  return highestPhaseReached;
}

// Build a system message from the loaded behavior pack, if present
// Per PDF: "Prefer fewer system messages; consolidate identity + stance"
// Per PDF: "Do not inject conversation phase labels into prompts"
function buildSystemFromBehaviorPack(phase = null, isKickoff = false) {
  if (!behaviorPacks) return '';
  const parts = [];

  // Only include identity and stance - consolidated single system prepend
  if (behaviorPacks.profile_intent?.identity) {
    parts.push(behaviorPacks.profile_intent.identity);
    console.log('[BEHAVIOR] Using profile_intent.identity from behavior_packs.json');
  }
  if (behaviorPacks.profile_intent?.stance) {
    parts.push(behaviorPacks.profile_intent.stance);
    console.log('[BEHAVIOR] Using profile_intent.stance from behavior_packs.json');
  }

  // Phase-sensitive guidance (read from behavior pack, do not name phases explicitly)
  // Phase-Based Prompt Subtraction: Each phase replaces prior guidance, not adds to it
  if (phase && behaviorPacks.conversation_phase?.phases?.[phase]) {
    const phaseConfig = behaviorPacks.conversation_phase.phases[phase];
    if (phase === 'reflection') {
      // Reflection has different guidance for kickoff vs subsequent turns
      const guidance = isKickoff 
        ? phaseConfig.guidance_kickoff 
        : phaseConfig.guidance_subsequent;
      if (guidance) {
        parts.push(guidance);
        console.log(`[BEHAVIOR] Using conversation_phase.phases.${phase}.guidance_${isKickoff ? 'kickoff' : 'subsequent'} from behavior_packs.json`);
      }
    } else {
      // Other phases have single guidance
      if (phaseConfig.guidance) {
        parts.push(phaseConfig.guidance);
        console.log(`[BEHAVIOR] Using conversation_phase.phases.${phase}.guidance from behavior_packs.json`);
      }
    }
  }
  
  // Repetition prevention (read from behavior pack)
  // Order matters: most critical instructions first
  if (behaviorPacks.repetition_prevention) {
    // Most critical: multi-sentence idea repetition
    if (behaviorPacks.repetition_prevention.multi_sentence_ban) {
      parts.push(behaviorPacks.repetition_prevention.multi_sentence_ban);
      console.log('[BEHAVIOR] Using repetition_prevention.multi_sentence_ban from behavior_packs.json');
    }
    if (behaviorPacks.repetition_prevention.idea_progression) {
      parts.push(behaviorPacks.repetition_prevention.idea_progression);
      console.log('[BEHAVIOR] Using repetition_prevention.idea_progression from behavior_packs.json');
    }
    if (behaviorPacks.repetition_prevention.within_response_ban) {
      parts.push(behaviorPacks.repetition_prevention.within_response_ban);
      console.log('[BEHAVIOR] Using repetition_prevention.within_response_ban from behavior_packs.json');
    }
    if (behaviorPacks.repetition_prevention.exact_phrase_ban) {
      parts.push(behaviorPacks.repetition_prevention.exact_phrase_ban);
      console.log('[BEHAVIOR] Using repetition_prevention.exact_phrase_ban from behavior_packs.json');
    }
    if (behaviorPacks.repetition_prevention.lexical_reuse_ban) {
      parts.push(behaviorPacks.repetition_prevention.lexical_reuse_ban);
      console.log('[BEHAVIOR] Using repetition_prevention.lexical_reuse_ban from behavior_packs.json');
    }
    if (behaviorPacks.repetition_prevention.sentence_variation) {
      parts.push(behaviorPacks.repetition_prevention.sentence_variation);
      console.log('[BEHAVIOR] Using repetition_prevention.sentence_variation from behavior_packs.json');
    }
  }

  // Enforce output_contract constraints without semantic interpretation
  if (behaviorPacks.output_contract?.rule) {
    parts.push(behaviorPacks.output_contract.rule);
    console.log('[BEHAVIOR] Using output_contract.rule from behavior_packs.json');
  }
  if (Array.isArray(behaviorPacks.output_contract?.must_not) && behaviorPacks.output_contract.must_not.length) {
    parts.push(`Must not: ${behaviorPacks.output_contract.must_not.join(', ')}`);
    console.log('[BEHAVIOR] Using output_contract.must_not from behavior_packs.json');
  }

  // Removed: conversation_phase.default injection (per PDF: no phase labels)
  // Removed: progression_required phase anchoring (per PDF: no phase labels)
  // Removed: avoid/allow lists (not needed in consolidated system message)

  if (parts.length > 0) {
    console.log('[BEHAVIOR] Built consolidated system message from behavior_packs.json (', parts.length, 'parts)');
  }
  return parts.join('\n');
}

// Format Shared Understanding block for prompt injection
// This is a contract of what is already understood, not a conversation log
function formatSharedUnderstanding() {
  if (!Array.isArray(sharedUnderstanding) || sharedUnderstanding.length === 0) {
    return '';
  }
  
  const bullets = sharedUnderstanding
    .filter(item => typeof item === "string" && item.trim())
    .slice(0, 12); // Limit to 12 points for conciseness
  
  if (bullets.length === 0) {
    return '';
  }
  
  const parts = [];
  const formatting = behaviorPacks?.memory_formatting || {};
  parts.push(formatting.shared_understanding_header || '--- Shared Understanding ---');
  parts.push(formatting.shared_understanding_instruction || 'The following points have already been established. Do not restate these points. Build on them.');
  parts.push(bullets.map(b => `- ${b.trim()}`).join('\n'));
  parts.push(formatting.shared_understanding_footer || '--- End Shared Understanding ---');
  
  return parts.join('\n');
}

// Format working memory for prompt injection (kept as SYSTEM content, not UI text)
function formatWorkingMemoryForPrompt() {
  const s = conversationWorkingMemory?.summary?.trim();
  const threads = Array.isArray(conversationWorkingMemory?.key_threads)
    ? conversationWorkingMemory.key_threads.filter(t => typeof t === "string" && t.trim()).slice(0, 9)
    : [];
  const parts = [];
  const formatting = behaviorPacks?.memory_formatting || {};
  // Reduce repetition: treat working memory as already-known background
  parts.push(formatting.working_memory_intro || 'Assume the following context is already understood. Do not restate it unless the user explicitly asks.');
  if (s) parts.push(`${formatting.working_memory_summary_label || 'Conversation summary:'}\n${s}`);
  if (threads.length) {
    parts.push(`${formatting.working_memory_threads_label || 'Key threads / open loops:'}\n- ${threads.join('\n- ')}`);
  }
  return parts.join('\n\n').trim();
}

// Fix 1 & 2: Compress conversation history - remove full assistant responses
// Replace with semantic summaries via Shared Understanding block (injected in system prompt)
// Preserve meaning, not phrasing. Keep only user messages to avoid lexical reuse.
function compressConversationHistory(history, currentUserMessage = '') {
  if (!Array.isArray(history) || history.length === 0) {
    return [];
  }
  
  // Check if current user message contains pronouns or indexicals
  // Pronouns: he, she, it, they, him, her, them, his, hers, its, their, this, that, these, those
  // Indexicals: this, that, these, those, here, there, now, then
  const pronounIndexicalPattern = /\b(he|she|it|they|him|her|them|his|hers|its|their|this|that|these|those|here|there|now|then)\b/gi;
  const hasPronounsOrIndexicals = currentUserMessage && pronounIndexicalPattern.test(currentUserMessage);
  
  if (hasPronounsOrIndexicals) {
    console.log(`[COMPRESS] Current user message contains pronouns/indexicals, preserving at least one assistant turn for context`);
  }
  
  // Remove all assistant messages - their semantic content is already captured in:
  // 1. Shared Understanding block (established facts/patterns)
  // 2. Working memory summary (rolling context)
  // This prevents Mistral from reusing exact phrases from prior assistant responses
  // EXCEPTION: If current message has pronouns/indexicals, preserve the most recent assistant message
  let compressed = [];
  let preservedAssistant = null;
  
  if (hasPronounsOrIndexicals) {
    // Find the most recent assistant message to preserve for context
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === 'assistant') {
        preservedAssistant = history[i];
        console.log(`[COMPRESS] Preserving assistant message at index ${i} for pronoun/indexical context`);
        break;
      }
    }
  }
  
  // Filter to user messages
  compressed = history.filter(msg => msg.role === 'user');
  
  // CRITICAL: Remove only the LAST trailing user message, since we're about to add a new user message
  // This prevents consecutive user messages which would violate the message sequence validation
  // The last user message will be added fresh as part of the current request
  // We keep all other user messages for context - only remove the very last one
  const originalCompressedLength = compressed.length;
  let removedUserCount = 0;
  if (compressed.length > 0 && compressed[compressed.length - 1].role === 'user') {
    compressed.pop();
    removedUserCount = 1;
    console.log(`[COMPRESS] Removed last user message from compressed history to prevent duplicate user role`);
  }
  
  // If we preserved an assistant message, insert it appropriately to maintain alternating pattern
  // The pattern should be: assistant -> user -> assistant -> user...
  // So if we have user messages, insert assistant before the first user message
  if (preservedAssistant) {
    if (compressed.length > 0) {
      // Insert before first user message to maintain: assistant -> user -> user -> ... pattern
      // But we need to ensure it doesn't create consecutive assistants, so check the last message before insertion
      compressed.unshift(preservedAssistant);
      console.log(`[COMPRESS] Inserted preserved assistant message before user messages for pronoun/indexical context`);
    } else {
      // No user messages, just add the assistant
      compressed.push(preservedAssistant);
    }
  }
  
  const assistantCount = history.length - originalCompressedLength - (preservedAssistant ? 1 : 0);
  const preservedCount = preservedAssistant ? 1 : 0;
  console.log(`[COMPRESS] Compressed history: ${history.length} messages -> ${compressed.length} messages (removed ${assistantCount} assistant messages + ${removedUserCount} last user message${preservedCount > 0 ? `, preserved ${preservedCount} assistant message for context` : ''})`);
  
  return compressed;
}

// Parse working memory from model response
function parseWorkingMemory(text) {
  const out = { summary: '', key_threads: [], sharedUnderstanding: [] };
  if (!text || typeof text !== "string") return out;
  
  const summaryMatch = text.match(/SUMMARY:\s*([\s\S]*?)(?:\n\s*(?:KEY_THREADS|SHARED_UNDERSTANDING):|$)/i);
  if (summaryMatch) out.summary = summaryMatch[1].trim();
  
  const threadsMatch = text.match(/KEY_THREADS:\s*([\s\S]*?)(?:\n\s*SHARED_UNDERSTANDING:|$)/i);
  if (threadsMatch) {
    const lines = threadsMatch[1].split(/\r?\n/).map(l => l.trim());
    out.key_threads = lines
      .filter(l => l.startsWith("-"))
      .map(l => l.replace(/^[-*]\s*/, "").trim())
      .filter(Boolean)
      .slice(0, 9);
  }
  
  const sharedMatch = text.match(/SHARED_UNDERSTANDING:\s*([\s\S]*)$/i);
  if (sharedMatch) {
    const lines = sharedMatch[1].split(/\r?\n/).map(l => l.trim());
    out.sharedUnderstanding = lines
      .filter(l => l.startsWith("-"))
      .map(l => l.replace(/^[-*]\s*/, "").trim())
      .filter(Boolean)
      .slice(0, 12);
  }
  
  // Fallback: if no markers, treat whole thing as summary
  if (!out.summary && !out.key_threads.length && !out.sharedUnderstanding.length) out.summary = text.trim();
  return out;
}
function getStyleRules() {
  const rules = behaviorPacks?.style_rules || {};
  if (Object.keys(rules).length > 0) {
    console.log('[BEHAVIOR] Using style_rules from behavior_packs.json');
  }
  return rules;
}

function violatesOpeningRules(text) {
  // Disabled for Mistral: opening behavior is enforced via training + behavior pack
      return false;
}

function escapeRegExp(str) {
  return String(str).replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
}
function containsForbiddenTerms(text, enforceStrict = true) {
  // Disabled for Mistral: semantic constraints handled in training
  return false;
}
let pendingExemplarSystemTexts = []; // One-shot exemplar(s) appended to the next request as system text
let exemplarRotationIndex = {}; // Track rotation index for each trigger
let isSending = false; // Prevent accidental double-send / concurrent sendMessage calls
let triggersQueuedThisTurn = new Set(); // de-dupe triggers per send
// Helper to kickoff conversation - only if behavior pack explicitly requests it
// Per PDF: "Do not inject assistant messages when a conversation starts; allow the model to generate the first response"
// This function is now disabled by default - the model will generate the first response when user sends a message
async function kickoffConversationIfEmpty() {
  // Check if behavior pack explicitly requests kickoff
  const shouldKickoff = behaviorPacks?.conversation_phase?.enable_kickoff === true;
  if (!shouldKickoff) {
    console.log('[KICKOFF][DEBUG] Kickoff disabled - model will generate first response when user sends message');
    return;
  }

  console.log('[KICKOFF][DEBUG] Entering kickoffConversationIfEmpty()', {
    historyLength: Array.isArray(conversationHistory) ? conversationHistory.length : 'not-array',
    hasAssistantHistory: Array.isArray(conversationHistory) && conversationHistory.some(m => m.role === 'assistant'),
    hasAssistantUI: !!document.getElementById('chatMessages')?.querySelector('.message.assistant'),
    modelInfoPresent: !!modelInfo
  });
  // Check if there's already an assistant message, either in history or UI
  // If so, return early; otherwise, proceed with kickoff
  const hasAssistantHistory = Array.isArray(conversationHistory)
    && conversationHistory.some(msg => msg.role === 'assistant');
  const chatMessagesDiv = document.getElementById('chatMessages');
  const hasAssistantUI = chatMessagesDiv
    ? !!chatMessagesDiv.querySelector('.message.assistant')
    : false;
  if (hasAssistantHistory || hasAssistantUI) {
    console.log('[KICKOFF][DEBUG] Aborting kickoff: assistant already present', {
      hasAssistantHistory,
      hasAssistantUI
    });
    return;
  }
  if (!modelInfo) {
    console.log('[KICKOFF][DEBUG] Aborting kickoff: modelInfo not yet available');
    return;
  }

  console.log('[CHAT] New conversation detected; issuing silent kickoff prompt');

  try {
    // Use behavior pack guidance for kickoff prompt, or default minimal prompt
    const kickoffPrompt = behaviorPacks?.conversation_phase?.kickoff_prompt || "Greet the user in one sentence.";

    // Kickoff turn: use reflection phase with isKickoff=true to get reflection instructions
    const behaviorSystem = buildSystemFromBehaviorPack('reflection', true);
    // No shared understanding on kickoff (first turn)
    const memoryPrepend = formatWorkingMemoryForPrompt();
    const combinedSystem = [behaviorSystem, memoryPrepend].filter(Boolean).join("\n\n");

    const result = await window.electronAPI.sendChatMessage({
      message: kickoffPrompt,
      version: window._autoSelectedVersion || 'base',
      prependedText: wrapSystemBlocks(combinedSystem, pendingExemplarSystemTexts),
      useSummary: false,
      conversationSummary: '',
      conversationHistory: [],
      temperature: getBehaviorPackTemperature(),
      max_tokens: 650
    });
    console.log('[KICKOFF][DEBUG] Kickoff LLM response received', {
      success: result?.success,
      responseLength: result?.response?.length ?? 0,
      rawResponse: result?.response
    });
    
    // Per PDF: "Do not rewrite responses by replacing them with renderer-authored text"
    // If validation fails, skip displaying the message rather than using a fallback
    if (result?.success && result.response?.trim()) {
      let assistantGreeting = result.response.trim();
      
      // Validate: reject non-ASCII characters - if present, skip message (don't use fallback)
      if (/[^\x00-\x7F]/.test(assistantGreeting)) {
        console.warn('[KICKOFF][WARN] Non-ASCII characters detected in greeting; skipping message (no fallback)');
        return; // Skip displaying invalid greeting
      }
      
      addMessage('assistant', assistantGreeting, true, { isOpening: true });
      kickoffIssued = true;
      console.log('[KICKOFF][DEBUG] Assistant greeting added to UI and history');
      saveConfig();
    } else {
      // No response or empty response - skip displaying (no fallback)
      console.warn('[KICKOFF][WARN] No valid greeting received; skipping (no fallback)');
    }
  } catch (error) {
    console.error('[CHAT] Error during kickoff conversation:', error);
    // Don't display any fallback message on error
  } finally {
    // One-shot semantics
    pendingExemplarSystemTexts = [];
    triggersQueuedThisTurn.clear();
  }
}

// Helper to get temperature from behavior packs with logging
function getBehaviorPackTemperature() {
  const temp = behaviorPacks?.generation_settings?.temperature;
  if (temp !== undefined) {
    console.log(`[BEHAVIOR] Using temperature ${temp} from behavior_packs.json generation_settings`);
  }
  return temp;
}

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
  if (!behaviorPacks || !behaviorPacks.exemplars) {
    console.log(`[BEHAVIOR] No exemplar found for trigger "${trigger}" (behavior packs not loaded or no exemplars)`);
    return null;
  }

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

  if (matches.length === 0) {
    console.log(`[BEHAVIOR] No exemplar found for trigger "${trigger}" in behavior_packs.json`);
    return null;
  }

  const index = exemplarRotationIndex[trigger] ?? 0;
  const selected = matches[index % matches.length];
  exemplarRotationIndex[trigger] = index + 1;
  console.log(`[BEHAVIOR] Retrieved exemplar for trigger "${trigger}" from behavior_packs.json (rotation index: ${index}, total matches: ${matches.length})`);
  return selected;
}

// Inject exemplar(s) as one-shot system text(s) (not shown in UI, not saved to history)
// Per PDF: "Avoid stacking exemplars unless a trigger explicitly fires"
// Per PDF: "Prefer fewer system messages" - cap at 1 exemplar for Mistral
function injectExemplar(trigger) {
  const exemplarText = getExemplarForTrigger(trigger);
  if (!exemplarText) return false;

  if (!Array.isArray(pendingExemplarSystemTexts)) pendingExemplarSystemTexts = [];
  // Per PDF: Avoid stacking - only add if not already present, cap at 1 for Mistral
  if (!pendingExemplarSystemTexts.includes(exemplarText)) {
    pendingExemplarSystemTexts.push(exemplarText);
    if (pendingExemplarSystemTexts.length > 1) {
      // Keep only the most recent exemplar (Mistral prefers fewer system messages)
      pendingExemplarSystemTexts = pendingExemplarSystemTexts.slice(-1);
    }
  }

  console.log(`[BEHAVIOR] Applied exemplar for trigger "${trigger}" (count: ${pendingExemplarSystemTexts.length})`);
  return true;
}

// Queue a trigger to be injected on the next request, deduped per send
function queueTrigger(trigger) {
  if (triggersQueuedThisTurn.has(trigger)) return;

  triggersQueuedThisTurn.add(trigger);
  injectExemplar(trigger);
}

// (Behavior-pack–driven forbidden checks and opening rules now used. See helpers above.)

// Detect lack of concrete, practical practice in assistant responses
function lacksConcretePractice(text) {
  // Disabled: Mistral handles concreteness via training
  return false;
}

// Detect if a response is too brief to be useful (behavior-pack–driven)
function isTooBrief(text) {
  // Disabled: brevity is acceptable and model-controlled
  return false;
}

// Detect short continuation prompts (e.g., "Tell me more", "Tell me more.")
function isContinuationPrompt(text) {
  if (!text) return false;

  // Normalize: lowercase, trim, collapse spaces, strip trailing punctuation
  let t = text
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[.!?]+$/g, '');

  // Also handle quoted / parenthetical stray punctuation at end
  t = t.replace(/["')\]]+$/g, '');

  // Keep this intentionally short to avoid false positives
  if (t.length > 40) return false;

  return (
    t === 'tell me more' ||
    t === 'tell me more please' ||
    t === 'say more' ||
    t === 'go deeper' ||
    t === 'expand' ||
    t.startsWith('can you expand') ||
    t.startsWith('say more about') ||
    t.startsWith('what might that look like')
  );
}

// Build a continuation instruction that extends the last assistant message
// Per PDF: Remove hardcoded coaching phrases - use neutral continuation request
// Fix 5: Redefine Continuation Mode
// Continuation should use working memory, not full assistant prose
function buildContinuationInstruction() {
  const summary = conversationWorkingMemory?.summary?.trim() || '';
  const threads = Array.isArray(conversationWorkingMemory?.key_threads)
    ? conversationWorkingMemory.key_threads.filter(t => typeof t === "string" && t.trim()).slice(0, 5)
    : [];
  
  const parts = [];
  const continuationConfig = behaviorPacks?.continuation_mode || {};
  parts.push(continuationConfig.instruction || "Please continue your previous response. Stay in the same voice and perspective.");
  
  if (summary) {
    parts.push(`\n${continuationConfig.context_label || 'Context from previous exchange:'}\n${summary}`);
  }
  
  if (threads.length > 0) {
    parts.push(`\n${continuationConfig.key_points_label || 'Key points to build on:'}\n${threads.map(t => `- ${t}`).join('\n')}`);
  }
  
  parts.push(`\n${continuationConfig.continue_prompt || 'Continue:'}`);
  
  return parts.join('\n').trim();
}

// Build a rewrite instruction using behavior_packs.json guidance
// Per PDF: "Rewrite prompts must request neutral transformation (e.g., language-only rewrite) and must not introduce new coaching content"
// Per PDF: "Only trigger a rewrite when a hard constraint is violated"
function buildRewriteInstruction(userMessage, badAssistantText) {
  const guidance = behaviorPacks?.rewrite_guidance || {};
  const basePrompt =
    guidance.rewrite_prompt ||
    guidance.purpose ||
    "Rewrite the response to use English only while preserving the meaning and tone.";

  // Per PDF: Neutral transformation only - no phase labels, no coaching content
  // Per PDF: Language-only rewrite for guardrail violations
  return `
${basePrompt}

Original response:
${badAssistantText}

Return only the revised response in English.
`;
}

// Wrap system content to reduce the model mistaking it for user content
// Per PDF: Use plain delimiters instead of XML-like tags for Mistral compatibility
function wrapSystemBlocks(userSystem, exemplars) {
  const blocks = [];
  const list = Array.isArray(exemplars)
    ? exemplars
    : (typeof exemplars === 'string' && exemplars.trim() ? [exemplars] : []);
  for (const ex of list) {
    if (ex && ex.trim()) {
      blocks.push(
        `--- SYSTEM EXEMPLAR (one-shot) ---\n` +
        `Use the following example to guide tone and structure. Do not mention, quote, or refer to the example.\n\n` +
        `${ex.trim()}\n` +
        `--- END SYSTEM EXEMPLAR ---`
      );
    }
  }
  if (userSystem && userSystem.trim()) {
    blocks.push(`--- SYSTEM INSTRUCTIONS ---\n${userSystem.trim()}\n--- END SYSTEM INSTRUCTIONS ---`);
  }
  return blocks.join('\n\n');
}

// Memory-safe normalization: preserve content by MERGING consecutive same-role messages
// rather than dropping them. This reduces "mechanical" forgetting while still keeping a clean history.
function normalizeConversation(messages) {
  const cleaned = [];
  for (const msg of (messages || [])) {
    if (!msg || !msg.role || typeof msg.content !== "string") continue;
    
    // First message must be user or system (same as before)
    if (cleaned.length === 0) {
      if (msg.role === "user" || msg.role === "system") {
        cleaned.push({ ...msg });
      }
      continue;
    }
    
    const last = cleaned[cleaned.length - 1];
    
    // If role repeats, MERGE with a clear separator
    if (msg.role === last.role) {
      last.content = `${last.content}\n\n---\n\n${msg.content}`.trim();
      // Preserve meta if present (prefer last meta)
      if (msg.meta) last.meta = { ...(last.meta || {}), ...(msg.meta || {}) };
      continue;
    }
    
    cleaned.push({ ...msg });
  }
  return cleaned;
}

// Save configuration to file
async function saveConfig() {
  try {
    // Read directly from input fields to ensure we save current values, not stale sshConfig
    const sshHostInput = document.getElementById('sshHost');
    const sshPortInput = document.getElementById('sshPort');
    // System messages now come entirely from behavior_packs.json - no user input field
    
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
      // System messages come from behavior_packs.json, not user input
      conversationHistory: conversationHistory || [], // Save the conversation history
      conversationSummary: conversationSummary || '', // Save summary for backward compatibility
      conversationWorkingMemory: conversationWorkingMemory || { summary: '', key_threads: [] }, // Save working memory
      sharedUnderstanding: sharedUnderstanding || [], // Save shared understanding
      highestPhaseReached: highestPhaseReached || 'reflection', // Save the highest phase reached
      selectedProfile: profileSelect ? profileSelect.value : '', // Save selected profile
      selectedVersion: window._autoSelectedVersion || 'base' // Auto-selected highest version (default to 'base')
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
      
      // System messages now come entirely from behavior_packs.json - no longer loading from config
      
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
      
      // Load highest phase reached
      if (config.highestPhaseReached !== undefined) {
        highestPhaseReached = config.highestPhaseReached;
        console.log('[CONFIG] Loaded highest phase reached:', highestPhaseReached);
      } else {
        // Reset to reflection if not found (new conversation or old config)
        highestPhaseReached = 'reflection';
      }
      
      // Load conversation summary (backward compatibility)
      if (config.conversationSummary !== undefined) {
        conversationSummary = config.conversationSummary || '';
      }
      
      // Load working memory
      if (config.conversationWorkingMemory !== undefined) {
        conversationWorkingMemory = {
          summary: config.conversationWorkingMemory.summary || conversationSummary || '',
          key_threads: Array.isArray(config.conversationWorkingMemory.key_threads) 
            ? config.conversationWorkingMemory.key_threads 
            : []
        };
        console.log('[CONFIG] Loaded working memory:', {
          summaryLength: conversationWorkingMemory.summary.length,
          threadsCount: conversationWorkingMemory.key_threads.length
        });
      } else if (conversationSummary) {
        // Migrate old summary to working memory format
        conversationWorkingMemory = {
          summary: conversationSummary,
          key_threads: []
        };
      }
      
      // Load shared understanding
      if (config.sharedUnderstanding !== undefined && Array.isArray(config.sharedUnderstanding)) {
        sharedUnderstanding = config.sharedUnderstanding.filter(item => typeof item === "string" && item.trim());
        console.log('[CONFIG] Loaded shared understanding:', sharedUnderstanding.length, 'points');
      } else {
        sharedUnderstanding = [];
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
  
  // System messages now come entirely from behavior_packs.json - no input field listeners needed
  const clearChatBtn = document.getElementById('clearChatBtn');
  
  // Set up listener for version selection
  const versionSelect = document.getElementById('versionSelect');
  if (versionSelect) {
    versionSelect.addEventListener('change', () => {
      console.log('[CONFIG] Version selection changed, saving config...');
      saveConfig();
    });
  }
  
  // System messages are loaded from behavior_packs.json when profile is selected
  
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', async () => {
      conversationHistory = [];
      highestPhaseReached = 'reflection'; // Reset phase when clearing conversation
      const chatMessages = document.getElementById('chatMessages');
      if (chatMessages) {
        chatMessages.innerHTML = '';
      }
      console.log('[CHAT] Conversation history cleared, phase reset to reflection');

      // Save config to persist the cleared history
      saveConfig();

      // Clear one-shot exemplar before voice_reset injection (one-shot semantics)
      pendingExemplarSystemTexts = [];
      triggersQueuedThisTurn.clear();

      // Load behavior packs for the current profile
      const profileName = modelInfo?.profileName || null;
      await loadBehaviorPacks(profileName);
      // Always run kickoff to ensure a greeting appears
      await kickoffConversationIfEmpty();
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
    document.getElementById('modelSelectionSection').style.display = 'none';
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
  const modelSelect = document.getElementById('modelSelect');
  const modelSelectionSection = document.getElementById('modelSelectionSection');
  
  console.log('[VERSIONS] Displaying models, version count:', profileVersions.length);
  
  if (!modelSelect) {
    console.error('[VERSIONS] Model select element not found');
    return;
  }
  
  // Clear existing options
  modelSelect.innerHTML = '';
  
  // Add base model option
  const baseOption = document.createElement('option');
  baseOption.value = 'base';
  baseOption.textContent = 'Base Model (from Hugging Face)';
  modelSelect.appendChild(baseOption);
  
  // Add version options (sorted by version number, highest first)
  const sortedVersions = [...profileVersions].sort((a, b) => b.version - a.version);
  sortedVersions.forEach(v => {
    const hasMerged = v.mergedBasePath ? ' (merged base + adapter)' : ' (adapter only)';
    const versionOption = document.createElement('option');
    versionOption.value = `v${v.version}`;
    versionOption.textContent = `Version ${v.version}${hasMerged}`;
    versionOption.setAttribute('data-version', v.version);
    modelSelect.appendChild(versionOption);
  });
  
  // Default to highest version if available, otherwise base
  if (sortedVersions.length > 0) {
    modelSelect.value = `v${sortedVersions[0].version}`;
    console.log('[VERSIONS] Default selected: highest version', sortedVersions[0].version);
  } else {
    modelSelect.value = 'base';
    console.log('[VERSIONS] Default selected: base model (no versions available)');
  }
  
  modelSelectionSection.style.display = 'block';
  console.log('[VERSIONS] Displayed', sortedVersions.length, 'version(s) + base model option');
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
  let useBaseModel = false;
  
  if (selectedProfile) {
    profileName = selectedProfile.name;
    baseModel = selectedProfile.baseModel;
    
    // Get selected model from dropdown
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect && modelSelect.value) {
      const selectedValue = modelSelect.value;
      
      if (selectedValue === 'base') {
        // Base model selected
        useBaseModel = true;
        versions = [];
        console.log('[INIT] Selected: Base model only');
      } else if (selectedValue.startsWith('v')) {
        // Version selected
        const versionNum = parseInt(selectedValue.substring(1));
        const selectedVersion = profileVersions.find(v => v.version === versionNum);
        if (selectedVersion) {
          versions = [selectedVersion];
          useBaseModel = false; // Don't download base model if using a version
          console.log('[INIT] Selected: Version', versionNum);
        }
      }
    }
    
    console.log('[INIT] Use base model:', useBaseModel);
    console.log('[INIT] Selected versions:', versions.map(v => `V${v.version}`).join(', '));
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
      useBaseModel: useBaseModel,
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
    summaryPrompt += `Please provide an updated summary and key threads that:\n`;
    summaryPrompt += `- Incorporates the new exchange above\n`;
    
    if (currentSummary && currentSummary.trim()) {
      summaryPrompt += `- Preserves important context from the previous summary\n`;
    }
    
    summaryPrompt += `- Identifies 5-9 key threads, open loops, or important facts\n`;
    summaryPrompt += `- Is concise (2-4 sentences for summary)\n`;
    summaryPrompt += `- Focuses on main topics, decisions, and key information\n`;
    summaryPrompt += `- Extracts established facts, patterns, or agreements that have been confirmed (for Shared Understanding)\n\n`;
    summaryPrompt += `Return format (plain text):\n`;
    summaryPrompt += `SUMMARY: <2-4 sentences>\n`;
    summaryPrompt += `KEY_THREADS:\n`;
    summaryPrompt += `- <short bullet>\n`;
    summaryPrompt += `- <short bullet>\n`;
    summaryPrompt += `SHARED_UNDERSTANDING:\n`;
    summaryPrompt += `- <established fact or pattern>\n`;
    summaryPrompt += `- <established fact or pattern>\n`;
    
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
      temperature: getBehaviorPackTemperature(),
      max_tokens: (() => {
        const max = behaviorPacks?.generation_settings?.max_tokens;
        if (max !== undefined) {
          console.log(`[BEHAVIOR] Using max_tokens ${max} from behavior_packs.json generation_settings`);
        }
        return max;
      })()
    });
    
    if (summaryResult.success && summaryResult.response) {
      // Parse working memory from response
      const parsed = parseWorkingMemory(summaryResult.response);
      conversationWorkingMemory = {
        summary: parsed.summary || '',
        key_threads: parsed.key_threads || []
      };
      
      // Update shared understanding (merge new points with existing, removing duplicates)
      if (Array.isArray(parsed.sharedUnderstanding) && parsed.sharedUnderstanding.length > 0) {
        const existingPoints = new Set(sharedUnderstanding.map(p => p.toLowerCase().trim()));
        const newPoints = parsed.sharedUnderstanding.filter(p => {
          const normalized = p.toLowerCase().trim();
          return normalized && !existingPoints.has(normalized);
        });
        sharedUnderstanding = [...sharedUnderstanding, ...newPoints].slice(0, 12); // Keep max 12 points
        console.log('[SUMMARY] Updated shared understanding:', sharedUnderstanding.length, 'points');
      }
      
      // Keep conversationSummary for backward compatibility
      conversationSummary = parsed.summary || summaryResult.response.trim();
      
      console.log('[SUMMARY] Updated working memory:');
      console.log('----------------------------------------');
      console.log('Summary:', conversationSummary);
      console.log('Key threads:', conversationWorkingMemory.key_threads);
      console.log('Shared understanding:', sharedUnderstanding);
      console.log('----------------------------------------');
      console.log('[SUMMARY] ========================================\n');
      
      // Save config to persist the updated working memory and shared understanding
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
      const fallbackSummary = recentMessages
        .map(m => {
          const roleLabel = m.role === 'user' ? 'User' : profileName;
          const content = m.content.length > 150 ? m.content.substring(0, 150) + '...' : m.content;
          return `${roleLabel}: ${content}`;
        })
        .join('\n\n');
      conversationSummary = fallbackSummary;
      conversationWorkingMemory = { summary: fallbackSummary, key_threads: [] };
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

    // Hide version selector - we always use highest version automatically
    const versionSelect = document.getElementById('versionSelect');
    if (versionSelect) {
      versionSelect.style.display = 'none';
      const versionLabel = versionSelect.previousElementSibling;
      if (versionLabel && versionLabel.tagName === 'LABEL') {
        versionLabel.style.display = 'none';
      }
    }
    
    // Automatically determine which version to use
    let selectedVersion = 'base'; // Default to base model
    if (modelInfo.highestVersion) {
      selectedVersion = `V${modelInfo.highestVersion.version}`;
      console.log(`[CHAT] Automatically using highest version: ${selectedVersion} (merged base model + adapter)`);
    } else if (modelInfo.versions && modelInfo.versions.length > 0) {
      // Fallback: use highest version from versions array if highestVersion not set
      const sortedVersions = modelInfo.versions.sort((a, b) => b.version - a.version);
      selectedVersion = `V${sortedVersions[0].version}`;
      console.log(`[CHAT] Using highest version from versions array: ${selectedVersion}`);
      } else {
      console.log(`[CHAT] No versions available, using base model`);
    }
    
    // Store selected version for use in chat requests
    window._autoSelectedVersion = selectedVersion;

    // Show chat interface
    document.getElementById('chatInterface').style.display = 'block';
    
    // Scroll to chat interface
    document.getElementById('chatInterface').scrollIntoView({ behavior: 'smooth' });
    
    // Load behavior packs for the current profile and inject exemplar if available
    const profileName = modelInfo?.profileName || null;
    await loadBehaviorPacks(profileName);

    // Diagnostic log before kickoff
    console.log('[CHAT][DEBUG] Calling kickoffConversationIfEmpty() from showChatInterface');
    // Trigger a single conversational opening if this is a brand-new chat
    await kickoffConversationIfEmpty();
  } catch (error) {
    console.error('Error showing chat interface:', error);
    showStatus('prepareStatus', 'Error loading chat interface: ' + error.message, 'error');
  }
}

// Clean up formatting artifacts from model output
function cleanFormattingArtifacts(text) {
  if (!text || typeof text !== 'string') return text;
  
  // Strip UI artifacts: Remove phrases like 'Show less', 'Nano message completed', etc.
  const uiArtifacts = [
    /show\s+less/gi,
    /nano\s+message\s+completed/gi,
    /message\s+completed/gi,
    /\[end\s+of\s+response\]/gi,
    /\[response\s+complete\]/gi,
    /\[finished\]/gi,
    /languages?:\s*en/gi, // Remove "languages: en" artifacts
    /<\/body>/gi, // Remove HTML body tags
    /<break\s*\/?>/gi, // Remove <break /> tags
    /add\s+a\s+break\s+after\s+your\s+response/gi, // Remove instruction artifacts
    /add\s+a\s+break/gi // Remove "add a break" instructions
  ];
  for (const pattern of uiArtifacts) {
    text = text.replace(pattern, '');
  }
  
  // Remove CSS-like syntax: {property: value} or {property:value}
  // This catches patterns like {margin-left: 0.75em}
  text = text.replace(/\{[^}]*\}/g, '');
  
  // Remove regex-like patterns: .* or .+ (standalone, not in context)
  // Match patterns like ".*" at start of lines or after spaces
  text = text.replace(/(?:^|\s)\.\*+(?:\s|$)/gm, ' ');
  text = text.replace(/(?:^|\s)\.\++(?:\s|$)/gm, ' ');
  
  // Remove CSS property patterns only when they appear in isolation
  // Match patterns like "margin-left: 0.75em" that appear standalone
  text = text.replace(/\b(margin|padding|font|color|background|border|width|height|display|position|top|left|right|bottom|text-align|line-height|white-space|opacity|z-index|float|clear|overflow|visibility|cursor|outline|box-shadow|text-shadow|transform|transition|animation|flex|grid|align|justify|gap|order|flex-direction|flex-wrap|align-items|align-content|justify-content|justify-items|grid-template|grid-area|grid-column|grid-row):\s*[^;}\s,]+(?:\s|$)/gi, '');
  
  // Clean up multiple spaces (but preserve intentional spacing)
  text = text.replace(/[ \t]{3,}/g, ' ');
  
  // Clean up spaces before punctuation (but keep space after sentence-ending punctuation)
  text = text.replace(/\s+([.,!?;:])/g, '$1');
  
  // Remove trailing whitespace from lines
  text = text.replace(/[ \t]+$/gm, '');
  
  // Clean up any double spaces that might have been created
  text = text.replace(/\s{2,}/g, ' ');
  
  return text.trim();
}

// Check if text has significant non-English content (threshold-based, not binary)
// Tier 3 Fix: Update non-English detection to allow Unicode punctuation
// Permit curly quotes, em/en dashes, ellipses, and typographic apostrophes
// Do not treat smart punctuation as language switching
function hasSignificantNonEnglish(text, threshold = 0.02) {
  if (!text || typeof text !== 'string') return false;
  const totalChars = text.length;
  if (totalChars === 0) return false;
  
  // Tier 3: Allow Unicode punctuation (curly quotes, dashes, ellipses, apostrophes)
  // These are common in English typography and should not trigger multilingual detection
  const unicodePunctuation = [
    /[\u2018\u2019\u201A\u201B]/g, // Typographic apostrophes and single quotes
    /[\u201C\u201D\u201E\u201F]/g, // Typographic double quotes
    /[\u2013\u2014]/g, // En dash and em dash
    /[\u2026]/g, // Ellipsis
    /[\u00A0]/g, // Non-breaking space
  ];
  
  // Remove Unicode punctuation from text before checking for non-English
  let textForCheck = text;
  for (const pattern of unicodePunctuation) {
    textForCheck = textForCheck.replace(pattern, '');
  }
  
  // Only trigger multilingual guardrails for actual non-Latin characters or known foreign alphabets
  // Exclude common Unicode punctuation that's valid in English
  const nonLatinChars = (textForCheck.match(/[^\x00-\x7F]/g) || []).length;
  const ratio = nonLatinChars / totalChars;
  return ratio > threshold;
}

// Sanitize HTML - remove dangerous tags but preserve safe formatting tags
function sanitizeHtml(html) {
  if (!html || typeof html !== 'string') return '';
  
  // List of allowed safe HTML tags
  const allowedTags = ['div', 'p', 'span', 'strong', 'em', 'b', 'i', 'u', 'br', 'hr', 
                       'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 
                       'blockquote', 'pre', 'code', 'a'];
  
  // Remove dangerous tags and their content (script, iframe, object, embed, etc.)
  html = html.replace(/<script[\s\S]*?<\/script>/gi, '');
  html = html.replace(/<iframe[\s\S]*?<\/iframe>/gi, '');
  html = html.replace(/<object[\s\S]*?<\/object>/gi, '');
  html = html.replace(/<embed[\s\S]*?<\/embed>/gi, '');
  html = html.replace(/<form[\s\S]*?<\/form>/gi, '');
  html = html.replace(/<input[^>]*>/gi, '');
  html = html.replace(/<button[^>]*>[\s\S]*?<\/button>/gi, '');
  
  // Remove dangerous attributes from allowed tags (onclick, onerror, etc.)
  html = html.replace(/\s*on\w+\s*=\s*["'][^"']*["']/gi, '');
  html = html.replace(/\s*on\w+\s*=\s*[^\s>]*/gi, '');
  html = html.replace(/\s*javascript:/gi, '');
  html = html.replace(/\s*data:text\/html/gi, '');
  
  // Remove style attributes that could contain dangerous CSS
  html = html.replace(/\s*style\s*=\s*["'][^"']*["']/gi, '');
  
  // Only keep allowed tags, remove others but preserve their content
  const tagPattern = /<\/?([a-zA-Z][a-zA-Z0-9]*)[^>]*>/g;
  html = html.replace(tagPattern, (match, tagName) => {
    const lowerTag = tagName.toLowerCase();
    if (allowedTags.includes(lowerTag)) {
      // Preserve allowed tags but remove all attributes for safety
      if (match.startsWith('</')) {
        return `</${lowerTag}>`;
      } else {
        return `<${lowerTag}>`;
      }
    } else {
      // Remove disallowed tags but keep their content
      return '';
    }
  });
  
  return html;
}

// Simple markdown to HTML converter with XSS protection
function markdownToHtml(text) {
  if (!text || typeof text !== 'string') return '';
  
  // Clean up formatting artifacts first
  text = cleanFormattingArtifacts(text);
  
  // Check if text already contains HTML tags
  const hasHtmlTags = /<[a-zA-Z][^>]*>/.test(text);
  
  // Escape HTML to prevent XSS
  // Note: We don't escape apostrophes (') as they're safe in HTML text content
  // Only escape them if needed in attribute values, but we're using textContent/innerHTML for content
  function escapeHtml(unsafe) {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
      // Removed apostrophe escaping - apostrophes are safe in HTML text content
  }
  
  // Store code blocks and inline code to preserve them
  const codeBlocks = [];
  const inlineCodes = [];
  let placeholderIndex = 0;
  
  // Replace code blocks with placeholders
  let html = text.replace(/```([\s\S]*?)```/g, (match, code) => {
    const placeholder = `__CODEBLOCK_${placeholderIndex}__`;
    codeBlocks[placeholderIndex] = escapeHtml(code.trim());
    placeholderIndex++;
    return placeholder;
  });
  
  // Replace inline code with placeholders
  html = html.replace(/`([^`\n]+)`/g, (match, code) => {
    const placeholder = `__INLINECODE_${placeholderIndex}__`;
    inlineCodes[placeholderIndex] = escapeHtml(code);
    placeholderIndex++;
    return placeholder;
  });
  
  // If text contains HTML tags, sanitize them first, then process markdown
  // Otherwise, escape all HTML and process markdown normally
  if (hasHtmlTags) {
    // Sanitize existing HTML (remove dangerous tags/attributes, keep safe ones)
    // This preserves tags like <div>, <p>, <strong>, etc. but removes <script>, etc.
    html = sanitizeHtml(html);
    // Don't escape HTML here - we want to preserve the safe tags
    // Markdown processing below will work on text content, and won't interfere with existing HTML tags
  } else {
    // Escape all HTML first (no existing HTML, so safe to escape)
    html = escapeHtml(html);
  }
  
  // Restore code blocks
  for (let i = 0; i < codeBlocks.length; i++) {
    html = html.replace(`__CODEBLOCK_${i}__`, `<pre><code>${codeBlocks[i]}</code></pre>`);
  }
  
  // Restore inline code
  for (let i = 0; i < inlineCodes.length; i++) {
    html = html.replace(`__INLINECODE_${i}__`, `<code>${inlineCodes[i]}</code>`);
  }
  
  // Process headers (must be on their own line)
  // Only process if not already inside HTML tags (simple check - if hasHtmlTags, skip headers)
  if (!hasHtmlTags) {
    html = html.replace(/^### (.+?)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+?)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+?)$/gm, '<h1>$1</h1>');
  }
  
  // Process links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  
  // Process bold (**text** or __text__) - but be careful not to double-process if already in HTML
  // Only process if we escaped HTML (i.e., no existing HTML tags)
  if (!hasHtmlTags) {
    html = html.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    
    // Process italic (*text* or _text_) - avoid conflicts with bold
    html = html.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
    html = html.replace(/(?<!_)_([^_\n]+)_(?!_)/g, '<em>$1</em>');
  }
  
  // Process lists line by line
  const lines = html.split('\n');
  const processedLines = [];
  let inUnorderedList = false;
  let inOrderedList = false;
  let orderedListStart = 1; // Track starting number for ordered lists
  let lastOrderedListNumber = 0; // Track the last list item number we processed
  let linesSinceLastListItem = 0; // Track how many non-list lines since last list item
  
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    let trimmedLine = line.trim();
    
    // Check for blank lines - these should close lists and create paragraph breaks
    if (trimmedLine === '') {
      if (inUnorderedList) {
        processedLines.push('</ul>');
        inUnorderedList = false;
      }
      if (inOrderedList) {
        processedLines.push('</ol>');
        inOrderedList = false;
        orderedListStart = 1;
        lastOrderedListNumber = 0;
        linesSinceLastListItem = 0;
      }
      processedLines.push(''); // Preserve blank lines for paragraph breaks
      linesSinceLastListItem++;
      continue;
    }
    
    // Check if line contains text followed by a list item (e.g., "text: 1. item" or "text: - item")
    // This handles cases where a list starts mid-line
    // Look for pattern: text ending with colon/period, then space, then list marker
    const inlineOrderedMatch = trimmedLine.match(/^(.+?)[:\.]\s+(\d+)\.\s+(.+)$/);
    const inlineUnorderedMatch = trimmedLine.match(/^(.+?)[:\.]\s+([\-\*])\s+(.+)$/);
    
    if (inlineOrderedMatch && !trimmedLine.match(/^(\d+)\. (.+)$/)) {
      // Split the line: text part and list item part
      const textPart = inlineOrderedMatch[1].trim();
      const listNumber = inlineOrderedMatch[2];
      const listContent = inlineOrderedMatch[3].trim();
      
      // Process the text part first (as a regular line)
      if (textPart) {
        // Close any open lists before the text
        if (inUnorderedList) {
          processedLines.push('</ul>');
          inUnorderedList = false;
        }
        if (inOrderedList) {
          processedLines.push('</ol>');
          inOrderedList = false;
          orderedListStart = 1;
          lastOrderedListNumber = 0;
          linesSinceLastListItem = 0;
        }
        processedLines.push(textPart);
        linesSinceLastListItem++;
      }
      
      // Now process the list item (will trigger list start if needed)
      trimmedLine = `${listNumber}. ${listContent}`;
      line = trimmedLine; // Update for processing below
    } else if (inlineUnorderedMatch && !trimmedLine.match(/^[\-\*] (.+)$/)) {
      // Split the line: text part and list item part
      const textPart = inlineUnorderedMatch[1].trim();
      const listMarker = inlineUnorderedMatch[2];
      const listContent = inlineUnorderedMatch[3].trim();
      
      // Process the text part first (as a regular line)
      if (textPart) {
        // Close any open lists before the text
        if (inUnorderedList) {
          processedLines.push('</ul>');
          inUnorderedList = false;
        }
        if (inOrderedList) {
          processedLines.push('</ol>');
          inOrderedList = false;
          orderedListStart = 1;
          lastOrderedListNumber = 0;
          linesSinceLastListItem = 0;
        }
        processedLines.push(textPart);
        linesSinceLastListItem++;
      }
      
      // Now process the list item (will trigger list start if needed)
      trimmedLine = `${listMarker} ${listContent}`;
      line = trimmedLine; // Update for processing below
    }
    
    const unorderedMatch = trimmedLine.match(/^[\-\*] (.+)$/);
    const orderedMatch = trimmedLine.match(/^(\d+)\. (.+)$/);
    
    if (unorderedMatch) {
      if (!inUnorderedList) {
        if (inOrderedList) {
          processedLines.push('</ol>');
          inOrderedList = false;
          orderedListStart = 1;
          lastOrderedListNumber = 0;
        }
        // Close any open paragraph before starting a list
        // Check if the last processed line was text (not a block element)
        if (processedLines.length > 0) {
          const lastLine = processedLines[processedLines.length - 1];
          // If last line is not a block element (opening or closing tag) and not empty, it's likely paragraph content
          if (lastLine && !lastLine.match(/^<\/?(ul|ol|h[1-6]|pre|blockquote|div|p|li)/) && lastLine.trim() && lastLine !== '') {
            // Insert a paragraph break marker to ensure proper separation
            processedLines.push('__PARAGRAPH_BREAK__');
          }
        }
        processedLines.push('<ul>');
        inUnorderedList = true;
      }
      processedLines.push(`<li>${unorderedMatch[1]}</li>`);
      linesSinceLastListItem = 0;
    } else if (orderedMatch) {
      const listNumber = parseInt(orderedMatch[1], 10);
      const listContent = orderedMatch[2];
      
      // If we're not in a list, start one
      if (!inOrderedList) {
        if (inUnorderedList) {
          processedLines.push('</ul>');
          inUnorderedList = false;
        }
        // Close any open paragraph before starting a list
        // Check if the last processed line was text (not a block element)
        if (processedLines.length > 0) {
          const lastLine = processedLines[processedLines.length - 1];
          // If last line is not a block element (opening or closing tag) and not empty, it's likely paragraph content
          if (lastLine && !lastLine.match(/^<\/?(ul|ol|h[1-6]|pre|blockquote|div|p|li)/) && lastLine.trim() && lastLine !== '') {
            // Insert a paragraph break marker to ensure proper separation
            processedLines.push('__PARAGRAPH_BREAK__');
          }
        }
        processedLines.push('<ol>');
        inOrderedList = true;
        orderedListStart = listNumber;
        lastOrderedListNumber = listNumber;
        linesSinceLastListItem = 0;
      } else {
        // We're already in a list
        // If we see "1." and we were recently in a list (within 3 lines), continue the list
        // This handles cases where the model restarts numbering but it's actually a continuation
        if (listNumber === 1 && lastOrderedListNumber > 0 && linesSinceLastListItem <= 3) {
          // Continue the list - the browser will auto-number this as the next item
          // The content already has the number stripped (listContent), so it will render correctly
          lastOrderedListNumber = listNumber;
          linesSinceLastListItem = 0;
        } else if (listNumber === 1 && orderedListStart !== 1 && linesSinceLastListItem > 3) {
          // It's been too long since the last item - start a new list
          processedLines.push('</ol>');
          processedLines.push('<ol>');
          orderedListStart = 1;
          lastOrderedListNumber = 1;
          linesSinceLastListItem = 0;
        } else {
          // Normal continuation - update tracking
          lastOrderedListNumber = listNumber;
          linesSinceLastListItem = 0;
        }
      }
      // Use listContent which already has the number prefix stripped by the regex
      processedLines.push(`<li>${listContent}</li>`);
    } else {
      // Non-list line
      // Only close lists if we've had multiple non-list lines (more than 1)
      // This allows lists to continue across single-line interruptions
      if (inUnorderedList && linesSinceLastListItem > 1) {
        processedLines.push('</ul>');
        inUnorderedList = false;
      }
      if (inOrderedList && linesSinceLastListItem > 1) {
        processedLines.push('</ol>');
        inOrderedList = false;
        orderedListStart = 1;
        lastOrderedListNumber = 0;
      }
      processedLines.push(line);
      linesSinceLastListItem++;
    }
  }
  
  // Close any open lists
  if (inUnorderedList) processedLines.push('</ul>');
  if (inOrderedList) processedLines.push('</ol>');
  
  html = processedLines.join('\n');
  
  // Handle paragraph break markers before processing line breaks
  // Replace markers with double line breaks to ensure proper paragraph separation
  html = html.replace(/__PARAGRAPH_BREAK__/g, '\n\n');
  
  // Normalize multiple blank lines to double line breaks
  html = html.replace(/\n\n+/g, '\n\n');
  
  // Split by double line breaks to identify paragraphs
  // Use a regex that matches double line breaks but doesn't split inside block elements
  // First, protect block elements by temporarily replacing them
  const blockElementPlaceholders = [];
  let blockPlaceholderIndex = 0;
  html = html.replace(/(<(ul|ol|h[1-6]|pre|blockquote|div|p)[^>]*>[\s\S]*?<\/\2>)/gi, (match) => {
    const placeholder = `__BLOCK_ELEMENT_${blockPlaceholderIndex}__`;
    blockElementPlaceholders[blockPlaceholderIndex] = match;
    blockPlaceholderIndex++;
    return placeholder;
  });
  
  // Now split by double line breaks
  const parts = html.split(/\n\n/);
  const processedParts = [];
  
  for (let part of parts) {
    if (!part.trim()) continue; // Skip empty parts
    
    // Restore block element placeholders
    for (let i = 0; i < blockElementPlaceholders.length; i++) {
      part = part.replace(`__BLOCK_ELEMENT_${i}__`, blockElementPlaceholders[i]);
    }
    
    const trimmedPart = part.trim();
    
    // Check if this part is already a block element
    const isBlockElement = /^\s*<(ul|ol|h[1-6]|pre|blockquote|div|p)/.test(trimmedPart);
    
    if (isBlockElement) {
      // Already a block element, keep as-is (convert internal line breaks to <br> if needed)
      processedParts.push(trimmedPart.replace(/\n/g, '<br>'));
    } else {
      // Regular text - convert single line breaks to <br> and wrap in paragraph
      const textWithBreaks = trimmedPart.replace(/\n/g, '<br>');
      processedParts.push(`<p>${textWithBreaks}</p>`);
    }
  }
  
  html = processedParts.join('\n');
  
  // Restore line breaks in code blocks and pre blocks
  html = html.replace(/(<pre>[\s\S]*?<\/pre>)/g, (match) => match.replace(/<br>/g, '\n'));
  
  // Handle line breaks within list items
  // Remove <br> that are at the start/end of list items (from line breaks between list items)
  // But preserve <br> in the middle of list item content for multi-line items
  html = html.replace(/(<li>)(<br>\s*)+/g, '$1'); // Remove leading <br> in list items
  html = html.replace(/(\s*<br>)+(<\/li>)/g, '$2'); // Remove trailing <br> in list items
  
  // Clean up empty paragraphs and excessive <br> tags
  html = html.replace(/<p>\s*<\/p>/g, '');
  html = html.replace(/<p>(<br>\s*)+<\/p>/g, '');
  html = html.replace(/(<br>\s*){3,}/g, '<br><br>'); // Max 2 consecutive <br>
  
  return html;
}

// Show non-linguistic error indicator (per PDF: "surface technical error states in logs or UI status indicators, but must not generate conversational language")
function showErrorIndicator(errorMessage) {
  const messagesDiv = document.getElementById('chatMessages');
  if (!messagesDiv) return;
  
  // Create a non-linguistic error indicator (icon + technical code, not conversational text)
  const errorDiv = document.createElement('div');
  errorDiv.className = 'message error-indicator';
  errorDiv.style.cssText = 'padding: 8px 12px; margin: 8px 0; background: #ffebee; border-left: 3px solid #f44336; color: #c62828; font-size: 12px; font-family: monospace;';
  errorDiv.innerHTML = `⚠️ <span style="font-weight: bold;">ERROR</span> ${errorMessage}`;
  
  messagesDiv.appendChild(errorDiv);
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (errorDiv.parentNode) {
      errorDiv.parentNode.removeChild(errorDiv);
    }
  }, 5000);
  
  // Scroll to bottom
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addMessage(role, content, addToHistory = true, meta = {}) {
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
  // Defensive trim to avoid trailing multilingual or meta leakage
  if (typeof content === 'string') {
    content = content.trim();
  }
  
  // Guardrail: Render-time language guard - check for non-English text (threshold-based)
  // Tier 2 Fix: Disable guardrails during expansion (meta.expansion indicates expansion)
  // Per PDF: "Do not rewrite responses by replacing them with renderer-authored text"
  // If non-English detected, log only for Mistral, do not block rendering
  if (role === 'assistant' && typeof content === 'string' && !meta.expansion) {
    if (hasSignificantNonEnglish(content, 0.02)) {
      console.warn('[GUARDRAIL] Non-English characters detected, but allowing render for Mistral');
    }
  }
  
  // Render markdown for assistant messages, plain text for user messages
  if (role === 'assistant') {
    contentDiv.innerHTML = markdownToHtml(content);
  } else {
    // User messages stay as plain text for security
    contentDiv.textContent = content;
  }

  messageDiv.appendChild(header);
  messageDiv.appendChild(contentDiv);

  // Add "Expand on this" button for assistant messages, except for opening message and expansions
  if (role === 'assistant' && !meta.isOpening && !meta.expansion) {
    // Check if this message has already been expanded by looking for expansions in history
    // We need to check if any expansion references this message's content
    let hasBeenExpanded = false;
    if (Array.isArray(conversationHistory)) {
      // Find the index of this message in history (if it's being added to history)
      const currentHistoryLength = conversationHistory.length;
      const messageIndex = addToHistory ? currentHistoryLength : -1;
      
      // Check if any expansion references this message
      hasBeenExpanded = conversationHistory.some(msg => 
        msg.role === 'assistant' && 
        msg.meta && 
        msg.meta.expansion &&
        msg.meta.expandedFrom !== undefined &&
        // Check if the expanded message has the same content as this one
        (messageIndex >= 0 ? msg.meta.expandedFrom === messageIndex : false)
      );
      
      // Also check by content match if we can't use index
      if (!hasBeenExpanded && messageIndex < 0) {
        hasBeenExpanded = conversationHistory.some(msg => 
          msg.role === 'assistant' && 
          msg.meta && 
          msg.meta.expansion &&
          // Find the original message that was expanded
          conversationHistory[msg.meta.expandedFrom] &&
          conversationHistory[msg.meta.expandedFrom].content === content
        );
      }
    }
    
    // Only show button if this message hasn't been expanded yet
    if (!hasBeenExpanded) {
      // Ensure message div uses flexbox for proper button positioning
      messageDiv.style.display = 'flex';
      messageDiv.style.flexDirection = 'column';
      
      // Store the message content as a data attribute for tracking
      messageDiv.setAttribute('data-message-content', content);
      
      const expandBtn = document.createElement('button');
      expandBtn.className = 'expand-btn';
      expandBtn.textContent = 'Expand on this';
      expandBtn.style.cssText =
        'margin-top:6px;font-size:11px;padding:3px 8px;' +
        'width:auto;align-self:flex-end;' +
        'background:#81c784;color:#fff;border:none;' +
        'border-radius:6px;cursor:pointer;opacity:0.7;';
      expandBtn.onmouseenter = () => {
        if (!expandBtn.disabled) {
          expandBtn.style.opacity = '0.9';
        }
      };
      expandBtn.onmouseleave = () => {
        if (!expandBtn.disabled) {
          expandBtn.style.opacity = '0.7';
        }
      };
      expandBtn.addEventListener('click', () => {
        // Disable button immediately to prevent multiple clicks
        expandBtn.disabled = true;
        expandBtn.style.opacity = '0.5';
        expandBtn.style.cursor = 'not-allowed';
        expandBtn.textContent = 'Expanding...';
        
        handleExpandAssistantMessage(content, expandBtn);
      });
      messageDiv.appendChild(expandBtn);
    }
  }

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
  // Reset per-turn state
  triggersQueuedThisTurn.clear();
  pendingExemplarSystemTexts = [];
  
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
  
  // Infer conversation phase
  const inferredPhase = inferConversationPhase(conversationHistory, behaviorPacks);
  console.log('[SEND] Inferred conversation phase:', inferredPhase);

  // --- Continuation mode detection ---
  const continuationMode = isContinuationPrompt(userText);
  console.log('[SEND] Continuation mode:', continuationMode);

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

  if (!continuationMode) {
    if (detectAskedForNextSteps(userText)) {
      // Prefer newer pack keys if present
      if (behaviorPacks?.triggers?.structuring_invitation) {
        queueTrigger('structuring_invitation');
        console.warn('[BEHAVIOR] Triggered structuring_invitation from user intent');
      } else if (behaviorPacks?.triggers?.expand_into_practice) {
        queueTrigger('expand_into_practice');
        console.warn('[BEHAVIOR] Triggered expand_into_practice from user intent');
      } else if (behaviorPacks?.triggers?.asked_for_next_steps) {
        // Back-compat
        queueTrigger('asked_for_next_steps');
        console.warn('[BEHAVIOR] Triggered asked_for_next_steps from user intent');
      }
    }
  }

  const sendBtn = document.getElementById('sendBtn');
  const versionSelect = document.getElementById('versionSelect');

  // Phase-aware system prompt (only include reflection instruction on kickoff)
  const isKickoffTurn = !kickoffIssued && conversationHistory.length === 0;
  const behaviorSystem = buildSystemFromBehaviorPack(inferredPhase, isKickoffTurn);
  if (!behaviorSystem) {
    console.log('[BEHAVIOR] No system message from behavior_packs.json (profile_intent may be empty)');
  }

  // Inject Shared Understanding on non-kickoff turns
  const sharedUnderstandingBlock = !isKickoffTurn ? formatSharedUnderstanding() : '';
  
  // Inject working memory as background
  const memoryPrepend = formatWorkingMemoryForPrompt();
  const combinedSystem = [behaviorSystem, sharedUnderstandingBlock, memoryPrepend].filter(Boolean).join("\n\n");
  
  // Fire diagnostic trigger only when appropriate
  if (inferredPhase === 'reflection' && !kickoffIssued && userText.length > 40) {
    queueTrigger('diagnostic_pause');
    console.log('[BEHAVIOR] Triggered diagnostic_pause for reflection phase');
  }
  
  // Prevent kickoff double nudging
  if (!kickoffIssued && behaviorPacks?.triggers?.conversation_start) {
    queueTrigger('conversation_start');
    console.log('[BEHAVIOR] Triggered conversation_start (first user message)');
  }
  
  // Wrap system and exemplars once
  const effectiveSystemMessage = wrapSystemBlocks(combinedSystem, pendingExemplarSystemTexts);
  
  // Enforce one-shot semantics
  pendingExemplarSystemTexts = [];
  triggersQueuedThisTurn.clear();

  // Log what we're sending to verify it's correct
  console.log('[SEND] User message (exactly as typed):', userText);
  console.log('[SEND] Message length:', userText.length);
  console.log('[SEND] System message length:', effectiveSystemMessage.length);
  console.log('[SEND] One-shot exemplar count:', pendingExemplarSystemTexts.length);
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
  // Tier 1 Fix: Reduce default max_new_tokens to 600 for normal responses
  // Reserve higher token limits (up to 900) only for explicit expansion requests
  const gen = behaviorPacks?.generation_defaults || {};
  if (Object.keys(gen).length > 0) {
    console.log('[BEHAVIOR] Using generation_defaults from behavior_packs.json:', gen);
  }
  // Per PDF: "Keep max_new_tokens in the 600-700 range to reduce looping"
  let requestedMaxTokens = 600; // Default to 600 for Mistral
  if (gen.soft_max_tokens) {
    // Cap at 700 for normal responses (Mistral range), only use higher for expansions
    requestedMaxTokens = Math.min(gen.soft_max_tokens, 700);
    console.log(`[BEHAVIOR] Using soft_max_tokens ${gen.soft_max_tokens} from behavior_packs.json (capped at 700 for Mistral)`);
  }
  // Remove verbosity multiplier - keep consistent token limits
  // Tier 1: Do not use 1200 tokens as default for first responses
  console.log(`[BEHAVIOR] Using max_new_tokens: ${requestedMaxTokens} for normal response`);

  // --- Rewrite attempt counter for hard stop (max 1 per turn) ---
  let rewriteAttempts = 0;

  try {
    // Conversation history is always sent (always enabled)
    let requestBody;

    // Fix 1 & 2: Compress conversation history - remove full assistant responses, use semantic summaries instead
    // Pass current user message to check for pronouns/indexicals
    let compressedHistory = compressConversationHistory(conversationHistory, userText);
    
    // Safety check: Ensure compressed history never ends with a user message
    // This is a final safeguard to prevent consecutive user messages
    // Only remove the last one if it exists (compressConversationHistory already removed the last one, but double-check)
    if (compressedHistory.length > 0 && compressedHistory[compressedHistory.length - 1].role === 'user') {
      compressedHistory.pop();
      console.log(`[COMPRESS] Safety check: Removed last user message from compressed history`);
    }
    
    // CRITICAL: Validate message sequence before sending
    // Build the full messages array as it will be sent (system + history + user)
    // This simulates what main.js will build to validate the sequence
    const testMessages = [];
    if (effectiveSystemMessage && effectiveSystemMessage.trim()) {
      testMessages.push({ role: 'system', content: effectiveSystemMessage.trim() });
    }
    testMessages.push(...compressedHistory);
    testMessages.push({ role: 'user', content: userText });
    
    // Validate: messages[0].role === "system" (MUST be true)
    // Validate: messages.slice(1).every((m, i) => i === 0 || m.role !== messages[i].role)
    // This means: after system, no two consecutive messages should have the same role
    // Note: messages[i] in the slice context refers to the message at index i in the sliced array
    // We need to compare with the previous message in the slice: messages.slice(1)[i - 1]
    // Validate according to user's specification:
    // messages[0].role === "system" (MUST be true)
    // messages.slice(1).every((m, i) => i === 0 || m.role !== messages[i].role)
    // Note: In the slice context, messages[i] refers to messages[i+1] in the original array
    // So we're checking: m.role !== messages[i+1].role, which means comparing with the message at index i+1
    // But we want to compare with the previous message, so we use messages[i+1] where i is the slice index
    // Actually, the condition m.role !== messages[i].role where i is the slice index means:
    // Compare current message role with the role of the message at index i in the original array
    // Since we sliced from index 1, messages[i] in original = messages[i+1]
    // So we're comparing m (at slice index i, original index i+1) with messages[i+1] - which is itself!
    // I think the user meant to compare with the previous message, so let's do that:
    const isValid = testMessages.length > 0 && 
                    testMessages[0].role === "system" &&
                    testMessages.slice(1).every((m, i) => {
                      if (i === 0) return true; // First message after system is always valid
                      // Compare with previous message: at slice index i-1, which is original index i
                      return m.role !== testMessages[i].role;
                    });
    
    if (!isValid) {
      console.warn('[COMPRESS] Message sequence validation failed, rebuilding compressed history...');
      console.warn('[COMPRESS] Test messages:', testMessages.map(m => m.role));
      
      // Rebuild: ensure alternating pattern after system message
      // Since compressedHistory only contains user messages (assistants were removed),
      // we need to rebuild to ensure no consecutive user messages
      const rebuilt = [];
      let lastRole = 'system'; // Start after system message
      
      // Add messages one by one, ensuring they alternate
      // Since we only have user messages in compressedHistory, we need to skip consecutive ones
      for (const msg of compressedHistory) {
        if (msg.role === 'user' && lastRole !== 'user') {
          rebuilt.push(msg);
          lastRole = 'user';
        } else if (msg.role === 'assistant' && lastRole !== 'assistant') {
          rebuilt.push(msg);
          lastRole = 'assistant';
        }
        // Skip messages that would create consecutive same-role
      }
      
      // Ensure it doesn't end with user (since we're adding user next)
      if (rebuilt.length > 0 && rebuilt[rebuilt.length - 1].role === 'user') {
        rebuilt.pop();
      }
      
      compressedHistory = rebuilt;
      console.log('[COMPRESS] Rebuilt compressed history:', compressedHistory.length, 'messages');
      
      // Re-validate
      const retestMessages = [];
      if (effectiveSystemMessage && effectiveSystemMessage.trim()) {
        retestMessages.push({ role: 'system', content: effectiveSystemMessage.trim() });
      }
      retestMessages.push(...compressedHistory);
      retestMessages.push({ role: 'user', content: userText });
      
      const retestValid = retestMessages.length > 0 && 
                          retestMessages[0].role === "system" &&
                          retestMessages.slice(1).every((m, i) => {
                            if (i === 0) return true;
                            // Compare with previous message: at slice index i-1, which is original index i
                            return m.role !== retestMessages[i].role;
                          });
      
      if (!retestValid) {
        console.error('[COMPRESS] Rebuilt history still invalid! Falling back to empty history.');
        compressedHistory = [];
      } else {
        console.log('[COMPRESS] Rebuilt history validated successfully');
      }
    } else {
      console.log('[COMPRESS] Message sequence validation passed');
    }

    if (continuationMode) {
      // Fix 5: Continuation uses working memory, not full assistant prose
      const continuationPrompt = buildContinuationInstruction();

      requestBody = {
        message: continuationPrompt,
        version: window._autoSelectedVersion || 'base',
        prependedText: effectiveSystemMessage,
        useSummary: false,
        conversationSummary: '',
        conversationHistory: compressedHistory, // Use compressed history
        temperature: getBehaviorPackTemperature(),
        max_tokens: requestedMaxTokens
      };

      console.log('[SEND] Continuation mode active; rewrite logic will be skipped');
    } else {
      requestBody = {
        message: userText,
        version: window._autoSelectedVersion || 'base',
        prependedText: effectiveSystemMessage,
        useSummary: false,
        conversationSummary: '',
        conversationHistory: compressedHistory, // Use compressed history
        temperature: getBehaviorPackTemperature(),
        max_tokens: requestedMaxTokens
      };
    }
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
      const initialAssistantText = assistantText;
      
      // Clean up formatting artifacts (CSS syntax, regex patterns, etc.) immediately
      if (assistantText) {
        assistantText = cleanFormattingArtifacts(assistantText);
      }
      
      // Guardrail: Generation-time hard guard - inspect output immediately after generation
      // Use threshold-based check (2% non-ASCII) instead of binary check
      const hasNonEnglish = !continuationMode && assistantText && hasSignificantNonEnglish(assistantText, 0.02);
      if (hasNonEnglish) {
        const nonAsciiCount = (assistantText.match(/[^\x00-\x7F]/g) || []).length;
        const ratio = (nonAsciiCount / assistantText.length * 100).toFixed(2);
        console.warn(`[GUARDRAIL] Significant non-English content detected (${ratio}% non-ASCII); will force rewrite`);
      }

      // If this was a continuation prompt, skip all rewrite / enforcement logic
      if (continuationMode) {
        console.warn('[CONTINUATION] Continuation prompt detected; skipping behavior triggers, rewrites, and enforcement checks');

        // Per PDF: "Do not rewrite responses by replacing them with renderer-authored text"
        // If no response, skip displaying rather than using fallback
        if (assistantText && assistantText.trim()) {
          addMessage('assistant', assistantText.trim());
          saveConfig();
        } else {
          console.warn('[CONTINUATION] Empty continuation response; skipping display (no fallback)');
        }
        pendingExemplarSystemTexts = [];
        return;
      }

      // Tier 2 Fix: Remove all automatic expansion triggers based on response evaluation
      // No automatic expansion triggers - expansion is only user-initiated
      // Removed automatic expansion triggers (expand_into_practice queueing)
      console.log("[BEHAVIOR] Automatic expansion triggers disabled - expansion is user-initiated only");
      
      // Per PDF: "Remove auto-expansion rewrites triggered solely by paragraph count"
      // Removed: min_paragraphs enforcement (no longer triggers rewrites)
      // Behavior pack respects generation_defaults without forcing expansion

      // Only call addMessage('assistant', ...) after all rewrites and checks.
      let handled = false;

      if (assistantText.length === 0) {
        // Per PDF: "Eliminate any default assistant content used when responses are short, empty, or malformed"
        // Empty response - skip displaying (no fallback)
        console.warn('[CHAT] Received empty assistant response - skipping display (no fallback)');
        handled = true;
      }

      // Helper to check rewrite limit
      function rewriteLimitReached() {
        return rewriteAttempts >= 1;
      }

      // Per PDF: "Remove forced rewrites unless a guardrail is violated (e.g., non-English output)"
      // Per PDF: "Remove diagnostic or corrective system messages injected mid-conversation"
      // Removed: diagnostic-opening rewrite (not a hard constraint violation)
      // Opening rules are informational only, not trigger for rewrites

      // Per PDF: "Only trigger a rewrite when a hard constraint is violated"
      // Per PDF: "Rewrite prompts must request neutral transformation (e.g., language-only rewrite) and must not introduce new coaching content"
      // Hard constraint: non-English output (guardrail violation)
      if (!handled && hasNonEnglish && !continuationMode) {
        if (rewriteLimitReached()) {
          console.warn('[GUARDRAIL] Rewrite limit reached; skipping display (no fallback)');
          handled = true;
        } else {
          rewriteAttempts++;
          console.warn('[GUARDRAIL] Non-English content detected. Forcing language-only rewrite.');
          // Per PDF: Neutral transformation only - no exemplars, no coaching content
          const behaviorSystem = buildSystemFromBehaviorPack();
          const memoryPrepend = formatWorkingMemoryForPrompt();
          const combinedSystem = [behaviorSystem, memoryPrepend].filter(Boolean).join("\n\n");
          const rewritePrompt = buildRewriteInstruction(userText, assistantText);
          try {
            const retry = await window.electronAPI.sendChatMessage({
              message: rewritePrompt,
              version: window._autoSelectedVersion || 'base',
              prependedText: wrapSystemBlocks(combinedSystem, []), // No exemplars for guardrail rewrites
              useSummary: false,
              conversationSummary: '',
              conversationHistory: conversationHistory,
              temperature: behaviorPacks?.generation_settings?.temperature,
              max_tokens: requestedMaxTokens
            });
            if (retry.success && (retry.response?.trim() ?? '').length > 0) {
              const rewritten = retry.response.trim();
              // Verify the rewrite has acceptable non-English content (threshold-based)
              if (!hasSignificantNonEnglish(rewritten, 0.02)) {
                assistantText = rewritten;
                handled = false; // allow next checks
              } else {
                console.warn('[GUARDRAIL] Rewrite still contains non-English; skipping display (no fallback)');
                handled = true;
              }
            } else {
              console.warn('[GUARDRAIL] Rewrite failed; skipping display (no fallback)');
              handled = true;
            }
          } catch (e) {
            console.error('[GUARDRAIL] Error during language-only rewrite:', e);
            handled = true;
          }
        }
      }

      // Per PDF: "Only trigger a rewrite when a hard constraint is violated"
      // Forbidden terms are not a hard constraint - they're style preferences
      // Removed: forbidden terms rewrite (not a hard constraint violation)
      // Style rules are informational only, not trigger for rewrites

      // Tier 2 Fix: Remove all automatic expansion triggers based on response evaluation
      // No automatic expansion rewrites - expansion is only user-initiated via button
      // Removed automatic expansion rewrite based on lacksConcretePractice
      console.log('[BEHAVIOR] Automatic expansion rewrites disabled - expansion is user-initiated only');

      // Final UI display logic:
      // Per PDF: "Eliminate any default assistant content used when responses are short, empty, or malformed"
      // - Any non-empty assistantText is displayed
      // - If empty or all rewrites fail, skip display (no fallback)
      if (!handled && assistantText && assistantText.trim().length > 0) {
        addMessage('assistant', assistantText.trim());
        saveConfig();
      } else if (!handled) {
        // If ALL rewrites fail and there's no usable output, skip displaying (no fallback)
        console.warn('[CHAT] No usable response after all rewrites; skipping display (no fallback)');
        // Do not display any message - handled flag already set
      }
    } else {
      // Per PDF: "In the event of timeouts, empty responses, or server errors, the renderer may surface technical error states in logs or UI status indicators, but must not generate conversational language"
      const errorMsg = result.error || 'Unknown error';
      console.error('[CHAT] Chat error:', errorMsg);
      // Show non-linguistic error indicator instead of conversational message
      showErrorIndicator(errorMsg);
      // Do NOT mutate conversation history here - errors are not part of the conversation
    }
    
    // Clear one-shot exemplar after request attempt (one-shot semantics)
    pendingExemplarSystemTexts = [];
  } catch (error) {
    // Per PDF: "In the event of timeouts, empty responses, or server errors, the renderer may surface technical error states in logs or UI status indicators, but must not generate conversational language"
    console.error('[CHAT] Error sending message:', error);
    // Show non-linguistic error indicator instead of conversational message
    showErrorIndicator(error.message);
    // Do NOT mutate conversation history here - errors are not part of the conversation
    
    // Clear one-shot exemplar on error (one-shot semantics)
    pendingExemplarSystemTexts = [];
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


// Handler for "Expand on this" button in assistant messages
async function handleExpandAssistantMessage(originalAssistantText, expandBtn = null) {
  if (!modelInfo) {
    console.warn('[EXPAND] Model not available; cannot expand message');
    if (expandBtn) {
      expandBtn.disabled = false;
      expandBtn.style.opacity = '0.7';
      expandBtn.style.cursor = 'pointer';
      expandBtn.textContent = 'Expand on this';
    }
    return;
  }

  // Find the last assistant message whose content matches originalAssistantText
  let expandedFromIndex = null;
  if (Array.isArray(conversationHistory)) {
    for (let i = conversationHistory.length - 1; i >= 0; i--) {
      const msg = conversationHistory[i];
      if (msg.role === 'assistant' && msg.content === originalAssistantText && !msg.meta?.expansion) {
        expandedFromIndex = i;
        break;
      }
    }
  }
  
  if (expandedFromIndex === null) {
    console.warn('[EXPAND] Could not find original message to expand');
    if (expandBtn) {
      expandBtn.disabled = false;
      expandBtn.style.opacity = '0.7';
      expandBtn.style.cursor = 'pointer';
      expandBtn.textContent = 'Expand on this';
    }
    return;
  }
  
  // Check if this message has already been expanded by checking if any expansion references this index
  const alreadyExpanded = Array.isArray(conversationHistory) && 
    conversationHistory.some(msg => 
      msg.role === 'assistant' && 
      msg.meta && 
      msg.meta.expansion &&
      msg.meta.expandedFrom === expandedFromIndex
    );
  
  if (alreadyExpanded) {
    console.warn('[EXPAND] This message has already been expanded; ignoring request');
    if (expandBtn) {
      expandBtn.disabled = true;
      expandBtn.style.opacity = '0.5';
      expandBtn.style.cursor = 'not-allowed';
      expandBtn.textContent = 'Already expanded';
    }
    return;
  }

  const expansionPrompt = `
You previously said:

${originalAssistantText}

Please expand on this:
- Add depth, examples, or clarification
- Stay consistent with the same voice and perspective
- Do not change or contradict the original meaning
`.trim();

  console.log('[EXPAND] Sending expansion prompt for assistant message');

  try {
    const versionSelect = document.getElementById('versionSelect');
    // Tier 2 Fix: Expansion must be a distinct mode, triggered only by user action
    // Tier 2 Fix: Disable all guardrails during expansion - use minimal system message
    // Expansion-specific system message - minimal, no guardrails
    const expansionSystemMessage = behaviorPacks?.profile_intent?.identity 
      ? `${behaviorPacks.profile_intent.identity}\n\nRespond only in English.`
      : 'Respond only in English.';
    // Include memory in expansion for context
    const memoryPrepend = formatWorkingMemoryForPrompt();
    const combinedSystem = [expansionSystemMessage, memoryPrepend].filter(Boolean).join("\n\n");
    const effectiveSystemMessage = wrapSystemBlocks(combinedSystem, []);

    // Tier 2 Fix: Expansion requests should send a fresh prompt referencing the prior assistant message
    // Expansion is a UI-level affordance and must not participate in conversational role alternation.
    // It must be sent as a stateless request (no conversation history) to avoid duplicate assistant roles.
    // Tier 2 Fix: Use max_new_tokens ≈ 900 and temperature ≈ 0.65 for expansions
    const result = await window.electronAPI.sendChatMessage({
      message: expansionPrompt,
      version: window._autoSelectedVersion || 'base',
      prependedText: effectiveSystemMessage,
      useSummary: false,
      conversationSummary: '',
      conversationHistory: [], // IMPORTANT: expansion must be stateless to avoid duplicate assistant roles
      temperature: 0.65, // Tier 2: Fixed temperature for expansions
      max_tokens: 900, // Tier 2: Reserve higher token limits (900) for explicit expansion requests
      repetition_penalty: 1.05 // Lower repetition penalty for expansion to allow natural elaboration
    });

    if (result.success && result.response?.trim()) {
      // Tier 2 Fix: Disable all guardrails during expansion
      // Clean up formatting artifacts but skip language checks for expansions
      let expandedText = result.response.trim();
      expandedText = cleanFormattingArtifacts(expandedText);
      
      // Append expanded response to UI and conversationHistory, tagging with meta
      // meta.expansion=true will skip language guardrails in addMessage()
      addMessage('assistant', expandedText, true, {
        expansion: true,
        expandedFrom: expandedFromIndex
      });
      conversationHistory.push({
        role: 'assistant',
        content: expandedText,
        meta: { expandedFrom: expandedFromIndex, expansion: true }
      });
      saveConfig();
      
      // Mark the button as used (it should already be disabled, but ensure it stays that way)
      if (expandBtn) {
        expandBtn.disabled = true;
        expandBtn.style.opacity = '0.5';
        expandBtn.style.cursor = 'not-allowed';
        expandBtn.textContent = 'Expanded';
      }
    } else {
      addMessage('assistant', '(No expansion generated)', false);
      // Re-enable button if expansion failed
      if (expandBtn) {
        expandBtn.disabled = false;
        expandBtn.style.opacity = '0.7';
        expandBtn.style.cursor = 'pointer';
        expandBtn.textContent = 'Expand on this';
      }
    }
  } catch (error) {
    console.error('[EXPAND] Error expanding assistant message:', error);
    addMessage('assistant', 'Error: ' + error.message, false);
    // Re-enable button if expansion failed
    if (expandBtn) {
      expandBtn.disabled = false;
      expandBtn.style.opacity = '0.7';
      expandBtn.style.cursor = 'pointer';
      expandBtn.textContent = 'Expand on this';
    }
  }
}
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getProfiles: () => ipcRenderer.invoke('get-profiles'),
  getProfileVersions: (profileName) => ipcRenderer.invoke('get-profile-versions', profileName),
  getVastApiKey: () => ipcRenderer.invoke('get-vast-api-key'),
  getSSHKeyPath: () => ipcRenderer.invoke('get-ssh-key-path'),
  initializeInferenceEnvironment: (config) => ipcRenderer.invoke('initialize-inference-environment', config),
  testVLLMUrl: (config) => ipcRenderer.invoke('test-vllm-url', config),
  prepareVLLM: (config) => ipcRenderer.invoke('prepare-inference-server', config),
  getModelInfo: () => ipcRenderer.invoke('get-model-info'),
  sendChatMessage: (config) => ipcRenderer.invoke('send-chat-message', config),
  loadConfig: () => ipcRenderer.invoke('load-config'),
  saveConfig: (config) => ipcRenderer.invoke('save-config', config),
});


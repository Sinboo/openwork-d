import { IpcMain, dialog } from 'electron'
import Store from 'electron-store'
import * as fs from 'fs/promises'
import * as path from 'path'
import type { ModelConfig, Provider, ProviderId } from '../types'
import { setWorkspacePath, getWorkspacePath, getCheckpointer } from '../agent/runtime'

// Encrypted store for API keys
const store = new Store({
  name: 'openwork-settings',
  encryptionKey: 'openwork-encryption-key-v1' // In production, derive from machine ID
})

// Provider configurations
const PROVIDERS: Omit<Provider, 'hasApiKey'>[] = [
  { id: 'anthropic', name: 'Anthropic' },
  { id: 'openai', name: 'OpenAI' },
  { id: 'google', name: 'Google AI' }
]

// Environment variable mapping
const ENV_VAR_MAP: Record<ProviderId, string> = {
  anthropic: 'ANTHROPIC_API_KEY',
  openai: 'OPENAI_API_KEY',
  google: 'GOOGLE_API_KEY',
  ollama: '' // Ollama doesn't need API key
}

// Available models configuration
const AVAILABLE_MODELS: ModelConfig[] = [
  {
    id: 'claude-sonnet-4-20250514',
    name: 'Claude Sonnet 4',
    provider: 'anthropic',
    model: 'claude-sonnet-4-20250514',
    description: 'Latest Claude model, best for complex tasks',
    available: true
  },
  {
    id: 'claude-3-5-sonnet-20241022',
    name: 'Claude 3.5 Sonnet',
    provider: 'anthropic',
    model: 'claude-3-5-sonnet-20241022',
    description: 'Excellent balance of speed and capability',
    available: true
  },
  {
    id: 'claude-3-5-haiku-20241022',
    name: 'Claude 3.5 Haiku',
    provider: 'anthropic',
    model: 'claude-3-5-haiku-20241022',
    description: 'Fast and efficient for simpler tasks',
    available: true
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: 'openai',
    model: 'gpt-4o',
    description: 'OpenAI flagship model',
    available: true
  },
  {
    id: 'gpt-4o-mini',
    name: 'GPT-4o Mini',
    provider: 'openai',
    model: 'gpt-4o-mini',
    description: 'Smaller, faster GPT-4o variant',
    available: true
  },
  {
    id: 'gemini-2.0-flash',
    name: 'Gemini 2.0 Flash',
    provider: 'google',
    model: 'gemini-2.0-flash',
    description: 'Google fast model',
    available: true
  }
]

export function registerModelHandlers(ipcMain: IpcMain) {
  // List available models
  ipcMain.handle('models:list', async () => {
    // Check which models have API keys configured
    return AVAILABLE_MODELS.map(model => ({
      ...model,
      available: hasApiKey(model.provider)
    }))
  })

  // Get default model
  ipcMain.handle('models:getDefault', async () => {
    return store.get('defaultModel', 'claude-sonnet-4-20250514') as string
  })

  // Set default model
  ipcMain.handle('models:setDefault', async (_event, modelId: string) => {
    store.set('defaultModel', modelId)
  })

  // Set API key for a provider
  ipcMain.handle('models:setApiKey', async (_event, { provider, apiKey }: { provider: string; apiKey: string }) => {
    store.set(`apiKeys.${provider}`, apiKey)
    
    // Also set as environment variable for the current session
    const envVar = ENV_VAR_MAP[provider as ProviderId]
    if (envVar) {
      process.env[envVar] = apiKey
    }
  })

  // Get API key for a provider
  ipcMain.handle('models:getApiKey', async (_event, provider: string) => {
    return store.get(`apiKeys.${provider}`, null) as string | null
  })

  // Delete API key for a provider
  ipcMain.handle('models:deleteApiKey', async (_event, provider: string) => {
    store.delete(`apiKeys.${provider}`)
    
    // Also clear environment variable for the current session
    const envVar = ENV_VAR_MAP[provider as ProviderId]
    if (envVar) {
      delete process.env[envVar]
    }
  })

  // List providers with their API key status
  ipcMain.handle('models:listProviders', async () => {
    return PROVIDERS.map(provider => ({
      ...provider,
      hasApiKey: hasApiKey(provider.id)
    }))
  })

  // Sync version info
  ipcMain.on('app:version', (event) => {
    event.returnValue = require('../../package.json').version
  })

  // Get workspace path for a thread (from thread metadata)
  ipcMain.handle('workspace:get', async (_event, threadId?: string) => {
    if (!threadId) {
      // Fallback to global setting for backwards compatibility
      return store.get('workspacePath', null) as string | null
    }
    
    // Get from thread metadata via threads:get
    const { getThread } = await import('../db')
    const thread = getThread(threadId)
    if (!thread?.metadata) return null
    
    const metadata = JSON.parse(thread.metadata)
    return metadata.workspacePath || null
  })

  // Set workspace path for a thread (stores in thread metadata)
  ipcMain.handle('workspace:set', async (_event, { threadId, path }: { threadId?: string; path: string | null }) => {
    if (!threadId) {
      // Fallback to global setting
      if (path) {
        store.set('workspacePath', path)
      } else {
        store.delete('workspacePath')
      }
      setWorkspacePath(path)
      return path
    }
    
    // Update thread metadata
    const { getThread, updateThread } = await import('../db')
    const thread = getThread(threadId)
    if (!thread) return null
    
    const metadata = thread.metadata ? JSON.parse(thread.metadata) : {}
    metadata.workspacePath = path
    
    updateThread(threadId, { metadata: JSON.stringify(metadata) })
    
    // Also update runtime for current sync
    setWorkspacePath(path)
    return path
  })

  // Select workspace folder via dialog (for a specific thread)
  ipcMain.handle('workspace:select', async (_event, threadId?: string) => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory', 'createDirectory'],
      title: 'Select Workspace Folder',
      message: 'Choose a folder to sync agent files to'
    })

    if (result.canceled || result.filePaths.length === 0) {
      return null
    }

    const selectedPath = result.filePaths[0]
    
    if (threadId) {
      // Store in thread metadata
      const { getThread, updateThread } = await import('../db')
      const thread = getThread(threadId)
      if (thread) {
        const metadata = thread.metadata ? JSON.parse(thread.metadata) : {}
        metadata.workspacePath = selectedPath
        updateThread(threadId, { metadata: JSON.stringify(metadata) })
      }
    } else {
      // Fallback to global
      store.set('workspacePath', selectedPath)
    }
    
    setWorkspacePath(selectedPath)
    return selectedPath
  })

  // Sync files from thread state to disk (on-demand)
  ipcMain.handle('workspace:syncToDisk', async (_event, { threadId }: { threadId: string }) => {
    const { getThread, updateThread } = await import('../db')
    
    // Get workspace path from thread metadata first
    const thread = getThread(threadId)
    const metadata = thread?.metadata ? JSON.parse(thread.metadata) : {}
    let targetPath = metadata.workspacePath as string | null
    
    // If no path set for this thread, prompt for one
    if (!targetPath) {
      const result = await dialog.showOpenDialog({
        properties: ['openDirectory', 'createDirectory'],
        title: 'Select Folder to Sync Files',
        message: 'Choose where to save the workspace files for this thread'
      })

      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, error: 'No folder selected' }
      }

      targetPath = result.filePaths[0]
      
      // Save to thread metadata
      metadata.workspacePath = targetPath
      updateThread(threadId, { metadata: JSON.stringify(metadata) })
      setWorkspacePath(targetPath)
    }

    try {
      // Get the current state from the checkpointer
      const checkpointer = await getCheckpointer()
      const config = { configurable: { thread_id: threadId } }
      const checkpoint = await checkpointer.getTuple(config)
      
      if (!checkpoint?.checkpoint?.channel_values) {
        return { success: false, error: 'No checkpoint found for thread' }
      }

      const state = checkpoint.checkpoint.channel_values as { 
        files?: Record<string, { content?: string[]; created_at?: string; modified_at?: string }> 
      }
      
      if (!state.files || Object.keys(state.files).length === 0) {
        return { success: false, error: 'No files to sync' }
      }

      // Write each file to disk
      const synced: string[] = []
      const errors: string[] = []

      for (const [filePath, fileData] of Object.entries(state.files)) {
        try {
          // Convert virtual path to disk path
          const relativePath = filePath.startsWith('/') ? filePath.slice(1) : filePath
          const fullPath = path.join(targetPath, relativePath)
          
          // Ensure directory exists
          await fs.mkdir(path.dirname(fullPath), { recursive: true })
          
          // Write file content (join lines with newlines)
          const content = Array.isArray(fileData.content) 
            ? fileData.content.join('\n') 
            : String(fileData.content || '')
          await fs.writeFile(fullPath, content, 'utf-8')
          
          synced.push(filePath)
        } catch (e) {
          errors.push(`${filePath}: ${e instanceof Error ? e.message : 'Unknown error'}`)
        }
      }

      return { 
        success: true, 
        synced, 
        errors,
        targetPath 
      }
    } catch (e) {
      return { 
        success: false, 
        error: e instanceof Error ? e.message : 'Unknown error' 
      }
    }
  })
}

function hasApiKey(provider: string): boolean {
  // Check store first
  const storedKey = store.get(`apiKeys.${provider}`) as string | undefined
  if (storedKey) return true

  // Check environment variables
  const envVar = ENV_VAR_MAP[provider as ProviderId]
  return envVar ? !!process.env[envVar] : false
}

// Export for use in agent runtime
export function getApiKey(provider: string): string | undefined {
  const storedKey = store.get(`apiKeys.${provider}`) as string | undefined
  if (storedKey) return storedKey

  const envVar = ENV_VAR_MAP[provider as ProviderId]
  return envVar ? process.env[envVar] : undefined
}

export function getDefaultModel(): string {
  return store.get('defaultModel', 'claude-sonnet-4-20250514') as string
}

/* eslint-disable @typescript-eslint/no-unused-vars */
import { createDeepAgent } from 'deepagents'
import { app } from 'electron'
import { join } from 'path'
import { getDefaultModel, getApiKey } from '../ipc/models'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatOpenAI } from '@langchain/openai'
import { SqlJsSaver } from '../checkpointer/sqljs-saver'
import { createSyncedBackendFactory } from './synced-backend'

import type * as _lcTypes from 'langchain'
import type * as _lcMessages from '@langchain/core/messages'
import type * as _lcLanggraph from '@langchain/langgraph'
import type * as _lcZodTypes from '@langchain/core/utils/types'

// Singleton checkpointer instance
let checkpointer: SqlJsSaver | null = null

// Global workspace path for filesystem sync
let globalWorkspacePath: string | null = null

/**
 * Set the workspace path for filesystem synchronization.
 * When set, files created by the agent will be synced to this directory.
 */
export function setWorkspacePath(path: string | null) {
  globalWorkspacePath = path
  console.log('[Runtime] Workspace path set to:', path)
}

/**
 * Get the current workspace path.
 */
export function getWorkspacePath(): string | null {
  return globalWorkspacePath
}

export async function getCheckpointer(): Promise<SqlJsSaver> {
  if (!checkpointer) {
    const dbPath = join(app.getPath('userData'), 'langgraph.sqlite')
    checkpointer = new SqlJsSaver(dbPath)
    await checkpointer.initialize()
  }
  return checkpointer
}

// Get the appropriate model instance based on configuration
function getModelInstance(modelId?: string): ChatAnthropic | ChatOpenAI | string {
  const model = modelId || getDefaultModel()
  console.log('[Runtime] Using model:', model)

  // Determine provider from model ID
  if (model.startsWith('claude')) {
    const apiKey = getApiKey('anthropic')
    console.log('[Runtime] Anthropic API key present:', !!apiKey)
    if (!apiKey) {
      throw new Error('Anthropic API key not configured')
    }
    return new ChatAnthropic({
      model,
      anthropicApiKey: apiKey
    })
  } else if (model.startsWith('gpt')) {
    const apiKey = getApiKey('openai')
    console.log('[Runtime] OpenAI API key present:', !!apiKey)
    if (!apiKey) {
      throw new Error('OpenAI API key not configured')
    }
    return new ChatOpenAI({
      model,
      openAIApiKey: apiKey
    })
  } else if (model.startsWith('gemini')) {
    // For Gemini, we'd need @langchain/google-genai
    throw new Error('Gemini support coming soon')
  }

  // Default to model string (let deepagents handle it)
  return model
}

export interface CreateAgentRuntimeOptions {
  /** Model ID to use (defaults to configured default model) */
  modelId?: string
  /** Workspace path to sync files to (overrides global setting) */
  workspacePath?: string | null
}

// Create agent runtime with configured model and checkpointer
export type AgentRuntime = ReturnType<typeof createDeepAgent>

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
export async function createAgentRuntime(options: CreateAgentRuntimeOptions = {}) {
  const { modelId, workspacePath } = options

  console.log('[Runtime] Creating agent runtime...')

  const model = getModelInstance(modelId)
  console.log('[Runtime] Model instance created:', typeof model)

  const saver = await getCheckpointer()
  console.log('[Runtime] Checkpointer ready')

  // Use provided workspace path, fall back to global, or null for no sync
  const syncPath = workspacePath !== undefined ? workspacePath : globalWorkspacePath
  console.log('[Runtime] Sync path:', syncPath)

  // Using type assertion to work around version compatibility issues
  // between @langchain packages and deepagentsjs types
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const agent = createDeepAgent({
    model: model as any,
    checkpointer: saver as any,
    // Use SyncedStateBackend to enable bidirectional disk sync
    backend: createSyncedBackendFactory(syncPath) as any
  })

  console.log('[Runtime] Deep agent created with', syncPath ? 'disk sync' : 'state-only storage')
  return agent
}

// Clean up resources
export async function closeRuntime(): Promise<void> {
  if (checkpointer) {
    await checkpointer.close()
    checkpointer = null
  }
}

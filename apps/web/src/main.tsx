import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App'
import './styles.css'
import { applyDesignTokens } from './ui/applyTokens'

applyDesignTokens()

const root = document.getElementById('root')
if (!root) throw new Error('#root missing from index.html')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

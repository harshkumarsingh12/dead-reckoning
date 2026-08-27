/**
 * WebSocket subscription to the gateway's /live stream.
 *
 * OWNER: Tanmay  |  MILESTONE: M4
 *
 * Reconnects on drop. During a demo the phone hotspot will hiccup at least once, and
 * a UI that needs a manual refresh at that moment is a UI that loses the room.
 */
import { useEffect, useState } from 'react'

import type { TelemetryFrame } from '../types'

const RECONNECT_DELAY_MS = 500

export function useTelemetry(path: string) {
  const [frame, setFrame] = useState<TelemetryFrame | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    // TODO(M4, Tanmay): open the socket, parse frames into setFrame, and reconnect
    // after RECONNECT_DELAY_MS on close. Relative URL on purpose — an absolute host
    // here is how the demo acquires an internet dependency by accident.
    void path
    void RECONNECT_DELAY_MS
    void setFrame
    setConnected(false)
  }, [path])

  return { frame, connected }
}

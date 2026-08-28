/**
 * WebSocket subscription to the gateway's /live stream.
 *
 * OWNER: Tanmay  |  MILESTONE: M4
 *
 * Reconnects on drop. During a demo the phone hotspot will hiccup at least once, and
 * a UI that needs a manual refresh at that moment is a UI that loses the room.
 */
import { useEffect, useRef, useState } from 'react'

import type { TelemetryFrame } from '../types'

const RECONNECT_DELAY_MS = 500

/** Native WebSocket needs an absolute ws(s):// URL; it cannot take a relative path
 * the way fetch() can. Deriving the host from the page itself (not a hardcoded one)
 * is what keeps this relative in spirit -- an absolute host baked in here is how the
 * demo acquires an accidental internet dependency. */
function resolveWsUrl(path: string): string {
  const url = new URL(path, window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

export function useTelemetry(path: string) {
  const [frame, setFrame] = useState<TelemetryFrame | null>(null)
  const [connected, setConnected] = useState(false)

  // Guards against setting state after a deliberate close (unmount / path change) --
  // onclose fires asynchronously and would otherwise race a fresh connect() below.
  const cancelledRef = useRef(false)

  useEffect(() => {
    cancelledRef.current = false
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      if (cancelledRef.current) return
      socket = new WebSocket(resolveWsUrl(path))

      socket.onopen = () => setConnected(true)

      socket.onmessage = (event) => {
        try {
          setFrame(JSON.parse(event.data as string) as TelemetryFrame)
        } catch {
          // A malformed frame must not tear down a live demo socket -- drop it and
          // keep waiting for the next one.
        }
      }

      socket.onclose = () => {
        setConnected(false)
        if (!cancelledRef.current) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS)
        }
      }

      socket.onerror = () => {
        socket?.close()
      }
    }

    connect()

    return () => {
      cancelledRef.current = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [path])

  return { frame, connected }
}

// llama-swap-auth: a tiny bearer-key reverse proxy.
//
// Fronts a keyless, localhost-only llama-swap (127.0.0.1:18080) on an optional LAN-facing
// port, requiring "Authorization: Bearer <key>" (or "x-api-key: <key>") for
// every request except /health. This gives "keys for the network, none for localhost":
// local tools hit llama-swap directly on :18080 (no key); remote machines come
// through here on :4096 and must present a key. llama-swap's own apiKeys are
// global (no localhost exemption), which is why auth lives here instead.
//
// Streaming-safe: FlushInterval=-1 flushes SSE/chat-completion chunks
// immediately, and there are no server write timeouts to cut off long
// generations.
//
// Config via env:
//
//	LSA_LISTEN     default "127.0.0.1:4096"; set explicitly for LAN access
//	LSA_UPSTREAM   default "http://127.0.0.1:18080"
//	LSA_KEYS_FILE  default "$XDG_CONFIG_HOME/llama-swap/api-keys"
//	               falls back to "$HOME/.config/llama-swap/api-keys" (Linux and macOS)
//	               one key per line; blank lines and #comments ignored
package main

import (
	"crypto/subtle"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func loadKeys(path string) map[string]struct{} {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("cannot read keys file %s: %v", path, err)
	}
	keys := map[string]struct{}{}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		keys[line] = struct{}{}
	}
	if len(keys) == 0 {
		log.Fatalf("no keys found in %s", path)
	}
	return keys
}

// presented extracts the token from Authorization: Bearer or x-api-key.
func presented(r *http.Request) string {
	if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
		return strings.TrimSpace(h[len("Bearer "):])
	}
	return strings.TrimSpace(r.Header.Get("x-api-key"))
}

func authorized(keys map[string]struct{}, tok string) bool {
	if tok == "" {
		return false
	}
	for k := range keys {
		if subtle.ConstantTimeCompare([]byte(k), []byte(tok)) == 1 {
			return true
		}
	}
	return false
}

func newHandler(upstream string, keys map[string]struct{}) (http.Handler, error) {
	target, err := url.Parse(upstream)
	if err != nil {
		return nil, fmt.Errorf("invalid upstream URL")
	}
	if (target.Scheme != "http" && target.Scheme != "https") || target.Hostname() == "" || target.User != nil || target.Fragment != "" {
		return nil, fmt.Errorf("upstream must be an absolute HTTP(S) URL with a host and no userinfo or fragment")
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1 // immediate flush: SSE / streaming chat completions
	director := proxy.Director
	proxy.Director = func(r *http.Request) {
		director(r)
		r.Header.Del("Authorization")
		r.Header.Del("X-Api-Key")
		r.Header.Del("Proxy-Authorization")
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Do not canonicalize paths or exempt encoded aliases of /health.
		health := r.URL.Path == "/health" && r.URL.EscapedPath() == "/health"
		if !health && !authorized(keys, presented(r)) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="llama-swap"`)
			http.Error(w, "unauthorized: provide Authorization: Bearer <api-key>",
				http.StatusUnauthorized)
			return
		}
		proxy.ServeHTTP(w, r)
	}), nil
}

// defaultKeysFile follows XDG on every supported platform. os.UserConfigDir is
// deliberately not used: on macOS it returns ~/Library/Application Support,
// which would split the proxy's config from the rest of llms and llama-swap.
func defaultKeysFile() (string, error) {
	configDir := os.Getenv("XDG_CONFIG_HOME")
	if configDir == "" {
		home := os.Getenv("HOME")
		if home == "" {
			return "", fmt.Errorf("neither $XDG_CONFIG_HOME nor $HOME is set")
		}
		configDir = filepath.Join(home, ".config")
	}
	return filepath.Join(configDir, "llama-swap", "api-keys"), nil
}

func main() {
	listen := env("LSA_LISTEN", "127.0.0.1:4096")
	upstream := env("LSA_UPSTREAM", "http://127.0.0.1:18080")
	keysFile := os.Getenv("LSA_KEYS_FILE")
	if keysFile == "" {
		var err error
		keysFile, err = defaultKeysFile()
		if err != nil {
			log.Fatalf("cannot locate config directory; set LSA_KEYS_FILE: %v", err)
		}
	}
	keys := loadKeys(keysFile)
	handler, err := newHandler(upstream, keys)
	if err != nil {
		log.Fatal(err)
	}

	srv := &http.Server{
		Addr:              listen,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       0, // long streaming generations must not be cut off
		WriteTimeout:      0,
		IdleTimeout:       120 * time.Second,
	}
	log.Printf("llama-swap-auth: %s -> %s  (%d key(s) from %s)",
		listen, upstream, len(keys), keysFile)
	log.Fatal(srv.ListenAndServe())
}

package main

import (
	"bufio"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestHandler(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, name := range []string{"Authorization", "X-Api-Key", "Proxy-Authorization"} {
			if r.Header.Get(name) != "" {
				t.Errorf("credential header %s reached upstream", name)
			}
		}
		w.Header().Set("X-Upstream", "yes")
		w.WriteHeader(http.StatusAccepted)
		body, _ := io.ReadAll(r.Body)
		_, _ = io.WriteString(w, r.Method+" "+r.URL.RequestURI()+" "+string(body))
	}))
	defer upstream.Close()
	handler, err := newHandler(upstream.URL, map[string]struct{}{"test-key": {}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, path, bearer, apiKey string
		want                       int
	}{
		{"missing", "/v1/models", "", "", 401},
		{"wrong", "/v1/models", "Bearer wrong", "", 401},
		{"bearer", "/v1/chat/completions?stream=true", "Bearer test-key", "", 202},
		{"api key", "/v1/models", "", "test-key", 202},
		{"both credentials stripped", "/v1/models", "Bearer test-key", "ignored", 202},
		{"bearer takes precedence", "/v1/models", "Bearer wrong", "test-key", 401},
		{"health", "/health", "", "", 202},
		{"health query", "/health?check=1", "", "", 202},
		{"health credentials", "/health", "Bearer ignored", "ignored", 202},
		{"encoded health", "/%68ealth", "", "", 401},
		{"encoded slash", "/%2fhealth", "", "", 401},
		{"double slash", "//health", "", "", 401},
		{"dot segment", "/v1/../health", "", "", 401},
		{"encoded dot segment", "/v1/%2e%2e/health", "", "", 401},
		{"trailing slash", "/health/", "", "", 401},
		{"health child", "/health/v1/models", "", "", 401},
		{"health parent", "/health/../v1/models", "", "", 401},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader("payload"))
			req.Header.Set("Authorization", tc.bearer)
			req.Header.Set("X-Api-Key", tc.apiKey)
			req.Header.Set("Proxy-Authorization", "Basic ignored")
			res := httptest.NewRecorder()
			handler.ServeHTTP(res, req)
			if res.Code != tc.want {
				t.Fatalf("status = %d, want %d", res.Code, tc.want)
			}
			if tc.want == 401 {
				if res.Header().Get("WWW-Authenticate") != `Bearer realm="llama-swap"` || res.Header().Get("X-Upstream") != "" {
					t.Fatal("missing challenge or unauthorized request reached upstream")
				}
			} else if want := "POST " + tc.path + " payload"; res.Body.String() != want {
				t.Fatalf("body = %q, want %q", res.Body.String(), want)
			}
		})
	}
}

func TestUpstreamValidation(t *testing.T) {
	for _, upstream := range []string{"", "/relative", "//localhost:8080", "localhost:8080", "ftp://localhost", "http:///path", "https://:8080", "http://%", "http://user:password@localhost", "http://localhost/#fragment"} {
		if _, err := newHandler(upstream, nil); err == nil {
			t.Errorf("accepted invalid upstream %q", upstream)
		}
	}
	for _, upstream := range []string{"http://localhost:8080", "https://example.com/base?param=value", "http://[::1]:8080"} {
		if _, err := newHandler(upstream, nil); err != nil {
			t.Errorf("rejected valid upstream %q: %v", upstream, err)
		}
	}
}

func TestDefaultKeysFile(t *testing.T) {
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" || runtime.GOOS == "plan9" {
		t.Skip("XDG defaults are Unix-specific")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	for _, xdg := range []string{"", filepath.Join(home, "custom-config")} {
		t.Setenv("XDG_CONFIG_HOME", xdg)
		base := xdg
		if base == "" {
			base = filepath.Join(home, ".config")
		}
		got, err := defaultKeysFile()
		if want := filepath.Join(base, "llama-swap", "api-keys"); err != nil || got != want {
			t.Fatalf("defaultKeysFile() = %q, %v; want %q", got, err, want)
		}
	}
}

func TestStreaming(t *testing.T) {
	release := make(chan struct{}, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, "data: last\n\n")
	}))
	defer upstream.Close()
	handler, err := newHandler(upstream.URL, map[string]struct{}{"test-key": {}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	defer server.Close()
	defer close(release)
	client := &http.Client{Timeout: 5 * time.Second}
	req, _ := http.NewRequest(http.MethodGet, server.URL+"/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", res.StatusCode)
	}
	reader := bufio.NewReader(res.Body)
	line, err := reader.ReadString('\n')
	if err != nil || line != "data: first\n" {
		t.Fatalf("first chunk before upstream completion = %q, %v", line, err)
	}
	release <- struct{}{}
	rest, err := io.ReadAll(reader)
	if err != nil || string(rest) != "\ndata: last\n\n" {
		t.Fatalf("remaining stream = %q, %v", rest, err)
	}
}

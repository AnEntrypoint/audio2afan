#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 8765


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".mjs"):
            return "application/javascript"
        if path.endswith(".js"):
            return "application/javascript"
        if path.endswith(".onnx"):
            return "application/octet-stream"
        return super().guess_type(path)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs


os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(f"Starting server on http://localhost:{PORT}")
print(f"Open http://localhost:{PORT} in your browser")
print("Press Ctrl+C to stop")

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        httpd.shutdown()

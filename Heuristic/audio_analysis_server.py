import http.server
import json

with open('BIS-2A__2019-07-17_12_10.json') as json_file:
    response = json.load(json_file)

response = json.dumps(response)


class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(response.encode())


httpd = http.server.HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
print("serving now...")
httpd.serve_forever()

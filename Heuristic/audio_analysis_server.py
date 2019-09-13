import http.server
import json

files = ['BIS-2A__2019-07-08_12_10', 'BIS-2A__2019-07-10_12_10', 'BIS-2A__2019-07-12_12_10', 'BIS-2A__2019-07-17_15_10', 'BIS-2A__2019-07-17_12_10']

i = 0
full_response = {}

for file in files:
    with open('{}.json'.format(files)) as json_file:
        response = json.load(json_file)

    response = json.dumps(response)
    full_response[file] = response

print(full_response)  # newwww


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

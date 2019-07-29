from http.server import HTTPServer, BaseHTTPRequestHandler
import json

response = {"percent leading/trailing silence trimmed": 20.97,
            "student talking time": 54.8,
            "professor talking time": 325.4,
            "class duration": 417.5,
            "words spoken": 5304,
            "average syllables per word": 1.27,
            "grade level": "average 9th or 10th-grade student",
            "words per minute": 175.10,
            "words per second": 2.92,
            "student labels": 1,  # FIX ME
            "professor labels": 2  # FIX ME
            }

# convert into JSON:
response = json.dumps(response)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(response.encode())


httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()

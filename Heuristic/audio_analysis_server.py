import http.server
import json
# from .main import convert_file

response = {"percent_leading/trailing_silence_trimmed": 20.97,
            "student_talking_time": 54.8,
            "professor_talking_time": 325.4,
            "class_duration": 417.5,
            "words_spoken": 5304,
            "average_syllables_per_word": 1.27,
            "grade_level": "average 9th or 10th-grade student",
            "words_per_minute": 175.10,
            "words_per_second": 2.92,
            "student_labels": 1,  # FIX ME
            "professor_labels": 2  # FIX ME
            }

# convert into JSON:
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
httpd.serve_forever()

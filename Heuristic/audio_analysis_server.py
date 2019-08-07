import http.server
import json
# from .main import convert_file

response = {"percent_leading_trailing_silence_trimmed": 20.97,
            "student_talking_time": 54.8,
            "professor_talking_time": 325.4,
            "class_duration": 417.5,
            "words_spoken": 5304,
            "average_syllables_per_word": 1.27,
            "grade_level": "average 9th or 10th-grade student",
            "words_per_minute": 175.10,
            "words_per_second": 2.92,
            "all_labels": [
                {"start": 0.09287981859410431, "end": 0.3250793650793651, "label": "student"},
                {"start": 0.3250793650793651, "end": 4.4117913832199545, "label": "professor"},
                {"start": 4.4117913832199545, "end": 6.548027210884354, "label": "student"},
                {"start": 6.548027210884354, "end": 7.383945578231293, "label": "professor"},
                {"start": 7.383945578231293, "end": 7.430385487528345, "label": "student"},
                {"start": 7.430385487528345, "end": 9.659501133786849, "label": "silence"},
                {"start": 9.659501133786849, "end": 9.7059410430839, "label": "student"},
                {"start": 9.7059410430839, "end": 18.297324263038547, "label": "professor"},
                {"start": 18.297324263038547, "end": 20.48, "label": "student"},
                {"start": 20.48, "end": 20.805079365079365, "label": "professor"},
                {"start": 20.805079365079365, "end": 20.94439909297052, "label": "student"},
                {"start": 20.94439909297052, "end": 25.402630385487527, "label": "silence"},
                {"start": 25.402630385487527, "end": 27.028027210884353, "label": "professor"},
                {"start": 27.028027210884353, "end": 27.492426303854874, "label": "student"},
                {"start": 27.492426303854874, "end": 29.58222222222222, "label": "silence"},
                {"start": 29.58222222222222, "end": 33.85469387755102, "label": "professor"},
                {"start": 33.85469387755102, "end": 35.898049886621315, "label": "student"},
                {"start": 35.898049886621315, "end": 36.87328798185941, "label": "professor"},
                {"start": 36.87328798185941, "end": 39.10240362811791, "label": "student"},
                {"start": 39.10240362811791, "end": 49.5978231292517, "label": "professor"},
                {"start": 49.5978231292517, "end": 56.610249433106574, "label": "silence"},
                {"start": 56.610249433106574, "end": 56.65668934240363, "label": "student"},
                {"start": 56.65668934240363, "end": 98.35972789115647, "label": "professor"},
                {"start": 98.35972789115647, "end": 100.63528344671202, "label": "student"},
                {"start": 100.63528344671202, "end": 106.67247165532879, "label": "professor"},
                {"start": 106.67247165532879, "end": 108.9015873015873, "label": "student"},
                {"start": 108.9015873015873, "end": 109.04090702947846, "label": "professor"},
                {"start": 109.04090702947846, "end": 112.01306122448979, "label": "student"},
                {"start": 112.01306122448979, "end": 115.68181405895692, "label": "professor"},
                {"start": 115.68181405895692, "end": 118.00380952380952, "label": "silence"},
                {"start": 118.00380952380952, "end": 121.9047619047619, "label": "professor"},
                {"start": 121.9047619047619, "end": 124.59827664399093, "label": "student"},
                {"start": 124.59827664399093, "end": 154.27337868480726, "label": "professor"},
                {"start": 154.27337868480726, "end": 156.50249433106575, "label": "student"},
                {"start": 156.50249433106575, "end": 169.180589569161, "label": "professor"},
                {"start": 169.180589569161, "end": 169.3663492063492, "label": "student"},
                {"start": 169.3663492063492, "end": 176.79673469387754, "label": "silence"},
                {"start": 176.79673469387754, "end": 176.88961451247167, "label": "student"},
                {"start": 176.88961451247167, "end": 190.17142857142858, "label": "professor"},
                {"start": 190.17142857142858, "end": 190.68226757369615, "label": "student"},
                {"start": 190.68226757369615, "end": 194.07238095238094, "label": "silence"},
                {"start": 194.07238095238094, "end": 194.11882086167802, "label": "student"},
                {"start": 194.11882086167802, "end": 197.46249433106576, "label": "professor"},
                {"start": 197.46249433106576, "end": 200.38820861678005, "label": "student"},
                {"start": 200.38820861678005, "end": 214.1907029478458, "label": "professor"}
            ]
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

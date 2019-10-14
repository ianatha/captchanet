import os
import sys

# Minimize Terraform and Keras logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import keras_utils
import greekcaptcha

import numpy
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

global model

def ocr_image(image, model):
    digit_images = greekcaptcha.chop_image(image)

    s = ""
    for digit in digit_images:
        res = model.predict(digit.reshape(1, greekcaptcha.img_rows, greekcaptcha.img_cols, 1))
        s = s + greekcaptcha.LEXICON[res.argmax(1)[0]]
    return s
        
class RestfulCaptchanet(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

    def _text(self, message):
        content = f"{message}"
        return content.encode("utf8")

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self._text("CAPTCHANET//1.0.1.34//240x60,120x30"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        filestr = self.rfile.read(content_length)
        if filestr is b'':
            self._set_headers(500)
            self.wfile.write(self._text("No Image"))
            return
        npimg = numpy.asarray(bytearray(filestr), dtype=numpy.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        if image is None:
            self._set_headers(500)
            self.wfile.write(self._text("Bad Image"))
            return
        self._set_headers(200)            
        self.wfile.write(self._text(ocr_image(image, model)))

class RestfulCaptchanetServer():
    def run(server_class=HTTPServer, handler_class=RestfulCaptchanet, addr='localhost', port=5151):
        server_address = (addr, port)
        httpd = server_class(server_address, handler_class)

        print(f"Starting http server on {addr}:{port}")
        httpd.serve_forever()

    def main():
        global model
        import argparse

        parser = argparse.ArgumentParser(description="Run a captcha guessing HTTP server")

        parser.add_argument(
            "-l",
            "--listen",
            default="0.0.0.0",
            help="Specify the IP address on which the server listens",
        )

        parser.add_argument(
            "-p",
            "--port",
            type=int,
            default=5151,
            help="Specify the port on which the server listens",
        )

        parser.add_argument(
            "-m",
            "--model",
            default="model",
            help="The path for the JSON and H5 files containing the model",
        )

        args = parser.parse_args()

        model = keras_utils.load_model(args.model)
        RestfulCaptchanetServer.run(addr=args.listen, port=args.port)

if __name__ == "__main__":
    RestfulCaptchanetServer.main()

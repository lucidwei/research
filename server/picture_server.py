# coding=gbk
# Time Created: 2023/4/18 14:06
# Author  : Lucid
# FileName: picture_server.py
# Software: PyCharm
# ������ΪͼƬ�����йܷ�������ֻ����κ��������������
from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route("/images/<path:filename>")
def serve_image(filename):
    image_directory = r"D:\WPS_cloud\WPS Cloud Files\����-���\shared_images"
    return send_from_directory(image_directory, filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)

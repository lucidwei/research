# coding=gbk
# Time Created: 2023/4/18 14:06
# Author  : Lucid
# FileName: picture_server.py
# Software: PyCharm
# 本程序为图片本地托管服务器，只需在魏广泽主机上运行
from flask import Flask, send_from_directory
import configparser, os


def is_running_in_docker():
    with open('/proc/1/cgroup', 'rt') as ifh:
        return 'docker' in ifh.read()


if is_running_in_docker():
    config_file_path = '/app/config.ini'
    image_folder = '/app/images'
else:
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    grandparent_directory = os.path.dirname(parent_directory)
    config_file_path = os.path.join(grandparent_directory, 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_file_path, encoding='utf-8')
    image_folder = config.get('image_config', 'folder')

app = Flask(__name__)


@app.route("/images/<path:filename>")
def serve_image(filename):
    image_directory = image_folder
    return send_from_directory(image_directory, filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)

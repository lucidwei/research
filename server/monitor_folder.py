# coding=gbk
# Time Created: 2023/5/9 15:51
# Author  : Lucid
# FileName: monitor_folder.py
# Software: PyCharm
import os
import time
import subprocess
from pathlib import Path
import shutil
import pymsgbox, configparser


def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def get_file_count(folder):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            file_count += 1
    return file_count


def delete_oldest_files(backup_folder, target_size=None, target_file_count=None):
    files = list(Path(backup_folder).glob('*'))
    files.sort(key=lambda x: x.stat().st_mtime)

    if target_size:
        while get_folder_size(backup_folder) > target_size:
            oldest_file = files.pop(0)
            if oldest_file.is_file():
                os.remove(oldest_file)
            elif oldest_file.is_dir():
                shutil.rmtree(oldest_file)

    if target_file_count:
        while get_file_count(backup_folder) > target_file_count:
            oldest_file = files.pop(0)
            if oldest_file.is_file():
                os.remove(oldest_file)
            elif oldest_file.is_dir():
                shutil.rmtree(oldest_file)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
grandparent_directory = os.path.dirname(parent_directory)
config_file_path = os.path.join(grandparent_directory, 'config.ini')
config = configparser.ConfigParser()
config.read(config_file_path, encoding='utf-8')

def main():
    FOLDER_PATH = config.get('image_config', 'folder')
    BACKUP_FOLDER_PATH = os.path.join(FOLDER_PATH, 'backup')
    INTERVAL = 60  # 监控间隔，单位：秒
    FOLDER_SIZE_LIMIT = 300 * 1024 * 1024  # 300 MB
    FILE_COUNT_LIMIT = 1000
    BACKUP_FOLDER_TARGET_SIZE = 100 * 1024 * 1024  # 100 MB
    BACKUP_FOLDER_TARGET_FILE_COUNT = 500
    CONTAINER_NAME = "chfs_container"
    CONTAINER_SIZE_LIMIT = 1 * 1024 * 1024 * 1024  # 1 GB

    while True:
        time.sleep(INTERVAL)

        # 检查容器大小
        container_size_command = f"docker ps -s -f name={CONTAINER_NAME} --format {{.Size}}"
        container_size_output = subprocess.check_output(container_size_command, shell=True).decode('utf-8')
        container_size_str = container_size_output.split()[0]
        container_size = int(container_size_str[:-1])

        if container_size > CONTAINER_SIZE_LIMIT:
            # 关闭容器
            subprocess.run(f"docker stop {CONTAINER_NAME}", shell=True)
            pymsgbox.alert('CHFS Docker container has been stopped due to size limit exceeded.', 'Warning')

        current_size = get_folder_size(FOLDER_PATH)
        if current_size > FOLDER_SIZE_LIMIT:
            # 删除备份文件夹中的旧文件
            delete_oldest_files(BACKUP_FOLDER_PATH, target_size=BACKUP_FOLDER_TARGET_SIZE)
            pymsgbox.alert('Old files have been deleted from the backup folder due to SIZE limit exceeded.', 'Warning')

        current_file_count = get_file_count(FOLDER_PATH)
        if current_file_count > FILE_COUNT_LIMIT:
            # 删除备份文件夹中的旧文件
            delete_oldest_files(BACKUP_FOLDER_PATH, target_file_count=BACKUP_FOLDER_TARGET_FILE_COUNT)
            pymsgbox.alert('Old files have been deleted from the backup folder due to FILE COUNT limit exceeded.', 'Warning')

if __name__ == "__main__":
    main()

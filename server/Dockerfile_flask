# 使用官方的 Python 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 文件复制到容器中
COPY requirements.txt requirements.txt

# 安装依赖项
RUN pip install -r requirements.txt

# 将应用程序代码复制到容器中
COPY . .

# 暴露 Flask 应用程序使用的端口
EXPOSE 8001

# 启动 gunicorn 服务器
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "picture_server:app"]

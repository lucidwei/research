import openpyxl
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

def format_cell_value(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")  # 格式化日期
    return value

def get_formatted_table_data(sheet, range_str):
    cells = sheet[range_str]
    table_data = []
    for row in cells:
        formatted_row = [format_cell_value(cell.value) for cell in row]
        # 如果行中除了第一列之外，其他单元格都为0，则忽略该行
        if all((cell == 0 or cell is None) for cell in formatted_row[1:]):
            continue
        # 对数值进行四舍五入到一位小数
        row_data = [round(cell, 1) if isinstance(cell, (int, float)) else cell for cell in formatted_row]
        table_data.append(row_data)
    return table_data

def save_excel_range_as_image(excel_path, range_str, image_path):
    wb = openpyxl.load_workbook(excel_path, data_only=True)  # 使用 data_only=True 以获取公式的计算结果
    sheet = wb.active
    table_data = get_formatted_table_data(sheet, range_str)

    # 创建表格
    fig, ax = plt.subplots()
    ax.axis('off')  # 隐藏坐标轴
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # 调整字体大小以适应单元格

    # 保存图片时提高DPI
    plt.savefig(image_path, dpi=300)
    plt.close(fig)

def send_email_with_image(image_path, receiver_email):
    msg = MIMEMultipart()
    msg['Subject'] = 'Daily Report'
    msg['From'] = 'your_email@example.com'
    msg['To'] = receiver_email

    with open(image_path, 'rb') as file:
        img = MIMEImage(file.read())
        msg.attach(img)

    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('your_email@example.com', 'your_password')
        server.send_message(msg)


def main():
    excel_path = r"H:\bat_used.xlsm"
    image_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\shared_images\北向两融_image.png'
    # receiver_email = 'your_receiver_email@example.com'

    save_excel_range_as_image(excel_path, 'A2:E10', image_path)
    # send_email_with_image(image_path, receiver_email)

if __name__ == '__main__':
    main()

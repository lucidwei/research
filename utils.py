# coding=gbk
# Time Created: 2023/3/17 8:36
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm
# A place to store static functions
from functools import wraps
import time, re, os, shutil
from datetime import datetime, timedelta, date
from chinese_calendar import is_holiday
from WindPy import w
import pandas as pd
import re


def get_image_url(filename):
    return f"http://localhost:8080/images/{filename.replace(' ', '%20')}"


def backup_file(filepath):
    # 可能的文件扩展名列表
    extensions = ['', '.txt', '.png', '.jpg', '.jpeg', '.pdf']

    # 检查给定的文件路径是否存在，如果不存在，则尝试添加扩展名
    for ext in extensions:
        temp_filepath = filepath + ext
        if os.path.exists(temp_filepath):
            filepath = temp_filepath
            break
    else:
        print(f"No file found with the provided path: {filepath}")
        return

    if os.path.exists(filepath):
        # 获取文件名和扩展名
        filename, ext = os.path.splitext(filepath)

        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 获取备份文件的保存目录
        backup_dir = os.path.join(os.path.dirname(filepath), 'backup')
        # 如果备份目录不存在，则创建目录
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        # 备份文件的新名称和保存路径
        backup_name = f'{os.path.basename(filename)}_{timestamp}{ext}'
        backup_path = os.path.join(backup_dir, backup_name)
        # 备份文件
        shutil.copy(filepath, backup_path)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        print(f'\nFunction {func.__name__} is starting...')
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.2f} seconds')
        return result

    return timeit_wrapper


def get_tradedays(start, end):
    '''
    计算两个日期间的工作日
    start:开始时间
    end:结束时间
    '''

    # 字符串格式日期的处理
    if type(start) == str:
        start = datetime.strptime(start, '%Y-%m-%d').date()
    if type(end) == str:
        end = datetime.strptime(end, '%Y-%m-%d').date()
    # 开始日期大，颠倒开始日期和结束日期
    if start > end:
        start, end = end, start

    counts = 0
    tradedays = []
    while True:
        if start > end:
            break
        if is_holiday(start) or start.weekday() == 5 or start.weekday() == 6:
            start += timedelta(days=1)
            continue
        counts += 1
        tradedays.append(start)
        start += timedelta(days=1)
    # return sorted(tradedays, reverse=True)
    return sorted(tradedays)


def check_wind():
    # 插件可以配置到虚拟环境，方法：量化-数据接口-API接口-手册-python接口-接口手册，添加路径指定python，配置后检查是否登陆成功
    while not w.isconnected():
        w.start()
        if not w.isconnected():
            raise Exception('WindPy连接失败，请查找原因')

    # 检查quota超限
    download_try = w.wsd('600519.SH', "industry_citic", f'2023-02-20', f'2023-02-20',
                         "unit=1;industryType=1")
    if download_try.Data[0][0].__contains__('quota exceeded'):
        raise Warning('取数报错，quota超限')
        pass
    else:
        print('Wind在本次运行中数据尚未超限，如取数报错请检查其它报错原因（如wind本身缺少数据）')


def get_nearest_dates_from_contract(code):
    # 提取年月信息
    match = re.match(r"^(T|TF|TS)?(\d{2})(\d{2})$", code)
    if not match:
        raise ValueError("Invalid code format")
    year = int(match.group(2)) + 2000
    month = int(match.group(3))

    # 计算月份范围
    end_of_month = datetime(year, month, 1) + timedelta(days=32)
    end_of_month = datetime(end_of_month.year, end_of_month.month, 1) - timedelta(days=1)
    start_of_prev_month = datetime(year, month, 1) - timedelta(days=1)
    start_of_prev_month = datetime(start_of_prev_month.year, start_of_prev_month.month, 1)

    # 格式化输出
    start_str = start_of_prev_month.strftime("%Y-%m-%d")
    end_str = end_of_month.strftime("%Y-%m-%d")

    return start_str, end_str


def is_date(value):
    try:
        pd.to_datetime(value)
        return True
    except Exception:
        return False


def has_large_date_gap(date_list):
    """
    判断由日期时间组成的列表中是否存在超过20天间隔的日期
    """
    sorted_dates = sorted(date_list)  # 对日期进行排序
    for i in range(len(sorted_dates) - 1):
        diff = sorted_dates[i + 1] - sorted_dates[i]
        if diff > timedelta(days=20):
            print("存在超过20天间隔的日期：")
            print("左边日期:", sorted_dates[i])
            print("右边日期:", sorted_dates[i + 1])
            return True
    return False

def match_recent_tradedays(date_list, tradedays):
    """
    匹配给定列表中最近的五个交易日和 self.tradedays 的最后五个交易日，并检查对应元素的值是否一致
    """
    recent_dates = date_list[-5:] if len(date_list) >= 5 else date_list
    for i in range(1, min(len(recent_dates)+1, 6)):
        if recent_dates[-i] != tradedays[-i]:
            return False
    return True


def get_month_end_dates(start_date: date, end_date: date):
    """
    返回一个包含给定时间范围内所有月末日期的元组(tuple)，元组包含 datetime 对象和 str 对象两种格式
    :param start_date: 开始日期，datetime.date 对象
    :param end_date: 结束日期，datetime.date 对象
    :return: 元组(tuple)，格式为 ([datetime.date], [str])
    """
    dates = []
    date_strs = []
    temp_date = start_date
    while temp_date <= end_date:
        last_day = date(temp_date.year, temp_date.month, 1) + timedelta(days=32)
        last_day = last_day.replace(day=1) - timedelta(days=1)
        if last_day <= end_date:
            dates.append(last_day)
            date_strs.append(last_day.strftime('%Y-%m-%d'))
        temp_date = last_day + timedelta(days=1)
    return dates, date_strs


def translate_and_convert_to_camel(column_names, manual_translations: dict):
    import translators as ts
    # 每次调用api最多处理的变量个数，防止文本过长。而且同时翻译过多时会有漏翻现象
    max_length = 20
    # 拆分为多个小于最大长度的段
    segments = [column_names[i:i + max_length] for i in range(0, len(column_names), max_length)]
    camel_case_names = []

    for segment in segments:
        # Join the column names with a separator that is unlikely to appear in the translations
        joined = ','.join(segment)
        # Before translating, replace any strings that have manual translations
        for original, translation in manual_translations.items():
            joined = joined.replace(original, translation)

        # Remove the Chinese comma (顿号) 来避免翻译后出现逗号被误识为分隔符
        joined = joined.replace('、', '')

        # Translate the entire string
        translated = ts.translate_text(joined, translator='youdao')

        # Remove the spaces around colons
        translated = translated.replace(': ', ':').replace(' :', ':')
        # Split the translated string by the separator
        translated_parts = translated.split(',')

        for translated_part in translated_parts:
            # 转换成首字母大写
            words = re.split(r'(?<=:)|\s+', translated_part)
            words[0] = words[0].lower()
            for i in range(0, len(words)):
                words[i] = words[i].capitalize()
            translated_part = ' '.join(words)

            formated = translated_part.replace(': ', '_').replace(':', '_').replace(' ', '')
            camel_case_names.append(formated)
    return camel_case_names


def generate_column_name_dict(chinese_column_names, manual_translations: dict):
    english_column_names = translate_and_convert_to_camel(chinese_column_names, manual_translations)
    column_name_dict = dict(zip(chinese_column_names, english_column_names))
    return column_name_dict

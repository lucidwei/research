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
    # ���ܵ��ļ���չ���б�
    extensions = ['', '.txt', '.png', '.jpg', '.jpeg', '.pdf']

    # ���������ļ�·���Ƿ���ڣ���������ڣ����������չ��
    for ext in extensions:
        temp_filepath = filepath + ext
        if os.path.exists(temp_filepath):
            filepath = temp_filepath
            break
    else:
        print(f"No file found with the provided path: {filepath}")
        return

    if os.path.exists(filepath):
        # ��ȡ�ļ�������չ��
        filename, ext = os.path.splitext(filepath)

        # ��ȡ��ǰʱ���
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # ��ȡ�����ļ��ı���Ŀ¼
        backup_dir = os.path.join(os.path.dirname(filepath), 'backup')
        # �������Ŀ¼�����ڣ��򴴽�Ŀ¼
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        # �����ļ��������ƺͱ���·��
        backup_name = f'{os.path.basename(filename)}_{timestamp}{ext}'
        backup_path = os.path.join(backup_dir, backup_name)
        # �����ļ�
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
    �����������ڼ�Ĺ�����
    start:��ʼʱ��
    end:����ʱ��
    '''

    # �ַ�����ʽ���ڵĴ���
    if type(start) == str:
        start = datetime.strptime(start, '%Y-%m-%d').date()
    if type(end) == str:
        end = datetime.strptime(end, '%Y-%m-%d').date()
    # ��ʼ���ڴ󣬵ߵ���ʼ���ںͽ�������
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
    # ����������õ����⻷��������������-���ݽӿ�-API�ӿ�-�ֲ�-python�ӿ�-�ӿ��ֲᣬ���·��ָ��python�����ú����Ƿ��½�ɹ�
    while not w.isconnected():
        w.start()
        if not w.isconnected():
            raise Exception('WindPy����ʧ�ܣ������ԭ��')

    # ���quota����
    download_try = w.wsd('600519.SH', "industry_citic", f'2023-02-20', f'2023-02-20',
                         "unit=1;industryType=1")
    if download_try.Data[0][0].__contains__('quota exceeded'):
        raise Warning('ȡ������quota����')
        pass
    else:
        print('Wind�ڱ���������������δ���ޣ���ȡ������������������ԭ����wind����ȱ�����ݣ�')


def get_nearest_dates_from_contract(code):
    # ��ȡ������Ϣ
    match = re.match(r"^(T|TF|TS)?(\d{2})(\d{2})$", code)
    if not match:
        raise ValueError("Invalid code format")
    year = int(match.group(2)) + 2000
    month = int(match.group(3))

    # �����·ݷ�Χ
    end_of_month = datetime(year, month, 1) + timedelta(days=32)
    end_of_month = datetime(end_of_month.year, end_of_month.month, 1) - timedelta(days=1)
    start_of_prev_month = datetime(year, month, 1) - timedelta(days=1)
    start_of_prev_month = datetime(start_of_prev_month.year, start_of_prev_month.month, 1)

    # ��ʽ�����
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
    �ж�������ʱ����ɵ��б����Ƿ���ڳ���20����������
    """
    sorted_dates = sorted(date_list)  # �����ڽ�������
    for i in range(len(sorted_dates) - 1):
        diff = sorted_dates[i + 1] - sorted_dates[i]
        if diff > timedelta(days=20):
            print("���ڳ���20���������ڣ�")
            print("�������:", sorted_dates[i])
            print("�ұ�����:", sorted_dates[i + 1])
            return True
    return False

def match_recent_tradedays(date_list, tradedays):
    """
    ƥ������б����������������պ� self.tradedays �������������գ�������ӦԪ�ص�ֵ�Ƿ�һ��
    """
    recent_dates = date_list[-5:] if len(date_list) >= 5 else date_list
    for i in range(1, min(len(recent_dates)+1, 6)):
        if recent_dates[-i] != tradedays[-i]:
            return False
    return True


def get_month_end_dates(start_date: date, end_date: date):
    """
    ����һ����������ʱ�䷶Χ��������ĩ���ڵ�Ԫ��(tuple)��Ԫ����� datetime ����� str �������ָ�ʽ
    :param start_date: ��ʼ���ڣ�datetime.date ����
    :param end_date: �������ڣ�datetime.date ����
    :return: Ԫ��(tuple)����ʽΪ ([datetime.date], [str])
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
    # ÿ�ε���api��ദ��ı�����������ֹ�ı�����������ͬʱ�������ʱ����©������
    max_length = 20
    # ���Ϊ���С����󳤶ȵĶ�
    segments = [column_names[i:i + max_length] for i in range(0, len(column_names), max_length)]
    camel_case_names = []

    for segment in segments:
        # Join the column names with a separator that is unlikely to appear in the translations
        joined = ','.join(segment)
        # Before translating, replace any strings that have manual translations
        for original, translation in manual_translations.items():
            joined = joined.replace(original, translation)

        # Remove the Chinese comma (�ٺ�) �����ⷭ�����ֶ��ű���ʶΪ�ָ���
        joined = joined.replace('��', '')

        # Translate the entire string
        translated = ts.translate_text(joined, translator='youdao')

        # Remove the spaces around colons
        translated = translated.replace(': ', ':').replace(' :', ':')
        # Split the translated string by the separator
        translated_parts = translated.split(',')

        for translated_part in translated_parts:
            # ת��������ĸ��д
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

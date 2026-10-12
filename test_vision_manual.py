"""视觉识别接口本地测试脚本

用法：先启动后端，再运行：
    .venv\\Scripts\\python.exe test_vision_manual.py

说明：/api/upload 需要登录，脚本会自动注册一个临时账号，跑完自动注销。
"""
import io
import time

import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = 'http://localhost:5000'


def make_image(text):
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('msyh.ttc', 40)
    except Exception:
        try:
            font = ImageFont.truetype('arial.ttf', 40)
        except Exception:
            font = ImageFont.load_default()
    draw.text((50, 70), text, fill='black', font=font)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def upload(session, filename, text):
    files = {'image': (filename, io.BytesIO(make_image(text)), 'image/jpeg')}
    r = session.post(f'{BASE_URL}/api/upload', files=files, timeout=90)
    return r.status_code, r.json()


def main():
    session = requests.Session()
    username = 'visiontest_%d' % int(time.time())
    reg = session.post(f'{BASE_URL}/api/register',
                       json={'username': username, 'password': 'test123456'}, timeout=30).json()
    if not reg.get('success'):
        print('注册临时账号失败：', reg)
        return
    token, uid = reg['token'], reg['user']['id']
    session.headers['Authorization'] = 'Bearer ' + token
    print('临时账号：', username)

    cases = [
        ('识别 84 消毒液', 'test_84.jpg', '84消毒液'),
        ('识别医用酒精', 'test_alcohol.jpg', '医用酒精'),
        ('无法识别的图片', 'test_unknown.jpg', 'Hello World'),
    ]
    for label, filename, text in cases:
        print(f'\n=== {label} ===')
        code, body = upload(session, filename, text)
        print(f'  HTTP {code}')
        print(f'  {body}')

    print('\n已注销临时账号：', session.delete(f'{BASE_URL}/api/account/{uid}', timeout=30).json())


if __name__ == '__main__':
    main()

"""直接调用视觉模型 API 测试图片识别

用法：直接运行
    python test_vision_direct.py
不需要启动后端，只需要 .env 里配置好 VISION_API_URL / VISION_API_KEY / VISION_MODEL。

注意：DeepSeek 的 deepseek-chat 是纯文本模型，不支持图片输入，不能用于本测试。
请填写任意 OpenAI 兼容的多模态接口（如 qwen-vl-max、glm-4v）。
"""
import os
import io
import base64
import json
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import requests

load_dotenv()

API_KEY = os.getenv('VISION_API_KEY', '')
API_URL = os.getenv('VISION_API_URL', '')
MODEL = os.getenv('VISION_MODEL', '')

if not (API_KEY and API_URL and MODEL):
    print('错误：未配置视觉模型，请先在 .env 中设置 VISION_API_URL / VISION_API_KEY / VISION_MODEL')
    exit(1)


def make_image(text):
    """生成一张带文字的图片"""
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


def call_vision(base64_image, label=''):
    """直接请求视觉模型"""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': '你是家庭化学品安全识别助手，请根据图片内容识别化学品名称或主要成分，只输出最可能的名称，不要多余解释。'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': '这是一张家庭化学品产品标签的照片。请识别并返回产品名称、主要成分或常见别名，只输出最可能的名称，不要解释。例如：84消毒液、洁厕灵、医用酒精、洗洁精。'},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
                ]
            }
        ],
        'max_tokens': 200
    }
    print(f'  请求模型: {MODEL}')
    r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data['choices'][0]['message']['content']


def test_case(text):
    print(f'\n=== 测试图片文字：{text} ===')
    img_bytes = make_image(text)
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    print(f'  图片大小: {len(img_bytes)} bytes, base64 长度: {len(b64)}')
    try:
        result = call_vision(b64, text)
        print(f'  视觉模型识别结果: {result}')
    except Exception as e:
        print(f'  调用失败: {e}')


if __name__ == '__main__':
    print(f'API URL: {API_URL}')
    print(f'Vision Model: {MODEL}')
    print(f'API Key: {API_KEY[:8]}...')

    test_case('84消毒液')
    test_case('医用酒精')
    test_case('洁厕灵')
    test_case('阿司匹林')
    test_case('Unknown Product')

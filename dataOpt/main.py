# -*- coding: utf-8 -*-
from selenium import webdriver
from engine.baidu import Baidu
from engine.bing import Bing

# 设置chromedriver不加载图片
chrome_opt = webdriver.ChromeOptions()
chrome_opt.add_argument('disable-infobars')  # 隐藏"Chrome正在受到自动软件的控制"
chrome_opt.add_argument('disable-plugins')  # 禁止加载所有插件，可以增加速度。
driver = webdriver.Chrome(executable_path='driver/chromedriver.exe', chrome_options=chrome_opt)

# # 爬取图像的保存位置
storing_folder = r'bing_data/'

# 搜索关键词
key = '人佩戴口罩'

# 选择使用Baidu
# image_engine = Baidu(web_driver=driver, key=key, storing_folder=storing_folder)
# image_engine.retrieve_image()

# 选择使用Biying
image_engine = Bing(web_driver=driver, key=key, storing_folder=storing_folder)
image_engine.retrieve_image()

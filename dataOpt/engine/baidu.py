from util.util import Util
from engine.engine import Engine
import time
import re


class Baidu(Engine):
    HOME_URL = 'http://image.baidu.com/'

    def __init__(self, web_driver, key, storing_folder):
        Engine.__init__(self, web_driver)
        self.storing_folder = storing_folder
        self.key = key

    def retrieve_image(self):
        # 定位搜索框与输入信息
        Engine.retrieve_image(self)
        self.web_driver.get(Baidu.HOME_URL)
        key_input_element = self.web_driver.find_element_by_id('kw')
        key_input_element.send_keys(self.key)
        key_input_element.submit()
        self.web_driver.find_elements_by_css_selector("[class='imgbox']")[0].click()

        # navi_url = self.web_driver.find_elements_by_class_name('main_img img-hover')
        # self.web_driver.get(navi_url.get_attribute('src'))

        # 获取当前图片的URL
        time.sleep(2)
        self.web_driver.switch_to.window(self.web_driver.window_handles[-1])
        url = self.web_driver.find_element_by_class_name("currentImg")
        current_index = 0

        # 循环获取图片
        while url is not None:
            url = url.get_attribute('src')
            image_format = re.findall(r'\wjpg|png|jpeg', url)
            # 如果图片满足格式规范,保存图片
            if (len(image_format) != 0):
                print('IMAGE URL: ', url)
                image_name = str(current_index) + '.' + image_format[0]
                if Util.download_image(url, self.storing_folder, image_name):
                    current_index += 1
            # 点击下一张图片按钮控件
            self.web_driver.find_elements_by_class_name("img-switch-btn")[1].click()
            time.sleep(2)
            url = self.web_driver.find_element_by_class_name("currentImg")

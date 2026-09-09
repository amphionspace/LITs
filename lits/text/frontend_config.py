#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   frontend_config.py
@Time    :   2024/09/05 09:10:20
@Author  :   Bo Jin 
@Version :   1.0
@Contact :   jinbo5650@gmail.com
@Brief   :   存放前端的配置文件
'''



import os


#设置添加blank的等级, {0："没有",1:"字", 2:"词"}
#目前仅中文支持该功能，其他语种待添加
BLANK_LEVEL = 2
#设置资源文件所处的路径,最好写绝对路径
resource_path = os.path.join(os.getcwd(), "lits", "text")
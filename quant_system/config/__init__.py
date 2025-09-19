import sys
import os

# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目的根目录
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
config_file='config/db_config.ini'
print( os.path.join(root_dir, config_file))
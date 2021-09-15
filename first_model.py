#encoding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from flask import Flask, render_template, request, url_for

import numpy as np
import re
import base64
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from datetime import timedelta
def load_image(image):
  img = Image.open(image)
  img = img.convert('L')
  plt.imshow(img, cmap="gray")
  plt.show()
  img = img.resize((128, 128))
  img = np.array(img)
  img = img / 255
  img = img.reshape((1,) + img.shape + (1,))  # reshape img to size(1, 128, 128, 1)
  return img
label_name=['且', '世', '东', '九', '亭', '今', '从', '令', '作', '使', '侯', '元', '光', '利', '印', '去', '受', '右', '司', '合', '名', '周', '命', '和', '唯', '堂', '士', '多', '夜', '奉', '女', '好', '始', '字', '孝', '守', '宗', '官', '定', '宜', '室', '家', '寒', '左', '常', '建', '徐', '御', '必', '思', '意', '我', '敬', '新', '易', '春', '更', '朝', '李', '来', '林', '正', '武', '氏', '永', '流', '海', '深', '清', '游', '父', '物', '玉', '用', '申', '白', '皇', '益', '福', '秋', '立', '章', '老', '臣', '良', '莫', '虎', '衣', '西', '起', '足', '身', '通', '遂', '重', '陵', '雨', '高', '黄', '鼎']

# 1. 初始化 flask app
# 强制性的
app = Flask(__name__)
# app.config['DEBUG']=True
# app.config['SEND_FILE_MAX_AGE_DEFAULT']=timedelta(seconds=1)
# 2. 初始化global variables
# sess = tf.Session()
# graph = tf.get_default_graph()

# 3. 将用户画的图输出成output.png
def convertImage(imgData1):
  imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
  with open('output.png', 'wb') as output:
    output.write(base64.b64decode(imgstr))

# 4. 搭建前端框架
# 装饰器，那个url调用相关的函数
@app.route('/')
def index():
  return render_template("index.html")

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
    if filename:
        file_path = os.path.join(app.root_path, endpoint, filename)
        values['q'] = int(os.stat(file_path).st_mtime)
        return url_for(endpoint, **values)

# 5. 定义预测函数
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
  # 这个函数会在用户点击‘predict’按钮时触发
  # 会将输出的output.png放入模型中进行预测
  # 同时在页面上输出预测结果
  imgData = request.get_data()
  convertImage(imgData)
  # 读取图片
  predict_img=load_image('output.png')
  # 加载模型
  model = load_model('model/best_weights_hanzi_my.hdf5')
  predict = model.predict(predict_img)
  predict = np.argmax(predict, axis=1)
  return str(label_name[predict[0]])
  # 调用训练好的模型和并进行预测
  # global graph
  # global sess
  # with graph.as_default():
  #   set_session(sess)
  #   model = load_model('model.h5')
  #   out = model.predict(x)
  #   response = np.argmax(out, axis=1)
  #   return str(response[0])

# 6. 返回本地访问地址
if __name__ == "__main__":
    # 让app在本地运行，定义了host和port
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=8080)
# -*- coding: utf-8 -*-
import numpy as np
import random
import struct
#import matplotlib.pyplot as plt
import math
from compiler.ast import flatten
import cv2
# 训练集文件
train_images_idx3_ubyte_file = '/Users/Jason/desktop/Melbourn py/Backpropagation/data for test/train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '/Users/Jason/desktop/Melbourn py/Backpropagation/data for test/train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '/Users/Jason/desktop/Melbourn py/Backpropagation/data for test/t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '/Users/Jason/desktop/Melbourn py/Backpropagation/data for test/t10k-labels-idx1-ubyte'
n=50 #recurse n times

def decode_idx3_ubyte(idx3_ubyte_file):
    """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols)
    
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张' % (magic_number, num_images)
    
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        
        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
        TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.
        
        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
        TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  10000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        
        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
        TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  10000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.
        
        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
    return decode_idx1_ubyte(idx_ubyte_file)


train_images = load_train_images()
train_labels = load_train_labels()
test_images=load_test_images()
test_labels=load_test_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()


print 'got all the data~'

input_number=784
output_lenth=10
#hide_len=int(math.sqrt(input_number+output_lenth)+4)
hide_len=15
input=[0]*784

sigma=0.2
w1=np.zeros((len(input),hide_len))
b1=np.zeros((len(input),hide_len))
w2=np.zeros((hide_len,output_lenth))
b2=np.zeros((hide_len,output_lenth))
for j in range(0,hide_len):
    for i in range(0,len(input)):
        w1[i][j]=random.uniform(-0.05,0.05)
        b1[i][j]=random.uniform(0,0.05)
for k in range(0,output_lenth):
    for j in range(0,hide_len):
        w2[j][k]=random.uniform(-0.1,0.1)
        b2[j][k]=random.uniform(0,0.1)

out1=[0.0]*hide_len
category=[0,1,2,3,4,5,6,7,8,9,10]
out2=[0.0]*output_lenth
e=[0.0]*output_lenth
def f(x):
    if x<-500:
        return 0
    if x>=-500:
        return 1/(1+math.exp(-x))

def f_deri(x):
    return x-x*x

def output(inpu):
    value1=[0.0]*hide_len
    value2=[0.0]*output_lenth
    for j in range(0,hide_len):
        for i in range(0,input_number):
            value1[j] += inpu[i]*w1[i][j]+b1[i][j]
    #print value1
    #a=raw_input()
        out1[j] = f(value1[j])
#print out1[j]
#print out1
#a=raw_input()
    for k in range(0,output_lenth):
        for j in range(0,hide_len):
            value2[k] += out1[j]*w2[j][k]+b2[j][k]
        out2[k] = f(value2[k])
#print out2
#print '输出'+str(out2[k])
#print out2
#a=raw_input()
    return out1,out2

ret=0
xi1=[0.0]*hide_len
xi2=[0.0]*output_lenth
#recurse to update w1,w2,b1,b2
while(ret<n):
    print '正进行第'+str(ret)+'次迭代'
    ret=ret+1
#calculate e
    for num in range(0,500):#len(train_images)
        #print '     迭代第'+str(num)+'张图片'
        input = flatten(test_images[num].tolist())
        for items_num in range(0,len(input)):
            if input[items_num]>127:
                input[items_num]=1
            else:
                input[items_num]=0
    #print input
        #a=raw_input()
        result = output(input)
        result1 = result[0]
        result2 = result[1]
#update w2,b2
        for i in range(0,len(e)):
            if i == test_labels[num]:
                e[i] = 1-result2[i]
            else:
                e[i] = -result2[i]
    #print e
    #a=raw_input()
        for k in range(0,output_lenth):
            xi2[k]=e[k]*f_deri(result2[k])
            '''print result2[k]
            print '--'
            print xi2[k]
            print '---'
            print f_deri(result2[k])
            print '000'''
            b2[k]=b2[k]+sigma*xi2[k]
            for j in range(0,hide_len):
                w2[j][k]=w2[j][k]+sigma*xi2[k]*result1[j]
#update w1,b1
        for j in range(0,hide_len):
            xi1[j]=f_deri(result1[j])*sum([xi2[k]*w2[j][k] for k in range(0,output_lenth)])
            b1[j]=b1[j]+sigma*xi1[j]
            for i in range(0,len(input)):
                w1[i][j]=w1[i][j]+sigma*xi1[j]*result1[j]
                    #print w1
#print '---'
#print w2
#print w1
z_n=0.0
test_num=500
print '现在进行测试，一共'+str(test_num)+'个样本'
for num in range(0,test_num):#len(test_images)
    input2=flatten(test_images[num].tolist())
    zx=output(input2)[1]
    print zx
    cal_result=zx.index(max(zx))
    cor_result=test_labels[num]
    print str(cal_result)+'-----------------------'+str(cor_result)
    if int(cal_result)==int(cor_result):
        print 'correct'
        z_n+=1
print z_n
rate=float(z_n/test_num)
print '测试样本'+str(test_num)+'个,'+'正确率:'+str(rate)


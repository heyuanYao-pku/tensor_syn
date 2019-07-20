import numpy as np
import matplotlib.pyplot as plt
import util.util as util
import cv2



tensor_result_file = open('temp_result_new/tensor_result/matres_156.txt','r')
image_nums = 156
save_path = 'temp_result_new/tensor_result/temp_%d'%image_nums
util.mkdir(save_path)
Q_index = []
image_list_old = np.load('temp_result_new/tensor_result/image_list.npy')
w,h = 256,256
while True:
    this_line = tensor_result_file.readline().strip()
    # print(this_line)
    if this_line=='':
        break
    this_line = this_line.split(',')

    this_line_inQ = []
    for i in this_line:
        this_line_inQ.append(int(i))

    Q_index.append(this_line_inQ)
# Q_index = np.array(Q_index)
# _,color_num = Q_index.shape
# print(Q_index.shape)
color_num = len(Q_index[0])
# print(image_list.shape)
# image_list = image_list[0:10]
cmap = plt.get_cmap('gist_ncar')
colors_set = cmap(np.linspace(0, 1,color_num+1))

image_label_index = [[0 for j in range(color_num)] for i in range(image_nums)]
# I need relabel
new_image_label_list = []
now_n = 0

for image_index in range(image_nums):
    this_image_old_label = image_list_old[image_index]
    this_image_part_nums = int(this_image_old_label.max())
    print(this_image_part_nums)
    temp_part_index = []
    new_this_image = np.zeros([w,h])
    for part_index in range(1,this_image_part_nums+1):
        this_part_map = Q_index[now_n]
        now_n+=1
        this_part_new_label = this_part_map.index(1)
        temp_part_index.append(this_part_new_label)
        new_this_image[this_image_old_label==part_index] = this_part_new_label+1
    new_image_label_list.append(new_this_image)
    print(temp_part_index)

for image_index in range(image_nums):


    final_im = new_image_label_list[image_index]
    draw_image = np.zeros([w, h, 3])
    max_color = int(final_im.max())
    for c in range(1, max_color+1):
        draw_image[final_im == c] = colors_set[c][0:3] * 255

    # source = cv2.resize(cv2.imread('%s/%s' % (source_image_path, all_image[final])), (256, 256))
    draw_image = draw_image.astype(np.uint8)
    # all = cv2.addWeighted(source, 0.5, draw_image, 0.5, 1)

    cv2.imwrite('%s/%s.png' % (save_path, str(image_index)), draw_image)
    # cv2.imwrite('%s/all_%s' % (savepath, all_image[final]), all)
    # video_write.write(all)












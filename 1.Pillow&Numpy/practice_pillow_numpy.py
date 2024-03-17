# import the packages we need for this assignment
# pillow : 파이썬으로 이미지를 다룰 수 있게 해주는 라이브러리 (픽셀 단위의 조작, 마스킹 및 투명도 제어)
# numpy : 행렬이나 대규모 다차원 배열을 쉽게 처리할 수 있도록 도와주는 파이썬 라이브러리
from PIL import Image
import numpy as np

# open the test image 
# Note: If you didn't launch Python from the same directory where you saved
#       the file, chipmunk.png, you'll need to provide the full path name as
#       the argument to Image.open
im = Image.open('chipmunk.png')

# display relevant Image class attributes: dimensions (width, height),
# pixel format and file format
# 이미지의 사이즈, 이미지 모드, 이미지 파일의 형식을 출력
# image mode (grayscale - L, true color - RGB)
print (im.size, im.mode, im.format)

# Note: PIL does not have a built-in image display tool.  Instead, principally
# for debugging, there's a show method which saves an image to a temporary file
# on disk and calls a platform dependent external display utility
# (the default being "xv" on unix, and the "Paint" program on Windows).

# display the image
im.show()

# if this does not work on your system, try the imshow function from
# matplotlib's pyplot module by uncommenting the next three lines
#import matplotlib.pyplot as plt
#plt.imshow(im)
#plt.show()

# convert the image to a black and white "luminance" greyscale image
# convert를 통해 RGB이미지를 grayscale 이미지로 변경
im = im.convert('L')

# im_test = im.convert('L')
# im_test.save('chipmunk_grayscale.png','PNG')

# select a 100x100 sub region (containing the chipmunk's head)
# crop -> 이미지 잘라내기 (다람쥐 머리 부분)
# crop(('좌표')) = crop((start_x, strat_y, start_x + width, start_y+ height))
# (150, 150) 사이즈로 이미지 잘라내기 
im2 = im.crop((280,150,430,300))

# save the selected region
# crop을 통해 자른 이미지를 'chipmunk_head.png' 이름으로 PNG 형식으로 이미지를 저장
im2.save('chipmunk_head.png','PNG')

# PIL and numpy use different internal representations
# convert the image to a numpy array (for subsequent processing)
# 이미지를 intensity 값을 갖는 Numpy 배열로 변환 (np.asarray(image) : image -> array)
im2_array = np.asarray(im2)
#print(im2_array)

# compute the average intensity
# 배열의 intensity값 평균 구하기
average = np.mean(im2_array)

# Note: we need to make a copy to change the values of an array created using
# np.asarray
# copy를 통해 배열 복사 (python 기본 함수)
im3_array = im2_array.copy()

# add 50 to each pixel value (clipping above at 255, the maximum uint8 value)
# 0 : Black, 255 : white -> 모든 픽셀에 50을 더해 전체적으로 이미지가 밝아진다.
# 증가한 intensity value가 255을 넘어가지 않도록 한다. -> min을 통해 해당 문제 해결
# Note: indentation matters
# min
for x in range(0,150):
    for y in range(0,150):
        im3_array[y,x] = min(im3_array[y,x] + 50, 255)

# convert the result back to a PIL image and save
# Numpy 배열을 이미지로 변환 (Image.fromarray(array) : array to image)
# fromarray를 통해 배열을 이미지파일로 바꾸고 이미지 파일을 저장(save)
im3 = Image.fromarray(im3_array)
im3.save('chipmunk_head_bright.png','PNG')

# again make a copy of the (original) 100x100 sub-region
im4_array = im2_array.copy()

# this time, reduce the intensity of each pixel by half
# Note: this converts the array to a float array
# 모든 픽셀 값(intensity)에 대하여 0.5를 곱해준다 
# intensity가 반으로 줄어듬에 따라 사진의 밝기가 어두워진다. (White - 255 / Black - 0)
im4_array = im4_array * 0.5

# convert the array back to a unit8 array so we can write to a file
# 배열의 값들이 0-255사이의 값을 가질 수 있도록 변환해준다.*(uint8)
im4_array = im4_array.astype('uint8')

# convert the numpy array back to a PIL image and save
# image.fromarray(array) : array -> image
im4 = Image.fromarray(im4_array)
im4.save('chipmunk_head_dark.png','PNG')

# let's generate our own image, a simple gradient test pattern
# make a 1-D array of length 256 with the values 0 - 255
# np.arange(시작점, 끝점, step size(생략 시 1))
# 0 ~ 255 의 값을 갖는 numpy 1차원 배열이 만들어진다.
# [0 1 2 ... 253 254 255] 점점 밝아짐 (0 : Black / 255 : White)
grad = np.arange(0,256)

# repeat this 1-D array 256 times to create a 256x256 2-D array
# tile : 동일한 배열을 반복해서 복사 붙여넣기 하는 함수
# [256, 1] -> 해당 배열을 256(세로) / 1(가로) 로 반복해서 붙여 넣는다.
grad = np.tile(grad,[256,1])
#print(grad)

# convert to uint8 and then to a PIL image and save
# fromarray를 통해 array를 image 파일로 변환 시킨다.
# * grad.astype('uint8') *

im5 = Image.fromarray(grad.astype('uint8'))
im5.save('gradient.png','PNG')

#test arange & tile
# array = [0 2 4 6 8]
array = np.arange(0, 10, 2)

# [0 2 4 6 8]
# [0 2 4 6 8]
# [0 2 4 6 8]

array_test = np.tile(array, [3, 1])
# print(array_test)

print("----------------------------")

# [0 2 4 6 8 0 2 4 6 8]
# [0 2 4 6 8 0 2 4 6 8]
# [0 2 4 6 8 0 2 4 6 8]

array_test_2 = np.tile(array, [3,2])
#print(array_test_2)


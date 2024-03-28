from PIL import Image
import numpy as np
import math

#모든 원소의 합이 1이되는 n * n 사이즈의 행렬 만들기 
#단,원소의 값은 모두 동일해야한다.
def boxfilter(n):
    assert (n % 2 != 0), "dimension must be odd"
    #모두 1로 채워져있는 n * n 행렬만들기
    #행렬을 행렬의 원소 개수로 나누어 모든 원소의 합이 1이 되도록해준다.
    filter = np.ones((n, n))
    box_filter = filter / (n * n)
    return box_filter

# 1차원 가우시안 필터 만들기
def gauss1d(sigma):
    #필터의 크기는 시그마의 6배보다 큰 최소 홀수
    n = math.ceil(sigma * 6)
    if (n % 2 == 0) : n = n + 1
    range = (n-1)//2

    #모든 원소의 합이 1이되게 하는 1차원 가우시안 필터 생성
    array = np.arange(-range, range + 1, 1) ** 2
    array = np.exp(-array/(2 * sigma * sigma))
    gauss1d_filter = np.ones(n)/np.sum(array) * array
    return gauss1d_filter



#outer product를 통해 1차원 가우시안 커널을 2차원으로 만들어준다.
def gauss2d(sigma):
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma)) 
    return gauss2d_filter

# convolution operator
def convolve2d(array, filter):
    #이미지의 행과 열의 사이즈 알아내기
    image_x = len(array[0])
    image_y = len(array)

    #1 zeropadding - filter의 크기를 고려
    padding = (len(filter)-1)//2
    array = np.pad(array,((padding,padding),(padding,padding)),'constant', constant_values= 0)
 
    #2 flip the filter
    filter = np.flip(filter)

    #3 convolution - cross-correlation with flipped filter
    #(열, 행)
    filtered_array = np.empty((image_y, image_x))

    # use two for-loops for cross-correlation
    for i in range(0, image_y, 1):
        for j in range(0, image_x, 1):
            matrix_multiply = array[i : i + len(filter), j : j + len(filter)] * filter
            filtered_array[i][j] = np.sum(matrix_multiply)
    
    #convolution이 끝난 배열(이미지)을 반환한다.
    return filtered_array 

# convolution with gaussian filter
def gaussconvolve2d(array, sigma):
    
    # first generating a filter with gauss2d
    filter = gauss2d(sigma)

    # applying it to the array with convolve2d   
    filtered_array = convolve2d(array, filter)
    return filtered_array 

# rgb 이미지를 Grayscale이미지로 변환하고 convolution 연산
def part1_4():
    original = Image.open('3a_lion.bmp')
    #convert rgb image to grayscale
    filtered_img = original.convert('L')
    #convert image to array
    array = np.asarray(filtered_img)

    #do convolution operator when sigma is 3
    filtered_array = gaussconvolve2d(array, 3)
    #convert the array back to unsigned integer format
    filtered_img = Image.fromarray(filtered_array.astype('uint8'))
    filtered_img.show();
    
    filtered_img.save('result1_4.bmp', "BMP")
    # print("--------- part1 ---------")

# RGB이미지에 convolution연산 적용하기
# image = 이미지 파일의 위치(경로)
def part2_1(image):
    # get a Guassian filtered low frequency RGB image
    # load a RGB Image (RGB image has a 3 channels)
    # return low_frequency_array
    im = Image.open(image)
    array = np.asarray(im)

    # with a relatively large sigma (sigma = 5)
    # filter each of the three color channels seperately
    #Red
    red_array = array[:,:,0]
    red_filtered = gaussconvolve2d(red_array, 5)

    #Green
    green_array = array[:,:,1]
    green_filtered = gaussconvolve2d(green_array, 5)

    #Blue
    blue_array = array[:,:,2]
    blue_filtered = gaussconvolve2d(blue_array, 5)

    #compose the channels back to the color image
    low_freq_array = np.empty(array.shape)
    low_freq_array[:,:,0] = red_filtered
    low_freq_array[:,:,1] = green_filtered
    low_freq_array[:,:,2] = blue_filtered
    
    low_freq_img = Image.fromarray(low_freq_array.astype('uint8'))
    low_freq_img.show()
    #low_freq_img.save( image + "lf_img.bmp", 'BMP')

    #print("1.get_low_frequency_array")
    return low_freq_array


# High-Frequency 이미지 얻기
# image = 이미지 파일의 위치(경로)
def part2_2(image):
    # 1.get_original image
    im = Image.open(image)
    array = np.asarray(im)
  
    # 2.get_low-frequency_image = (blured image)
    low_freq_array = part2_1(image)

    #3.high_frequencyimage = original_image - LowFrequency_image
    high_frequency = array - low_freq_array
    #plus margin(+128) to avoid minus value
    hf_image = Image.fromarray((high_frequency+128).astype('uint8'))
    hf_image.show()
    #hf_image.save(image + "hf.bmp", 'BMP')
    #print("2.get_high_frequency_array")
    
    return high_frequency

def part2_3():
    # get_low_frequency_image
    lf = part2_1('1b_mandela.bmp')

    # get_high_frequency_image
    hf = part2_2('1a_steve.bmp')
    

    # get_hybrid_image
    hybrid_img = lf + hf

    # clip(array, min, max) 
    # min 값 보다 작은 값들은 min 값으로, max보다 큰 값들은 max값으로 바꿔주는 함수
    hybrid_img = np.clip(hybrid_img, 0, 255)
    hybrid_img = Image.fromarray(hybrid_img.astype('uint8'))
    hybrid_img.show()
    hybrid_img.save('hybrid_img.bmp', 'BMP')
    return hybrid_img

# def my_naive_sub_sampling():
#     im = Image.open('hybrid_img.bmp')
#     array = np.asarray(im)
#     #짝수행, 짝수열만 선택하는 naive한 방식의 subsampling
#     sampled = array[::2, ::2]
#     Image.fromarray(sampled).save('half.bmp', 'BMP')
#     sampled = sampled[::2, ::2]
#     Image.fromarray(sampled).save('half_half.bmp', 'BMP')
#     sampled = sampled[::2, ::2]
#     Image.fromarray(sampled).save('half_half_half.bmp', 'BMP')

#part1_1
# print(boxfilter(3))
# print(boxfilter(4))
# print(boxfilter(7))

#part1_2
# print(gauss1d(0.3))
# print(gauss1d(0.5))
# print(gauss1d(1))
# print(gauss1d(2))

#part1_3
# print(gauss2d(0.5))
# print(gauss2d(1))

part1_4()

# part2_1('1b_mandela.bmp')
# part2_2('1a_steve.bmp')

part2_3()
# my_naive_sub_sampling()

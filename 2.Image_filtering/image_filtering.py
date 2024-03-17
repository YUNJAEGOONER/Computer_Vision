from PIL import Image
import numpy as np;
import math




def boxfilter(n):
    assert (n % 2 != 0), "dimension must be odd"
    filter = np.ones((n, n))
    return filter / (n * n)

 
def gauss1d(sigma):
    n = math.ceil(sigma * 6)
    if (n % 2 == 0) : n = n + 1
    range = (n-1)//2
    array = np.arange(-range, range + 1, 1) ** 2
    array = np.exp(-array/(2 * sigma * sigma))
    gauss1d_filter = np.ones(n)/np.sum(array) * array
    return gauss1d_filter

#outer product를 통해 1차원 가우시안 커널을 2차원으로 만들어준다.
def gauss2d(sigma):
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma)) 
    return gauss2d_filter

def convolve2d(array, filter):

    image_x = len(array[0])
    image_y = len(array)

    #1 image - zeropadding
    padding = len(filter) // 2
    array = np.pad(array,((padding,padding),(padding,padding)),'constant', constant_values= 0)
 
    #2 flip the filter
    filter = np.flip(filter)

    #3 convolution - cross-correlation with fliped filter
    # (열, 행)
    filtered_array = np.empty((image_y, image_x))

    for i in range(0, image_y, 1):
        for j in range(0, image_x, 1):
            matrix_multiply = array[i : i + len(filter), j : j + len(filter)] * filter
            #print(array[i : i + len(filter), j : j + len(filter)])
            #print(np.sum(matrix_multiply))
            #print("--------------------------------------")
            filtered_array[i][j] = np.sum(matrix_multiply)
    return filtered_array 

def gaussconvolve2d(array, sigma):
    # first generating a filter with gauss2d
    filter = gauss2d(sigma)
    filtered_array = convolve2d(array, filter)
    # applying it to the array with convolve2d
    return filtered_array 

def part1_4():
    im = Image.open('0b_marilyn.bmp')
    #print(im.size, im.mode, im.format)
    #convert rgb image to grayscale
    im = im.convert('L')
    array = np.asarray(im)

    filtered_array = gaussconvolve2d(array, 10)
    #filtered_array = gaussconvolve2d(array, 10).astype('uint8')
    filtered_imag = Image.fromarray(filtered_array)
    filtered_imag.show
    print("--------- part1 ---------")

def part2_1(image):
    # get a Guassian filtered low frequency RGB image
    # load a RGB Image (RGB image has a 3 channels)
    # return low_frequency_array
    im = Image.open(image)
    array = np.asarray(im)
    # print(array.shape)

    # with a relatively large sigma (sigma = 3)
    # filter each of the three color channels seperately
    #R
    red_array = array[:,:,0]
    red_filtered = gaussconvolve2d(red_array, 5)

    #G
    green_array = array[:,:,1]
    green_filtered = gaussconvolve2d(green_array, 5)

    #B
    blue_array = array[:,:,2]
    blue_filtered = gaussconvolve2d(blue_array, 5)

    #compose the channels back to the color image
    low_freq_array = np.empty(array.shape)
    low_freq_array[:,:,0] = red_filtered
    low_freq_array[:,:,1] = green_filtered
    low_freq_array[:,:,2] = blue_filtered
    Image.fromarray(low_freq_array.astype('uint8')).show()

    print("1.get_low_frequency_array")
    return low_freq_array



def part2_2(image):
    # high_frequencyimage = original_image - LowFrequency_image
    
    # 1.get_original image
    im = Image.open(image)
  
    # 2.get_low-frequency_image = (blured image)
    array = np.asarray(im)
    # print(array.shape)
    low_freq_array = part2_1(image)


    high_frequency = array - low_freq_array
    # print(np.mean(high_frequency))
    hf_image = Image.fromarray((high_frequency+128).astype('uint8'))
    hf_image.show()

    print("2.get_high_frequency_array")
    return high_frequency

def part2_3():
    # low_frequency_image
    lf = part2_1('1a_steve.bmp')

    # high_frequency_image
    hf = part2_2('1b_mandela.bmp')

    hybrid_img = lf + hf
    hybrid_img = np.clip(hybrid_img, 0, 255)
    hybrid_img = Image.fromarray(hybrid_img.astype('uint8'))
    hybrid_img.show()
    return hybrid_img
    

part2_3()
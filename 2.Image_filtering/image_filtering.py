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
    return np.ones(n)/np.sum(array) * array

#outer product를 통해 1차원 가우시안 커널을 2차원으로 만들어준다.
def gauss2d(s):
    return np.outer(gauss1d(s), gauss1d(s))

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
    convolved_image = np.empty((image_y, image_x))

    for i in range(0, image_y, 1):
        for j in range(0, image_x, 1):
            matrix_multiply = array[i : i + len(filter), j : j + len(filter)] * filter
            #print(array[i : i + len(filter), j : j + len(filter)])
            #print(np.sum(matrix_multiply))
            #print("--------------------------------------")
            convolved_image[i][j] = np.sum(matrix_multiply)
    return convolved_image

def gaussconvolve2d(array, sigma):
    # first generating a filter with gauss2d
    filter = gauss2d(sigma)
    # applying it to the array with convolve2d
    return convolve2d(array, filter)

def part1_4():
    im = Image.open('0b_marilyn.bmp')
    print(im.size, im.mode, im.format)
    #convert rgb image to grayscale
    im = im.convert('L')
    array = np.asarray(im)
    im_convolved = Image.fromarray(gaussconvolve2d(array, 10).astype('uint8'))
    im_convolved.save('test.bmp', 'BMP')
    print("--------- part1 ---------")

def part2_1():
    im = Image.open('2a_mangosteen.bmp')
    array = np.asarray(im)
    print(im.size, im.mode, im.format)

    # red_array = array[:,:,0]
    # red_convolved = Image.fromarray(gaussconvolve2d(red_array, 0.5).astype('uint8'))

    red_array = array[:,:,0]
    red_convolved = gaussconvolve2d(red_array, 2)

    green_array = array[:,:,1]
    green_convolved = gaussconvolve2d(green_array, 3)

    blue_array = array[:,:,2]
    blue_convolved = gaussconvolve2d(blue_array, 4)
    
    new_image = np.empty((607,607,3))
    new_image[:,:,0] = red_convolved
    new_image[:,:,1] = green_convolved
    new_image[:,:,2] = blue_convolved

    blured_image = Image.fromarray(new_image.astype('uint8'))

    #high_frequencyimage (original_image - LowFrequency_image)
    high_frequency = array - new_image + 150
    High = Image.fromarray(high_frequency.astype('uint8'))
    High.show()


    
    


# test_array = np.array([[1,2,3,0,1],[0,1,5,1,0],
#                        [1,0,2,2,1],[1,1,2,0,0],
#                        [1,0,1,1,1]])

# test_filter = np.array([[1,0,1],[0,1,0],[1,0,1]])
part2_1()

#print(gauss1d(0.5))
#print(boxfilter(7)),,
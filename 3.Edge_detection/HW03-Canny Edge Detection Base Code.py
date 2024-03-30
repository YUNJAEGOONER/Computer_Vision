from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""

# 1차원 가우시안 필터
def gauss1d(sigma):
    n = math.ceil(sigma * 6)
    if (n % 2 == 0) : n = n + 1
    range = (n-1)//2
    array = np.arange(-range, range + 1, 1) ** 2
    array = np.exp(-array/(2 * sigma * sigma))
    gauss1d_filter = np.ones(n)/np.sum(array) * array

    return gauss1d_filter

# 2차원 가우시안 필터
def gauss2d(sigma):
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma)) 

    return gauss2d_filter

#convolution 
def convolve2d(array,filter):
    image_x = len(array[0])
    image_y = len(array)
    padding = (len(filter)-1)//2
    array = np.pad(array,((padding,padding),(padding,padding)),'constant', constant_values= 0)
    filter = np.flip(filter)
    filtered_array = np.empty((image_y, image_x))

    for i in range(0, image_y, 1):
        for j in range(0, image_x, 1):
            matrix_multiply = array[i : i + len(filter), j : j + len(filter)] * filter
            filtered_array[i][j] = np.sum(matrix_multiply)
    
    return filtered_array 

#convolution using gaussian 2d filter
def gaussconvolve2d(array,sigma):
    filter = gauss2d(sigma)   
    filtered_array = convolve2d(array, filter)

    return filtered_array 

def reduce_noise(img):

    """ Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
    """
    original = img

    # convert rgb image to grayscale image
    grayscale = img.convert('L')

    #convert image to array
    array = np.asarray(grayscale, dtype=np.float32)

    #denoise using gaussian filter  
    res = gaussconvolve2d(array, 1.6)

    gray_blured = Image.fromarray(res.astype('uint8'))

    merged = Image.new('RGB',(len(array[0]) * 2, len(array)))
    merged.paste(original, (0,0))
    merged.paste(gray_blured, (480,0))
    merged.show()

    #return array(=blurred image)
    return res

# sobel 필터 : edge 계산할 때 사용하는 필터
# 가우시안필터에 미분필터를 사용해 convolution
# 노이즈제거 + edgedetection (소벨필터에서의 가우시안의 효과는 미비)
def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """

    #sobel filter
    X_filter = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]], np.float32)
    Y_filter = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]], np.float32)

    #calculate the gradient
    X_gradient = convolve2d(img, X_filter)
    Y_gradient = convolve2d(img, Y_filter)

    #elements of G = gradient magnitude
    #value = 0에서 255사이의 값을 갖는다.
    G = np.hypot(X_gradient, Y_gradient)
    G = G / np.max(G) * 255

    #elements of theta = theta value
    theta = np.arctan2(Y_gradient, X_gradient)
    
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    # convert radian value to dgree
    # 1-2사분면에 대해서는 양의 값 / 3-4분면에 대해서는 음의 값을 갖는다.
    dgree = (theta * 180)/np.pi

    NMS = np.zeros(np.shape(G))
    count = 0

    for i in range(1, len(dgree) - 1, 1):
        for j in range(1, len(dgree[0]) - 1, 1):
            #green
            if 22.5 < dgree[i][j] < 67.5 or -157.5 < dgree[i][j] < -112.5:
                cmp1 = G[i - 1][j + 1]
                cmp2 = G[i + 1][j - 1]
                count = count + 1
            #red
            elif 67.5 < dgree[i][j] < 112.5 or -112.5 < dgree[i][j] < -67.5: 
                cmp1 = G[i - 1][j]
                cmp2 = G[i + 1][j]
                count = count + 1
            #black
            elif 112.5 < dgree[i][j] < 157.5 or -67.5 < dgree[i][j] < -22.5:
                cmp1 = G[i - 1][j - 1]
                cmp2 = G[i + 1][j + 1]
                count = count + 1
            #blue
            else:
                cmp1 = G[i][j + 1]
                cmp2 = G[i][j - 1]
                count = count + 1
            # 자신의 값이 최댓값일 경우에만 Pixel의 intensity value를 살려둔다.
            if(G[i][j] > cmp1 and G[i][j] > cmp2):
                NMS[i][j] = G[i][j] 
    print("478 X 269 = ", count)
    G = NMS
    return G

# NMS 결과 이미지를 3가지 카테고리로 구분
# Strong / Weak / non-relevant
def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    diff = np.max(img) - np.min(img)
    T_H = np.min(img) + diff * 0.15
    T_L = np.min(img) + diff * 0.03

    check = 0;

    for i in range(0, len(img), 1):
        for j in range(0, len(img[0]), 1):
            if T_H <= img[i][j] <= 255:
                img[i][j] = 255
                check = check + 1
            elif T_L < img[i][j] < T_H:
                img[i][j] = 80
                check = check + 1
            elif 0 <= img[i][j] <= T_L:
                img[i][j] = 0
                check = check + 1
    
    print(check)
    return img

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors 주변 8픽셀을 탐색
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                # Strong 주변에 존재하는 pixel 중 Weak에 해당하는 픽셀은 Strong으로 변경
                # 재귀 호출을 통해 res[i, j] = 255 이 수행되면서 Weak(80)에 해당하는 픽셀이 Strong(255)으로 변경
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    #implement 
    res = np.zeros(img.shape)
    visited = []

    for i in range(1, len(img)-1, 1):
        for j in range(1, len(img[0])-1,1):
            # Strong 주변의 weak는 Strong으로 바꾸어준다
            # Strong 주변 픽셀을 탐색하기 위하여 depth first search 수행 
            if img[i][j] == 255:
                dfs(img, res, i, j, visited)
    return res

def main():
    RGB_img = Image.open('iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')

main()


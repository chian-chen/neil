import numpy as np
from scipy.stats import skew
import cv2
import os
from scipy.signal import medfilt2d
from scipy.ndimage import laplace, convolve
from scipy.signal import convolve2d
import scipy.io
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import json

def evaluate(errors):
# some index should by -1, stupid GPT :(
    errors = np.sort(errors)
    n = len(errors)
    f05 = errors[int(np.floor(0.5 * n)) - 1]
    f025 = errors[int(np.floor(0.25 * n)) - 1]
    f075 = errors[int(np.floor(0.75 * n)) - 1]
    med = np.median(errors)
    men = np.mean(errors)
    trimean = 0.25 * (f025 + 2 * f05 + f075)
    bst25 = np.mean(errors[:int(np.floor(0.25 * n))])
    wst25 = np.mean(errors[int(np.floor(0.75 * n) - 1) :])

    return men, med, trimean, bst25, wst25

def angerr2(l1, l2):

    l1 = l1 / (np.linalg.norm(l1) + 1e-12)
    l2 = l2 / (np.linalg.norm(l2) + 1e-12)
    rec = np.degrees(np.arccos(np.clip(np.sum(l1 * l2), -1, 1)))
    LL = l2 / (l1 + 1e-12)
    rep = np.degrees(np.arccos(np.dot(LL, np.ones(3)) /
                                 (np.sqrt(3) * np.sqrt(np.sum(LL ** 2)))))
    return rec, rep

def set_border(inp, width, method=1):
    temp = np.ones_like(inp)
    rr, cc = inp.shape
    y, x = np.ogrid[:rr, :cc]
    temp *= ((x < (cc - width)) & (x + 1 > width))
    temp *= ((y < (rr - width)) & (y + 1 > width))
    out = temp * inp
    if method == 1:
        if np.sum(temp) != 0:
            avg_val = np.sum(out) / np.sum(temp)
        else:
            avg_val = 0
        out = out + avg_val * (np.ones_like(inp) - temp)
    return out

def dilation33(inp, it=1):
    inp = np.array(inp)
    hh, ll = inp.shape
    
    for _ in range(it):

        channel0 = np.vstack((inp[1:, :], inp[-1:, :]))
        channel1 = inp.copy()
        channel2 = np.vstack((inp[0:1, :], inp[:-1, :]))

        temp = np.stack((channel0, channel1, channel2), axis=2)
        out2 = np.max(temp, axis=2)
        

        channel0_h = np.hstack((out2[:, 1:], out2[:, -1:]))
        channel1_h = out2.copy()
        channel2_h = np.hstack((out2[:, 0:1], out2[:, :ll-1]))
        
        temp2 = np.stack((channel0_h, channel1_h, channel2_h), axis=2)
        inp = np.max(temp2, axis=2)
    
    return inp

def updated_saliency_map(sRGBImage, VarThreshold, ColorThreshold):
    # Compute the logarithm of each color channel
    r_ln = np.log(sRGBImage[:, :, 0] + 1)
    g_ln = np.log(sRGBImage[:, :, 1] + 1)
    b_ln = np.log(sRGBImage[:, :, 2] + 1)

    # Compute the variance map along the third dimension (channels)
    stacked = np.stack((r_ln, g_ln, b_ln), axis=2)
    variance_map = np.var(stacked, axis=2, ddof=1)
    
    
    # Create an initial saliency map: pixels with variance > VarThreshold are set to 1
    updated_saliencyMap = np.zeros_like(variance_map)  
    updated_saliencyMap[variance_map > VarThreshold] = 1
    
    # Apply a median filter with an 11x11 kernel
    # updated_saliencyMap = medfilt2d(updated_saliencyMap, kernel_size=7)
    
    # Compute the means for each channel and the minimum mean
    Mr = np.mean(r_ln, axis=0)
    Mg = np.mean(g_ln, axis=0)
    Mb = np.mean(b_ln, axis=0)
    Minimum = np.min(np.concatenate([Mr, Mg, Mb]))
    
    # Compute absolute differences from the mean for each channel
    Xr = np.abs(r_ln - Mr)
    Xg = np.abs(g_ln - Mg)
    Xb = np.abs(b_ln - Mb)
        
    # Determine a threshold based on the minimum mean and ColorThreshold factor
    threshold = ColorThreshold * Minimum
    
    # Compute a difference map taking the maximum difference across channels
    difference_map = np.maximum(np.maximum(Xr, Xg), Xb)
    
    # Identify pixels considered "not important"
    not_important_mask = difference_map > threshold

    # Zero out not-important pixels in the saliency map
    updated_saliencyMap2 = updated_saliencyMap.copy()
    updated_saliencyMap2[not_important_mask] = 0
    
    return updated_saliencyMap2

def compute_edge_confidence(image, mask, bitDepth):
    # Scale image channels according to bitDepth
    scale = 2 ** bitDepth
    R = image[:, :, 0] / scale
    G = image[:, :, 1] / scale
    B = image[:, :, 2] / scale
    
    # Compute the average intensity (OW)
    OW = (R + G + B) / 3.0
    OW = OW * mask  # apply the mask
    
    # Extract non-zero elements for skewness and mean calculation
    nonzero = OW[OW != 0]
    if nonzero.size == 0:
        # Avoid division by zero if mask removes all pixels
        OWskew = 0
        m = 1.0
    else:
        OWskew = skew(nonzero)
        m = np.mean(nonzero)
    
    # Determine exponent E based on the skewness
    if OWskew > 1.5:
        E = 1.0
    elif OWskew > 0.2:
        E = 2.0
    else:
        E = 4.0
    
    # Compute the edge weights
    edge_weights = 1 - np.exp(-((OW / m) ** E))
    edge_weights[edge_weights < 0.90] = 0
    
    # Recompute if all values are zero (as in the MATLAB code)
    if np.sum(edge_weights) == 0:
        edge_weights = 1 - np.exp(-((OW / m) ** E))
    
    return edge_weights

def compute_derivative(channel, order, sigma):
    # Determine kernel size: ceil(6*sigma); make it odd if necessary.
    kernel_size = int(np.ceil(6 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create a Gaussian kernel using OpenCV. cv2.getGaussianKernel returns a column vector.
    k = cv2.getGaussianKernel(kernel_size, sigma)
    G = k * k.T  # Create 2D kernel by outer product.
    
    # Smooth the channel using cv2.filter2D with border replication.
    smoothed = cv2.filter2D(channel, -1, G, borderType=cv2.BORDER_REPLICATE)
    
    if order == 1:
        grad_y, grad_x = np.gradient(smoothed)
        derivative = np.sqrt(grad_x**2 + grad_y**2)
    else:
        derivative = np.abs(laplace(smoothed))
        
    return derivative

def estimate_illuminant_pixelwise(image, order, p, sigma, mask):
    # Convert image to double if necessary (im2double equivalent)
    if image.dtype != np.float64:
        image = image.astype(np.float64) / 255.0
    
    h, w, _ = image.shape
    # Compute derivatives for each channel
    derivatives = np.zeros_like(image)
    for c in range(3):
        derivatives[:, :, c] = compute_derivative(image[:, :, c], order, sigma)
    
    # Compute edge confidence measures
    bitDepth = 14
    edge_weights = compute_edge_confidence(image, mask, bitDepth)
    
    # Local processing parameters
    window_size = 3
    pad_size = window_size // 2
    # Pad derivatives and edge_weights using replicate padding
    padded_derivatives = np.pad(derivatives, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    # Note: padded_weights is computed but not used; we use the original edge_weights.
    
    numerator = np.zeros(3)
    denominator = np.zeros(3)
    
    # Process each pixel with edge confidence weighting
    for y in range(h):
        for x in range(w):
            for c in range(3):
                window = padded_derivatives[y:y+window_size, x:x+window_size, c]
                max_val = np.max(window)
                if max_val > 0:
                    nonzero_elements = window[window != 0]
                    center_val2 = np.mean(nonzero_elements) if nonzero_elements.size > 0 else 0
                    center_weight = edge_weights[y, x]
                    norm_center = center_val2 / max_val
                    weighted_val = center_val2 * center_weight
                    weighted_norm = norm_center * center_weight
                    numerator[c] += np.abs(weighted_val)**p
                    denominator[c] += np.abs(weighted_norm)**p
    
    illuminant = np.zeros(3)
    for c in range(3):
        if denominator[c] > 0:
            illuminant[c] = (numerator[c] / denominator[c]) ** (1/p)
    
    # Normalize the illuminant vector
    norm_val = np.linalg.norm(illuminant)
    if norm_val > 0:
        illuminant = illuminant / norm_val
    return illuminant

# accelerated version of the above function
def estimate_illuminant_pixelwise_accelerated(image, order, p, sigma, mask):
    # Convert image to float64 (im2double equivalent)
    if image.dtype != np.float64:
        image = image.astype(np.float64) / 255.0

    # h, w, _ = image.shape
    ## Compute derivatives for each channel
    #derivatives = np.empty_like(image)
    #for c in range(3):
    #    derivatives[..., c] = compute_derivative(image[..., c], order, sigma)

    # Compute edge confidence measures
    bitDepth = 14
    edge_weights = compute_edge_confidence(image, mask, bitDepth)

    # Local processing parameters
    window_size = 3
    pad_size = window_size // 2
    # Replicate padding ("edge" mode)
    # padded_derivatives = np.pad(derivatives, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    padded_derivatives = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

    numerator = np.zeros(3)
    denominator = np.zeros(3)

    # Process each channel with vectorized sliding window operations
    for c in range(3):
        # Extract sliding windows for the current channel.
        # Resulting shape is (h, w, window_size, window_size)
        windows = sliding_window_view(padded_derivatives[:, :, c], (window_size, window_size))
        
        # Compute maximum value in each window
        max_vals = np.max(windows, axis=(-1, -2))
        
        # Compute mean of nonzero elements in each window:
        nonzero_mask = windows != 0
        # Sum only the nonzero elements
        sum_nonzero = np.sum(np.where(nonzero_mask, windows, 0), axis=(-1, -2))
        # Count of nonzero elements in each window
        count_nonzero = np.sum(nonzero_mask, axis=(-1, -2))
        # Compute mean safely; if no nonzero elements, mean is set to 0.
        mean_vals = np.divide(sum_nonzero, count_nonzero, out=np.zeros_like(sum_nonzero), where=(count_nonzero != 0))
        
        # Only consider pixels where the maximum is positive
        valid = max_vals > 0
        
        # Edge weight for each pixel (broadcast over the h x w grid)
        center_weight = edge_weights
        # Compute the normalized center value where valid
        norm_center = np.zeros_like(mean_vals)
        norm_center[valid] = mean_vals[valid] / max_vals[valid]
        
        # Compute weighted values using the edge confidence
        weighted_val = mean_vals * center_weight
        weighted_norm = norm_center * center_weight

        # Accumulate numerator and denominator over valid pixels only
        numerator[c] = np.sum(np.abs(weighted_val[valid]) ** p)
        denominator[c] = np.sum(np.abs(weighted_norm[valid]) ** p)

    # Compute per-channel illuminant estimate
    illuminant = np.zeros(3)
    for c in range(3):
        if denominator[c] > 0:
            illuminant[c] = (numerator[c] / denominator[c]) ** (1/p)

    # Normalize the illuminant vector
    norm_val = np.linalg.norm(illuminant)
    if norm_val > 0:
        illuminant = illuminant / norm_val

    return illuminant

def deriv_gauss(img, sigma):
    GaussianDieOff = 1e-6
    # Possible widths from 1 to 50.
    pw = np.arange(1, 51)  # equivalent to 1:50 in MATLAB
    ssq = sigma ** 2
    exp_vals = np.exp(-(pw**2) / (2 * ssq))
    valid = np.where(exp_vals > GaussianDieOff)[0]
    if valid.size > 0:
        # valid indices start at 0, so add 1 to match MATLAB indexing range.
        width = valid[-1] + 1
    else:
        width = 1  # user entered a really small sigma
    
    # Create meshgrid for kernel indices from -width to width.
    xs = np.arange(-width, width+1)
    ys = np.arange(-width, width+1)
    x, y = np.meshgrid(xs, ys)
    
    # Construct the derivative Gaussian filter.
    dgau2D = -x * np.exp(-(x**2 + y**2) / (2 * ssq)) / (np.pi * ssq)
    
    # Convolve the image with the kernel and its transpose.
    ax = convolve(img, dgau2D, mode='nearest')
    ay = convolve(img, dgau2D.T, mode='nearest')
    
    # Compute the magnitude.
    mag = np.sqrt(ax**2 + ay**2)
    return mag

def normr(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = np.finfo(float).eps
    return data / norms

def matlab_prctile(data, percentage):
    data_sorted = np.sort(data) 
    n = len(data_sorted)
    p = percentage / 100.0
    rank = p * (n - 1) + 1
    k = int(np.floor(rank))
    d = rank - k
    if k - 1 < 0:
        return data_sorted[0]
    elif k >= n:
        return data_sorted[-1]
    else:
        return data_sorted[k - 1] + d * (data_sorted[k] - data_sorted[k - 1])

def gray_index_angular(img, mask, sigma, percentage):

    eps_val = np.finfo(float).eps
    rr, cc, dd = img.shape
    
    # Separate the color channels (MATLAB is 1-indexed; Python is 0-indexed)
    R = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    B = img[:, :, 2].copy()
    
    # Replace zeros with eps
    R[R == 0] = eps_val
    G[G == 0] = eps_val
    B[B == 0] = eps_val
    
    # Compute the Gaussian derivative magnitude of the logarithm of each channel.
    Mr = deriv_gauss(np.log(R), sigma)
    Mg = deriv_gauss(np.log(G), sigma)
    Mb = deriv_gauss(np.log(B), sigma)
    
    # Create a data matrix (each row corresponds to a pixel, each column a channel)
    data = np.column_stack((Mr.ravel(order='F'), Mg.ravel(order='F'), Mb.ravel(order='F'))).astype(np.float64)

    
    # Replace zeros in each channel (column) with eps
    data[data[:, 0] == 0, 0] = eps_val
    data[data[:, 1] == 0, 1] = eps_val
    data[data[:, 2] == 0, 2] = eps_val
    
    # Row-normalize the data
    data_normed = normr(data)
    gt1 = normr(np.ones_like(data))
    
    dot_product = np.sum(data_normed * gt1, axis=1)
    dot_product = np.clip(dot_product, -1, 1)
    angular_error = np.arccos(dot_product)
    
    # Reshape the angular error into the image shape.
    Greyidx_angular = angular_error.reshape((rr, cc), order='F').astype(np.float64)
    
    # Normalize Greyidx_angular to get Greyidx.
    max_val = np.max(Greyidx_angular)
    Greyidx = Greyidx_angular / (max_val + eps_val)
    
    # For pixels where all derivative responses are almost zero, set to max.
    condition = (Mr < eps_val) & (Mg < eps_val) & (Mb < eps_val)
    Greyidx[condition] = np.max(Greyidx)
    Greyidx_angular[condition] = np.max(Greyidx_angular)
    
    # Create a 7x7 averaging kernel and apply circular filtering.
    kernel = np.ones((7, 7), dtype=np.float64) / 49.0
    Greyidx = convolve2d(Greyidx, kernel, mode='same', boundary='wrap')
    Greyidx_angular = convolve2d(Greyidx_angular, kernel, mode='same', boundary='wrap')
    
    # If a mask is provided, force the angular index to its maximum where mask is true.
    if mask is not None and mask.size > 0:
        Greyidx_angular[mask.astype(bool)] = np.max(Greyidx_angular)
    
    # Determine the threshold based on the given percentile.
    # threshold = np.percentile(Greyidx_angular.ravel(order='F'), percentage, method='linear')

    threshold = np.percentile(Greyidx_angular.ravel(order='F'), percentage )

    binary_mask = np.zeros_like(Greyidx_angular).astype(np.float64)
    binary_mask[Greyidx_angular <= threshold] = 1
    
    return binary_mask

def imresize_nearest(img, scale):

    in_h, in_w = img.shape[:2]
    out_h = int(np.round(in_h * scale))
    out_w = int(np.round(in_w * scale))
    
    row_indices = np.clip(np.round((np.arange(out_h) + 0.5) / scale - 0.5).astype(int), 0, in_h - 1)
    col_indices = np.clip(np.round((np.arange(out_w) + 0.5) / scale - 0.5).astype(int), 0, in_w - 1)
    
    if img.ndim == 3:
        resized = img[row_indices[:, np.newaxis], col_indices, :]
    else:
        resized = img[row_indices[:, np.newaxis], col_indices]
    
    return resized

def save_arrays_to_single_csv(array1, array2, filename="output.csv"):
    max_len = max(len(array1), len(array2))
    
    array1 = array1.flatten()
    array2 = array2.flatten()
    
    array1 = np.pad(array1, (0, max_len - len(array1)), constant_values=np.nan)
    array2 = np.pad(array2, (0, max_len - len(array2)), constant_values=np.nan)

    df = pd.DataFrame({"Perf": array1, "Perf_rep": array2})

    df.to_csv(filename, index=False, float_format="%.6f")

def main():
    # Set paths
    base_path = "./"
    gt_mat_path = os.path.join(base_path, "NCCdataset", "gt.mat")
    img_path = os.path.join(base_path, "NCCdataset", "img")
    msk_path = os.path.join(base_path, "NCCdataset", "msk")
    
    # Load ground truth data
    gt_data = scipy.io.loadmat(gt_mat_path)
 
    if "gts" in gt_data:
        gt = gt_data["gts"]
    elif "gt" in gt_data:
        gt = gt_data["gt"]
    else:
        gt = None
    
    # image_indices = range(1, 5)
    image_indices = range(1, 514)    
    Nimg = len(image_indices)
    Perf = []
    Perf_rep = []
    
    for i in image_indices:
        print(f"Processing image {i}/{Nimg}...", flush=True)
        imname = f"{i}.png"  # In MATLAB: num2str(set(i-set(1)+1)) + '.png'
        img_full_path = os.path.join(img_path, imname)
        mask_full_path = os.path.join(msk_path, imname)
        
        # Read image and mask
        img = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Image not found: {img_full_path}")
            continue
        # Convert BGR to RGB and to float64
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Mask not found: {mask_full_path}")
            continue
        # Convert mask to boolean
        mask = mask > 0

        img = imresize_nearest(img, 0.125)
        mask = imresize_nearest(mask.astype(np.uint8), 0.125).astype(bool)
        
        # Compute saturation threshold and mask processing
        saturation_threshold = np.max(img) * 0.95
        # Compute maximum across channels
        max_img = np.max(img, axis=2)
        # Apply dilation33 to the thresholded image
        dilated = dilation33((max_img >= saturation_threshold).astype(np.float64))
        mask_im2 = mask.astype(np.float64) + dilated
        mask_im2 = (mask_im2 == 0).astype(np.float64)
        mask_proc = set_border(mask_im2, 1, method=0)
        mask_proc = 1 - mask_proc
        
        # Parameters for GrayIndexAngular
        sigma_val = 1.00
        percentage = 1.7
        binary_mask = gray_index_angular(img, mask_proc, sigma_val, percentage)
        
        # Compute saliency map using the updated function
        saliencyMap = updated_saliency_map(img, 0.05, 0.30)
        a = binary_mask * saliencyMap
        
        # If 'a' is entirely zero, fallback to binary_mask
        if np.count_nonzero(a) == 0:
            a = binary_mask
        
        # Apply the mask 'a' to the image (broadcasting to 3 channels)
        img_masked = img * a[:, :, np.newaxis]
        
        # Set parameters for illuminant estimation
        order = 2      # 2nd order derivatives
        p = 7         # Minkowski norm
        sigma_est = 3  # Gaussian sigma
        
        illuminant = estimate_illuminant_pixelwise_accelerated(img_masked, order, p, sigma_est, a)
        EvaLum = illuminant
        
        # Compute angular error metrics using angerr2
        if gt is not None:
            # Adjust for zero-indexing (MATLAB 1-indexing)
            gt_val = gt[i - 1, :].flatten()
            arr, arr_rep = angerr2(EvaLum, gt_val)
            print(f"Image {i}: arr = {arr}, arr_rep = {arr_rep}")
            Perf.append(arr)
            Perf_rep.append(arr_rep)
    
    # Evaluate performance metrics if any results were collected
    if Perf:
        mean_perf, median_perf, trimean_perf, bst25, wst25 = evaluate(np.array(Perf))
        print("Performance (binary errors) [median, mean, trimean, best25%, worst25%]:", 
              median_perf, mean_perf, trimean_perf, bst25, wst25)
    if Perf_rep:
        mean_rep, median_rep, trimean_rep, bst25_rep, wst25_rep = evaluate(np.array(Perf_rep))
        print("Performance (rep errors) [median, mean, trimean, best25%, worst25%]:", 
              median_rep, mean_rep, trimean_rep, bst25_rep, wst25_rep)
        
    save_arrays_to_single_csv(np.array(Perf), np.array(Perf_rep), filename="output.csv")

main()
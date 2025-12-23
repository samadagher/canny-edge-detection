import cv2
import numpy as np   
from collections import deque 

def non_maximum_suppression(grad_mag, grad_angle):
    """
    Apply non-maximum suppression to gradient magnitudes.
    
    Args:
        grad_mag: 2D array of gradient magnitudes
        grad_angle: 2D array of gradient angles (in radians)
    
    Returns:
        Z: Suppressed gradient magnitude array
    """
    M, N = grad_mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    
    # Convert gradient angle to degrees and normalize to [0, 180)
    angle = np.rad2deg(grad_angle) % 180
    
    # Process interior pixels (exclude borders: no padding)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q, r = 255, 255  # Initialize comparison values (to suppress pixels in unexpected cases)
            
            current_angle = angle[i, j]
            
            # Determine neighboring pixels based on gradient direction
            # 0° or 180°: horizontal edge
            if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                q = grad_mag[i, j + 1]
                r = grad_mag[i, j - 1]
            
            # 45°: diagonal edge (northeast-southwest)
            elif 22.5 <= current_angle < 67.5:
                q = grad_mag[i + 1, j - 1]
                r = grad_mag[i - 1, j + 1]
            
            # 90°: vertical edge
            elif 67.5 <= current_angle < 112.5:
                q = grad_mag[i + 1, j]
                r = grad_mag[i - 1, j]
            
            # 135°: diagonal edge (northwest-southeast)
            elif 112.5 <= current_angle < 157.5:
                q = grad_mag[i - 1, j - 1]
                r = grad_mag[i + 1, j + 1]
            
            # Keep pixel if it's a local maximum along gradient direction
            if (grad_mag[i, j] >= q) and (grad_mag[i, j] >= r):
                Z[i, j] = grad_mag[i, j]
            # Otherwise, suppress it (already 0)
    
    return Z

def double_threshold(image, low_threshold, high_threshold):
    """
    Apply double thresholding and return binary masks for hysteresis tracking.

    Args:
        image: 2D array of gradient magnitudes
        low_threshold: Lower threshold value
        high_threshold: Upper threshold value
    
    Returns:
        strong_edges: Binary mask of strong edge pixels
        weak_edges: Binary mask of weak edge pixels
    """
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be less than high_threshold")
    
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    
    return strong_edges, weak_edges

def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    """
    Connect weak edges to strong edges using hysteresis thresholding.
    
    Fast BFS implementation - processes each pixel exactly once.
    
    Args:
        strong_edges: Binary mask of strong edge pixels
        weak_edges: Binary mask of weak edge pixels
    
    Returns:
        result: Final edge map with all connected edges
    """
    M, N = strong_edges.shape
    
    strong_edges = strong_edges.astype(bool)
    weak_edges = weak_edges.astype(bool)
    
    result = strong_edges.copy()
    visited = np.zeros((M, N), dtype=bool)
    
    # Queue for BFS
    queue = deque() #Uses a deque (double-ended queue) for efficient BFS
    
    # Add all strong edge pixels to queue as starting points
    strong_coords = np.argwhere(strong_edges)
    for coord in strong_coords:
        i, j = coord
        queue.append((i, j))
        visited[i, j] = True #add all strong coords to visited
    
    # 8-connected neighbors (all directions)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    # BFS to track connected weak edges
    while queue:
        i, j = queue.popleft()
        
        # Check all 8 neighbors
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            
            # Check bounds (neighbor is inside image)
            if 0 <= ni < M and 0 <= nj < N:
                # If neighbor is weak edge and not yet visited
                if weak_edges[ni, nj] and not visited[ni, nj]:
                    result[ni, nj] = True  # Promote weak edge to strong
                    visited[ni, nj] = True
                    queue.append((ni, nj))  # Continue searching from this pixel
    
    return result

def canny_algorithm(image, ksize=3):

	# Step 1: Noise reduction
	blurred = cv2.GaussianBlur(image, (ksize,ksize), 0)

	# Step 2: Gradient calculation
	grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
	grad_mag = np.sqrt(grad_x**2 + grad_y**2)
	grad_angle = np.arctan2(grad_y, grad_x)

	# Step 3: Non-maximum suppression
	non_max_image = non_maximum_suppression(grad_mag, grad_angle)

	# Step 4: Double thresholding
	strong_edges, weak_edges = double_threshold(non_max_image, 30, 60)
	
	# Step 5: Hysteresis edge tracking
	final_edges = edge_tracking_by_hysteresis(strong_edges, weak_edges)

	result = final_edges.astype(np.uint8) * 255

	return result

def ILPF(image, D0=50):
    M, N = image.shape
    mask = np.zeros((M, N, 2), np.uint8) #2 channels for real and imaginary parts (to match DFT output format)
    x, y = np.ogrid[:M, :N] #gives: x with shape (M, 1) containing: 0,1,...,M-1 & likewise y [alternative for nested loops]
    mask_area = (x-M//2)**2 + (y-N//2)**2 <= D0**2 #True inside circle (low frequencies), False outside
    mask[mask_area] = 1 #if True -> both channels = 1; gives us a circle of 1s in the middle, 0/black outside
    
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft)
    fshift = shifted*mask #keeps the values of "shifted" within the circle

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX) #contrast stretching

    return img_back

def IHPF(image, D0=50):
    M, N = image.shape
    mask = np.zeros((M, N, 2), np.uint8) #2 channels for real and imaginary parts (to match DFT output format)
    x, y = np.ogrid[:M, :N] #gives: x with shape (M, 1) containing: 0,1,...,M-1 & likewise y [alternative for nested loops]
    mask_area = (x-M//2)**2 + (y-N//2)**2 > D0**2 #True inside circle (low frequencies), False outside
    mask[mask_area] = 1 #if True -> both channels = 1; gives us a circle of 1s in the middle, 0/black outside
    
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft)
    fshift = shifted*mask #keeps the values of "shifted" within the circle

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX) #contrast stretching

    return img_back

def GLPF(image, D0=50):
    M, N = image.shape
    mask = np.zeros((M, N, 2), np.float32) #gaussian mask contains float weights bc smoother (not just 0/1) 
    x, y = np.ogrid[:M, :N]
    D2 = (x-M//2)**2 + (y-N//2)**2 #D2: D squared
    gaussian = np.exp(-D2/(2*(D0**2))) #instead of sqrting then sqring, just keep D as it is
    mask[:,:,0] = gaussian
    mask[:,:,1] = gaussian #both real and imaginary parts of every frequency are multiplied by the same Gaussian weight when we do shifted*mask

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft)
    fshift = shifted*mask #keeps the values of "shifted" within the circle

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX) #contrast stretching

    return img_back

def GHPF(image, D0=50):
    M, N = image.shape
    mask = np.zeros((M, N, 2), np.float32) #gaussian mask contains float weights bc smoother (not just 0/1) 
    x, y = np.ogrid[:M, :N]
    D2 = (x-M//2)**2 + (y-N//2)**2 #D2: D squared
    gaussian = 1-np.exp(-D2/(2*(D0**2))) #instead of sqrting then sqring, just keep D as it is
    mask[:,:,0] = gaussian
    mask[:,:,1] = gaussian #both real and imaginary parts of every frequency are multiplied by the same Gaussian weight when we do shifted*mask

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft)
    fshift = shifted*mask 

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX) 

    return img_back

def BLPF(image, D0=50, n=1):
    M, N = image.shape
    mask = np.zeros((M, N, 2), np.float32) 
    x, y = np.ogrid[:M, :N]
    D = np.sqrt((x-M//2)**2 + (y-N//2)**2)
    butterworth = 1.0/(1.0+(D/D0)**(2*n))
    mask[:,:,0] = butterworth
    mask[:,:,1] = butterworth

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft)
    fshift = shifted*mask 

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX) 

    return img_back

def BHPF(image, D0=50, n=1):
    M, N = image.shape
    mask = np.zeros((M, N, 2), np.float32)
    x, y = np.ogrid[:M, :N]
    D = np.sqrt((x-M//2)**2 + (y-N//2)**2)
    butterworth = 1.0/(1.0+(D0/D)**(2*n))
    mask[:,:,0] = butterworth
    mask[:,:,1] = butterworth

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft)
    fshift = shifted*mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return img_back
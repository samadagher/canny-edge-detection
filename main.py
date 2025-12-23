"""
Webcam demo for Canny edge detection.

Controls:
'o' - Show original
'i' - Show Ideal LPF
'I' - Show Ideal HPF
'g' - Show Gaussian LPF
'G' - Show Gaussian HPF
'b' - Show Butterworth LPF
'B' - Show Butterworth HPF
'c' - Show Canny edges
'x' - Show Sobel X (vertical edges)
'y' - Show Sobel Y (horizontal edges)
'+' - Increase D0
'-' - Decrease D0
'1'-'9' - Set Butterworth order (n)
'q' - Quit
"""

import cv2
import numpy as np
from canny import canny_algorithm, ILPF, IHPF, GLPF, GHPF, BLPF, BHPF

cap = cv2.VideoCapture(0) # Open webcam (0 is usually the default camera)

if not cap.isOpened():
    print("Error: Could not open webcam")
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Still cannot open webcam. Exiting.")
        exit()

mode = 'o'
D0 = 50  # default cutoff frequency
n = 2    # default Butterworth order
print("Webcam opened successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == 'o':
        display = frame
        text = "Mode: Original"

    elif mode == 'c':
        edges = canny_algorithm(gray)
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        text = "Mode: Canny Edges"

    elif mode == 'x':
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)  # Convert to uint8 for display
        display = cv2.cvtColor(grad_x, cv2.COLOR_GRAY2BGR)
        text = "Mode: Sobel X (Vertical Edges)"
    
    elif mode == 'y':
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_y = cv2.convertScaleAbs(grad_y)  # Convert to uint8 for display
        display = cv2.cvtColor(grad_y, cv2.COLOR_GRAY2BGR)
        text = "Mode: Sobel Y (Horizontal Edges)"

    elif mode == 'i':
        ilpf_result = ILPF(gray, D0=D0)
        display = cv2.cvtColor(ilpf_result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text = "Mode: Ideal LPF"
        text2 = f"D0 = {D0}"

    elif mode == 'I':
        ihpf_result = IHPF(gray, D0=D0)
        display = cv2.cvtColor(ihpf_result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text = "Mode: Ideal HPF"
        text2 = f"D0 = {D0}"

    elif mode == 'g':
        glpf_result = GLPF(gray, D0=D0)
        display = cv2.cvtColor(glpf_result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text = "Mode: Gaussian LPF"
        text2 = f"D0 = {D0}"

    elif mode == 'G':
        ghpf_result = GHPF(gray, D0=D0)
        display = cv2.cvtColor(ghpf_result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text = "Mode: Gaussian HPF"
        text2 = f"D0 = {D0}"

    elif mode == 'b':
        blpf_result = BLPF(gray, D0=D0, n=n)
        display = cv2.cvtColor(blpf_result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text = "Mode: Butterworth LPF"
        text2 = f"D0 = {D0}, n = {n}"

    elif mode == 'B':
        bhpf_result = BHPF(gray, D0=D0, n=n)
        display = cv2.cvtColor(bhpf_result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text = "Mode: Butterworth HPF"
        text2 = f"D0 = {D0}, n = {n}"

    cv2.putText(display, text, (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if mode in ['i', 'I', 'g', 'G', 'b', 'B']:
        cv2.putText(display, text2, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Webcam - Canny Edge Detection', display)

    key = cv2.waitKey(1) & 0xFF # keeps only the last byte of the result (for some systems)

    if key == ord('q'):
        break
    elif key == ord('+'):
        D0 += 5  # Increase D0
    elif key == ord('-'):
        D0 = max(5, D0 - 5)  # Decrease D0, minimum 5
    elif key >= ord('1') and key <= ord('9'):
        n = key - ord('0')  # Set Butterworth order from 1-9
    elif key in [ord('o'), ord('i'), ord('I'), ord('g'), ord('G'), 
                ord('b'), ord('B'), ord('c'), ord('x'), ord('y')]:
        mode = chr(key)

cap.release()
cv2.destroyAllWindows()
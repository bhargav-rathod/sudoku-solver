import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
import os

app = Flask(__name__)
model = load_model('digit_model.h5')

def preprocess_image(img):
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Apply adaptive thresholding with different parameters
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    # Combine both thresholding results
    combined = cv2.bitwise_or(thresh1, thresh2)
    
    # Clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return processed

def find_grid_contour(processed_img, original_img):
    # Find all contours
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort contours by area and check the top candidates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Check if it's quadrilateral
        if len(approx) == 4:
            # Additional checks for Sudoku grid
            area = cv2.contourArea(contour)
            if area < 10000:  # Minimum area threshold
                continue
                
            # Check convexity
            if not cv2.isContourConvex(approx):
                continue
                
            # Check aspect ratio (should be roughly square)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue
                
            return approx
    
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective(image, contour):
    ordered_pts = order_points(contour.reshape(4, 2))
    (tl, tr, br, bl) = ordered_pts
    width = height = 450
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix

def extract_digits(warped_grid):
    board = np.zeros((9, 9), dtype="int")
    cell_size = warped_grid.shape[0] // 9
    gray_warped = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
    
    # Try multiple thresholding methods
    thresh1 = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    for i in range(9):
        for j in range(9):
            # Try both thresholding results
            for thresh in [thresh1, thresh2]:
                cell = thresh[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell = cv2.copyMakeBorder(cell, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                # Find contours
                contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # More lenient size requirements
                    if w > 10 and h > 10 and cv2.contourArea(c) > 50:
                        digit_roi = cell[y:y+h, x:x+w]
                        digit_img = cv2.resize(digit_roi, (28, 28))
                        
                        # Additional preprocessing for the digit
                        digit_img = cv2.erode(digit_img, np.ones((2,2), np.uint8), iterations=1)
                        digit_img = cv2.dilate(digit_img, np.ones((2,2), np.uint8), iterations=1)
                        
                        digit_img = digit_img.astype("float") / 255.0
                        digit_img = np.expand_dims(digit_img, axis=-1)
                        digit_img = np.expand_dims(digit_img, axis=0)
                        
                        prediction = model.predict(digit_img, verbose=0)
                        confidence = np.max(prediction)
                        
                        # Only accept predictions with reasonable confidence
                        if confidence > 0.7:
                            board[i, j] = np.argmax(prediction)
                            break  # Use the first good prediction
    
    return board

def solve_sudoku(board):
    def find_empty(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(board, num, pos):
        # Check row
        for j in range(9):
            if board[pos[0]][j] == num and pos[1] != j:
                return False
        # Check column
        for i in range(9):
            if board[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x*3, box_x*3 + 3):
                if board[i][j] == num and (i,j) != pos:
                    return False
        return True

    find = find_empty(board)
    if not find:
        return True
    row, col = find

    for i in range(1,10):
        if is_valid(board, i, (row, col)):
            board[row][col] = i
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Try multiple preprocessing approaches
        processed = preprocess_image(img)
        grid_contour = find_grid_contour(processed, img)
        
        if grid_contour is None:
            # Try alternative approach if first attempt fails
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            grid_contour = find_grid_contour(edges, img)
            
            if grid_contour is None:
                return jsonify({'error': 'No Sudoku grid detected. Try a clearer image.'}), 400
        
        warped, _ = warp_perspective(img, grid_contour)
        board = extract_digits(warped)
        
        # Solve
        solved_board = board.copy()
        if not solve_sudoku(solved_board):
            return jsonify({'error': 'Unsolvable puzzle'}), 400
        
        return jsonify({
            'original': board.tolist(),
            'solution': solved_board.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
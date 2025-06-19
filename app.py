import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with error handling
try:
    model = load_model('digit_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def preprocess_image(img):
    """Enhanced image preprocessing with multiple techniques"""
    # Resize image if too large
    height, width = img.shape[:2]
    if width > 1000 or height > 1000:
        scale = min(1000/width, 1000/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Multiple thresholding approaches
    # 1. Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 2. Otsu's thresholding
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Another adaptive threshold with different parameters
    thresh3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 15, 3)
    
    # Combine thresholds
    combined = cv2.bitwise_and(thresh1, thresh2)
    combined = cv2.bitwise_or(combined, thresh3)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return processed, gray

def find_grid_contour(processed_img, original_img):
    """Improved grid detection with multiple fallback methods"""
    # Method 1: Find contours in processed image
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Sort by area and try largest contours first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:10]:  # Check top 10 largest
            # Approximate contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly quadrilateral
            if len(approx) >= 4:
                # If more than 4 points, try to find the best 4
                if len(approx) > 4:
                    # Use convex hull and find corners
                    hull = cv2.convexHull(contour)
                    epsilon = 0.02 * cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    # More lenient area threshold
                    if area > 5000:
                        # Check if it's roughly square
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        if 0.7 <= aspect_ratio <= 1.3:  # More lenient aspect ratio
                            return approx
    
    # Method 2: Use Hough Line Transform as fallback
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None and len(lines) >= 4:
        # This is a simplified approach - in practice, you'd want to
        # find the intersection points of the strongest horizontal and vertical lines
        height, width = gray.shape
        margin = min(width, height) // 10
        
        # Create a rough grid boundary
        approx = np.array([
            [[margin, margin]],
            [[width-margin, margin]],
            [[width-margin, height-margin]],
            [[margin, height-margin]]
        ], dtype=np.int32)
        
        return approx
    
    return None

def order_points(pts):
    """Order points in clockwise order starting from top-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left has smallest sum, bottom-right has largest sum
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, bottom-left has largest difference
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def warp_perspective(image, contour):
    """Extract and warp the Sudoku grid"""
    ordered_pts = order_points(contour.reshape(4, 2))
    
    # Calculate the width and height of the new image
    (tl, tr, br, bl) = ordered_pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Use the larger dimension to ensure square output
    size = max(maxWidth, maxHeight, 450)  # Minimum 450px
    
    dst = np.array([
        [0, 0],
        [size-1, 0],
        [size-1, size-1],
        [0, size-1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    
    return warped, matrix

def extract_digits(warped_grid):
    """Enhanced digit extraction with better preprocessing"""
    board = np.zeros((9, 9), dtype="int")
    cell_size = warped_grid.shape[0] // 9
    
    # Convert to grayscale if needed
    if len(warped_grid.shape) == 3:
        gray_warped = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
    else:
        gray_warped = warped_grid
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_warped = clahe.apply(gray_warped)
    
    # Multiple thresholding approaches
    thresh_methods = []
    
    # Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh_methods.append(thresh1)
    
    thresh2 = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    thresh_methods.append(thresh2)
    
    # Otsu thresholding
    _, thresh3 = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_methods.append(thresh3)
    
    for i in range(9):
        for j in range(9):
            cell_predictions = []
            
            # Try each thresholding method
            for thresh in thresh_methods:
                # Extract cell with some padding
                padding = cell_size // 10
                y_start = max(0, i * cell_size + padding)
                y_end = min(thresh.shape[0], (i + 1) * cell_size - padding)
                x_start = max(0, j * cell_size + padding)
                x_end = min(thresh.shape[1], (j + 1) * cell_size - padding)
                
                cell = thresh[y_start:y_end, x_start:x_end]
                
                if cell.size == 0:
                    continue
                
                # Find contours in the cell
                contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Filter out noise and grid lines
                    contour_area = cv2.contourArea(largest_contour)
                    cell_area = cell.shape[0] * cell.shape[1]
                    
                    # Check if contour is significant enough to be a digit
                    if (w > cell.shape[1] * 0.2 and h > cell.shape[0] * 0.2 and
                        contour_area > cell_area * 0.05 and
                        contour_area < cell_area * 0.8):
                        
                        # Extract digit region
                        digit_roi = cell[y:y+h, x:x+w]
                        
                        # Resize to 28x28 for model input
                        digit_img = cv2.resize(digit_roi, (28, 28))
                        
                        # Additional preprocessing
                        # Remove small noise
                        kernel = np.ones((2,2), np.uint8)
                        digit_img = cv2.morphologyEx(digit_img, cv2.MORPH_OPEN, kernel)
                        
                        # Normalize
                        digit_img = digit_img.astype("float32") / 255.0
                        digit_img = np.expand_dims(digit_img, axis=-1)
                        digit_img = np.expand_dims(digit_img, axis=0)
                        
                        # Predict
                        if model is not None:
                            try:
                                prediction = model.predict(digit_img, verbose=0)
                                confidence = np.max(prediction)
                                predicted_digit = np.argmax(prediction)
                                
                                # Only consider high-confidence predictions
                                if confidence > 0.6 and predicted_digit > 0:  # Exclude 0 (empty)
                                    cell_predictions.append((predicted_digit, confidence))
                            except Exception as e:
                                logger.warning(f"Prediction error for cell ({i},{j}): {e}")
            
            # Choose the best prediction
            if cell_predictions:
                # Sort by confidence and take the best
                cell_predictions.sort(key=lambda x: x[1], reverse=True)
                board[i, j] = cell_predictions[0][0]
    
    return board

def is_valid_sudoku(board):
    """Check if the extracted Sudoku board is valid"""
    def is_valid_unit(unit):
        unit = [i for i in unit if i != 0]
        return len(unit) == len(set(unit))
    
    # Check rows
    for row in board:
        if not is_valid_unit(row):
            return False
    
    # Check columns
    for col in range(9):
        if not is_valid_unit([board[row][col] for row in range(9)]):
            return False
    
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for row in range(box_row*3, box_row*3 + 3):
                for col in range(box_col*3, box_col*3 + 3):
                    box.append(board[row][col])
            if not is_valid_unit(box):
                return False
    
    return True

def solve_sudoku(board):
    """Solve Sudoku using backtracking"""
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
        
        # Check 3x3 box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x*3, box_x*3 + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False
        
        return True

    find = find_empty(board)
    if not find:
        return True
    
    row, col = find

    for i in range(1, 10):
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
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure digit_model.h5 is available.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.info(f"Image loaded: {img.shape}")
        
        # Preprocess image
        processed, gray = preprocess_image(img)
        
        # Find grid contour
        grid_contour = find_grid_contour(processed, img)
        
        if grid_contour is None:
            # Try with different preprocessing
            logger.info("First attempt failed, trying alternative preprocessing")
            
            # Alternative preprocessing
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilate edges to connect broken lines
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            grid_contour = find_grid_contour(edges, img)
            
            if grid_contour is None:
                return jsonify({'error': 'Could not detect Sudoku grid. Please ensure the image shows a clear, complete Sudoku grid.'}), 400
        
        logger.info("Grid contour found")
        
        # Warp perspective to get clean grid
        warped, _ = warp_perspective(img, grid_contour)
        
        # Extract digits
        board = extract_digits(warped)
        logger.info(f"Extracted board:\n{board}")
        
        # Validate the extracted board
        if not is_valid_sudoku(board):
            logger.warning("Invalid Sudoku detected, but attempting to solve anyway")
        
        # Check if board has enough clues (at least 17 for a valid Sudoku)
        filled_cells = np.count_nonzero(board)
        if filled_cells < 10:
            return jsonify({'error': 'Not enough digits detected. Please use a clearer image.'}), 400
        
        # Solve the puzzle
        solved_board = board.copy()
        if solve_sudoku(solved_board):
            return jsonify({
                'original': board.tolist(),
                'solution': solved_board.tolist(),
                'detected_digits': int(filled_cells)
            })
        else:
            return jsonify({
                'error': 'Could not solve the puzzle. This might be due to incorrect digit recognition or an invalid puzzle.',
                'original': board.tolist(),
                'detected_digits': int(filled_cells)
            }), 400
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
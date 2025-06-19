import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import threading

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

# Global thread pool for faster processing
executor = ThreadPoolExecutor(max_workers=4)

def preprocess_image_fast(img):
    """Fast image preprocessing optimized for speed"""
    start_time = time.time()
    
    # Resize image if too large (more aggressive resizing)
    height, width = img.shape[:2]
    max_size = 600  # Reduced from 1000
    if width > max_size or height > max_size:
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image to: {new_width}x{new_height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Fast contrast enhancement
    gray = cv2.equalizeHist(gray)
    
    # Single, fast thresholding method
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    logger.info(f"Preprocessing took: {time.time() - start_time:.2f}s")
    return thresh, gray

def find_grid_contour_fast(processed_img, original_img):
    """Fast grid detection with early exit"""
    start_time = time.time()
    
    # Find contours
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort by area and check only top 3 candidates for speed
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3000:  # Skip small contours early
            continue
            
        # Quick approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Quick aspect ratio check
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.7 <= aspect_ratio <= 1.3:
                logger.info(f"Grid detection took: {time.time() - start_time:.2f}s")
                return approx
    
    # Fast fallback: use image bounds
    height, width = processed_img.shape
    margin = min(width, height) // 15
    approx = np.array([
        [[margin, margin]],
        [[width-margin, margin]],
        [[width-margin, height-margin]],
        [[margin, height-margin]]
    ], dtype=np.int32)
    
    logger.info(f"Grid detection (fallback) took: {time.time() - start_time:.2f}s")
    return approx

def order_points_fast(pts):
    """Fast point ordering"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    
    return rect

def warp_perspective_fast(image, contour):
    """Fast perspective warping with fixed size"""
    ordered_pts = order_points_fast(contour.reshape(4, 2))
    
    # Fixed size for speed (smaller than before)
    size = 360  # Reduced from 450
    dst = np.array([
        [0, 0],
        [size-1, 0],
        [size-1, size-1],
        [0, size-1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    
    return warped, matrix

def extract_single_digit(cell_img, row, col):
    """Extract digit from a single cell - optimized for threading"""
    try:
        if cell_img.size == 0:
            return 0
        
        # Fast thresholding
        if len(cell_img.shape) == 3:
            cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            cell_gray = cell_img
        
        # Single threshold method for speed
        thresh = cv2.adaptiveThreshold(cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Quick filtering
        contour_area = cv2.contourArea(largest_contour)
        cell_area = thresh.shape[0] * thresh.shape[1]
        
        if (w < thresh.shape[1] * 0.2 or h < thresh.shape[0] * 0.2 or
            contour_area < cell_area * 0.05 or contour_area > cell_area * 0.8):
            return 0
        
        # Extract and preprocess digit
        digit_roi = thresh[y:y+h, x:x+w]
        digit_img = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Simple morphological operation
        kernel = np.ones((2,2), np.uint8)
        digit_img = cv2.morphologyEx(digit_img, cv2.MORPH_OPEN, kernel)
        
        # Prepare for model
        digit_img = digit_img.astype("float32") / 255.0
        digit_img = np.expand_dims(digit_img, axis=-1)
        digit_img = np.expand_dims(digit_img, axis=0)
        
        # Predict
        if model is not None:
            prediction = model.predict(digit_img, verbose=0)
            confidence = np.max(prediction)
            predicted_digit = np.argmax(prediction)
            
            # Lower confidence threshold for speed
            if confidence > 0.5 and predicted_digit > 0:
                return predicted_digit
        
        return 0
        
    except Exception as e:
        logger.warning(f"Error processing cell ({row},{col}): {e}")
        return 0

def extract_digits_fast(warped_grid):
    """Fast digit extraction using parallel processing"""
    start_time = time.time()
    
    board = np.zeros((9, 9), dtype="int")
    cell_size = warped_grid.shape[0] // 9
    
    # Convert to grayscale if needed
    if len(warped_grid.shape) == 3:
        gray_warped = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
    else:
        gray_warped = warped_grid
    
    # Prepare all cells for parallel processing
    cell_tasks = []
    for i in range(9):
        for j in range(9):
            # Extract cell with padding
            padding = max(1, cell_size // 20)  # Minimal padding
            y_start = max(0, i * cell_size + padding)
            y_end = min(gray_warped.shape[0], (i + 1) * cell_size - padding)
            x_start = max(0, j * cell_size + padding)
            x_end = min(gray_warped.shape[1], (j + 1) * cell_size - padding)
            
            cell = gray_warped[y_start:y_end, x_start:x_end]
            cell_tasks.append((cell.copy(), i, j))
    
    # Process cells in parallel (limited to avoid timeout)
    results = {}
    batch_size = 9  # Process 9 cells at a time
    
    for batch_start in range(0, len(cell_tasks), batch_size):
        batch = cell_tasks[batch_start:batch_start + batch_size]
        
        # Use threading for this batch
        futures = []
        for cell_img, row, col in batch:
            future = executor.submit(extract_single_digit, cell_img, row, col)
            futures.append((future, row, col))
        
        # Collect results with timeout
        for future, row, col in futures:
            try:
                result = future.result(timeout=1.0)  # 1 second timeout per cell
                results[(row, col)] = result
            except Exception as e:
                logger.warning(f"Timeout or error for cell ({row},{col}): {e}")
                results[(row, col)] = 0
    
    # Fill board
    for i in range(9):
        for j in range(9):
            board[i, j] = results.get((i, j), 0)
    
    logger.info(f"Digit extraction took: {time.time() - start_time:.2f}s")
    return board

def solve_sudoku_fast(board):
    """Fast Sudoku solver with early termination"""
    start_time = time.time()
    
    def find_empty_optimized(board):
        # Find empty cell with fewest possibilities (MRV heuristic)
        min_possibilities = 10
        best_cell = None
        
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    possibilities = 0
                    for num in range(1, 10):
                        if is_valid_fast(board, num, (i, j)):
                            possibilities += 1
                    
                    if possibilities < min_possibilities:
                        min_possibilities = possibilities
                        best_cell = (i, j)
                        
                        # If only one possibility, return immediately
                        if possibilities == 1:
                            return best_cell
        
        return best_cell

    def is_valid_fast(board, num, pos):
        row, col = pos
        
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True

    def solve_recursive(board, depth=0):
        # Prevent infinite recursion
        if depth > 81:
            return False
            
        # Check timeout (15 seconds max)
        if time.time() - start_time > 15:
            return False
        
        find = find_empty_optimized(board)
        if not find:
            return True
        
        row, col = find
        
        for num in range(1, 10):
            if is_valid_fast(board, num, (row, col)):
                board[row][col] = num
                
                if solve_recursive(board, depth + 1):
                    return True
                
                board[row][col] = 0
        
        return False

    success = solve_recursive(board)
    logger.info(f"Sudoku solving took: {time.time() - start_time:.2f}s")
    return success

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    overall_start = time.time()
    
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
        
        # Fast preprocessing
        processed, gray = preprocess_image_fast(img)
        
        # Fast grid detection
        grid_contour = find_grid_contour_fast(processed, img)
        
        if grid_contour is None:
            return jsonify({'error': 'Could not detect Sudoku grid. Please ensure the image shows a clear, complete Sudoku grid.'}), 400
        
        logger.info("Grid contour found")
        
        # Fast perspective warp
        warped, _ = warp_perspective_fast(img, grid_contour)
        
        # Fast digit extraction
        board = extract_digits_fast(warped)
        logger.info(f"Extracted board:\n{board}")
        
        # Check if we have enough digits
        filled_cells = np.count_nonzero(board)
        if filled_cells < 8:  # Reduced threshold
            return jsonify({'error': 'Not enough digits detected. Please use a clearer image.'}), 400
        
        # Fast solve
        solved_board = board.copy()
        solve_success = solve_sudoku_fast(solved_board)
        
        total_time = time.time() - overall_start
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        if solve_success:
            return jsonify({
                'original': board.tolist(),
                'solution': solved_board.tolist(),
                'detected_digits': int(filled_cells),
                'processing_time': round(total_time, 2)
            })
        else:
            return jsonify({
                'error': 'Could not solve the puzzle within time limit. This might be due to incorrect digit recognition.',
                'original': board.tolist(),
                'detected_digits': int(filled_cells),
                'processing_time': round(total_time, 2)
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
    # Increased timeout for development
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
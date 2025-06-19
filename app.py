import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
import os

app = Flask(__name__)
model = load_model('digit_model.h5')

# --- Your Original Functions (Minimal Changes) ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def find_grid_contour(processed_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    if len(approx) == 4 and cv2.contourArea(largest_contour) > 2000:
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
    thresh_warped = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    
    for i in range(9):
        for j in range(9):
            cell = thresh_warped[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell = cv2.copyMakeBorder(cell, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if w > 15 and h > 15 and cv2.contourArea(c) > 100:
                    digit_roi = cell[y:y+h, x:x+w]
                    digit_img = cv2.resize(digit_roi, (28, 28))
                    digit_img = digit_img.astype("float") / 255.0
                    digit_img = np.expand_dims(digit_img, axis=-1)
                    digit_img = np.expand_dims(digit_img, axis=0)
                    prediction = model.predict(digit_img, verbose=0)
                    board[i, j] = np.argmax(prediction)
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

# --- Flask Routes ---
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
        
        # Process image
        processed = preprocess_image(img)
        grid_contour = find_grid_contour(processed)
        
        if grid_contour is None:
            return jsonify({'error': 'No Sudoku grid detected'}), 400
            
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
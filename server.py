#!/usr/bin/env python3
"""
ChessScan Analysis Server
=========================
Full pipeline server using YOLO models.

Place models in same folder as this script:
  1.pt - Board detection
  2.pt - Corner detection  
  
  === COLOR PIPELINE (default) ===
  3.pt - Piece TYPE classification (P/N/B/R/Q/K/empty)
  P.pt - Pawn COLOR (W vs B)
  N.pt - Knight COLOR (W vs B)
  B.pt - Bishop COLOR (W vs B)
  R.pt - Rook COLOR (W vs B)
  Q.pt - Queen COLOR (W vs B)
  K.pt - King COLOR (W vs B)
  
  === B/W BOOK PIPELINE ===
  Z0.pt - Board type classifier (book_bw vs color) - DECIDES WHICH PIPELINE
  Z3.pt - Piece TYPE for B/W books (K/Q/R/B/N/P/empty)
  Z3_K_color.pt - King color for B/W books (white vs black)
  Z3_Q_color.pt - Queen color for B/W books
  Z3_R_color.pt - Rook color for B/W books
  Z3_B_color.pt - Bishop color for B/W books
  Z3_N_color.pt - Knight color for B/W books
  Z3_P_color.pt - Pawn color for B/W books

Endpoints:
  GET  /health          - Health check
  POST /analyze         - Full analysis (multipart image upload)

Returns JSON with base64 images for each pipeline step.

Pipeline:
  1. Board detection (1.pt)
  2. Corner detection (2.pt)
  3. Perspective warp
  4. CV corner refinement + grid extrapolation
  5. Square extraction using CV grid
  6. Z0.pt → Decide pipeline (book_bw or color)
  7a. COLOR: 3.pt (piece type) + P/N/B/R/Q/K.pt (color)
  7b. BOOK_BW: Z3.pt (piece type) + Z3_X_color.pt (color)
  8. Combine → FEN (uppercase=White, lowercase=Black)

Usage:
  python3 server.py
  # Server runs on http://localhost:5555
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os

app = Flask(__name__)
CORS(app)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BOARD_MODEL_PATH = SCRIPT_DIR / '1.pt'
CORNER_MODEL_PATH = SCRIPT_DIR / '2.pt'

# === COLOR PIPELINE (default) ===
PIECE_MODEL_PATH = SCRIPT_DIR / '3.pt'  # TYPE classification (P/N/B/R/Q/K/empty)
COLOR_MODEL_PATHS = {
    'P': SCRIPT_DIR / 'P.pt',
    'N': SCRIPT_DIR / 'N.pt',
    'B': SCRIPT_DIR / 'B.pt',
    'R': SCRIPT_DIR / 'R.pt',
    'Q': SCRIPT_DIR / 'Q.pt',
    'K': SCRIPT_DIR / 'K.pt',
}

# === B/W BOOK PIPELINE ===
Z0_MODEL_PATH = SCRIPT_DIR / 'Z0.pt'  # Board type classifier (decides pipeline)
Z3_PIECE_MODEL_PATH = SCRIPT_DIR / 'Z3.pt'  # Piece TYPE for B/W books
Z3_COLOR_MODEL_PATHS = {
    'K': SCRIPT_DIR / 'Z3_K_color.pt',
    'Q': SCRIPT_DIR / 'Z3_Q_color.pt',
    'R': SCRIPT_DIR / 'Z3_R_color.pt',
    'B': SCRIPT_DIR / 'Z3_B_color.pt',
    'N': SCRIPT_DIR / 'Z3_N_color.pt',
    'P': SCRIPT_DIR / 'Z3_P_color.pt',
}

CLASS_NAMES = ['empty', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b']
# Z3.pt class names (will be loaded from model)
Z3_CLASS_NAMES = ['B', 'K', 'N', 'P', 'Q', 'R', 'empty']  # Default, updated at load

FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']

SQUARE_SIZE = 64
WARP_SIZE = 512

# Colors (BGR)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

# =============================================================================
# BOARD SQUARE COLOR DETECTION (for orientation check)
# =============================================================================

def get_board_square_color(square_img):
    """
    Determine if the BOARD SQUARE is light or dark.
    Checks corners/edges where the board color shows (not center where piece is).
    Returns: float 0-1 (0=dark square, 1=light square)
    """
    h, w = square_img.shape[:2]
    
    corner_size = max(int(0.1 * min(h, w)), 3)
    
    corners = [
        square_img[0:corner_size, 0:corner_size],
        square_img[0:corner_size, w-corner_size:w],
        square_img[h-corner_size:h, 0:corner_size],
        square_img[h-corner_size:h, w-corner_size:w],
    ]
    
    brightnesses = []
    for corner in corners:
        if len(corner.shape) == 3:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        else:
            gray = corner
        brightnesses.append(np.mean(gray) / 255.0)
    
    return np.median(brightnesses)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
BOARD_MODEL = None
CORNER_MODEL = None

# Color pipeline models
PIECE_MODEL = None
COLOR_MODELS = {}

# B/W Book pipeline models
Z0_MODEL = None
Z3_PIECE_MODEL = None
Z3_COLOR_MODELS = {}

# --- Board & Corner (shared) ---
if BOARD_MODEL_PATH.exists():
    BOARD_MODEL = YOLO(str(BOARD_MODEL_PATH))
    print(f"  ✅ 1.pt Board detection")
else:
    print(f"  ❌ 1.pt not found")

if CORNER_MODEL_PATH.exists():
    CORNER_MODEL = YOLO(str(CORNER_MODEL_PATH))
    print(f"  ✅ 2.pt Corner detection")
else:
    print(f"  ❌ 2.pt not found")

# --- Z0 Pipeline Selector ---
if Z0_MODEL_PATH.exists():
    Z0_MODEL = YOLO(str(Z0_MODEL_PATH))
    z0_names = Z0_MODEL.names if hasattr(Z0_MODEL, 'names') else {}
    print(f"  ✅ Z0.pt Pipeline selector - classes: {z0_names}")
else:
    print(f"  ⚠️  Z0.pt not found (will use COLOR pipeline by default)")

# --- COLOR PIPELINE ---
print("\nLoading COLOR pipeline models...")
if PIECE_MODEL_PATH.exists():
    PIECE_MODEL = YOLO(str(PIECE_MODEL_PATH))
    print(f"  ✅ 3.pt Piece TYPE classification")
else:
    print(f"  ❌ 3.pt not found")

for piece_type, model_path in COLOR_MODEL_PATHS.items():
    if model_path.exists():
        COLOR_MODELS[piece_type] = YOLO(str(model_path))
        names = COLOR_MODELS[piece_type].names if hasattr(COLOR_MODELS[piece_type], 'names') else {}
        print(f"  ✅ {piece_type}.pt ({piece_type} color) - classes: {names}")
    else:
        print(f"  ❌ {piece_type}.pt not found")

# --- B/W BOOK PIPELINE ---
print("\nLoading B/W BOOK pipeline models...")
if Z3_PIECE_MODEL_PATH.exists():
    Z3_PIECE_MODEL = YOLO(str(Z3_PIECE_MODEL_PATH))
    if hasattr(Z3_PIECE_MODEL, 'names'):
        Z3_CLASS_NAMES = list(Z3_PIECE_MODEL.names.values())
    print(f"  ✅ Z3.pt Piece TYPE (B/W) - classes: {Z3_CLASS_NAMES}")
else:
    print(f"  ❌ Z3.pt not found")

for piece_type, model_path in Z3_COLOR_MODEL_PATHS.items():
    if model_path.exists():
        Z3_COLOR_MODELS[piece_type] = YOLO(str(model_path))
        names = Z3_COLOR_MODELS[piece_type].names if hasattr(Z3_COLOR_MODELS[piece_type], 'names') else {}
        print(f"  ✅ Z3_{piece_type}_color.pt - classes: {names}")
    else:
        print(f"  ❌ Z3_{piece_type}_color.pt not found")

# =============================================================================
# HELPERS
# =============================================================================

def image_to_base64(img_array):
    """Convert numpy array (BGR) to base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

def decode_image(file_or_base64):
    """Decode image from file upload or base64"""
    if hasattr(file_or_base64, 'read'):
        img_bytes = file_or_base64.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img_bytes = base64.b64decode(file_or_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# =============================================================================
# PIPELINE SELECTOR (Z0)
# =============================================================================

def detect_board_type(warped_img):
    """
    Use Z0.pt to determine if board is B/W book style or color.
    Returns: 'book_bw' or 'color'
    """
    if Z0_MODEL is None:
        return 'color'  # Default to color pipeline if Z0 not available
    
    # Save temp image for YOLO
    temp_path = '/tmp/board_temp.jpg'
    cv2.imwrite(temp_path, warped_img)
    
    results = Z0_MODEL(temp_path, verbose=False)
    idx = results[0].probs.top1
    conf = float(results[0].probs.top1conf.cpu().numpy())
    
    # Get class name
    class_names = Z0_MODEL.names if hasattr(Z0_MODEL, 'names') else {0: 'book_bw', 1: 'color'}
    board_type = class_names.get(idx, 'color')
    
    # Normalize the class name
    board_type_lower = board_type.lower()
    if 'book' in board_type_lower or 'bw' in board_type_lower or 'b/w' in board_type_lower or 'black' in board_type_lower:
        result = 'book_bw'
    else:
        result = 'color'
    
    print(f"[Z0] Board type: {board_type} (conf={conf:.2f}) → using {result.upper()} pipeline")
    
    return result, conf

# =============================================================================
# PIPELINE STEPS
# =============================================================================

def detect_board(img):
    """Step 1: Detect board bounding box"""
    if BOARD_MODEL is None:
        return None, None
    
    results = BOARD_MODEL(img, conf=0.3, verbose=False)
    if len(results[0].boxes) == 0:
        return None, None
    
    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    conf = float(results[0].boxes[0].conf[0].cpu().numpy())
    return box, conf

def detect_corners(img):
    """Step 2: Detect 4 corners using YOLO pose"""
    if CORNER_MODEL is None:
        return None, None
    
    results = CORNER_MODEL(img, conf=0.1, verbose=False)
    if len(results[0].keypoints) == 0:
        return None, None
    
    corners = results[0].keypoints.xy[0].cpu().numpy()
    conf = float(results[0].boxes[0].conf[0].cpu().numpy()) if len(results[0].boxes) > 0 else 0.0
    
    corners = sort_corners(corners)
    
    return corners, conf


def sort_corners(corners):
    """
    Sort 4 corners into consistent order: TL, TR, BR, BL
    """
    if len(corners) != 4:
        return corners
    
    corners = np.array(corners, dtype=np.float32)
    
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).flatten()
    
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmin(d)]
    bl = corners[np.argmax(d)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_perspective(img, corners, size=512, margin_pct=0.035):
    """Step 3: Warp to flat square with margin."""
    margin = int(size * margin_pct)
    full_size = size + 2 * margin
    
    dst = np.array([
        [margin, margin],
        [size + margin, margin],
        [size + margin, size + margin],
        [margin, size + margin]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (full_size, full_size))
    
    return warped, margin


def find_cv_corners(gray):
    """Step 4: Find inner corners using OpenCV"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    methods = [enhanced, gray, cv2.GaussianBlur(gray, (5, 5), 0)]
    patterns = [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3)]
    
    for processed in methods:
        for pattern in patterns:
            ret, corners = cv2.findChessboardCorners(
                processed, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return refined, pattern
    
    return None, None


def simple_grid(size, margin=0):
    """Fallback: simple 8x8 divide, accounting for margin"""
    board_size = size - 2 * margin
    step = board_size / 8
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            grid[i, j] = [margin + j * step, margin + i * step]
    return grid


def create_hybrid_grid(cv_corners, pattern, full_size, margin):
    """Create 9x9 grid using CV corners as primary source."""
    if cv_corners is None:
        grid = simple_grid(full_size, margin)
        grid_info = {'cv_indices': [], 'extrapolated_indices': [(i, j) for i in range(9) for j in range(9)]}
        return grid, None, grid_info
    
    rows, cols = pattern
    cv_grid = cv_corners.reshape(rows, cols, 2)
    
    h_steps = []
    for i in range(rows):
        for j in range(cols - 1):
            h_steps.append(cv_grid[i, j+1] - cv_grid[i, j])
    
    v_steps = []
    for i in range(rows - 1):
        for j in range(cols):
            v_steps.append(cv_grid[i+1, j] - cv_grid[i, j])
    
    avg_h = np.median(h_steps, axis=0) if h_steps else np.array([50, 0])
    avg_v = np.median(v_steps, axis=0) if v_steps else np.array([0, 50])
    
    print(f"[GRID] CV {rows}x{cols}, raw h={avg_h}, v={avg_v}")
    
    h_is_horizontal = abs(avg_h[0]) > abs(avg_h[1])
    v_is_vertical = abs(avg_v[1]) > abs(avg_v[0])
    
    print(f"[GRID] h_is_horizontal={h_is_horizontal}, v_is_vertical={v_is_vertical}")
    
    if not h_is_horizontal and not v_is_vertical:
        print(f"[GRID] TRANSPOSE: h was vertical, v was horizontal (90° rotation)")
        cv_grid = cv_grid.transpose(1, 0, 2)
        rows, cols = cols, rows
        avg_h, avg_v = avg_v.copy(), avg_h.copy()
    
    if avg_h[0] < 0:
        print(f"[GRID] Flipping columns (h pointed left)")
        cv_grid = cv_grid[:, ::-1, :]
        avg_h = -avg_h
    
    if avg_v[1] < 0:
        print(f"[GRID] Flipping rows (v pointed up)")
        cv_grid = cv_grid[::-1, :, :]
        avg_v = -avg_v
    
    print(f"[GRID] Corrected h={avg_h}, v={avg_v}")
    
    detected_h = cols - 1
    detected_v = rows - 1
    
    need_h = 8 - detected_h
    need_v = 8 - detected_v
    
    add_left = need_h // 2
    add_up = need_v // 2
    
    print(f"[GRID] Adding: left={add_left}, up={add_up}")
    
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    cv_indices = []
    extrapolated_indices = []
    
    for i in range(rows):
        for j in range(cols):
            grid_i = i + add_up
            grid_j = j + add_left
            if 0 <= grid_i < 9 and 0 <= grid_j < 9:
                grid[grid_i, grid_j] = cv_grid[i, j]
                cv_indices.append((grid_i, grid_j))
    
    anchor_cv_i = rows // 2
    anchor_cv_j = cols // 2
    anchor_grid_i = anchor_cv_i + add_up
    anchor_grid_j = anchor_cv_j + add_left
    anchor_pos = cv_grid[anchor_cv_i, anchor_cv_j].copy()
    
    for i in range(9):
        for j in range(9):
            if (i, j) not in cv_indices:
                di = i - anchor_grid_i
                dj = j - anchor_grid_j
                grid[i, j] = anchor_pos + dj * avg_h + di * avg_v
                extrapolated_indices.append((i, j))
    
    min_x, min_y = np.min(grid[:,:,0]), np.min(grid[:,:,1])
    max_x, max_y = np.max(grid[:,:,0]), np.max(grid[:,:,1])
    
    if min_x < margin:
        grid[:,:,0] += (margin - min_x)
    if min_y < margin:
        grid[:,:,1] += (margin - min_y)
    if max_x > full_size - margin:
        grid[:,:,0] -= (max_x - (full_size - margin))
    if max_y > full_size - margin:
        grid[:,:,1] -= (max_y - (full_size - margin))
    
    grid_info = {
        'cv_indices': cv_indices,
        'extrapolated_indices': extrapolated_indices,
        'pattern': f"{rows}x{cols}"
    }
    
    print(f"[GRID] Final: {len(cv_indices)} CV pts, {len(extrapolated_indices)} extrapolated")
    
    return grid, cv_grid, grid_info

def extract_squares(img, grid):
    """Step 6: Extract 64 squares"""
    squares = {}
    for row in range(8):
        for col in range(8):
            tl = grid[row, col]
            tr = grid[row, col + 1]
            bl = grid[row + 1, col]
            br = grid[row + 1, col + 1]
            
            src = np.array([tl, tr, br, bl], dtype=np.float32)
            dst = np.array([[0, 0], [SQUARE_SIZE, 0], 
                           [SQUARE_SIZE, SQUARE_SIZE], [0, SQUARE_SIZE]], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src, dst)
            square = cv2.warpPerspective(img, M, (SQUARE_SIZE, SQUARE_SIZE))
            
            label = f"{FILES[col]}{RANKS[row]}"
            squares[label] = square
    
    return squares

def stitch_squares(squares):
    """Step 7: Stitch 64 squares"""
    stitched = np.zeros((SQUARE_SIZE * 8, SQUARE_SIZE * 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            sq = squares[label]
            y = row * SQUARE_SIZE
            x = col * SQUARE_SIZE
            stitched[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE] = sq
    return stitched


# =============================================================================
# CLASSIFICATION - COLOR PIPELINE
# =============================================================================

def classify_squares_color(squares):
    """
    COLOR PIPELINE: Classify squares using 3.pt + P/N/B/R/Q/K.pt
    Includes king logic: exactly 2 kings, highest confidence keeps color, other gets opposite.
    """
    if PIECE_MODEL is None:
        return None, None, None
    
    board = [['' for _ in range(8)] for _ in range(8)]
    predictions = {}
    square_colors = {}
    kings_found = []  # Track kings: [(label, row, col, is_white, color_conf)]
    
    for label, sq_img in squares.items():
        temp_path = '/tmp/sq_temp.jpg'
        cv2.imwrite(temp_path, sq_img)
        
        # Step 1: Get piece TYPE from 3.pt
        results = PIECE_MODEL(temp_path, verbose=False)
        idx = results[0].probs.top1
        conf = float(results[0].probs.top1conf.cpu().numpy())
        ml_piece = CLASS_NAMES[idx]
        
        col = FILES.index(label[0])
        row = RANKS.index(label[1])
        
        is_dark_square = (row + col) % 2 == 1
        
        if ml_piece != 'empty':
            piece_type = ml_piece.upper()
            
            # Step 2: Use color model
            if piece_type in COLOR_MODELS:
                color_model = COLOR_MODELS[piece_type]
                color_results = color_model(temp_path, verbose=False)
                color_idx = color_results[0].probs.top1
                color_conf = float(color_results[0].probs.top1conf.cpu().numpy())
                
                color_names = color_model.names if hasattr(color_model, 'names') else {0: 'B', 1: 'W'}
                color_pred = color_names.get(color_idx, str(color_idx))
                
                is_white = color_pred.upper().startswith('W') or color_pred == '1'
                
                print(f"[COLOR] {label}: {piece_type} (conf={conf:.2f}) + {piece_type}.pt → {color_pred} (conf={color_conf:.2f}) → {'W' if is_white else 'B'}")
            else:
                is_white = ml_piece.isupper()
                color_conf = 0.0
                print(f"[COLOR] {label}: {piece_type} (conf={conf:.2f}) - no {piece_type}.pt, using 3.pt guess → {'W' if is_white else 'B'}")
            
            # Track kings for later correction
            if piece_type == 'K':
                kings_found.append((label, row, col, is_white, color_conf))
            
            if is_white:
                final_piece = piece_type.upper()
            else:
                final_piece = piece_type.lower()
        else:
            final_piece = 'empty'
            is_white = None
            color_conf = 0.0
        
        board[row][col] = final_piece if final_piece != 'empty' else '.'
        
        predictions[label] = {
            'piece': final_piece, 
            'confidence': float(conf),
            'color_confidence': float(color_conf) if ml_piece != 'empty' else None,
            'square_type': 'dark' if is_dark_square else 'light',
            'ml_type': ml_piece.upper() if ml_piece != 'empty' else 'empty',
            'is_white_piece': bool(is_white) if is_white is not None else None,
            'pipeline': 'color'
        }
        square_colors[label] = 'dark' if is_dark_square else 'light'
    
    # === KING LOGIC: Ensure one white king and one black king ===
    if len(kings_found) == 2:
        # Sort by color confidence (highest first)
        kings_found.sort(key=lambda x: x[4], reverse=True)
        
        king1_label, king1_row, king1_col, king1_is_white, king1_conf = kings_found[0]
        king2_label, king2_row, king2_col, king2_is_white, king2_conf = kings_found[1]
        
        # If both kings are same color, flip the lower confidence one
        if king1_is_white == king2_is_white:
            print(f"[KING LOGIC] Both kings detected as {'White' if king1_is_white else 'Black'}!")
            print(f"[KING LOGIC] {king1_label} (conf={king1_conf:.2f}) keeps color, {king2_label} (conf={king2_conf:.2f}) flipped")
            
            # Flip king2's color
            king2_is_white = not king1_is_white
            
            # Update board
            if king2_is_white:
                board[king2_row][king2_col] = 'K'
            else:
                board[king2_row][king2_col] = 'k'
            
            # Update predictions
            predictions[king2_label]['piece'] = 'K' if king2_is_white else 'k'
            predictions[king2_label]['is_white_piece'] = king2_is_white
            predictions[king2_label]['king_color_corrected'] = True
            
            print(f"[KING LOGIC] Result: {king1_label}={'K' if king1_is_white else 'k'}, {king2_label}={'K' if king2_is_white else 'k'}")
    
    return board, predictions, square_colors


# =============================================================================
# CLASSIFICATION - B/W BOOK PIPELINE
# =============================================================================

def classify_squares_book_bw(squares):
    """
    B/W BOOK PIPELINE: Classify squares using Z3.pt + Z3_X_color.pt
    Includes king logic: exactly 2 kings, highest confidence keeps color, other gets opposite.
    """
    if Z3_PIECE_MODEL is None:
        print("[BOOK_BW] Z3.pt not available, falling back to COLOR pipeline")
        return classify_squares_color(squares)
    
    board = [['' for _ in range(8)] for _ in range(8)]
    predictions = {}
    square_colors = {}
    kings_found = []  # Track kings: [(label, row, col, is_white, color_conf)]
    
    for label, sq_img in squares.items():
        temp_path = '/tmp/sq_temp.jpg'
        cv2.imwrite(temp_path, sq_img)
        
        # Step 1: Get piece TYPE from Z3.pt
        results = Z3_PIECE_MODEL(temp_path, verbose=False)
        idx = results[0].probs.top1
        conf = float(results[0].probs.top1conf.cpu().numpy())
        
        # Get class name from Z3 model
        z3_names = Z3_PIECE_MODEL.names if hasattr(Z3_PIECE_MODEL, 'names') else {}
        ml_piece = z3_names.get(idx, 'empty')
        
        col = FILES.index(label[0])
        row = RANKS.index(label[1])
        
        is_dark_square = (row + col) % 2 == 1
        
        if ml_piece.lower() != 'empty':
            piece_type = ml_piece.upper()  # K, Q, R, B, N, P
            
            # Step 2: Use Z3_X_color model
            if piece_type in Z3_COLOR_MODELS:
                color_model = Z3_COLOR_MODELS[piece_type]
                color_results = color_model(temp_path, verbose=False)
                color_idx = color_results[0].probs.top1
                color_conf = float(color_results[0].probs.top1conf.cpu().numpy())
                
                color_names = color_model.names if hasattr(color_model, 'names') else {0: 'black', 1: 'white'}
                color_pred = color_names.get(color_idx, str(color_idx))
                
                # Check for 'white' in class name
                is_white = 'white' in color_pred.lower() or color_pred == '1'
                
                print(f"[BOOK_BW] {label}: {piece_type} (conf={conf:.2f}) + Z3_{piece_type}_color.pt → {color_pred} (conf={color_conf:.2f}) → {'W' if is_white else 'B'}")
            else:
                # No color model - default to white (can't determine)
                is_white = True
                color_conf = 0.0
                print(f"[BOOK_BW] {label}: {piece_type} (conf={conf:.2f}) - no Z3_{piece_type}_color.pt, defaulting to W")
            
            # Track kings for later correction
            if piece_type == 'K':
                kings_found.append((label, row, col, is_white, color_conf))
            
            if is_white:
                final_piece = piece_type.upper()
            else:
                final_piece = piece_type.lower()
        else:
            final_piece = 'empty'
            is_white = None
            color_conf = 0.0
        
        board[row][col] = final_piece if final_piece != 'empty' else '.'
        
        predictions[label] = {
            'piece': final_piece, 
            'confidence': float(conf),
            'color_confidence': float(color_conf) if ml_piece.lower() != 'empty' else None,
            'square_type': 'dark' if is_dark_square else 'light',
            'ml_type': ml_piece.upper() if ml_piece.lower() != 'empty' else 'empty',
            'is_white_piece': bool(is_white) if is_white is not None else None,
            'pipeline': 'book_bw'
        }
        square_colors[label] = 'dark' if is_dark_square else 'light'
    
    # === KING LOGIC: Ensure one white king and one black king ===
    if len(kings_found) == 2:
        # Sort by color confidence (highest first)
        kings_found.sort(key=lambda x: x[4], reverse=True)
        
        king1_label, king1_row, king1_col, king1_is_white, king1_conf = kings_found[0]
        king2_label, king2_row, king2_col, king2_is_white, king2_conf = kings_found[1]
        
        # If both kings are same color, flip the lower confidence one
        if king1_is_white == king2_is_white:
            print(f"[KING LOGIC] Both kings detected as {'White' if king1_is_white else 'Black'}!")
            print(f"[KING LOGIC] {king1_label} (conf={king1_conf:.2f}) keeps color, {king2_label} (conf={king2_conf:.2f}) flipped")
            
            # Flip king2's color
            king2_is_white = not king1_is_white
            
            # Update board
            if king2_is_white:
                board[king2_row][king2_col] = 'K'
            else:
                board[king2_row][king2_col] = 'k'
            
            # Update predictions
            predictions[king2_label]['piece'] = 'K' if king2_is_white else 'k'
            predictions[king2_label]['is_white_piece'] = king2_is_white
            predictions[king2_label]['king_color_corrected'] = True
            
            print(f"[KING LOGIC] Result: {king1_label}={'K' if king1_is_white else 'k'}, {king2_label}={'K' if king2_is_white else 'k'}")
    
    return board, predictions, square_colors


# =============================================================================
# UNIFIED CLASSIFY FUNCTION
# =============================================================================

def classify_squares(squares, board_type='color'):
    """
    Route to appropriate classification pipeline based on board type.
    """
    if board_type == 'book_bw':
        return classify_squares_book_bw(squares)
    else:
        return classify_squares_color(squares)


def board_to_fen(board):
    """Convert board to FEN"""
    fen_rows = []
    for row in board:
        fen_row = ''
        empty = 0
        for cell in row:
            if cell == '.':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_box(img, box, color=GREEN, thickness=4):
    result = img.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    return result

def draw_corners(img, corners, color=GREEN, radius=15):
    result = img.copy()
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, (x, y) in enumerate(corners):
        cv2.circle(result, (int(x), int(y)), radius, color, -1)
        cv2.putText(result, labels[i], (int(x)+20, int(y)+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return result


def draw_grid_detailed(img, grid, cv_grid, grid_info):
    """Draw grid with color-coded points."""
    result = img.copy()
    
    CYAN = (255, 255, 0)
    
    for i in range(9):
        for j in range(8):
            pt1 = tuple(grid[i, j].astype(int))
            pt2 = tuple(grid[i, j + 1].astype(int))
            cv2.line(result, pt1, pt2, BLUE, 2)
    for i in range(8):
        for j in range(9):
            pt1 = tuple(grid[i, j].astype(int))
            pt2 = tuple(grid[i + 1, j].astype(int))
            cv2.line(result, pt1, pt2, BLUE, 2)
    
    cv_indices = set(tuple(x) for x in grid_info.get('cv_indices', []))
    extrapolated_indices = set(tuple(x) for x in grid_info.get('extrapolated_indices', []))
    
    for (i, j) in extrapolated_indices:
        pt = tuple(grid[i, j].astype(int))
        cv2.circle(result, pt, 6, CYAN, -1)
        cv2.circle(result, pt, 6, BLUE, 1)
    
    for (i, j) in cv_indices:
        pt = tuple(grid[i, j].astype(int))
        cv2.circle(result, pt, 8, MAGENTA, -1)
        cv2.circle(result, pt, 8, (255, 255, 255), 2)
    
    legend_y = 20
    cv2.putText(result, f"CV found: {len(cv_indices)} pts", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 2)
    cv2.putText(result, f"Extrapolated: {len(extrapolated_indices)} pts", (10, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 2)
    
    if 'pattern' in grid_info:
        cv2.putText(result, f"Pattern: {grid_info['pattern']}", (10, legend_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result

def draw_classifications(stitched, predictions):
    """Draw the piece classification results on stitched image."""
    result = stitched.copy()
    
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            pred = predictions[label]
            piece = pred['piece']
            is_white = pred.get('is_white_piece')
            pipeline = pred.get('pipeline', 'color')
            
            cx = col * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = row * SQUARE_SIZE + SQUARE_SIZE // 2
            
            if piece != 'empty':
                if is_white:
                    color = (0, 255, 0)   # Green for White
                    text = "W"
                else:
                    color = (0, 0, 255)   # Red for Black
                    text = "B"
                
                cv2.circle(result, (cx, cy), 8, color, -1)
                cv2.putText(result, text, (col * SQUARE_SIZE + 2, row * SQUARE_SIZE + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                piece_label = f"{piece.upper()}"
                cv2.putText(result, piece_label, 
                           (col * SQUARE_SIZE + SQUARE_SIZE - 14, row * SQUARE_SIZE + SQUARE_SIZE - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                cv2.circle(result, (cx, cy), 3, (100, 100, 100), -1)
    
    return result

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': {
            'board': BOARD_MODEL is not None,
            'corner': CORNER_MODEL is not None,
            'z0_selector': Z0_MODEL is not None,
            'color_pipeline': {
                'piece': PIECE_MODEL is not None,
                'color_models': {pt: (pt in COLOR_MODELS) for pt in ['P', 'N', 'B', 'R', 'Q', 'K']}
            },
            'book_bw_pipeline': {
                'piece': Z3_PIECE_MODEL is not None,
                'color_models': {pt: (pt in Z3_COLOR_MODELS) for pt in ['K', 'Q', 'R', 'B', 'N', 'P']}
            }
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Full analysis pipeline with Z0 fork"""
    try:
        # === INCOMING REQUEST LOGGING ===
        print("\n" + "=" * 60)
        print("[REQUEST] New analysis request received")
        print("=" * 60)
        print(f"[REQUEST] Content-Type: {request.content_type}")
        print(f"[REQUEST] Files: {list(request.files.keys())}")
        print(f"[REQUEST] Is JSON: {request.is_json}")
        
        # Detect source (iOS app vs web)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        if 'iPhone' in user_agent or 'iPad' in user_agent or 'iOS' in user_agent:
            source = 'iOS App'
        elif 'Android' in user_agent:
            source = 'Android App'
        elif 'Mozilla' in user_agent or 'Chrome' in user_agent or 'Safari' in user_agent:
            source = 'Web Browser'
        else:
            source = 'Unknown'
        print(f"[REQUEST] Source: {source}")
        print(f"[REQUEST] User-Agent: {user_agent[:80]}...")
        
        # Get image
        if 'image' in request.files:
            print("[REQUEST] ✅ Image found in multipart form")
            img = decode_image(request.files['image'])
        elif request.is_json and 'image' in request.get_json():
            print("[REQUEST] ✅ Image found in JSON body")
            img = decode_image(request.get_json()['image'])
        else:
            print("[REQUEST] ❌ No image found!")
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        if img is None:
            print("[REQUEST] ❌ Failed to decode image")
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        result = {
            'success': True,
            'steps': {}
        }
        
        h, w = img.shape[:2]
        result['original_size'] = f"{w}x{h}"
        print(f"[IMAGE] Size: {w}x{h}")
        
        # Step 1: Original
        result['steps']['1_original'] = image_to_base64(img)
        
        # Step 2: Board detection
        box, box_conf = detect_board(img)
        if box is not None:
            x1, y1, x2, y2 = box
            print(f"[BOARD] ✅ Detected (conf={box_conf:.2f}) box=[{x1},{y1},{x2},{y2}]")
            margin = int((x2 - x1) * 0.05)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            board_vis = draw_box(img, [x1, y1, x2, y2])
            result['steps']['2_board_detected'] = image_to_base64(board_vis)
            result['board_confidence'] = box_conf
            
            cropped = img[y1:y2, x1:x2]
        else:
            print(f"[BOARD] ⚠️ Not detected, using full image")
            result['steps']['2_board_detected'] = image_to_base64(img)
            result['board_confidence'] = 0
            cropped = img
        
        # Step 3: Cropped
        result['steps']['3_cropped'] = image_to_base64(cropped)
        
        # Step 4: Corner detection
        corners, corner_conf = detect_corners(cropped)
        if corners is None:
            ch, cw = cropped.shape[:2]
            corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
            result['corner_confidence'] = 0
            print(f"[CORNERS] ⚠️ Not detected, using image corners")
        else:
            result['corner_confidence'] = corner_conf
            print(f"[CORNERS] ✅ Detected (conf={corner_conf:.2f})")
        
        corners_vis = draw_corners(cropped, corners)
        result['steps']['4_corners'] = image_to_base64(corners_vis)
        
        # Step 5: Warp
        warped, margin = warp_perspective(cropped, corners, WARP_SIZE)
        full_size = WARP_SIZE + 2 * margin
        result['steps']['5_warped'] = image_to_base64(warped)
        result['warp_margin'] = margin
        print(f"[WARP] Done, size={full_size}x{full_size}, margin={margin}")
        
        # Step 6: CV corners + hybrid grid
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        cv_corners, pattern = find_cv_corners(gray)
        
        if cv_corners is not None:
            grid, cv_grid, grid_info = create_hybrid_grid(cv_corners, pattern, full_size, margin)
            result['cv_pattern'] = f"{pattern[0]}x{pattern[1]}"
            result['grid_method'] = 'CV + extrapolation'
            print(f"[GRID] ✅ CV pattern {pattern[0]}x{pattern[1]} found")
        else:
            grid = simple_grid(full_size, margin)
            cv_grid = None
            grid_info = {'cv_indices': [], 'extrapolated_indices': [(i, j) for i in range(9) for j in range(9)]}
            result['cv_pattern'] = 'none'
            result['grid_method'] = 'simple 8x8 (CV failed)'
            print(f"[GRID] ⚠️ CV failed, using simple 8x8")
        
        grid_vis = draw_grid_detailed(warped, grid, cv_grid, grid_info)
        result['steps']['6_grid'] = image_to_base64(grid_vis)
        
        # Step 7: Extract squares
        squares = extract_squares(warped, grid)
        
        result['detected_orientation'] = 'as_detected'
        
        # Save individual squares
        squares_b64 = {}
        for label, sq in squares.items():
            squares_b64[label] = image_to_base64(sq)
        result['squares'] = squares_b64
        
        # Step 8: Stitch
        stitched = stitch_squares(squares)
        result['steps']['7_stitched'] = image_to_base64(stitched)
        
        # =====================================================================
        # STEP 9: Z0 FORK - DECIDE WHICH PIPELINE TO USE
        # =====================================================================
        board_type, z0_conf = detect_board_type(warped)
        result['board_type'] = board_type
        result['board_type_confidence'] = z0_conf
        
        # Step 10: Classify using appropriate pipeline
        print(f"\n[PIPELINE] Using {board_type.upper()} pipeline")
        print("-" * 40)
        board, predictions, square_colors = classify_squares(squares, board_type)
        
        if board is not None:
            result['board'] = board
            result['predictions'] = predictions
            result['square_colors'] = square_colors
            result['fen'] = board_to_fen(board)
            result['pipeline_used'] = board_type
            
            classified_vis = draw_classifications(stitched, predictions)
            result['steps']['8_classified'] = image_to_base64(classified_vis)
            
            # === SUMMARY LOGGING ===
            piece_count = sum(1 for row in board for cell in row if cell != '.')
            white_count = sum(1 for row in board for cell in row if cell.isupper() and cell != '.')
            black_count = sum(1 for row in board for cell in row if cell.islower())
            
            print("-" * 40)
            print(f"[RESULT] ✅ Analysis complete")
            print(f"[RESULT] FEN: {result['fen']}")
            print(f"[RESULT] Pieces: {piece_count} total ({white_count} white, {black_count} black)")
            print(f"[RESULT] Pipeline: {board_type}")
            print(f"[RESULT] Z0 confidence: {z0_conf:.2f}")
            print("=" * 60 + "\n")
        else:
            result['board'] = [['.'] * 8 for _ in range(8)]
            result['fen'] = '8/8/8/8/8/8/8/8'
            result['pipeline_used'] = 'none'
            print(f"[RESULT] ❌ Classification failed")
            print("=" * 60 + "\n")
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(f"[ERROR] ❌ Exception: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5555))
    print(f"\n🚀 ChessScan Server running on http://localhost:{port}")
    print(f"\nEndpoints:")
    print(f"  GET  /health  - Health check")
    print(f"  POST /analyze - Full analysis")
    print(f"\nPipelines:")
    print(f"  COLOR (default): 3.pt + P/N/B/R/Q/K.pt")
    print(f"  BOOK_BW:         Z3.pt + Z3_X_color.pt")
    print(f"  Selector:        Z0.pt")
    print(f"\nModels loaded from: {SCRIPT_DIR}")
    app.run(host='0.0.0.0', port=port, debug=True)

#!/usr/bin/env python3
"""
ChessScan Analysis Server
=========================
Full pipeline server using YOLO models.

Place models in same folder as this script:
  1.pt - Board detection
  2.pt - Corner detection  
  3.pt - Piece TYPE classification (P/N/B/R/Q/K/empty)
  P.pt - Pawn COLOR (W vs B)
  N.pt - Knight COLOR (W vs B)
  B.pt - Bishop COLOR (W vs B)
  R.pt - Rook COLOR (W vs B)
  Q.pt - Queen COLOR (W vs B)
  K.pt - King COLOR (W vs B)

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
  6. Piece TYPE classification (3.pt)
  7. Piece COLOR classification (P/N/B/R/Q/K.pt based on type)
  8. Combine → FEN (uppercase=White, lowercase=Black)

Usage:
  python3 server.py
  # Server runs on http://localhost:5555
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BOARD_MODEL_PATH = SCRIPT_DIR / '1.pt'
CORNER_MODEL_PATH = SCRIPT_DIR / '2.pt'
PIECE_MODEL_PATH = SCRIPT_DIR / '3.pt'  # TYPE classification (P/N/B/R/Q/K/empty)

# Per-piece COLOR classifiers (W vs B)
COLOR_MODEL_PATHS = {
    'P': SCRIPT_DIR / 'P.pt',  # Pawn color
    'N': SCRIPT_DIR / 'N.pt',  # Knight color
    'B': SCRIPT_DIR / 'B.pt',  # Bishop color
    'R': SCRIPT_DIR / 'R.pt',  # Rook color
    'Q': SCRIPT_DIR / 'Q.pt',  # Queen color
    'K': SCRIPT_DIR / 'K.pt',  # King color
}

CLASS_NAMES = ['empty', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b']
# 3.pt output - we only care about the TYPE (uppercase), color comes from P/N/B/R/Q/K.pt
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']

SQUARE_SIZE = 64
WARP_SIZE = 512  # Base size for warped board (actual image will be larger with 6% margin)

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
    
    # Sample the 4 corners (10% regions)
    corner_size = max(int(0.1 * min(h, w)), 3)
    
    corners = [
        square_img[0:corner_size, 0:corner_size],                    # Top-left
        square_img[0:corner_size, w-corner_size:w],                  # Top-right
        square_img[h-corner_size:h, 0:corner_size],                  # Bottom-left
        square_img[h-corner_size:h, w-corner_size:w],                # Bottom-right
    ]
    
    # Get brightness of each corner
    brightnesses = []
    for corner in corners:
        if len(corner.shape) == 3:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        else:
            gray = corner
        brightnesses.append(np.mean(gray) / 255.0)
    
    # Return median to be robust against pieces overlapping corners
    return np.median(brightnesses)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
BOARD_MODEL = None
CORNER_MODEL = None
PIECE_MODEL = None
COLOR_MODELS = {}  # Dict: 'P' -> model, 'N' -> model, etc.

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

if PIECE_MODEL_PATH.exists():
    PIECE_MODEL = YOLO(str(PIECE_MODEL_PATH))
    print(f"  ✅ 3.pt Piece TYPE classification")
else:
    print(f"  ❌ 3.pt not found")

# Load per-piece color models
print("Loading color models...")
for piece_type, model_path in COLOR_MODEL_PATHS.items():
    if model_path.exists():
        COLOR_MODELS[piece_type] = YOLO(str(model_path))
        # Show class names so we know what 0/1 mean
        names = COLOR_MODELS[piece_type].names if hasattr(COLOR_MODELS[piece_type], 'names') else {}
        print(f"  ✅ {piece_type}.pt ({piece_type} color) - classes: {names}")
    else:
        print(f"  ❌ {piece_type}.pt not found")

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
        # File upload
        img_bytes = file_or_base64.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Base64
        img_bytes = base64.b64decode(file_or_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
    
    # Sort corners to ensure consistent order: TL, TR, BR, BL
    corners = sort_corners(corners)
    
    return corners, conf


def sort_corners(corners):
    """
    Sort 4 corners into consistent order: TL, TR, BR, BL
    Uses geometric sorting:
    - TL has smallest sum (x+y)
    - BR has largest sum (x+y)
    - TR has smallest difference (y-x)
    - BL has largest difference (y-x)
    """
    if len(corners) != 4:
        return corners
    
    corners = np.array(corners, dtype=np.float32)
    
    # Sum and difference
    s = corners.sum(axis=1)  # x + y
    d = np.diff(corners, axis=1).flatten()  # y - x
    
    # TL = smallest sum, BR = largest sum
    # TR = smallest diff, BL = largest diff
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmin(d)]
    bl = corners[np.argmax(d)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_perspective(img, corners, size=512, margin_pct=0.035):
    """
    Step 3: Warp to flat square with margin.
    Adding margin helps CV find corners near edges.
    """
    # Calculate the size with margin
    margin = int(size * margin_pct)
    full_size = size + 2 * margin
    
    # Destination points with margin offset
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

def extrapolate_to_9x9(corners, pattern, img_size):
    """Step 5: Extrapolate CV corners to full 9x9 grid"""
    rows, cols = pattern
    corners_grid = corners.reshape(rows, cols, 2)
    
    # Calculate step sizes
    h_steps = []
    for i in range(rows):
        for j in range(cols - 1):
            h_steps.append(corners_grid[i, j+1] - corners_grid[i, j])
    
    v_steps = []
    for i in range(rows - 1):
        for j in range(cols):
            v_steps.append(corners_grid[i+1, j] - corners_grid[i, j])
    
    avg_h = np.median(h_steps, axis=0)
    avg_v = np.median(v_steps, axis=0)
    
    # Calculate detected squares
    detected_h = cols - 1
    detected_v = rows - 1
    
    # Need 8 squares total
    need_h = 8 - detected_h
    need_v = 8 - detected_v
    
    # Distribute evenly
    add_left = need_h // 2
    add_right = need_h - add_left
    add_up = need_v // 2
    add_down = need_v - add_up
    
    # Calculate top-left of full grid
    tl_detected = corners_grid[0, 0]
    tl_full = tl_detected - add_left * avg_h - add_up * avg_v
    
    # Check bounds and shift if needed
    margin = 5
    br_full = tl_full + 8 * avg_h + 8 * avg_v
    
    if tl_full[0] < margin:
        tl_full[0] = margin
    if tl_full[1] < margin:
        tl_full[1] = margin
    
    br_full = tl_full + 8 * avg_h + 8 * avg_v
    if br_full[0] > img_size - margin:
        tl_full[0] -= (br_full[0] - img_size + margin)
    if br_full[1] > img_size - margin:
        tl_full[1] -= (br_full[1] - img_size + margin)
    
    # Build 9x9 grid
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            grid[i, j] = tl_full + j * avg_h + i * avg_v
    
    return grid, corners_grid, f"{rows}x{cols}"

def simple_grid(size, margin=0):
    """Fallback: simple 8x8 divide, accounting for margin"""
    # The actual board is from margin to size-margin
    board_size = size - 2 * margin
    step = board_size / 8
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            grid[i, j] = [margin + j * step, margin + i * step]
    return grid


def create_hybrid_grid(cv_corners, pattern, full_size, margin):
    """
    Create 9x9 grid using CV corners as primary source.
    CV findChessboardCorners returns corners row-by-row, but the "rows" depend on 
    board orientation. We need to ensure our grid has:
    - h (horizontal) pointing RIGHT (+x)
    - v (vertical) pointing DOWN (+y)
    """
    if cv_corners is None:
        grid = simple_grid(full_size, margin)
        grid_info = {'cv_indices': [], 'extrapolated_indices': [(i, j) for i in range(9) for j in range(9)]}
        return grid, None, grid_info
    
    rows, cols = pattern
    cv_grid = cv_corners.reshape(rows, cols, 2)
    
    # Calculate step vectors from CV corners
    # h_step: moving along columns (j+1 - j)
    # v_step: moving along rows (i+1 - i)
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
    
    # Check if h/v need to be swapped or flipped
    h_is_horizontal = abs(avg_h[0]) > abs(avg_h[1])  # h has larger x component
    v_is_vertical = abs(avg_v[1]) > abs(avg_v[0])    # v has larger y component
    
    print(f"[GRID] h_is_horizontal={h_is_horizontal}, v_is_vertical={v_is_vertical}")
    
    # If h is vertical and v is horizontal, we need to transpose (90° rotation)
    if not h_is_horizontal and not v_is_vertical:
        print(f"[GRID] TRANSPOSE: h was vertical, v was horizontal (90° rotation)")
        cv_grid = cv_grid.transpose(1, 0, 2)
        rows, cols = cols, rows
        avg_h, avg_v = avg_v.copy(), avg_h.copy()
    
    # Now ensure h points RIGHT (+x) and v points DOWN (+y)
    if avg_h[0] < 0:
        print(f"[GRID] Flipping columns (h pointed left)")
        cv_grid = cv_grid[:, ::-1, :]
        avg_h = -avg_h
    
    if avg_v[1] < 0:
        print(f"[GRID] Flipping rows (v pointed up)")
        cv_grid = cv_grid[::-1, :, :]
        avg_v = -avg_v
    
    print(f"[GRID] Corrected h={avg_h}, v={avg_v}")
    
    # Now cv_grid[0,0] is top-left, and steps point right/down
    # Calculate how many squares to add on each side
    detected_h = cols - 1
    detected_v = rows - 1
    
    need_h = 8 - detected_h
    need_v = 8 - detected_v
    
    add_left = need_h // 2
    add_up = need_v // 2
    
    print(f"[GRID] Adding: left={add_left}, up={add_up}")
    
    # Build 9x9 grid
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    cv_indices = []
    extrapolated_indices = []
    
    # Place CV corners in grid
    for i in range(rows):
        for j in range(cols):
            grid_i = i + add_up
            grid_j = j + add_left
            if 0 <= grid_i < 9 and 0 <= grid_j < 9:
                grid[grid_i, grid_j] = cv_grid[i, j]
                cv_indices.append((grid_i, grid_j))
    
    # Extrapolate missing points from center anchor
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
    
    # Clamp to bounds
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


def check_board_orientation(squares):
    """
    Check board orientation based on BOARD SQUARE colors (not piece colors).
    
    CV findChessboardCorners ALWAYS returns corners left-to-right, top-to-bottom.
    So the only possible issue is 180° flip (board upside down).
    
    Standard chess: a1=dark square, h1=light square, a8=light, h8=dark
    If flipped 180°: a1=light, h1=dark (what we read as a1 is actually h8)
    
    Returns: 'correct' or 'rotated_180'
    """
    # Check corner squares - use BOARD color, not piece color
    a1 = squares.get('a1')
    h1 = squares.get('h1')
    b1 = squares.get('b1')
    a2 = squares.get('a2')
    
    if a1 is None or h1 is None:
        return 'correct'  # Can't determine, assume correct
    
    # Check BOARD SQUARE brightness (corners, not center)
    a1_bright = get_board_square_color(a1)
    h1_bright = get_board_square_color(h1)
    b1_bright = get_board_square_color(b1) if b1 is not None else 0.5
    a2_bright = get_board_square_color(a2) if a2 is not None else 0.5
    
    print(f"[ORIENTATION] Board square colors: a1={a1_bright:.2f}, h1={h1_bright:.2f}, b1={b1_bright:.2f}, a2={a2_bright:.2f}")
    
    # In correct orientation (a1 is DARK square):
    # a1=dark (<0.5), h1=light (>=0.5), b1=light, a2=light
    correct_score = 0
    if a1_bright < 0.5: correct_score += 1  # a1 dark square
    if h1_bright >= 0.5: correct_score += 1  # h1 light square
    if b1_bright >= 0.5: correct_score += 1  # b1 light square
    if a2_bright >= 0.5: correct_score += 1  # a2 light square
    
    # In 180° rotated (what we call a1 is actually h8, which is dark):
    # Wait - h8 is also dark! So a1 bright vs dark doesn't change with 180° flip
    # But b1 (adjacent) should swap: b1 is light in correct, g8 is dark
    # Actually: a1↔h8 (both dark), b1↔g8 (light↔dark), h1↔a8 (light↔light)
    
    # Better check: the DIAGONAL pattern
    # Correct: a1=dark, b2=dark, c1=dark (same color squares)
    # Let's check a1+b2 vs b1+a2
    b2 = squares.get('b2')
    b2_bright = get_board_square_color(b2) if b2 is not None else 0.5
    
    # a1 and b2 should be same color (both dark in correct orientation)
    # b1 and a2 should be same color (both light in correct orientation)
    same_color_score = 0
    if abs(a1_bright - b2_bright) < 0.3: same_color_score += 1  # a1 ≈ b2
    if abs(b1_bright - a2_bright) < 0.3: same_color_score += 1  # b1 ≈ a2
    if a1_bright < b1_bright: same_color_score += 1  # a1 darker than b1
    if a1_bright < a2_bright: same_color_score += 1  # a1 darker than a2
    
    print(f"[ORIENTATION] correct_score={correct_score}, same_color_score={same_color_score}")
    
    # If a1 is lighter than expected, board might be 180° rotated
    if correct_score <= 1 and a1_bright > h1_bright:
        print("[ORIENTATION] Detected 180° rotation (a1 is light, should be dark)")
        return 'rotated_180'
    else:
        return 'correct'


def rotate_board_90_cw(squares):
    """Rotate board 90° clockwise"""
    rotated = {}
    for label, img in squares.items():
        col = FILES.index(label[0])
        row = RANKS.index(label[1])
        
        # 90° CW: (row, col) -> (col, 7-row)
        new_col = 7 - row
        new_row = col
        new_label = f"{FILES[new_col]}{RANKS[new_row]}"
        
        rotated[new_label] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    return rotated


def rotate_board_90_ccw(squares):
    """Rotate board 90° counter-clockwise"""
    rotated = {}
    for label, img in squares.items():
        col = FILES.index(label[0])
        row = RANKS.index(label[1])
        
        # 90° CCW: (row, col) -> (7-col, row)
        new_col = row
        new_row = 7 - col
        new_label = f"{FILES[new_col]}{RANKS[new_row]}"
        
        rotated[new_label] = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return rotated


def rotate_board_180(squares):
    """Rotate board 180 degrees by swapping square labels"""
    rotated = {}
    for label, img in squares.items():
        col = FILES.index(label[0])
        row = RANKS.index(label[1])
        
        # Flip both row and column
        new_col = 7 - col
        new_row = 7 - row
        new_label = f"{FILES[new_col]}{RANKS[new_row]}"
        
        # Also rotate the image itself 180°
        rotated[new_label] = cv2.rotate(img, cv2.ROTATE_180)
    
    return rotated

def classify_squares(squares, warped_img=None, grid=None, color_grid=None):
    """
    Step 8: Classify all squares
    1. 3.pt → piece TYPE (P/N/B/R/Q/K or empty)
    2. Per-piece color model (P.pt, N.pt, etc.) → W or B
    3. Combine: uppercase=White, lowercase=Black
    
    Uses CV grid squares (already extracted with proper alignment).
    """
    if PIECE_MODEL is None:
        return None, None, None
    
    board = [['' for _ in range(8)] for _ in range(8)]
    predictions = {}
    square_colors = {}
    
    for label, sq_img in squares.items():
        # Save temp for YOLO
        temp_path = '/tmp/sq_temp.jpg'
        cv2.imwrite(temp_path, sq_img)
        
        # Step 1: Get piece TYPE from 3.pt
        results = PIECE_MODEL(temp_path, verbose=False)
        idx = results[0].probs.top1
        conf = float(results[0].probs.top1conf.cpu().numpy())
        ml_piece = CLASS_NAMES[idx]
        
        col = FILES.index(label[0])
        row = RANKS.index(label[1])
        
        # Board square color (for reference only)
        is_dark_square = (row + col) % 2 == 1
        
        if ml_piece != 'empty':
            # Get piece TYPE (uppercase)
            piece_type = ml_piece.upper()  # P, N, B, R, Q, or K
            
            # Step 2: Use the matching color model for this piece type
            if piece_type in COLOR_MODELS:
                color_model = COLOR_MODELS[piece_type]
                color_results = color_model(temp_path, verbose=False)
                color_idx = color_results[0].probs.top1
                color_conf = float(color_results[0].probs.top1conf.cpu().numpy())
                
                # Get class name from model (should be 'W'/'B' or 'White'/'Black' or '0'/'1')
                color_names = color_model.names if hasattr(color_model, 'names') else {0: 'B', 1: 'W'}
                color_pred = color_names.get(color_idx, str(color_idx))
                
                # Determine if white: check if prediction contains 'W' or 'white' or is '1'
                is_white = color_pred.upper().startswith('W') or color_pred == '1'
                
                print(f"[PIECE] {label}: {piece_type} (conf={conf:.2f}) + {piece_type}.pt → {color_pred} (conf={color_conf:.2f}) → {'W' if is_white else 'B'}")
            else:
                # Fallback: no color model, use 3.pt's guess
                is_white = ml_piece.isupper()
                color_conf = 0.0
                print(f"[PIECE] {label}: {piece_type} (conf={conf:.2f}) - no {piece_type}.pt, using 3.pt guess → {'W' if is_white else 'B'}")
            
            # Step 3: Combine type + color
            if is_white:
                final_piece = piece_type.upper()  # White = uppercase
            else:
                final_piece = piece_type.lower()  # Black = lowercase
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
            'is_white_piece': bool(is_white) if is_white is not None else None
        }
        square_colors[label] = 'dark' if is_dark_square else 'light'
    
    return board, predictions, square_colors

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

def draw_grid(img, grid, cv_corners=None):
    result = img.copy()
    
    # Blue lines
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
    
    # Green points (extrapolated)
    for i in range(9):
        for j in range(9):
            pt = tuple(grid[i, j].astype(int))
            cv2.circle(result, pt, 5, GREEN, -1)
    
    # Magenta rings (CV detected)
    if cv_corners is not None:
        rows, cols = cv_corners.shape[:2]
        for i in range(rows):
            for j in range(cols):
                pt = tuple(cv_corners[i, j].astype(int))
                cv2.circle(result, pt, 10, MAGENTA, 2)
    
    return result


def draw_grid_detailed(img, grid, cv_grid, grid_info):
    """
    Draw grid with color-coded points:
    - MAGENTA filled: CV-detected corners (what OpenCV found)
    - CYAN filled: Extrapolated corners (calculated from CV step sizes)
    - BLUE lines: Grid lines connecting all points
    """
    result = img.copy()
    
    # Define colors
    CYAN = (255, 255, 0)  # BGR for cyan
    
    # Blue lines for entire grid
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
    
    # Get CV vs extrapolated indices
    cv_indices = set(tuple(x) for x in grid_info.get('cv_indices', []))
    extrapolated_indices = set(tuple(x) for x in grid_info.get('extrapolated_indices', []))
    
    # Draw extrapolated points first (CYAN) - smaller
    for (i, j) in extrapolated_indices:
        pt = tuple(grid[i, j].astype(int))
        cv2.circle(result, pt, 6, CYAN, -1)  # Filled cyan
        cv2.circle(result, pt, 6, BLUE, 1)   # Blue border
    
    # Draw CV-detected points on top (MAGENTA) - larger
    for (i, j) in cv_indices:
        pt = tuple(grid[i, j].astype(int))
        cv2.circle(result, pt, 8, MAGENTA, -1)  # Filled magenta
        cv2.circle(result, pt, 8, (255, 255, 255), 2)  # White border
    
    # Add legend
    legend_y = 20
    cv2.putText(result, f"CV found: {len(cv_indices)} pts", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 2)
    cv2.putText(result, f"Extrapolated: {len(extrapolated_indices)} pts", (10, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 2)
    
    if 'pattern' in grid_info:
        cv2.putText(result, f"Pattern: {grid_info['pattern']}", (10, legend_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result

def draw_classifications(stitched, predictions, color_grid=None):
    """
    Draw the piece classification results on stitched image.
    Shows: W/B color indicator, piece type
    """
    result = stitched.copy()
    
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            pred = predictions[label]
            piece = pred['piece']
            is_white = pred.get('is_white_piece')
            conf = pred.get('confidence', 0)
            
            # Stitched is uniform 8x8 grid of SQUARE_SIZE
            cx = col * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = row * SQUARE_SIZE + SQUARE_SIZE // 2
            
            if piece != 'empty':
                # Color based on detected piece color
                if is_white:
                    color = (0, 255, 0)   # Green for White
                    text = "W"
                else:
                    color = (0, 0, 255)   # Red for Black
                    text = "B"
                
                # Draw small filled circle at center
                cv2.circle(result, (cx, cy), 8, color, -1)
                
                # Draw W/B label in top-left corner
                cv2.putText(result, text, (col * SQUARE_SIZE + 2, row * SQUARE_SIZE + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw piece type + confidence in bottom-right corner
                piece_label = f"{piece.upper()}"
                cv2.putText(result, piece_label, 
                           (col * SQUARE_SIZE + SQUARE_SIZE - 14, row * SQUARE_SIZE + SQUARE_SIZE - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                # Empty - faint dot
                cv2.circle(result, (cx, cy), 3, (100, 100, 100), -1)
    
    return result


def detect_board_orientation(square_colors):
    """
    Detect if board needs rotation based on a1 square color.
    Standard chess: a1 is ALWAYS dark, h1 is ALWAYS light.
    Returns: 'correct', 'rotated_180', or 'unknown'
    """
    a1_color = square_colors.get('a1', 'unknown')
    h1_color = square_colors.get('h1', 'unknown')
    a8_color = square_colors.get('a8', 'unknown')
    h8_color = square_colors.get('h8', 'unknown')
    
    # Standard orientation: a1=dark, h1=light
    if a1_color == 'dark' and h1_color == 'light':
        return 'correct'
    # Rotated 180: a1=light, h1=dark (board is upside down)
    elif a1_color == 'light' and h1_color == 'dark':
        return 'rotated_180'
    else:
        return 'unknown'

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
            'piece': PIECE_MODEL is not None,
            'color_models': {pt: (pt in COLOR_MODELS) for pt in ['P', 'N', 'B', 'R', 'Q', 'K']}
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Full analysis pipeline"""
    try:
        # Debug: log what we received
        print(f"[DEBUG] request.files: {list(request.files.keys())}")
        print(f"[DEBUG] request.is_json: {request.is_json}")
        print(f"[DEBUG] request.content_type: {request.content_type}")
        
        # Get image
        if 'image' in request.files:
            print("[DEBUG] Found image in files")
            img = decode_image(request.files['image'])
        elif request.is_json and 'image' in request.get_json():
            print("[DEBUG] Found image in JSON")
            img = decode_image(request.get_json()['image'])
        else:
            print("[DEBUG] No image found!")
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        result = {
            'success': True,
            'steps': {}
        }
        
        h, w = img.shape[:2]
        result['original_size'] = f"{w}x{h}"
        
        # Step 1: Original
        result['steps']['1_original'] = image_to_base64(img)
        
        # Step 2: Board detection
        box, box_conf = detect_board(img)
        if box is not None:
            x1, y1, x2, y2 = box
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
            result['steps']['2_board_detected'] = image_to_base64(img)
            result['board_confidence'] = 0
            cropped = img
        
        # Step 3: Cropped
        result['steps']['3_cropped'] = image_to_base64(cropped)
        
        # Step 4: Corner detection
        corners, corner_conf = detect_corners(cropped)
        if corners is None:
            # Fallback to image corners
            ch, cw = cropped.shape[:2]
            corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
            result['corner_confidence'] = 0
        else:
            result['corner_confidence'] = corner_conf
        
        corners_vis = draw_corners(cropped, corners)
        result['steps']['4_corners'] = image_to_base64(corners_vis)
        
        # Step 5: Warp with margin for better CV detection
        warped, margin = warp_perspective(cropped, corners, WARP_SIZE)
        full_size = WARP_SIZE + 2 * margin
        result['steps']['5_warped'] = image_to_base64(warped)
        result['warp_margin'] = margin
        
        # Step 6: CV corners + hybrid grid
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        cv_corners, pattern = find_cv_corners(gray)
        
        if cv_corners is not None:
            # USE CV AS PRIMARY - extrapolate missing squares
            grid, cv_grid, grid_info = create_hybrid_grid(cv_corners, pattern, full_size, margin)
            result['cv_pattern'] = f"{pattern[0]}x{pattern[1]}"
            result['grid_method'] = 'CV + extrapolation'
        else:
            # FALLBACK: simple 8x8 divide only if CV found nothing
            grid = simple_grid(full_size, margin)
            cv_grid = None
            grid_info = {'cv_indices': [], 'extrapolated_indices': [(i, j) for i in range(9) for j in range(9)]}
            result['cv_pattern'] = 'none'
            result['grid_method'] = 'simple 8x8 (CV failed)'
        
        grid_vis = draw_grid_detailed(warped, grid, cv_grid, grid_info)
        result['steps']['6_grid'] = image_to_base64(grid_vis)
        
        # Step 7: Extract squares using CV grid (proper alignment)
        squares = extract_squares(warped, grid)
        
        # NO ROTATION - CV corners define correct orientation
        # Just detect for info, but NEVER rotate
        result['detected_orientation'] = 'as_detected'
        
        # Save individual squares
        squares_b64 = {}
        for label, sq in squares.items():
            squares_b64[label] = image_to_base64(sq)
        result['squares'] = squares_b64
        
        # Step 8: Stitch
        stitched = stitch_squares(squares)
        result['steps']['7_stitched'] = image_to_base64(stitched)
        
        # Step 9: Classify using 3.pt (TYPE + COLOR in one model)
        board, predictions, square_colors = classify_squares(squares)
        if board is not None:
            result['board'] = board
            result['predictions'] = predictions
            result['square_colors'] = square_colors
            result['fen'] = board_to_fen(board)
            
            # Classified visualization
            classified_vis = draw_classifications(stitched, predictions)
            result['steps']['8_classified'] = image_to_base64(classified_vis)
        else:
            result['board'] = [['.'] * 8 for _ in range(8)]
            result['fen'] = '8/8/8/8/8/8/8/8'
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
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
    print(f"\nModels loaded from: {SCRIPT_DIR}")
    app.run(host='0.0.0.0', port=port, debug=True)

#!/usr/bin/env python3
"""
ChessScan Analysis Server
=========================
Full pipeline server using YOLO models + PyTorch CNN color classifiers.

Place models in same folder as this script:
  1.pt - Board detection (YOLO)
  2.pt - Corner detection (YOLO)
  3.pt - Piece TYPE classification (YOLO - P/N/B/R/Q/K/empty)
  K_classifier.pt - King COLOR (PyTorch CNN - black vs white)
  Q_classifier.pt - Queen COLOR (PyTorch CNN)
  R_classifier.pt - Rook COLOR (PyTorch CNN)
  B_classifier.pt - Bishop COLOR (PyTorch CNN)
  N_classifier.pt - Knight COLOR (PyTorch CNN)
  P_classifier.pt - Pawn COLOR (PyTorch CNN)

Endpoints:
  GET  /health          - Health check
  POST /analyze         - Full analysis (multipart image upload)

Returns JSON with base64 images for each pipeline step.
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
import torch
import torch.nn as nn
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BOARD_MODEL_PATH = SCRIPT_DIR / '1.pt'
CORNER_MODEL_PATH = SCRIPT_DIR / '2.pt'
PIECE_MODEL_PATH = SCRIPT_DIR / '3.pt'  # TYPE classification (P/N/B/R/Q/K/empty)

# Per-piece COLOR classifiers (PyTorch CNN - black vs white)
COLOR_MODEL_PATHS = {
    'P': SCRIPT_DIR / 'P.pt',
    'N': SCRIPT_DIR / 'N.pt',
    'B': SCRIPT_DIR / 'B.pt',
    'R': SCRIPT_DIR / 'R.pt',
    'Q': SCRIPT_DIR / 'Q.pt',
    'K': SCRIPT_DIR / 'K.pt',
}

CLASS_NAMES = ['empty', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b']
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
# PYTORCH CNN MODEL DEFINITION (must match training architecture)
# =============================================================================

class PieceColorCNN(nn.Module):
    """CNN for binary classification (black vs white piece)"""
    
    def __init__(self):
        super(PieceColorCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============================================================================
# COLOR MODEL WRAPPER
# =============================================================================

class ColorClassifier:
    """Wrapper for PyTorch CNN color classifiers"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = PieceColorCNN()
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, img):
        """
        Predict color from image (numpy BGR or PIL).
        Returns: (is_white: bool, confidence: float)
        """
        # Convert numpy BGR to PIL RGB
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        # Transform and predict
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
        
        # probability > 0.5 = white (label 1), else black (label 0)
        is_white = probability > 0.5
        confidence = probability if is_white else (1 - probability)
        
        return is_white, confidence

# =============================================================================
# BOARD SQUARE COLOR DETECTION (for orientation check)
# =============================================================================

def get_board_square_color(square_img):
    """Determine if the BOARD SQUARE is light or dark."""
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
PIECE_MODEL = None
COLOR_MODELS = {}

# Detect device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Using device: {DEVICE}")

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

# Load per-piece color models (PyTorch CNN)
print("Loading color models (PyTorch CNN)...")
for piece_type, model_path in COLOR_MODEL_PATHS.items():
    if model_path.exists():
        try:
            COLOR_MODELS[piece_type] = ColorClassifier(str(model_path), device=DEVICE)
            print(f"  ✅ {piece_type}.pt loaded")
        except Exception as e:
            print(f"  ❌ {piece_type}.pt error: {e}")
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
        img_bytes = file_or_base64.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
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
    
    corners = sort_corners(corners)
    return corners, conf


def sort_corners(corners):
    """Sort 4 corners into consistent order: TL, TR, BR, BL"""
    corners = np.array(corners)
    
    # Sort by y first (top vs bottom)
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    
    # Sort each pair by x (left vs right)
    top_two = top_two[np.argsort(top_two[:, 0])]
    bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
    
    # TL, TR, BR, BL
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])


def warp_perspective(img, corners, size):
    """Warp board to square with margin"""
    margin = int(size * 0.06)
    full_size = size + 2 * margin
    
    dst = np.array([
        [margin, margin],
        [margin + size, margin],
        [margin + size, margin + size],
        [margin, margin + size]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (full_size, full_size))
    
    return warped, margin


def find_cv_corners(gray):
    """Find chessboard corners using OpenCV"""
    for pattern in [(7, 7), (6, 6), (5, 5), (7, 6), (6, 7)]:
        found, corners = cv2.findChessboardCorners(gray, pattern, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(pattern[1], pattern[0], 2), pattern
    return None, None


def create_hybrid_grid(cv_corners, pattern, full_size, margin):
    """Create 9x9 grid using CV corners + extrapolation"""
    rows, cols = cv_corners.shape[:2]
    
    # Calculate step sizes from CV corners
    h_steps = []
    v_steps = []
    for i in range(rows):
        for j in range(cols - 1):
            h_steps.append(cv_corners[i, j+1] - cv_corners[i, j])
    for i in range(rows - 1):
        for j in range(cols):
            v_steps.append(cv_corners[i+1, j] - cv_corners[i, j])
    
    avg_h_step = np.mean(h_steps, axis=0)
    avg_v_step = np.mean(v_steps, axis=0)
    
    # Create full 9x9 grid
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    
    # Place CV corners (offset by 1 since CV finds inner corners)
    cv_indices = []
    for i in range(rows):
        for j in range(cols):
            grid_i = i + 1
            grid_j = j + 1
            grid[grid_i, grid_j] = cv_corners[i, j]
            cv_indices.append((grid_i, grid_j))
    
    # Extrapolate edges
    extrapolated_indices = []
    
    # Top row
    for j in range(1, cols + 1):
        grid[0, j] = grid[1, j] - avg_v_step
        extrapolated_indices.append((0, j))
    
    # Bottom row
    for j in range(1, cols + 1):
        grid[8, j] = grid[7, j] + avg_v_step
        extrapolated_indices.append((8, j))
    
    # Left column
    for i in range(9):
        grid[i, 0] = grid[i, 1] - avg_h_step
        extrapolated_indices.append((i, 0))
    
    # Right column
    for i in range(9):
        grid[i, 8] = grid[i, 7] + avg_h_step
        extrapolated_indices.append((i, 8))
    
    grid_info = {
        'cv_indices': cv_indices,
        'extrapolated_indices': extrapolated_indices,
        'pattern': f"{pattern[0]}x{pattern[1]}"
    }
    
    return grid, cv_corners, grid_info


def simple_grid(full_size, margin):
    """Fallback: simple 8x8 grid"""
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    step = (full_size - 2 * margin) / 8
    
    for i in range(9):
        for j in range(9):
            grid[i, j] = [margin + j * step, margin + i * step]
    
    return grid


def extract_squares(warped, grid):
    """Extract 64 squares from warped board using grid"""
    squares = {}
    
    for row in range(8):
        for col in range(8):
            tl = grid[row, col]
            br = grid[row + 1, col + 1]
            
            x1, y1 = int(tl[0]), int(tl[1])
            x2, y2 = int(br[0]), int(br[1])
            
            # Ensure valid bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(warped.shape[1], x2), min(warped.shape[0], y2)
            
            square = warped[y1:y2, x1:x2]
            if square.size > 0:
                square = cv2.resize(square, (SQUARE_SIZE, SQUARE_SIZE))
            else:
                square = np.zeros((SQUARE_SIZE, SQUARE_SIZE, 3), dtype=np.uint8)
            
            label = f"{FILES[col]}{RANKS[row]}"
            squares[label] = square
    
    return squares


def stitch_squares(squares):
    """Stitch squares into single image"""
    result = np.zeros((8 * SQUARE_SIZE, 8 * SQUARE_SIZE, 3), dtype=np.uint8)
    
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            if label in squares:
                y1 = row * SQUARE_SIZE
                x1 = col * SQUARE_SIZE
                result[y1:y1+SQUARE_SIZE, x1:x1+SQUARE_SIZE] = squares[label]
    
    return result


def classify_squares(squares, warped_img=None, grid=None, color_grid=None):
    """
    Classify all squares:
    1. 3.pt → piece TYPE (P/N/B/R/Q/K or empty)
    2. PyTorch CNN color model → black or white
    3. Combine: uppercase=White, lowercase=Black
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
        
        is_dark_square = (row + col) % 2 == 1
        
        if ml_piece != 'empty':
            piece_type = ml_piece.upper()  # P, N, B, R, Q, or K
            
            # Step 2: Use PyTorch CNN color classifier
            if piece_type in COLOR_MODELS:
                color_classifier = COLOR_MODELS[piece_type]
                is_white, color_conf = color_classifier.predict(sq_img)
                
                print(f"[PIECE] {label}: {piece_type} (conf={conf:.2f}) + CNN → {'W' if is_white else 'B'} (conf={color_conf:.2f})")
            else:
                # Fallback: no color model, use 3.pt's guess
                is_white = ml_piece.isupper()
                color_conf = 0.0
                print(f"[PIECE] {label}: {piece_type} (conf={conf:.2f}) - no CNN model, using 3.pt guess → {'W' if is_white else 'B'}")
            
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


def draw_grid_detailed(img, grid, cv_grid, grid_info):
    """Draw grid with color-coded points"""
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
    
    return result


def draw_classifications(stitched, predictions, color_grid=None):
    """Draw the piece classification results on stitched image"""
    result = stitched.copy()
    
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            pred = predictions[label]
            piece = pred['piece']
            is_white = pred.get('is_white_piece')
            conf = pred.get('confidence', 0)
            
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
        'device': DEVICE,
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
        print(f"[DEBUG] request.files: {list(request.files.keys())}")
        print(f"[DEBUG] request.content_type: {request.content_type}")
        
        if 'image' in request.files:
            img = decode_image(request.files['image'])
        elif request.is_json and 'image' in request.get_json():
            img = decode_image(request.get_json()['image'])
        else:
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
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            
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
            ch, cw = cropped.shape[:2]
            corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
            result['corner_confidence'] = 0
        else:
            result['corner_confidence'] = corner_conf
        
        corners_vis = draw_corners(cropped, corners)
        result['steps']['4_corners'] = image_to_base64(corners_vis)
        
        # Step 5: Warp
        warped, margin = warp_perspective(cropped, corners, WARP_SIZE)
        full_size = WARP_SIZE + 2 * margin
        result['steps']['5_warped'] = image_to_base64(warped)
        
        # Step 6: CV corners + grid
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        cv_corners, pattern = find_cv_corners(gray)
        
        if cv_corners is not None:
            grid, cv_grid, grid_info = create_hybrid_grid(cv_corners, pattern, full_size, margin)
            result['cv_pattern'] = f"{pattern[0]}x{pattern[1]}"
            result['grid_method'] = 'CV + extrapolation'
        else:
            grid = simple_grid(full_size, margin)
            cv_grid = None
            grid_info = {'cv_indices': [], 'extrapolated_indices': [(i, j) for i in range(9) for j in range(9)]}
            result['cv_pattern'] = 'none'
            result['grid_method'] = 'simple 8x8'
        
        grid_vis = draw_grid_detailed(warped, grid, cv_grid, grid_info)
        result['steps']['6_grid'] = image_to_base64(grid_vis)
        
        # Step 7: Extract squares
        squares = extract_squares(warped, grid)
        
        squares_b64 = {}
        for label, sq in squares.items():
            squares_b64[label] = image_to_base64(sq)
        result['squares'] = squares_b64
        
        # Step 8: Stitch
        stitched = stitch_squares(squares)
        result['steps']['7_stitched'] = image_to_base64(stitched)
        
        # Step 9: Classify
        board, predictions, square_colors = classify_squares(squares)
        if board is not None:
            result['board'] = board
            result['predictions'] = predictions
            result['square_colors'] = square_colors
            result['fen'] = board_to_fen(board)
            
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
    print(f"  Device: {DEVICE}")
    print(f"\nEndpoints:")
    print(f"  GET  /health  - Health check")
    print(f"  POST /analyze - Full analysis")
    app.run(host='0.0.0.0', port=port, debug=True)

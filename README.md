# ChessScan Server

Chess position recognition using YOLO models.

## Files Needed

```
chessscan-server/
├── server.py
├── requirements.txt
├── render.yaml
├── .gitignore
├── 1.pt          # Board detection
├── 2.pt          # Corner detection
├── 3.pt          # Piece TYPE classification
├── P.pt          # Pawn color (W/B)
├── N.pt          # Knight color (W/B)
├── B.pt          # Bishop color (W/B)
├── R.pt          # Rook color (W/B)
├── Q.pt          # Queen color (W/B)
└── K.pt          # King color (W/B)
```

## Local Setup (Mac)

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
# → http://localhost:5555
```

## Render Deployment

1. Create GitHub repo with all files above
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects render.yaml
5. Deploy!

**Note:** Models (.pt files) must be in the repo. They'll be ~50-100MB each.

## API Endpoints

```
GET  /health   - Check server status
POST /analyze  - Analyze chess board image (multipart form: 'image')
```

## Test with curl

```bash
# Health check
curl http://localhost:5555/health

# Analyze image
curl -X POST -F "image=@chess_photo.jpg" http://localhost:5555/analyze
```

# YOLOv8 Table Layout Detector

A full-stack document intelligence system that detects and segments table structures — including column headers and projected row headers — from document images using a custom-trained YOLOv8 model.

---

## Project Overview

This project fine-tunes YOLOv8 on a dataset of ~3,500 annotated document images to identify three structural elements inside tables:

| Class | Description |
|---|---|
| `table` | Full table bounding region |
| `table column header` | Horizontal header row at the top of the table |
| `table projected row header` | Vertical/left-side row labels |

The system exposes a FastAPI backend for inference, a Next.js frontend for visual interaction, and a GitHub Actions CI pipeline for automated quality checks.

---

## Repository Structure

```
YOLO/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application & YOLO inference engine
│   ├── requirements.txt            # Python dependencies
│   ├── weights/
│   │   └── best.pt                 # Trained YOLOv8 model weights (~3,500 images)
│   └── tests/
│       └── test_main.py            # Pytest unit & integration tests
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Main UI — upload, inference, canvas overlay
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── public/
│   ├── package.json
│   ├── next.config.ts
│   └── tsconfig.json
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
└── README.md
```

---

## Tech Stack

### Backend
| Package | Role |
|---|---|
| `fastapi >= 0.111.0` | REST API framework — routes, request handling, interactive `/docs` UI |
| `uvicorn >= 0.30.1` | ASGI web server — binds to a port and serves live traffic |
| `ultralytics >= 8.4.57` | YOLOv8 engine — loads `best.pt` weights and runs inference |
| `python-multipart >= 0.0.9` | Parses incoming binary image file uploads over HTTP |
| `pillow >= 10.4.0` | Converts uploaded bytes into clean RGB images for the model |
| `pytest` | Automated test runner |
| `httpx` | HTTP client used to simulate API requests in tests |

### Frontend
- **Next.js** (App Router) with TypeScript
- Canvas API for bounding-box overlay rendering
- Lucide React for icons
- Tailwind CSS for styling

### CI/CD
- **GitHub Actions** — runs on push/PR to `main` or `master`
- Python 3.13 environment
- `flake8` linting (fatal errors: E9, F63, F7, F82)
- Full dependency install from `backend/requirements.txt`

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- `best.pt` weights file placed at `backend/weights/best.pt`

### 1. Backend Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd YOLO

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the API server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The UI will be available at `http://localhost:3000`.

---

## API Reference

### `GET /`
Health check — returns a confirmation that the service is running.

**Response:**
```json
{ "message": "Service is running and ready to accept requests." }
```

### `GET /status`
Returns the current model load state.

**Response:**
```json
{ "message": "YOLO FastAPI is running", "model": "best.pt loaded" }
```

### `POST /predict`
Accepts a `.jpg`, `.jpeg`, or `.png` image and returns all detected table structures.

**Request:** `multipart/form-data` with a `file` field containing the image.

**Response:**
```json
{
  "filename": "document.png",
  "width": 2550,
  "height": 3300,
  "detections_count": 3,
  "detections": [
    {
      "class_name": "table",
      "class_id": 0,
      "confidence": 0.97,
      "bbox": { "x1": 200.0, "y1": 1000.0, "x2": 2000.0, "y2": 2500.0 }
    },
    {
      "class_name": "table column header",
      "class_id": 1,
      "confidence": 0.91,
      "bbox": { "x1": 200.0, "y1": 1000.0, "x2": 2000.0, "y2": 1100.0 }
    },
    {
      "class_name": "table projected row header",
      "class_id": 2,
      "confidence": 0.88,
      "bbox": { "x1": 200.0, "y1": 1100.0, "x2": 600.0, "y2": 2500.0 }
    }
  ]
}
```

**Error responses:**
- `400` — Invalid file format (not JPG/PNG)
- `503` — Model weights not loaded
- `500` — Inference engine error

### `PUT /update_model?weight_path=<path>`
Dynamically swaps the active model weights at runtime without restarting the server.

### `DELETE /delete_model`
Unloads the active model from RAM and disables the `/predict` endpoint.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8 |
| Training images | ~3,500 annotated document pages |
| Inference resolution | 800px (matches training setup) |
| Confidence threshold | 0.5 |
| IoU threshold | 0.5 |
| Weights file | `backend/weights/best.pt` |

The model detects three classes corresponding to structural regions of tables in scanned or digital document images.

---

## Running Tests

```bash
# From the project root with the venv active
pytest backend/tests/ -v
```

The test suite covers:

- **Health endpoints** — `GET /` and `GET /status` return expected responses
- **Input validation** — invalid file extensions are rejected with `400`
- **Mocked inference** — a mock YOLO model verifies all three detection classes are returned correctly with accurate confidence scores and bounding boxes
- **Lifecycle endpoints** — `DELETE /delete_model` unloads the model cleanly

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push or pull request to `main` or `master`.

**Steps:**
1. Check out repository code
2. Set up Python 3.13
3. Install all backend dependencies from `requirements.txt`
4. Run `flake8` to catch fatal syntax and import errors (E9, F63, F7, F82)
5. Run a secondary `flake8` pass for complexity and line-length warnings (non-blocking)

---

## Frontend Usage

1. Open `http://localhost:3000` in your browser.
2. Click **"Click to browse filesystem"** and select a document image (JPG or PNG).
3. Click **"Run Layout Analysis"**.
4. Detected regions are drawn directly on the image canvas with colour-coded bounding boxes:
   - 🔵 **Blue** — full table
   - 🟢 **Green** — column header
   - 🟡 **Amber** — projected row header
5. Detection count, image dimensions, and per-detection confidence scores appear in the left panel.

---

## Sample Output

The screenshot below shows a real inference run on a medical benefits document (`assets/Screenshot From 2026-06-08 09-33-42.png`, 2550 × 3300 px).

### Inference Visualizer Canvas

The model correctly identified all three structural layers of the table with high confidence:

| Detection | Confidence | Colour |
|---|---|---|
| Table | **98.5%** | 🔵 Blue border around the full table region |
| Table Projected Row Header | **95.6%** | 🟡 Amber border on the left-side row label column |
| Table Column Header | **88.4%** | 🟢 Green border on the top header row |

**What the annotated output looks like:**

```
┌─────────────────────────────────────────────────────────┐  ← table 98.5%
│ table projected row header 96% │                        │  ← row header (amber)
│─────────────────────────────────────────────────────────│
│ table column header 88% │  Plan 2  │  PPO  │  Non-PPO  │  ← col header (green)
│─────────────────────────────────────────────────────────│
│ Covered Services             │ PPO Providers │ Non-PPO  │
│ Physician Office Visit...    │    100%       │  70%     │
│ Primary Care Physician       │  no Copay     │  after   │
│ Specialist                   │  after $20    │ Deduct.  │
│ ...                          │    ...        │  ...     │
└─────────────────────────────────────────────────────────┘
```

The left panel simultaneously displays:

- **Targets Located:** `3` — total detections in the image
- **Image Geometry:** `2550 × 3300 px`
- Per-class confidence badges (Table · Table Projected Row Header · Table Column Header)

> The model handles dense, multi-column insurance/benefits documents with footnotes and merged cells, correctly separating the structural skeleton (headers and row labels) from the body content.

---

## CORS Configuration

The backend allows requests from the following origins by default:

- `http://localhost:3000`
- `http://localhost:3001`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:3001`
- `http://192.168.1.84:3000`
- `http://192.168.1.84:3001`

To add additional origins, update the `allow_origins` list in `backend/main.py`.

---

## License

This project is proprietary. All rights reserved.
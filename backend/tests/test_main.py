import io
from pathlib import Path
import sys
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch  # MagicMock for creating fake objects, patch for replacing real classes/functions with our fakes during tests   
import backend.main
from backend.main import app


# Ensure the backend folder is visible to Python's pathing system
sys.path.append(str(Path(__file__).resolve().parent))

# Initialize the test client with our FastAPI instance
client = TestClient(app)


# 1. HEALTH ENDPOINT TESTS
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    # Matches your exact main.py string response
    assert response.json() == {"message": "Service is running and ready to accept requests."}

def test_read_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert "model" in response.json()


# 2. VALIDATION & ERROR HANDLING TESTS
def test_predict_invalid_file_extension():
    # Attempt to upload a text file instead of an image
    file_payload = {"file": ("document.txt", b"dummy file content", "text/plain")}
    response = client.post("/predict", files=file_payload)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image format. Please upload a JPG or PNG."



# 3. MOCKED DETECTION RUNTIME TESTS
@patch('backend.main.YOLO') 
def test_predict_all_classes(mock_yolo_class):
    """
    This test creates a fake AI model (stunt double) that instantly sends back 
    3 types of boxes: a full table, a column header, and a project row header.
    """
    mock_model_instance = MagicMock()
    
    # # Fake box for "table"
    mock_box_table = MagicMock()
    mock_box_table.xyxy = np.array([[200.0, 1000.0, 2000.0, 2500.0]])
    mock_box_table.conf = np.array([0.98])
    mock_box_table.cls = np.array([0])

    # Fake box for "table column header"
    mock_box_col = MagicMock()
    mock_box_col.xyxy = np.array([[200.0, 1000.0, 2000.0, 1100.0]])
    mock_box_col.conf = np.array([0.90])
    mock_box_col.cls = np.array([1])

    # Fake box for "table projected row header"
    mock_box_row = MagicMock()
    mock_box_row.xyxy = np.array([[200.0, 1100.0, 600.0, 2500.0]])
    mock_box_row.conf = np.array([0.88])
    mock_box_row.cls = np.array([2])
        
    # Bundle all fake boxes together
    mock_result = MagicMock()
    mock_result.boxes = [mock_box_table, mock_box_col, mock_box_row]
    mock_result.names = {
        0: "table",
        1: "table column header",
        2: "table projected row header"
    }
    mock_result.width = 2550
    mock_result.height = 3300

    mock_model_instance.predict.return_value = [mock_result]
    
    # import backend.main
    backend.main.model = mock_model_instance

    # Create a tiny dummy image in memory
    dummy_image = io.BytesIO()
    from PIL import Image
    Image.new('RGB', (1, 1), color='white').save(dummy_image, format='PNG')
    dummy_image.seek(0)

    file_payload = {"file": ("test_doc.png", dummy_image, "image/png")}
    response = client.post("/predict", files=file_payload)

    assert response.status_code == 200
    json_data = response.json()
    
    assert json_data["detections_count"] == 3
    assert json_data["detections"][0]["class_name"] == "table"
    assert json_data["detections"][0]["confidence"] == 0.98
    assert json_data["detections"][1]["class_name"] == "table column header"
    assert json_data["detections"][1]["confidence"] == 0.90
    assert json_data["detections"][2]["class_name"] == "table projected row header"
    assert json_data["detections"][2]["confidence"] == 0.88


# 4. LIFECYCLE ROUTE TESTS

def test_delete_model_endpoint():
    """
    Send a request to delete the model from system RAM
    """
    response = client.delete("/delete_model")
    assert response.status_code == 200
    assert "unloaded" in response.json()["message"] or "No active model" in response.json()["message"]

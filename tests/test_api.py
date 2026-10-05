"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns status 200 and valid schema."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is True


def test_samples_endpoint():
    """Verify pre-canned clinical samples endpoint."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "cases" in data
    assert len(data["cases"]) >= 2


def test_predict_endpoint_with_preset():
    """Verify inference endpoint runs successfully on preset case."""
    payload = {
        "demographics": {
            "subject_id": "TEST_CASE_ASD",
            "age": 10.5,
            "sex": 1,
            "full_scale_iq": 95.0,
            "site_id": "TEST_CLINIC"
        },
        "preset_case": "asd_sample"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["subject_id"] == "TEST_CASE_ASD"
    assert data["predicted_class"] in [0, 1]
    assert "asd_probability" in data
    assert "control_probability" in data
    assert len(data["top_pathways"]) > 0
    assert "connectome_graph" in data
    assert len(data["connectome_graph"]["nodes"]) == 116


def test_predict_image_resnet50():
    """Verify image screening endpoint with ResNet-50 backbone."""
    import io
    from PIL import Image

    # Create dummy 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict/image?model_type=resnet50",
        files={"file": ("test_portrait.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in ["ASD", "Typical"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_asd"] <= 1.0
    assert 0.0 <= data["probability_typical"] <= 1.0
    assert round(data["probability_asd"] + data["probability_typical"], 2) == 1.00
    assert data["model_used"] == "resnet50"
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] > 0


def test_predict_image_mobilenet_v2():
    """Verify image screening endpoint with MobileNet-V2 backbone."""
    import io
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict/image?model_type=mobilenet_v2",
        files={"file": ("test_portrait.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in ["ASD", "Typical"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_used"] == "mobilenet_v2"
    assert "inference_time_ms" in data


def test_predict_image_empty_file():
    """Verify that uploading an empty file returns HTTP 400 Bad Request."""
    import io

    empty_buf = io.BytesIO(b"")
    response = client.post(
        "/api/predict/image",
        files={"file": ("empty.jpg", empty_buf, "image/jpeg")}
    )
    assert response.status_code == 400


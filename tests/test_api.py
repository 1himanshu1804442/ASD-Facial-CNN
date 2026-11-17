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
    assert data["version"] in ["1.0.0", "1.1.0"]
    assert data["model_loaded"] is True


def test_samples_endpoint():
    """Verify pre-canned clinical samples endpoint."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "cases" in data
    assert len(data["cases"]) >= 2


def test_predict_image_invalid_payload():
    """Verify that posting non-image garbage returns HTTP 400 Bad Request."""
    import io
    bad_buf = io.BytesIO(b"NotAnImageFileDataAtAll")
    response = client.post(
        "/api/predict/image",
        files={"file": ("corrupt.txt", bad_buf, "text/plain")}
    )
    assert response.status_code == 400


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

    assert data["predicted_class"] in [1, 0, "ASD", "Typical"]
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

    assert data["predicted_class"] in [1, 0, "ASD", "Typical"]
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


def test_predict_facial_gcn_endpoint():
    """Verify facial GCN endpoint returns 68 landmarks and top pathways."""
    import io
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(180, 120, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict/facial-gcn",
        files={"file": ("child_face.jpg", buf, "image/jpeg")},
        data={"subject_id": "TEST_CHILD_001", "age": 7.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "connectome_graph" in data or "landmarks" in data
    nodes = data["connectome_graph"]["nodes"] if "connectome_graph" in data else data["landmarks"]
    assert len(nodes) == 68
    assert "top_pathways" in data
    assert len(data["top_pathways"]) >= 4
    assert data["predicted_label"] in ["Autism Spectrum Disorder", "Typical Control"]


def test_predict_image_rgba_png():
    """Verify that RGBA images with transparency are cleanly converted and predicted."""
    import io
    from PIL import Image

    img = Image.new("RGBA", (300, 300), color=(120, 180, 220, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/predict/image",
        files={"file": ("transparent_portrait.png", buf, "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["predicted_class"] in [1, 0, "ASD", "Typical"]


def test_predict_image_grayscale():
    """Verify that grayscale (single-channel L mode) portraits are properly normalized."""
    import io
    from PIL import Image

    img = Image.new("L", (256, 256), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict/image",
        files={"file": ("bw_photo.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["predicted_class"] in [1, 0, "ASD", "Typical"]


def test_predict_image_large_resolution():
    """Verify that high-resolution phone camera portraits (1920x1080) execute without memory or shape issues."""
    import io
    from PIL import Image

    img = Image.new("RGB", (1920, 1080), color=(210, 170, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict/image",
        files={"file": ("hd_phone_selfie.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200


def test_predict_image_svg_sample_portrait():
    """Verify that preset SVG sample portraits from the React dashboard do not throw HTTP 400."""
    import io

    svg_content = b'<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><circle cx="112" cy="112" r="80" fill="#fcd34d"/></svg>'
    buf = io.BytesIO(svg_content)

    response = client.post(
        "/api/predict/image",
        files={"file": ("pediatric_asd_portrait.svg", buf, "image/svg+xml")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert "landmarks_68" in data
    assert "gcn_saliency_edges" in data
    assert "gradcam_hotspots" in data
    assert response.json()["predicted_class"] in [1, 0, "ASD", "Typical"]


def test_v1_health_and_image_endpoints():
    """Verify that both /api/v1/health and /api/v1/predict/image route cleanly."""
    import io
    from PIL import Image

    resp_h = client.get("/api/v1/health")
    assert resp_h.status_code == 200
    assert resp_h.json()["status"] == "healthy"

    img = Image.new("RGB", (224, 224), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp_img = client.post(
        "/api/v1/predict/image",
        files={"file": ("test_portrait.jpg", buf, "image/jpeg")}
    )
    assert resp_img.status_code == 200
    assert resp_img.json()["predicted_class"] in [0, 1]


def test_models_benchmark_endpoint():
    """Verify that /api/v1/models/benchmark returns empirical literature metrics for all 6 backbones."""
    resp = client.get("/api/v1/models/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "models" in data
    models_dict = data["models"]
    for arch in ["modstem_lstm", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "vit"]:
        assert arch in models_dict
        assert "accuracy" in models_dict[arch]
        assert "latency" in str(models_dict[arch]).lower() or "inference_latency_ms" in models_dict[arch]


def test_predict_demo_endpoint():
    """Verify rapid 1-click preset demo endpoint for both ASD and Control cases."""
    # GET ASD case
    resp_asd = client.get("/api/v1/predict/demo?case=asd&model_type=modstem_lstm")
    assert resp_asd.status_code == 200
    d_asd = resp_asd.json()
    assert d_asd["predicted_class"] == 1
    assert d_asd["predicted_label"] == "Autism Spectrum Disorder"
    assert len(d_asd["landmarks_68"]) == 68

    # GET Control case
    resp_ctrl = client.get("/api/v1/predict/demo?case=control&model_type=mobilenet_v2")
    assert resp_ctrl.status_code == 200
    d_ctrl = resp_ctrl.json()
    assert d_ctrl["predicted_class"] == 0
    assert d_ctrl["predicted_label"] == "Typical Control"

    # POST demo
    resp_post = client.post("/api/v1/predict/demo", data={"case": "asd"})
    assert resp_post.status_code == 200


def test_detect_and_crop_face_edge_cases():
    """Unit test verify detect_and_crop_face against boundary conditions, mode conversions, and zero crops."""
    from PIL import Image
    from src.api.server import detect_and_crop_face

    # 1. Non-RGB image modes (RGBA, Grayscale L, 1-bit, CMYK)
    for mode, color in [("RGBA", (120, 150, 200, 255)), ("L", 128), ("1", 1), ("CMYK", (0, 100, 100, 0))]:
        im = Image.new(mode, (120, 120), color=color)
        cx, cy, fw, fh, cr = detect_and_crop_face(im)
        assert cr.mode == "RGB", f"Expected RGB output for mode {mode}, got {cr.mode}"
        assert cr.size[0] > 0 and cr.size[1] > 0
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0

    # 2. None input handling
    cx, cy, fw, fh, cr = detect_and_crop_face(None)
    assert cr is not None
    assert cr.mode == "RGB"

    # 3. Tiny input image (1x1)
    tiny = Image.new("RGB", (1, 1), color=(100, 100, 100))
    cx, cy, fw, fh, cr = detect_and_crop_face(tiny)
    assert cr.size == (1, 1)



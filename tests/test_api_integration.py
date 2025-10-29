"""
Integration tests for Vehicle Damage Detection API.
Tests API endpoints with real HTTP requests.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import httpx
from PIL import Image
import io
import json


class TestVehicleDamageAPI:
    """Integration tests for the vehicle damage detection API."""

    @pytest.fixture
    async def client(self):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            yield client

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes

    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "database_connected" in data

    async def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

    async def test_analyze_image_with_sample(self, client, sample_image):
        """Test image analysis with sample data."""
        files = {
            "file": ("test_image.jpg", sample_image, "image/jpeg")
        }
        
        response = await client.post("/api/analyze", files=files)
        
        # This might return 503 if model isn't loaded, which is acceptable for integration tests
        if response.status_code == 503:
            pytest.skip("Model not loaded - skipping integration test")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "detection" in data
        assert "classification" in data
        assert "cost_estimate" in data
        assert "total_processing_time" in data

    async def test_get_nonexistent_result(self, client):
        """Test getting a non-existent result."""
        response = await client.get("/api/results/99999")
        assert response.status_code == 404

    async def test_history_endpoint(self, client):
        """Test history endpoint."""
        response = await client.get("/api/history")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)

    async def test_delete_nonexistent_result(self, client):
        """Test deleting a non-existent result."""
        response = await client.delete("/api/results/99999")
        assert response.status_code == 404

    async def test_invalid_file_upload(self, client):
        """Test uploading an invalid file."""
        files = {
            "file": ("test.txt", b"not an image", "text/plain")
        }
        
        response = await client.post("/api/analyze", files=files)
        assert response.status_code == 400

    async def test_large_file_upload(self, client):
        """Test uploading a very large file."""
        # Create a large fake image file (simulating 100MB)
        large_data = b"fake_image_data" * 1024 * 1024 * 10  # ~160MB
        
        files = {
            "file": ("large_image.jpg", io.BytesIO(large_data), "image/jpeg")
        }
        
        response = await client.post("/api/analyze", files=files)
        # Should either be 400 (too large) or 413 (payload too large)
        assert response.status_code in [400, 413, 422]

    async def test_missing_file_upload(self, client):
        """Test uploading without file."""
        response = await client.post("/api/analyze")
        assert response.status_code == 422  # Validation error

    async def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = await client.options("/api/analyze")
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    async def test_api_documentation_available(self, client):
        """Test that API documentation is available."""
        response = await client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    async def test_openapi_spec(self, client):
        """Test OpenAPI specification availability."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        assert "components" in data


class TestAPIValidation:
    """Test API input validation and error handling."""

    @pytest.fixture
    async def client(self):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            yield client

    async def test_invalid_json(self, client):
        """Test invalid JSON request."""
        response = await client.post(
            "/api/analyze",
            json={"invalid": "data"},
            files={"file": ("dummy.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 422

    async def test_empty_file_upload(self, client):
        """Test uploading empty file."""
        files = {
            "file": ("empty.jpg", b"", "image/jpeg")
        }
        
        response = await client.post("/api/analyze", files=files)
        # Should return 400 for invalid image
        assert response.status_code == 400

    async def test_unsupported_image_format(self, client):
        """Test uploading unsupported image format."""
        # Create a fake GIF file
        gif_data = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,"
        
        files = {
            "file": ("test.gif", io.BytesIO(gif_data), "image/gif")
        }
        
        response = await client.post("/api/analyze", files=files)
        assert response.status_code == 400


class TestStreamlitUI:
    """Test Streamlit UI connectivity."""

    @pytest.fixture
    async def ui_client(self):
        """Create HTTP client for Streamlit UI."""
        async with httpx.AsyncClient(base_url="http://localhost:8501") as client:
            yield client

    async def test_ui_health(self, ui_client):
        """Test Streamlit UI health check."""
        try:
            response = await ui_client.get("/_stcore/health", timeout=5.0)
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Streamlit UI not running")

    async def test_ui_main_page(self, ui_client):
        """Test main UI page accessibility."""
        try:
            response = await ui_client.get("/", timeout=5.0)
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Streamlit UI not running")


# Async test runner
def run_async_tests():
    """Helper to run async tests manually."""
    import asyncio
    
    async def run_tests():
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get("http://localhost:8000/health")
            print(f"Health check: {response.status_code}")
            print(f"Response: {response.json()}")
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    # Run tests manually
    run_async_tests()
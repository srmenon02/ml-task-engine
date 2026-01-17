import pytest

@pytest.mark.integration
class TestAPIVersioning:
    def test_v1_endpoint_accessible(self, client, auth_headers):
        response = client.get(
            "/api/v1/jobs",
            headers = auth_headers
        )

        assert response.status_code == 200

    def test_legacy_endpoint_shows_depracation(self, client, auth_headers):
        response = client.get("/jobs", headers = auth_headers)

        assert "X-API-Warn" in response.headers
        assert "depracated" in response.headers["X-API-Warn"].lower()

    def test_version_in_response_headers(self, client):
        response = client.get("/health")

        assert "X-API-Version" in response.headers
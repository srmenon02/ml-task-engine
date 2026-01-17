import pytest
from factories.job_factory import JobFactory

@pytest.mark.integration
class TestPagination:
    def test_pagination_first_page(self, client, auth_headers, test_db):
        jobs = JobFactory.create_batch(75, user_id = "user_123")
        test_db.add_all(jobs)
        test_db.commit()

        response = client.get(
            "/api/v1/jobs?page_size=50",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) == 50
        assert data["total"] == 75
        assert data["page"] == 1
        assert data["total_pages"] == 2
        assert data["has_next"] is True
        assert data["has_prev"] is False

    def test_pagination_second_page(self, client, auth_headers, test_db):
        jobs = JobFactory.create_batch(75, user_id = "user123")
        test_db.add_all(jobs)
        test_db.commit()

        response = client.get(
            "/api/v1/jobs?page=2&page_size=50",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) == 25
        assert data["page"] == 2
        assert data["has_next"] is False
        assert data["has_prev"] is True
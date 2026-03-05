import apiClient from "./client";

export const fetchJobs = async ({ page = 1, pageSize = 20, status } = {}) => {
    const params = { page, page_size: pageSize };
    if (status) params.status = status;
    const { data } = await apiClient.get('/jobs', { params });
    return data;
}

export const fetchJob = async (jobId) => {
    const { data } = await apiClient.get(`/jobs/${jobId}`);
    return data;
}

export const createJob = async (jobData) => {
    const { data } = await apiClient.post('/jobs', jobData);
    return data;
}

export const cancelJob = async (jobId) => {
    const { data } = await apiClient.post(`/jobs/${jobId}/cancel`);
    return data;
}

export const fetchSystemStats = async () => {
    const { data } = await apiClient.get('/system/stats');
    return data;
}

export const fetchHealth = async () => {
    const { data } = await apiClient.get('/health');
    return data;
}

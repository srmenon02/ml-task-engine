import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL
const API_KEY = import.meta.env.VITE_API_KEY

const apiClient = axios.create({
    baseURL: BASE_URL,
    headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
    },
});

apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        console.error('API error: ', error.response?.status, error.response?.data);
        return Promise.reject(error);
    }
);

export default apiClient;
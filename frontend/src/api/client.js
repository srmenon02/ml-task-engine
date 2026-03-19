import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL;

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API error:', error.response?.status, error.response?.data);
    return Promise.reject(error);
  }
);

export default apiClient;
import { useAuth } from '@clerk/clerk-react';
import { useMemo } from 'react';

export function useApiClient() {
  const { getToken } = useAuth();

  const client = useMemo(() => {
    const instance = axios.create({
      baseURL: BASE_URL,
      headers: { 'Content-Type': 'application/json' },
    });

    instance.interceptors.request.use(async (config) => {
      const token = await getToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    instance.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API error:', error.response?.status, error.response?.data);
        return Promise.reject(error);
      }
    );

    return instance;
  }, [getToken]);

  return client;
}
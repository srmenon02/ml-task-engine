import { useApiClient } from './client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export function useJobs({ page = 1, pageSize = 20, status } = {}) {
  const apiClient = useApiClient();

  return useQuery({
    queryKey: ['jobs', page, pageSize, status],
    queryFn: async () => {
      const params = { page, page_size: pageSize };
      if (status) params.status = status;
      try {
        const { data } = await apiClient.get('/api/v1/jobs', { params });
        console.log('jobs response:', data);
        return data;
      } catch (err) {
        console.log('jobs error:', err.response?.status, err.response?.data);
        throw err;
      }
    },
  });
}

export function useJob(jobId) {
  const apiClient = useApiClient();

  return useQuery({
    queryKey: ['job', jobId],
    queryFn: async () => {
      const { data } = await apiClient.get(`/jobs/${jobId}`);
      return data;
    },
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'pending' ? 5000 : false;
    },
    enabled: !!jobId,
  });
}

export function useCreateJob() {
  const apiClient = useApiClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobData) => {
      const { data } = await apiClient.post('/jobs', jobData);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['jobs']);
    },
  });
}

export function useCancelJob(jobId) {
  const apiClient = useApiClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const { data } = await apiClient.post(`/jobs/${jobId}/cancel`);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['job', jobId]);
      queryClient.invalidateQueries(['jobs']);
    },
  });
}

export function useSystemStats() {
  const apiClient = useApiClient();

  return useQuery({
    queryKey: ['systemStats'],
    queryFn: async () => {
      const { data } = await apiClient.get('/system/stats');
      return data;
    },
    refetchInterval: 5000,
  });
}
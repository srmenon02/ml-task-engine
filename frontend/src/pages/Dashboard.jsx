import { useQuery } from '@tanstack/react-query';
import { fetchSystemStats } from '../api/jobs';

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['systemStats'],
    queryFn: fetchSystemStats,
    refetchInterval: 30000, // refresh every 30 seconds
  });

  if (isLoading) return <p className="text-gray-500">Loading stats...</p>;
  if (error) return <p className="text-red-500">Failed to load stats.</p>;

  const { jobs, workers } = data;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Jobs" value={jobs.total} />
        <StatCard label="Completed" value={jobs.completed} color="green" />
        <StatCard label="Failed" value={jobs.failed} color="red" />
        <StatCard label="Pending" value={jobs.pending} color="yellow" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <StatCard label="Active Workers" value={workers.active} />
        <StatCard
          label="Success Rate"
          value={`${jobs.success_rate.toFixed(1)}%`}
          color="green"
        />
      </div>
    </div>
  );
}

function StatCard({ label, value, color = 'gray' }) {
  const colors = {
    green: 'text-green-600',
    red: 'text-red-600',
    yellow: 'text-yellow-600',
    gray: 'text-gray-800',
  };

  return (
    <div className="bg-white rounded-lg border p-4 shadow-sm">
      <p className="text-sm text-gray-500">{label}</p>
      <p className={`text-3xl font-bold mt-1 ${colors[color]}`}>{value}</p>
    </div>
  );
}
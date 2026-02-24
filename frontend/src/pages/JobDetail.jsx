import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchJob, cancelJob } from '../api/jobs';
import JobStatusBadge from '../components/JobStatusBadge';
import LoadingSpinner from '../components/LoadingSpinner';

export default function JobDetail() {
    const  { id } = useParams();
    const queryClient = useQueryClient();

    const { data: job, isLoading, error } = useQuery({
        queryKey: ['job', id],
        queryFn: () => fetchJob(id),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === 'running' || status === 'pending' ? 5000 : false;
        },
    });

    const cancelMutation = useMutation({
        mutationFn: () => cancelJob(id),
        onSuccess: () => {
            queryClient.invalidateQueries(['job', id]);
            queryClient.invalidateQueries(['jobs']);
        },
    });

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleString();
    };

    const formatDuration = (start, end) => {
        if (!start || !end) return 'N/A';
        const ms = new Date(end) - new Date(start);
        return `${(ms / 1000).toFixed(2)}s`;
    };

    if (isLoading) return <LoadingSpinner text = "Loading job details..." />;

    if (error) {
        return (
            <div className = "text-center py-8">
                <p className = "text-red-500 mb-4">Failed to load job #{id}</p>
                <Link
                    to="/jobs"
                    className = "text-blue-600 hover:text-blue-700 underline"
                >
                    Back to Jobs
                </Link>
            </div>
        );
    }

    const canCancel = job.status === 'pending' || job.status === 'running';

    return (
        <div>
            <Link to="/jobs" className="text-blue-600 hover:text-blue-700 text-sm mb-4 inline-block">
                Back to Jobs
            </Link>

            <div className="bg-white border rounded-lg p-6">
                <div className="flex items-start justify-between mb-6 pb-6 border-b">
                    <div>
                        <h1 className = "text-2xl font-bold mb-2">Job #{job.id}</h1>
                        <p className = "text-gray-600">{job.job_type}</p>
                    </div>
                    <div className = "flex flex-col items-end gap-2">
                        <JobStatusBadge status={job.status} />
                        {canCancel && (
                            <button
                                onClick={() => cancelMutation.mutate()}
                                disabled={cancelMutation.isPending}
                                className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
                            >
                                {cancelMutation.isPending ? 'Cancelling...' : 'Cancel Job'}
                            </button>
                        )}
                    </div>
                </div>

                <div className = "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <InfoItem label="Priority" value={job.priority} />
                    <InfoItem label="Created At" value={formatDate(job.created_at)} />
                    <InfoItem label="Started At" value={formatDate(job.started_at)} />
                    <InfoItem label="CompletedAt" value={formatDate(job.completed_at)} />
                </div>

                {job.started_at && job.completed_at && (
                    <div className="mb-6">
                        <InfoItem
                        label = "Execution Time"
                        value = {formatDuration(job.started_at, job.completed_at)}
                        />
                    </div>
                )}

                <section className = "mb-6">
                    <h2 className = "text-lg font-semibold mb-3"> Configuration </h2>
                    <div className = "bg-gray-50 rounded-lg p-4">
                        <pre className = "text-sm text-gray-700 overflow-x-auto">
                            {JSON.stringify(job.config, null, 2)}
                        </pre>
                     </div>
                </section>

                {job.predicted_memory_db || job.predicted_cpu_percent ? (
                    <section className = "mb-6">
                        <h2 className = "text-lg font-semibold mb-3"> Resource Predictions</h2>
                        <div className = "grid grid-cols-2 gap-4">
                            {job.predicted_memory_db && (
                                <div className = "bg-blue-50 rounded-lg p-4">
                                    <p className = "text-sm text-gray-600 mb-1"> Predicted Memory</p>
                                    <p className = "text-2xl font-bold text-blue-700">
                                        {job.predicted_memory_db.toFixed(1)} MB
                                    </p>
                                </div>
                            )}
                            {job.predicted_cpu_percent && (
                                <div className = "bg-green-50 rounded-lg p-4">
                                    <p className = "text-sm text-gray-600 mb-1"> Predicted CPU</p>
                                    <p className = "text-2xl font-bold text-green-700">
                                        {job.predicted_cpu_percent.toFixed(1)}%
                                    </p>
                                </div>
                            )}
                        </div>
                    </section>
                ) : null}
                
                {job.results && (
                    <section className = "mb-6">
                        <h2 className = "text-lg font-semibold mb-3"> Results </h2>
                        <div className = "bg-green-50 rounded-lg p-4">
                            <pre className = "text-sm text-gray-700 overflow-x-auto">
                                {JSON.stringify(job.results, null, 2)}
                            </pre>
                        </div>
                    </section>
                )}

                {job.error_message && (
                    <section>
                        <h2 className = "text-lg font-semibold mb-3"> Error </h2>
                        <div className = "bg-red-50 border border-red-200 rounded-lg p-4">
                            <p className = "text-sm text-red-700">{job.error_message}</p>
                        </div>
                    </section>
                )}

                {job.cancelled_by && (
                    <section className = "mt-6 pt-6 border-t">
                        <p className = "text-sm text-gray-600">
                            Cancelled by <span className = "font-medium">{job.cancelled_by}</span> at {''}
                            {formatDate(job.cancelled_at)}
                        </p>
                    </section>
                )}
            </div>
        </div>
    );
}

function InfoItem( { label, value}) {
    return (
        <div>
            <p className = "text-sm text-gray-500 mb-1">{label}</p>
            <p className = "font-medium text-gray-900">{value}</p>
        </div>
    )
}
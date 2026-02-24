import { Link } from 'react-router-dom';
import JobStatusBadge from './JobStatusBadge';

export default function JobCard({ job }) {
    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <Link
        to={`/jobs/${job.id}`}
        className="block bg-white border rounded-lg p-4 hover:shadow-md transition-shadow"
        >
            <div className="flex items-start justify-between mb-3">
                <div>
                    <h3 className="font-semibold text-gray-900">Job #{job.id}</h3>
                    <p className="text-sm text-gray-500">{job.job_type}</p>
                </div>
                <JobStatusBadge status={job.status} />
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                    <p className="text-gray-500">Priority</p>
                    <p className="font-medium text-gray-900">{job.priority}</p>
                </div>
                <div>
                    <p className="text-gray-500">Created</p>
                    <p className="font-medium text-gray-900">{formatDate(job.created_at)}</p>
                </div>
            </div>

            {job.config && (
                <div className="mt-3 pt-3 border-t">
                    <p className="text-xs text-gray-500 mb-1">Configuration</p>
                    <div className="flex gap-4 text-xs">
                        {job.config.n_estimators && (
                            <span className="text-gray-700">
                                <span className="font-medium">Estimators</span>: {job.config.n_estimators}
                            </span>
                        )}
                        {job.config.dataset_rows && (
                            <span className="text-gray-700">
                                <span className="font-medium">Rows</span>: {job.config.dataset_rows.toLocaleString()}
                            </span>
                        )}
                    </div>
                </div>
            )}
        </Link>
    );
}
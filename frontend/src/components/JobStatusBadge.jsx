export default function JobStatusBadge({ status}) {
    const statusConfig = {
        pending: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Pending' },
        running: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Running' },
        completed: { bg: 'bg-green-100', text: 'text-green-800', label: 'Completed' },
        failed: { bg: 'bg-red-100', text: 'text-red800', label: 'Failed' },
        timeout: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'Timeout' },
        retrying: { bg: 'bg-purple-100', text: 'text-purple-800', label: 'Retrying' },
        cancelled: { bg: 'bg-gray-100', text: 'text-gray-800', label: 'Cancelled' },
    };

    const config = statusConfig[status?.toLowerCase()] || statusConfig.pending;

    return (
        <span
        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text}`}
        >
            {config.label}
        </span>
    );
}
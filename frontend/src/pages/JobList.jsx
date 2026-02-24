import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchJobs } from "../api/jobs";
import JobCard from "../components/JobCard";
import LoadingSpinner from "../components/LoadingSpinner";

export default function JobList() {
    const [page, setPage] = useState(1);
    const [statusFilter, setStatusFilter] = useState('');
    const pageSize = 20;

    const { data, isLoading, error, refetch } = useQuery({
        queryKey: ['jobs', page, statusFilter],
        queryFn: () => fetchJobs({ page, pageSize, status: statusFilter || undefined}),
        refetchInterval: 10000,
    });

    const statusOptions = [
        { value: '', label: 'All'},
        { value: 'pending', label: 'Pending'},
        { value: 'running', label: 'Running'},
        { value: 'completed', label: 'Completed'},
        { value: 'failed', label: 'Failed'},
    ]

    const handleFilterChange = (newStatus) => {
        setStatusFilter(newStatus);
        setPage(1);
    };

    if (error) {
        return (
            <div className = "text-center py-8">
                <p className = "text-red-500 mb-4">Failed to load jobs</p>
                <button
                    onClick = {() => refetch()}
                    className = "px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                    Try Again
                </button>
            </div>
        );
    }

    return (
        <div>
            <div className = "flex items-center justify-between mb-6">
                <h1 className = "text-2xl font-bold">Jobs</h1>
                <div className = "flex items-center gap-3">
                    <label className = "text-sm text-gray-600">Filter:</label>
                    <select
                      value = {statusFilter}
                      onChange = {(e) => handleFilterChange(e.target.value)}
                      className = "border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        {statusOptions.map((option) => (
                            <option key = {option.value} value = {option.value}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {isLoading ? (
                <LoadingSpinner text = "Loading Jobs..." />
            ) : (
                <>
                    {data.items.length == 0 ? (
                        <div className = "text-center py-12 bg-gray-50 rounded-lg">
                            <p className = "text-gray-500">No jobs found</p>
                        </div>
                    ) : (
                        <div className = "space-y-3">
                            {data.items.map((job) => (
                                <JobCard key = {job.id} job = {job} />
                            ))}
                        </div>
                    )}

                    {data.total_pages > 1 && (
                        <div className = "flex items-center justify-center gap-2 mt-6">
                            <button
                            onClick = {() => setPage((p) => Math.max(1, p-1))}
                            disabled = {!data.has_prev}
                            className = "px-4 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Previous
                            </button>

                            <span className="px-4 py-2 text-sm text-gray-600">
                                Page {data.page} of {data.total_pages}
                            </span>

                            <button
                            onClick = {() => setPage((p) => p +1)}
                            disabled = {!data.has_next}
                            className = "px-4 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Next
                            </button>
                        </div>
                    )}

                    <p className="text-center text-sm text-gray-500 mt-4">
                        Showing {data.items.length} of {data.total} jobs
                    </p>
                </>
            )}
        </div>
    );
}
import React from "react";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createJob } from "../api/jobs";
import { data, useNavigate } from "react-router-dom";

export default function CreateJob() {
    const navigate = useNavigate();
    const queryClient = useQueryClient();

    const [formData, setFormData] = useState({
        n_estimators: 100,
        dataset_rows: 10000,
        priority: 5,
    });

    const mutation = useMutation({
        mutationFn: () =>
            createJob({
                job_type: 'train_sklearn_model',
                config: {
                    model: 'RandomForest',
                    n_estimators: Number(formData.n_estimators),
                    dataset_rows: Number(formData.dataset_rows),
                },
                priority: Number(formData.priority),
            }),
        onSuccess: (data) => {
        queryClient.invalidateQueries(['jobs']);
        navigate(`/jobs/${data.id}`);
        },
    });

    const handleSubmit = (e) => {
        e.preventDefault();
        mutation.mutate();
    };

    return (
        <div className = "max-w-md">
            <h1 className="text-2xl font-bold mb-6">Submit New Job</h1>
            <form onSubmit={handleSubmit} className="space-y-4">
                <Field 
                    label="Number of Estiamtors (trees)"
                    value={formData.n_estimators}
                    min={1} max={1000000}
                    onChange={(v) => setFormData({...formData, n_estimators: v})}
                />
                <Field
                    label="Dataset Rows"
                    value={formData.dataset_rows}
                    min={1} max={10000000}
                    onChange={(v) => setFormData({ ...formData, dataset_rows: v })}
                />
                <Field
                    label="Priority (0-20)"
                    value={formData.priority}
                    min={0} max={20}
                    onChange={(v) => setFormData({ ...formData, priority: v })}
                />

                {mutation.isError && (
                    <p className="text-red-500 text-sm">
                        {mutation.error?.response?.data?.detail || "Something went wrong"}
                    </p>
                )}

                <button
                    type="submit"
                    disabled={mutation.isPending}
                    className="w-full bg-blue-600 text-black py-2 rounded-lg hover:bg-blue-700"
                >
                    {mutation.isPending ? 'Submitting...' : 'Submit Job'}
                </button>
            </form>
        </div>
    );
}

function Field({ label, value, min, max, onChange }) {
    return (
        <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
            <input
                type="number"
                value={value}
                min={min}
                max={max}
                onChange={(e) => onChange(e.target.value)}
                className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
        </div>
    )
}
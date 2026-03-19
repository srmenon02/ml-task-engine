import { useState } from "react";
import { useJobs } from "../api/jobs";
import JobCard from "../components/JobCard";
import LoadingSpinner from "../components/LoadingSpinner";

const STATUS_OPTIONS = [
  { value: '', label: 'ALL' },
  { value: 'pending', label: 'PENDING' },
  { value: 'running', label: 'RUNNING' },
  { value: 'completed', label: 'COMPLETED' },
  { value: 'failed', label: 'FAILED' },
];

const STATUS_COLORS = {
  '': 'var(--text-secondary)',
  pending: 'var(--yellow-signal)',
  running: 'var(--blue-signal)',
  completed: 'var(--green-signal)',
  failed: 'var(--red-signal)',
};

export default function JobList() {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const pageSize = 20;

  const { data, isLoading, error, refetch } = useJobs({
    page,
    pageSize,
    status: statusFilter || undefined,
  });

  const handleFilterChange = (newStatus) => {
    setStatusFilter(newStatus);
    setPage(1);
  };

  return (
    <div style={{ padding: '40px 0' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'space-between',
        marginBottom: '28px',
        animation: 'fadeSlideUp 0.5s ease forwards',
      }}>
        <div>
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '11px',
            color: 'var(--amber)',
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            marginBottom: '6px',
          }}>Job Queue</div>
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: '28px',
            fontWeight: 800,
            color: 'var(--text-primary)',
            letterSpacing: '-0.03em',
          }}>Execution Log</h1>
        </div>

        {data && (
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '12px',
            color: 'var(--text-muted)',
          }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{data.total}</span>
            {' '}total jobs
          </div>
        )}
      </div>

      {/* Filter tabs */}
      <div style={{
        display: 'flex',
        gap: '4px',
        marginBottom: '20px',
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border-dim)',
        borderRadius: 'var(--radius-md)',
        padding: '4px',
        width: 'fit-content',
        animation: 'fadeSlideUp 0.5s ease forwards',
        animationDelay: '0.06s',
        opacity: 0,
      }}>
        {STATUS_OPTIONS.map(opt => (
          <button
            key={opt.value}
            onClick={() => handleFilterChange(opt.value)}
            style={{
              padding: '5px 12px',
              borderRadius: '4px',
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--font-body)',
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '0.08em',
              transition: 'all 0.12s ease',
              backgroundColor: statusFilter === opt.value ? 'var(--bg-elevated)' : 'transparent',
              color: statusFilter === opt.value ? (STATUS_COLORS[opt.value] || 'var(--text-primary)') : 'var(--text-muted)',
              boxShadow: statusFilter === opt.value ? '0 0 0 1px var(--border-mid)' : 'none',
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {isLoading && <LoadingSpinner text="Fetching jobs..." />}

      {error && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '16px',
          padding: '48px',
          textAlign: 'center',
        }}>
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '12px',
            color: 'var(--red-signal)',
          }}>Failed to load jobs</div>
          <button
            onClick={() => refetch()}
            style={{
              padding: '8px 18px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-mid)',
              backgroundColor: 'var(--bg-raised)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-body)',
              fontSize: '11px',
              cursor: 'pointer',
              letterSpacing: '0.06em',
            }}
          >
            Retry
          </button>
        </div>
      )}

      {data && !isLoading && (
        <>
          {data.items.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '64px',
              backgroundColor: 'var(--bg-surface)',
              border: '1px dashed var(--border-dim)',
              borderRadius: 'var(--radius-lg)',
            }}>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: '11px',
                color: 'var(--text-faint)',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}>No jobs found</div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {data.items.map((job, i) => (
                <JobCard key={job.id} job={job} index={i} />
              ))}
            </div>
          )}

          {/* Pagination */}
          {data.total_pages > 1 && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginTop: '24px',
              padding: '0 4px',
              animation: 'fadeIn 0.4s ease forwards',
            }}>
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={!data.has_prev}
                style={paginationButtonStyle(!data.has_prev)}
              >
                ← Prev
              </button>

              <span style={{
                fontFamily: 'var(--font-body)',
                fontSize: '11px',
                color: 'var(--text-muted)',
                letterSpacing: '0.06em',
              }}>
                Page <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{data.page}</span>
                {' '}/ {data.total_pages}
                {' '}· {data.items.length} of <span style={{ color: 'var(--text-secondary)' }}>{data.total}</span>
              </span>

              <button
                onClick={() => setPage(p => p + 1)}
                disabled={!data.has_next}
                style={paginationButtonStyle(!data.has_next)}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function paginationButtonStyle(disabled) {
  return {
    padding: '7px 16px',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-dim)',
    backgroundColor: disabled ? 'transparent' : 'var(--bg-surface)',
    color: disabled ? 'var(--text-faint)' : 'var(--text-secondary)',
    fontFamily: 'var(--font-body)',
    fontSize: '11px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    letterSpacing: '0.04em',
    transition: 'all 0.12s ease',
  };
}
import { useParams, Link } from 'react-router-dom';
import { useJob, useCancelJob } from '../api/jobs';
import JobStatusBadge from '../components/JobStatusBadge';
import LoadingSpinner from '../components/LoadingSpinner';

export default function JobDetail() {
  const { id } = useParams();
  const { data: job, isLoading, error } = useJob(id);
  const cancelMutation = useCancelJob(id);

  const formatDate = (ds) => {
    if (!ds) return '—';
    const utcString = ds.endsWith('Z') || ds.includes('+') ? ds : ds + 'Z';
    const d = new Date(utcString);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      + ' · ' + d.toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: true,
      });
  };

  const formatDuration = (start, end) => {
    if (!start || !end) return '—';
    const ms = new Date(end) - new Date(start);
    const s = ms / 1000;
    if (s < 60) return `${s.toFixed(2)}s`;
    return `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`;
  };

  if (isLoading) return (
    <div style={{ padding: '60px 0' }}>
      <LoadingSpinner text="Loading job details..." />
    </div>
  );

  if (error) return (
    <div style={{ padding: '40px 0', textAlign: 'center' }}>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '12px',
        color: 'var(--red-signal)',
        marginBottom: '16px',
      }}>
        Failed to load job #{id}
      </div>
      <Link to="/jobs" style={backLinkStyle}>← Back to jobs</Link>
    </div>
  );

  const canCancel = job.status === 'pending' || job.status === 'running';

  return (
    <div style={{ padding: '40px 0' }}>
      {/* Breadcrumb */}
      <Link to="/jobs" style={backLinkStyle}>← Jobs</Link>

      {/* Header card */}
      <div style={{
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border-dim)',
        borderRadius: 'var(--radius-lg)',
        padding: '28px',
        marginTop: '20px',
        marginBottom: '12px',
        position: 'relative',
        overflow: 'hidden',
        animation: 'fadeSlideUp 0.5s ease forwards',
        opacity: 0,
      }}>
        {/* Accent gradient */}
        <div style={{
          position: 'absolute',
          top: 0, right: 0,
          width: '200px', height: '120px',
          background: 'radial-gradient(ellipse at top right, var(--amber-glow) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '20px' }}>
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              marginBottom: '10px',
            }}>
              <span style={{
                fontFamily: 'var(--font-body)',
                fontSize: '11px',
                color: 'var(--text-muted)',
                letterSpacing: '0.08em',
              }}>#{String(job.id).padStart(4, '0')}</span>
              <JobStatusBadge status={job.status} />
            </div>
            <h1 style={{
              fontFamily: 'var(--font-display)',
              fontSize: '24px',
              fontWeight: 800,
              color: 'var(--text-primary)',
              letterSpacing: '-0.03em',
            }}>
              {job.job_type.replace(/_/g, ' ')}
            </h1>
          </div>

          {canCancel && (
            <button
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
              style={{
                padding: '8px 16px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid rgba(239,68,68,0.35)',
                backgroundColor: 'var(--red-dim)',
                color: 'var(--red-signal)',
                fontFamily: 'var(--font-body)',
                fontSize: '11px',
                fontWeight: 600,
                letterSpacing: '0.06em',
                cursor: cancelMutation.isPending ? 'not-allowed' : 'pointer',
                opacity: cancelMutation.isPending ? 0.6 : 1,
                transition: 'all 0.15s ease',
                flexShrink: 0,
              }}
            >
              {cancelMutation.isPending ? 'Cancelling...' : '✕ Cancel Job'}
            </button>
          )}
        </div>
      </div>

      {/* Timing row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '10px',
        marginBottom: '12px',
        animation: 'fadeSlideUp 0.5s ease forwards',
        animationDelay: '0.06s',
        opacity: 0,
      }}>
        <TimeCard label="Created" value={formatDate(job.created_at)} />
        <TimeCard label="Started" value={formatDate(job.started_at)} />
        <TimeCard label="Completed" value={formatDate(job.completed_at)} />
        <TimeCard
          label="Duration"
          value={formatDuration(job.started_at, job.completed_at)}
          accent="var(--amber)"
        />
      </div>

      {/* Main content grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '12px',
        marginBottom: '12px',
        animation: 'fadeSlideUp 0.5s ease forwards',
        animationDelay: '0.12s',
        opacity: 0,
      }}>
        {/* Config */}
        <Section title="Configuration" icon="⚙">
          <CodeBlock data={job.config} />
        </Section>

        {/* Resource predictions */}
        {(job.predicted_memory_db || job.predicted_cpu_percent) && (
          <Section title="Resource Predictions" icon="◈">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {job.predicted_memory_db && (
                <ResourceBar
                  label="Memory"
                  value={`${job.predicted_memory_db.toFixed(1)} MB`}
                  pct={Math.min(100, job.predicted_memory_db / 20)}
                  color="var(--blue-signal)"
                />
              )}
              {job.predicted_cpu_percent && (
                <ResourceBar
                  label="CPU"
                  value={`${job.predicted_cpu_percent.toFixed(1)}%`}
                  pct={job.predicted_cpu_percent}
                  color="var(--amber)"
                />
              )}
              <div style={{ display: 'flex', gap: '12px' }}>
                <MetaItem label="Priority" value={job.priority} />
                <MetaItem label="Max Memory" value={job.max_memory_mb ? `${job.max_memory_mb.toFixed(0)} MB` : '—'} />
                <MetaItem label="Max Time" value={`${job.max_execution_time_sec}s`} />
              </div>
            </div>
          </Section>
        )}
      </div>

      {/* Results */}
      {job.results && (
        <div style={{
          animation: 'fadeSlideUp 0.5s ease forwards',
          animationDelay: '0.18s',
          opacity: 0,
          marginBottom: '12px',
        }}>
          <Section title="Results" icon="✓" accentColor="var(--green-signal)">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
              {Object.entries(job.results).map(([key, val]) => (
                <div key={key} style={{
                  padding: '12px',
                  backgroundColor: 'var(--bg-elevated)',
                  border: '1px solid var(--border-dim)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: '10px',
                    color: 'var(--text-faint)',
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                    marginBottom: '4px',
                  }}>
                    {String(key).replace(/_/g, ' ')}
                  </div>
                  <div style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: '14px',
                    fontWeight: 600,
                    color: 'var(--green-signal)',
                  }}>
                    {typeof val === 'number' ? val.toFixed(typeof val === 'number' && val < 10 ? 4 : 2) : String(val)}
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Error */}
      {(job.error_message || job.error_msg) && (
        <div style={{
          animation: 'fadeSlideUp 0.5s ease forwards',
          animationDelay: '0.2s',
          opacity: 0,
          marginBottom: '12px',
        }}>
          <Section title="Error" icon="⚠" accentColor="var(--red-signal)">
            <div style={{
              fontFamily: 'var(--font-body)',
              fontSize: '12px',
              color: 'var(--red-signal)',
              lineHeight: 1.7,
            }}>
              {job.error_message || job.error_msg}
            </div>
          </Section>
        </div>
      )}

      {/* Cancellation info */}
      {job.cancelled_by && (
        <div style={{
          padding: '12px 16px',
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-dim)',
          borderRadius: 'var(--radius-md)',
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          color: 'var(--text-muted)',
        }}>
          Cancelled by <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{job.cancelled_by}</span>
          {job.cancelled_at && <> · {formatDate(job.cancelled_at)}</>}
        </div>
      )}
    </div>
  );
}

const backLinkStyle = {
  textDecoration: 'none',
  fontFamily: 'var(--font-body)',
  fontSize: '11px',
  color: 'var(--text-muted)',
  letterSpacing: '0.06em',
  display: 'inline-flex',
  alignItems: 'center',
  transition: 'color 0.12s ease',
};

function Section({ title, icon, children, accentColor = 'var(--amber)' }) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-surface)',
      border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-lg)',
      padding: '20px',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '16px',
        paddingBottom: '12px',
        borderBottom: '1px solid var(--border-dim)',
      }}>
        <span style={{ color: accentColor, fontSize: '13px' }}>{icon}</span>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: '13px',
          fontWeight: 700,
          color: 'var(--text-primary)',
          letterSpacing: '-0.01em',
        }}>{title}</span>
      </div>
      {children}
    </div>
  );
}

function CodeBlock({ data }) {
  return (
    <pre style={{
      fontFamily: 'var(--font-body)',
      fontSize: '11px',
      color: 'var(--text-secondary)',
      lineHeight: 1.7,
      backgroundColor: 'var(--bg-elevated)',
      border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-sm)',
      padding: '12px 14px',
      overflow: 'auto',
      whiteSpace: 'pre-wrap',
    }}>
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function TimeCard({ label, value, accent }) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-surface)',
      border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-md)',
      padding: '14px',
    }}>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '10px',
        color: 'var(--text-faint)',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        marginBottom: '4px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '12px',
        color: accent || 'var(--text-secondary)',
        fontWeight: accent ? 600 : 400,
      }}>{value}</div>
    </div>
  );
}

function ResourceBar({ label, value, pct, color }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '10px',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>{label}</span>
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '12px',
          fontWeight: 600,
          color,
        }}>{value}</span>
      </div>
      <div style={{
        height: '3px',
        backgroundColor: 'var(--bg-elevated)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${Math.min(100, pct)}%`,
          backgroundColor: color,
          boxShadow: `0 0 6px ${color}`,
          borderRadius: '2px',
          transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  );
}

function MetaItem({ label, value }) {
  return (
    <div>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '10px',
        color: 'var(--text-faint)',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        marginBottom: '2px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '12px',
        fontWeight: 600,
        color: 'var(--text-secondary)',
      }}>{value}</div>
    </div>
  );
}
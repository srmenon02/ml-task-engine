import { Link } from 'react-router-dom';
import JobStatusBadge from './JobStatusBadge';

export default function JobCard({ job, index = 0 }) {
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

  const priorityColor = (p) => {
    if (p >= 15) return 'var(--red-signal)';
    if (p >= 10) return 'var(--amber)';
    if (p >= 5) return 'var(--blue-signal)';
    return 'var(--text-muted)';
  };

  return (
    <Link
      to={`/jobs/${job.id}`}
      style={{
        display: 'block',
        textDecoration: 'none',
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border-dim)',
        borderRadius: 'var(--radius-md)',
        padding: '14px 18px',
        transition: 'all 0.15s ease',
        position: 'relative',
        overflow: 'hidden',
        animation: 'fadeSlideUp 0.4s ease forwards',
        animationDelay: `${index * 0.04}s`,
        opacity: 0,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.backgroundColor = 'var(--bg-raised)';
        e.currentTarget.style.borderColor = 'var(--border-mid)';
        e.currentTarget.style.transform = 'translateY(-1px)';
        e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
        e.currentTarget.style.borderColor = 'var(--border-dim)';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      {/* Ambient left border accent for running */}
      {job.status === 'running' && (
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: '2px',
          backgroundColor: 'var(--blue-signal)',
          boxShadow: '0 0 8px var(--blue-signal)',
        }} />
      )}

      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        {/* Left info */}
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            {/* Job ID */}
            <span style={{
              fontFamily: 'var(--font-body)',
              fontSize: '11px',
              color: 'var(--text-muted)',
              letterSpacing: '0.06em',
            }}>
              #{String(job.id).padStart(4, '0')}
            </span>
            {/* Job type */}
            <span style={{
              fontFamily: 'var(--font-display)',
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--text-primary)',
              letterSpacing: '-0.01em',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {job.job_type.replace(/_/g, ' ')}
            </span>
          </div>

          {/* Config preview */}
          {job.config && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {job.config.n_estimators && (
                <ConfigPill label="n_est" value={job.config.n_estimators} />
              )}
              {job.config.dataset_rows && (
                <ConfigPill label="rows" value={job.config.dataset_rows.toLocaleString()} />
              )}
              {job.config.model && (
                <ConfigPill label="model" value={job.config.model} />
              )}
            </div>
          )}
        </div>

        {/* Right meta */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px', flexShrink: 0 }}>
          <JobStatusBadge status={job.status} size="small" />
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            {/* Priority */}
            <div style={{ textAlign: 'right' }}>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: '10px',
                color: 'var(--text-faint)',
                letterSpacing: '0.06em',
                marginBottom: '2px',
              }}>PRIORITY</div>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: '13px',
                fontWeight: 600,
                color: priorityColor(job.priority),
              }}>{job.priority}</div>
            </div>
            {/* Created */}
            <div style={{ textAlign: 'right' }}>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: '10px',
                color: 'var(--text-faint)',
                letterSpacing: '0.06em',
                marginBottom: '2px',
              }}>CREATED</div>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: '11px',
                color: 'var(--text-secondary)',
              }}>{formatDate(job.created_at)}</div>
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
}

function ConfigPill({ label, value }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '4px',
      padding: '2px 7px',
      borderRadius: '2px',
      backgroundColor: 'var(--bg-elevated)',
      border: '1px solid var(--border-dim)',
      fontFamily: 'var(--font-body)',
      fontSize: '10px',
    }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{value}</span>
    </span>
  );
}
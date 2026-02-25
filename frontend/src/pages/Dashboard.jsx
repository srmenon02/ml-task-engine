import { useQuery } from '@tanstack/react-query';
import { fetchSystemStats } from '../api/jobs';

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['systemStats'],
    queryFn: fetchSystemStats,
    refetchInterval: 5000,
  });

  return (
    <div style={{ padding: '40px 0' }}>
      {/* Header */}
      <div style={{
        marginBottom: '40px',
        animation: 'fadeSlideUp 0.5s ease forwards',
      }}>
        <div style={{
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          color: 'var(--amber)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          marginBottom: '8px',
        }}>
          System Overview
        </div>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '32px',
          fontWeight: 800,
          color: 'var(--text-primary)',
          letterSpacing: '-0.03em',
          lineHeight: 1.1,
        }}>
          Execution Dashboard
        </h1>
        <div style={{
          fontFamily: 'var(--font-editorial)',
          fontStyle: 'italic',
          fontSize: '15px',
          color: 'var(--text-muted)',
          marginTop: '6px',
          fontWeight: 300,
        }}>
          ML-based resource prediction & distributed task orchestration
        </div>
      </div>

      {isLoading && <SkeletonDashboard />}
      {error && <ErrorState message="Failed to fetch system statistics" />}

      {data && (
        <>
          {/* Primary metrics row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '12px',
            marginBottom: '12px',
          }}>
            <BigMetric
              label="Total Jobs"
              value={data.jobs.total}
              subtext="all time"
              delay={0.05}
            />
            <BigMetric
              label="Success Rate"
              value={`${data.jobs.success_rate.toFixed(1)}%`}
              subtext="completed / total"
              accent="var(--green-signal)"
              glowColor="rgba(34,197,94,0.1)"
              delay={0.1}
            />
          </div>

          {/* Secondary metrics row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '12px',
            marginBottom: '32px',
          }}>
            <SmallMetric label="Completed" value={data.jobs.completed} color="var(--green-signal)" delay={0.15} />
            <SmallMetric label="Running" value={data.jobs.running} color="var(--blue-signal)" pulse delay={0.18} />
            <SmallMetric label="Pending" value={data.jobs.pending} color="var(--yellow-signal)" delay={0.21} />
            <SmallMetric label="Failed" value={data.jobs.failed} color="var(--red-signal)" delay={0.24} />
          </div>

          {/* Worker status section */}
          <div style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-dim)',
            borderRadius: 'var(--radius-lg)',
            padding: '24px',
            animation: 'fadeSlideUp 0.5s ease forwards',
            animationDelay: '0.3s',
            opacity: 0,
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '20px',
            }}>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontSize: '14px',
                fontWeight: 700,
                color: 'var(--text-primary)',
                letterSpacing: '-0.01em',
              }}>Worker Infrastructure</div>
              <span style={{
                fontFamily: 'var(--font-body)',
                fontSize: '10px',
                color: 'var(--text-muted)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}>Live Status</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
              <WorkerStat label="Total Workers" value={data.workers.total} />
              <WorkerStat label="Active" value={data.workers.active} accent="var(--green-signal)" />
              <WorkerStat label="Stale" value={data.workers.stale} accent={data.workers.stale > 0 ? 'var(--red-signal)' : 'var(--text-muted)'} />
            </div>

            {/* Visual worker capacity bar */}
            <div style={{ marginTop: '20px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontFamily: 'var(--font-body)',
                fontSize: '10px',
                color: 'var(--text-muted)',
                letterSpacing: '0.08em',
                marginBottom: '6px',
              }}>
                <span>WORKER UTILIZATION</span>
                <span>{data.workers.total > 0 ? Math.round(data.workers.active / data.workers.total * 100) : 0}%</span>
              </div>
              <div style={{
                height: '4px',
                backgroundColor: 'var(--bg-elevated)',
                borderRadius: '2px',
                overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%',
                  width: data.workers.total > 0 ? `${data.workers.active / data.workers.total * 100}%` : '0%',
                  backgroundColor: 'var(--green-signal)',
                  borderRadius: '2px',
                  boxShadow: '0 0 8px var(--green-signal)',
                  transition: 'width 0.8s ease',
                }} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function BigMetric({ label, value, subtext, accent = 'var(--amber)', glowColor = 'var(--amber-glow)', delay = 0 }) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-surface)',
      border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-lg)',
      padding: '28px',
      position: 'relative',
      overflow: 'hidden',
      animation: 'fadeSlideUp 0.5s ease forwards',
      animationDelay: `${delay}s`,
      opacity: 0,
    }}>
      {/* Background glow */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        right: 0,
        width: '120px',
        height: '120px',
        borderRadius: '50%',
        background: glowColor,
        filter: 'blur(40px)',
        pointerEvents: 'none',
      }} />
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '10px',
        color: 'var(--text-muted)',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        marginBottom: '12px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'var(--font-display)',
        fontSize: '48px',
        fontWeight: 800,
        color: accent,
        lineHeight: 1,
        letterSpacing: '-0.04em',
      }}>{value}</div>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '11px',
        color: 'var(--text-faint)',
        marginTop: '8px',
      }}>{subtext}</div>
    </div>
  );
}

function SmallMetric({ label, color = 'var(--text-primary)', value, pulse = false, delay = 0 }) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-surface)',
      border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-md)',
      padding: '16px',
      animation: 'fadeSlideUp 0.5s ease forwards',
      animationDelay: `${delay}s`,
      opacity: 0,
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginBottom: '8px',
      }}>
        <div style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          backgroundColor: color,
          flexShrink: 0,
          ...(pulse ? { animation: 'statusPulse 2s ease infinite' } : {}),
        }} />
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '10px',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>{label}</span>
      </div>
      <div style={{
        fontFamily: 'var(--font-display)',
        fontSize: '28px',
        fontWeight: 700,
        color: color,
        letterSpacing: '-0.03em',
        lineHeight: 1,
      }}>{value}</div>
    </div>
  );
}

function WorkerStat({ label, value, accent = 'var(--text-secondary)' }) {
  return (
    <div>
      <div style={{
        fontFamily: 'var(--font-body)',
        fontSize: '10px',
        color: 'var(--text-faint)',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        marginBottom: '4px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'var(--font-display)',
        fontSize: '22px',
        fontWeight: 700,
        color: accent,
        letterSpacing: '-0.02em',
      }}>{value}</div>
    </div>
  );
}

function SkeletonDashboard() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {[1, 2, 3].map(i => (
        <div key={i} style={{
          height: '80px',
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-dim)',
          borderRadius: 'var(--radius-md)',
          animation: 'pulse-amber 1.5s ease infinite',
        }} />
      ))}
    </div>
  );
}

function ErrorState({ message }) {
  return (
    <div style={{
      backgroundColor: 'var(--red-dim)',
      border: '1px solid rgba(239,68,68,0.3)',
      borderRadius: 'var(--radius-md)',
      padding: '20px',
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      fontFamily: 'var(--font-body)',
      fontSize: '12px',
      color: 'var(--red-signal)',
    }}>
      <span style={{ fontSize: '16px' }}>⚠</span>
      {message}
    </div>
  );
}
export default function JobStatusBadge({ status, size = 'normal' }) {
  const statusConfig = {
    pending: {
      color: 'var(--yellow-signal)',
      bg: 'var(--yellow-dim)',
      border: 'rgba(234, 179, 8, 0.3)',
      label: 'PENDING',
      dot: true,
    },
    running: {
      color: 'var(--blue-signal)',
      bg: 'var(--blue-dim)',
      border: 'rgba(59, 130, 246, 0.3)',
      label: 'RUNNING',
      dot: true,
      pulse: true,
    },
    completed: {
      color: 'var(--green-signal)',
      bg: 'var(--green-dim)',
      border: 'rgba(34, 197, 94, 0.3)',
      label: 'COMPLETED',
      dot: false,
    },
    failed: {
      color: 'var(--red-signal)',
      bg: 'var(--red-dim)',
      border: 'rgba(239, 68, 68, 0.3)',
      label: 'FAILED',
      dot: false,
    },
    timeout: {
      color: '#f97316',
      bg: 'rgba(249, 115, 22, 0.12)',
      border: 'rgba(249, 115, 22, 0.3)',
      label: 'TIMEOUT',
      dot: false,
    },
    retrying: {
      color: 'var(--purple-signal)',
      bg: 'var(--purple-dim)',
      border: 'rgba(168, 85, 247, 0.3)',
      label: 'RETRYING',
      dot: true,
      pulse: true,
    },
    cancelled: {
      color: 'var(--text-muted)',
      bg: 'rgba(90, 90, 112, 0.15)',
      border: 'rgba(90, 90, 112, 0.25)',
      label: 'CANCELLED',
      dot: false,
    },
  };

  const cfg = statusConfig[status?.toLowerCase()] || statusConfig.pending;
  const isSmall = size === 'small';

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: isSmall ? '4px' : '6px',
      padding: isSmall ? '2px 6px' : '3px 9px',
      borderRadius: 'var(--radius-sm)',
      backgroundColor: cfg.bg,
      border: `1px solid ${cfg.border}`,
      fontFamily: 'var(--font-body)',
      fontSize: isSmall ? '10px' : '11px',
      fontWeight: 600,
      letterSpacing: '0.08em',
      color: cfg.color,
    }}>
      {cfg.dot && (
        <span style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          backgroundColor: cfg.color,
          flexShrink: 0,
          ...(cfg.pulse ? {
            animation: 'statusPulse 2s ease infinite',
          } : {}),
        }} />
      )}
      {!cfg.dot && (
        <span style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          backgroundColor: cfg.color,
          opacity: 0.7,
          flexShrink: 0,
        }} />
      )}
      {cfg.label}
    </span>
  );
}
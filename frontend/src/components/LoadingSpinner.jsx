export default function LoadingSpinner({ size = 'medium', text = 'Loading...' }) {
  const sizes = {
    small: 20,
    medium: 32,
    large: 48,
  };

  const dim = sizes[size] || sizes.medium;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px',
      padding: '40px',
    }}>
      {/* Terminal-style spinner */}
      <div style={{ position: 'relative', width: dim, height: dim }}>
        {/* Outer ring */}
        <div style={{
          position: 'absolute',
          inset: 0,
          borderRadius: '50%',
          border: `1px solid var(--border-dim)`,
        }} />
        {/* Spinning arc */}
        <div style={{
          position: 'absolute',
          inset: 0,
          borderRadius: '50%',
          border: `2px solid transparent`,
          borderTopColor: 'var(--amber)',
          borderRightColor: 'rgba(245,158,11,0.3)',
          animation: 'spin 0.8s linear infinite',
        }} />
        {/* Center dot */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '4px',
          height: '4px',
          borderRadius: '50%',
          backgroundColor: 'var(--amber)',
          boxShadow: '0 0 6px var(--amber)',
        }} />
      </div>

      {text && (
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          color: 'var(--text-muted)',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
        }}>
          {text}
        </span>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
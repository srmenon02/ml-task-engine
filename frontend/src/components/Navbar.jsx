import { Link, useLocation } from 'react-router-dom';
import { UserButton, SignedIn, SignedOut, SignInButton } from '@clerk/clerk-react';

export default function Navbar() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Overview' },
    { path: '/jobs', label: 'Jobs' },
    { path: '/jobs/new', label: 'New Job' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      backgroundColor: 'rgba(10, 10, 11, 0.92)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border-dim)',
    }}>
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '1px',
        background: 'linear-gradient(90deg, transparent, var(--amber), transparent)',
        opacity: 0.6,
      }} />

      <div style={{
        maxWidth: '1100px',
        margin: '0 auto',
        padding: '0 24px',
        height: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        {/* Logo */}
        <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '28px',
            height: '28px',
            background: 'var(--amber-glow)',
            border: '1px solid var(--amber)',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 12px var(--amber-glow)',
          }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M2 7L5 4L9 8L12 5" stroke="var(--amber)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="7" cy="7" r="1.5" fill="var(--amber)" opacity="0.6"/>
            </svg>
          </div>
          <div>
            <span style={{
              fontFamily: 'var(--font-display)',
              fontSize: '15px',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.02em',
            }}>ML Task Engine</span>
            <span style={{
              fontFamily: 'var(--font-body)',
              fontSize: '9px',
              color: 'var(--amber)',
              marginLeft: '8px',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              opacity: 0.8,
            }}>v2.0</span>
          </div>
        </Link>

        {/* Nav items */}
        <SignedIn>
          <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                style={{
                  textDecoration: 'none',
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-sm)',
                  fontFamily: 'var(--font-body)',
                  fontSize: '12px',
                  fontWeight: isActive(item.path) ? 600 : 400,
                  letterSpacing: '0.04em',
                  color: isActive(item.path) ? 'var(--amber)' : 'var(--text-secondary)',
                  backgroundColor: isActive(item.path) ? 'var(--amber-glow)' : 'transparent',
                  border: isActive(item.path) ? '1px solid rgba(245,158,11,0.3)' : '1px solid transparent',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={e => {
                  if (!isActive(item.path)) {
                    e.currentTarget.style.color = 'var(--text-primary)';
                    e.currentTarget.style.backgroundColor = 'var(--bg-raised)';
                    e.currentTarget.style.borderColor = 'var(--border-mid)';
                  }
                }}
                onMouseLeave={e => {
                  if (!isActive(item.path)) {
                    e.currentTarget.style.color = 'var(--text-secondary)';
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.borderColor = 'transparent';
                  }
                }}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </SignedIn>

        {/* Right side */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <SignedIn>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontFamily: 'var(--font-body)',
              fontSize: '11px',
              color: 'var(--text-muted)',
              marginRight: '4px',
            }}>
              <div style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: 'var(--green-signal)',
                boxShadow: '0 0 6px var(--green-signal)',
                animation: 'statusPulse 2s ease infinite',
              }} />
              <span style={{ letterSpacing: '0.06em' }}>OPERATIONAL</span>
            </div>
            <UserButton
              appearance={{
                elements: {
                  avatarBox: {
                    width: '28px',
                    height: '28px',
                  }
                }
              }}
              afterSignOutUrl="/"
            />
          </SignedIn>

          <SignedOut>
            <SignInButton mode="modal">
              <button style={{
                padding: '6px 14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-mid)',
                backgroundColor: 'var(--bg-raised)',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-body)',
                fontSize: '12px',
                letterSpacing: '0.04em',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.color = 'var(--text-primary)';
                e.currentTarget.style.borderColor = 'var(--border-bright)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.color = 'var(--text-secondary)';
                e.currentTarget.style.borderColor = 'var(--border-mid)';
              }}
              >
                Sign In
              </button>
            </SignInButton>
          </SignedOut>
        </div>
      </div>
    </nav>
  );
}
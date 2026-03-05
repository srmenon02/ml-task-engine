import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createJob } from "../api/jobs";
import { useNavigate } from "react-router-dom";

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

  const complexityScore = Math.min(
    100,
    Math.round((Number(formData.n_estimators) / 1000 + Number(formData.dataset_rows) / 100000) * 50)
  );

  const complexityColor =
    complexityScore > 70 ? 'var(--red-signal)' :
    complexityScore > 40 ? 'var(--amber)' :
    'var(--green-signal)';

  return (
    <div style={{ padding: '40px 0', maxWidth: '520px' }}>
      {/* Header */}
      <div style={{
        marginBottom: '36px',
        animation: 'fadeSlideUp 0.5s ease forwards',
      }}>
        <div style={{
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          color: 'var(--amber)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          marginBottom: '6px',
        }}>Job Submission</div>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '28px',
          fontWeight: 800,
          color: 'var(--text-primary)',
          letterSpacing: '-0.03em',
        }}>New Training Job</h1>
        <p style={{
          fontFamily: 'var(--font-editorial)',
          fontStyle: 'italic',
          fontSize: '14px',
          color: 'var(--text-muted)',
          marginTop: '6px',
          fontWeight: 300,
        }}>
          Configure a RandomForest training run on synthetic classification data.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Job type display (read-only) */}
        <div style={{
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-dim)',
          borderRadius: 'var(--radius-md)',
          padding: '16px',
          marginBottom: '16px',
          animation: 'fadeSlideUp 0.5s ease forwards',
          animationDelay: '0.06s',
          opacity: 0,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <FieldLabel>Job Type</FieldLabel>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--text-primary)',
                marginTop: '4px',
              }}>train_sklearn_model</div>
            </div>
            <div style={{
              padding: '4px 10px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'var(--amber-glow)',
              border: '1px solid rgba(245,158,11,0.3)',
              fontFamily: 'var(--font-body)',
              fontSize: '10px',
              color: 'var(--amber)',
              letterSpacing: '0.08em',
            }}>RandomForest</div>
          </div>
        </div>

        {/* Fields */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
          marginBottom: '16px',
        }}>
          <NumberField
            label="n_estimators"
            description="Number of decision trees"
            value={formData.n_estimators}
            min={1} max={100000}
            delay={0.1}
            onChange={(v) => setFormData({ ...formData, n_estimators: v })}
          />
          <NumberField
            label="dataset_rows"
            description="Synthetic training samples"
            value={formData.dataset_rows}
            min={1} max={1000000}
            delay={0.14}
            onChange={(v) => setFormData({ ...formData, dataset_rows: v })}
          />
          <NumberField
            label="priority"
            description="Queue priority (0 = low, 20 = urgent)"
            value={formData.priority}
            min={0} max={20}
            delay={0.18}
            onChange={(v) => setFormData({ ...formData, priority: v })}
          />
        </div>

        {/* Complexity estimate */}
        <div style={{
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-dim)',
          borderRadius: 'var(--radius-md)',
          padding: '16px',
          marginBottom: '20px',
          animation: 'fadeSlideUp 0.5s ease forwards',
          animationDelay: '0.24s',
          opacity: 0,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <FieldLabel>Estimated Complexity</FieldLabel>
            <span style={{
              fontFamily: 'var(--font-body)',
              fontSize: '12px',
              fontWeight: 600,
              color: complexityColor,
            }}>{complexityScore}/100</span>
          </div>
          <div style={{
            height: '3px',
            backgroundColor: 'var(--bg-elevated)',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              height: '100%',
              width: `${complexityScore}%`,
              backgroundColor: complexityColor,
              borderRadius: '2px',
              boxShadow: `0 0 8px ${complexityColor}`,
              transition: 'width 0.3s ease, background-color 0.3s ease',
            }} />
          </div>
        </div>

        {/* Error */}
        {mutation.isError && (
          <div style={{
            backgroundColor: 'var(--red-dim)',
            border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 'var(--radius-sm)',
            padding: '12px 14px',
            marginBottom: '16px',
            fontFamily: 'var(--font-body)',
            fontSize: '11px',
            color: 'var(--red-signal)',
          }}>
            {mutation.error?.response?.data?.detail || "Submission failed"}
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={mutation.isPending}
          style={{
            width: '100%',
            padding: '12px',
            borderRadius: 'var(--radius-md)',
            border: mutation.isPending ? '1px solid var(--border-dim)' : '1px solid var(--amber)',
            backgroundColor: mutation.isPending ? 'var(--bg-raised)' : 'var(--amber-glow)',
            color: mutation.isPending ? 'var(--text-muted)' : 'var(--amber)',
            fontFamily: 'var(--font-display)',
            fontSize: '13px',
            fontWeight: 700,
            letterSpacing: '0.04em',
            cursor: mutation.isPending ? 'not-allowed' : 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: mutation.isPending ? 'none' : '0 0 16px var(--amber-glow)',
            animation: 'fadeSlideUp 0.5s ease forwards',
            animationDelay: '0.28s',
            opacity: 0,
          }}
          onMouseEnter={e => {
            if (!mutation.isPending) {
              e.currentTarget.style.backgroundColor = 'rgba(245,158,11,0.22)';
              e.currentTarget.style.boxShadow = '0 0 24px rgba(245,158,11,0.3)';
            }
          }}
          onMouseLeave={e => {
            if (!mutation.isPending) {
              e.currentTarget.style.backgroundColor = 'var(--amber-glow)';
              e.currentTarget.style.boxShadow = '0 0 16px var(--amber-glow)';
            }
          }}
        >
          {mutation.isPending ? '⟳  Submitting...' : '↗  Submit Job'}
        </button>
      </form>
    </div>
  );
}

function FieldLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-body)',
      fontSize: '10px',
      color: 'var(--text-muted)',
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
    }}>
      {children}
    </div>
  );
}

function NumberField({ label, description, value, min, max, onChange, delay = 0 }) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-surface)',
      border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-md)',
      padding: '16px',
      transition: 'border-color 0.15s ease',
      animation: 'fadeSlideUp 0.5s ease forwards',
      animationDelay: `${delay}s`,
      opacity: 0,
    }}
    onFocusCapture={e => e.currentTarget.style.borderColor = 'var(--border-bright)'}
    onBlurCapture={e => e.currentTarget.style.borderColor = 'var(--border-dim)'}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
        <div>
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '12px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            marginBottom: '2px',
          }}>
            {label}
          </div>
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '10px',
            color: 'var(--text-muted)',
          }}>
            {description}
          </div>
        </div>
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '10px',
          color: 'var(--text-faint)',
        }}>
          {min}–{max.toLocaleString()}
        </span>
      </div>

      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%',
          backgroundColor: 'var(--bg-elevated)',
          border: '1px solid var(--border-dim)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 12px',
          fontFamily: 'var(--font-body)',
          fontSize: '14px',
          fontWeight: 600,
          color: 'var(--text-primary)',
          outline: 'none',
          transition: 'border-color 0.15s ease',
        }}
        onFocus={e => e.target.style.borderColor = 'var(--amber)'}
        onBlur={e => e.target.style.borderColor = 'var(--border-dim)'}
      />
    </div>
  );
}
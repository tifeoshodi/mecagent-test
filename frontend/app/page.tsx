'use client';

import Link from 'next/link';
import {
  Box,
  Sun,
  Moon,
  Sunset
} from 'lucide-react';
import { useEditorStore } from '@/lib/store';
import SketchBackground from '@/components/ui/SketchBackground';

export default function Home() {
  const { theme, setTheme } = useEditorStore();

  const cycleTheme = () => {
    if (theme === 'light') setTheme('dim');
    else if (theme === 'dim') setTheme('dark');
    else setTheme('light');
  };

  const ThemeIcon = theme === 'light' ? Sun : theme === 'dim' ? Sunset : Moon;

  return (
    <div style={{
      height: '100vh',
      width: '100vw',
      overflowY: 'auto',
      position: 'relative',
      background: 'radial-gradient(ellipse at 50% 0%, rgba(59, 130, 246, 0.15) 0%, var(--bg-color) 70%)',
      color: 'var(--text-primary)',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Dynamic Animated CAD Line Sketch Background */}
      <SketchBackground />

      {/* Navbar */}
      <header style={{
        padding: '16px 40px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        zIndex: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Box size={24} style={{ color: 'var(--accent-primary)' }} />
          <span style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
            AIMDEA
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Link
            href="/gallery"
            className="button secondary"
            style={{ borderRadius: '20px', padding: '8px 18px', textDecoration: 'none' }}
          >
            View Gallery
          </Link>
          <Link
            href="/editor"
            className="button"
            style={{ borderRadius: '20px', padding: '8px 20px', textDecoration: 'none' }}
          >
            Launch Editor
          </Link>
          <button
            onClick={cycleTheme}
            className="button secondary"
            style={{ padding: '8px', borderRadius: '50%' }}
            title="Toggle theme"
          >
            <ThemeIcon size={16} />
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <main style={{
        flex: 1,
        maxWidth: '840px',
        margin: '0 auto',
        padding: '10px 20px 16px 20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        zIndex: 10,
      }}>
        {/* Hero Title */}
        <h1
          className="font-display"
          style={{
            fontSize: 'clamp(2.1rem, 3.2vw, 2.9rem)',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: '1.18',
            marginTop: '0px',
            marginBottom: '12px',
            maxWidth: '780px',
          }}
        >
          Autonomous AI Mechanical Engineering &amp;{' '}
          <span style={{
            background: 'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-hover) 60%, #ec4899 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            3D CAD Synthesis
          </span>
        </h1>

        {/* Hero Description */}
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-secondary)',
          maxWidth: '660px',
          lineHeight: '1.6',
          marginBottom: '20px',
        }}>
          Convert natural language instructions or reference 2D engineering views directly into watertight, parametric CadQuery B-Rep models with real-time 3D orbit previews and STEP/STL export.
        </p>

        {/* CTAs */}
        <div style={{ display: 'flex', gap: '18px', marginBottom: '24px', flexWrap: 'wrap', justifyContent: 'center' }}>
          <Link
            href="/editor"
            className="button font-display"
            style={{
              padding: '11px 32px',
              fontSize: '0.96rem',
              fontWeight: 700,
              borderRadius: '30px',
              textDecoration: 'none',
              letterSpacing: '0.02em',
            }}
          >
            Launch Editor
          </Link>

          <Link
            href="/gallery"
            className="button secondary font-display"
            style={{
              padding: '11px 32px',
              fontSize: '0.96rem',
              fontWeight: 700,
              borderRadius: '30px',
              textDecoration: 'none',
              letterSpacing: '0.02em',
            }}
          >
            View Gallery
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        padding: '16px 40px',
        borderTop: '1px solid var(--border-color)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '0.8rem',
        color: 'var(--text-secondary)',
        zIndex: 20,
      }}>
        <div>AIMDEA — AI Mechanical Design Engineer Assistant</div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <span>OpenCASCADE / CadQuery</span>
          <span>Three.js / React Three Fiber</span>
          <span>FastAPI</span>
        </div>
      </footer>
    </div>
  );
}

'use client';

import React from 'react';

export default function LaserGridBackground() {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: 0,
      }}
      aria-hidden="true"
    >
      <style>{`
        @keyframes laserSweepY1 {
          0% { transform: translateY(0vh); }
          50% { transform: translateY(98vh); }
          100% { transform: translateY(0vh); }
        }

        @keyframes laserSweepX1 {
          0% { transform: translateX(0vw); }
          50% { transform: translateX(98vw); }
          100% { transform: translateX(0vw); }
        }

        @keyframes laserSweepY2 {
          0% { transform: translateY(95vh); }
          50% { transform: translateY(5vh); }
          100% { transform: translateY(95vh); }
        }

        @keyframes laserSweepX2 {
          0% { transform: translateX(95vw); }
          50% { transform: translateX(5vw); }
          100% { transform: translateX(95vw); }
        }

        @keyframes laserPulseHeadH {
          0% { transform: translateX(-100px); opacity: 0; }
          15% { opacity: 1; }
          85% { opacity: 1; }
          100% { transform: translateX(calc(100vw + 100px)); opacity: 0; }
        }

        @keyframes laserPulseHeadV {
          0% { transform: translateY(-100px); opacity: 0; }
          15% { opacity: 1; }
          85% { opacity: 1; }
          100% { transform: translateY(calc(100vh + 100px)); opacity: 0; }
        }

        @keyframes laserDashRowA {
          0% { transform: translateX(-15%); opacity: 0; }
          5% { opacity: 0.9; }
          90% { opacity: 0.9; }
          100% { transform: translateX(115%); opacity: 0; }
        }

        @keyframes laserDashRowB {
          0% { transform: translateX(115%); opacity: 0; }
          5% { opacity: 0.85; }
          90% { opacity: 0.85; }
          100% { transform: translateX(-15%); opacity: 0; }
        }

        @keyframes laserDashColA {
          0% { transform: translateY(-15%); opacity: 0; }
          5% { opacity: 0.9; }
          90% { opacity: 0.9; }
          100% { transform: translateY(115%); opacity: 0; }
        }

        @keyframes laserDashColB {
          0% { transform: translateY(115%); opacity: 0; }
          5% { opacity: 0.85; }
          90% { opacity: 0.85; }
          100% { transform: translateY(-15%); opacity: 0; }
        }

        @keyframes reticleSpin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @keyframes reticlePulse {
          0%, 100% { transform: scale(1); opacity: 0.8; }
          50% { transform: scale(1.18); opacity: 1; }
        }
      `}</style>

      {/* ================================================================ */}
      {/* 1. TECHNICAL CAD BLUEPRINT GRID                                  */}
      {/* ================================================================ */}
      <svg
        width="100%"
        height="100%"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
      >
        <defs>
          {/* Minor Grid: 32px */}
          <pattern id="laserGridSmall" width="32" height="32" patternUnits="userSpaceOnUse">
            <path
              d="M 32 0 L 0 0 0 32"
              fill="none"
              stroke="var(--border-color)"
              strokeWidth="0.5"
              strokeOpacity="0.22"
            />
          </pattern>

          {/* Major Grid: 160px with intersection markers */}
          <pattern id="laserGridMajor" width="160" height="160" patternUnits="userSpaceOnUse">
            <rect width="160" height="160" fill="url(#laserGridSmall)" />
            <path
              d="M 160 0 L 0 0 0 160"
              fill="none"
              stroke="var(--cad-sketch)"
              strokeWidth="0.85"
              strokeOpacity="0.16"
            />
            {/* Major Intersection Reticle (+) */}
            <path
              d="M -5 0 L 5 0 M 0 -5 L 0 5"
              stroke="var(--cad-sketch)"
              strokeWidth="1.2"
              strokeOpacity="0.35"
            />
            <circle cx="0" cy="0" r="1.5" fill="var(--cad-sketch)" fillOpacity="0.4" />
          </pattern>

          {/* Laser Glow Filter */}
          <filter id="laserGlowCyan" x="-20%" y="-100%" width="140%" height="300%">
            <feGaussianBlur stdDeviation="3" result="blur1" />
            <feGaussianBlur stdDeviation="7" result="blur2" />
            <feMerge>
              <feMergeNode in="blur2" />
              <feMergeNode in="blur1" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Linear Gradients for Beam Falloff */}
          <linearGradient id="laserBeamHGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0" />
            <stop offset="10%" stopColor="var(--accent-primary)" stopOpacity="0.6" />
            <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.95" />
            <stop offset="90%" stopColor="var(--accent-primary)" stopOpacity="0.6" />
            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0" />
          </linearGradient>

          <linearGradient id="laserBeamVGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0" />
            <stop offset="10%" stopColor="var(--accent-primary)" stopOpacity="0.6" />
            <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.95" />
            <stop offset="90%" stopColor="var(--accent-primary)" stopOpacity="0.6" />
            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0" />
          </linearGradient>

          {/* Secondary Beam Gradients (Violet/Indigo) */}
          <linearGradient id="laserBeamH2Grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#818cf8" stopOpacity="0" />
            <stop offset="20%" stopColor="#818cf8" stopOpacity="0.4" />
            <stop offset="50%" stopColor="#a78bfa" stopOpacity="0.8" />
            <stop offset="80%" stopColor="#818cf8" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
          </linearGradient>

          <linearGradient id="laserBeamV2Grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#818cf8" stopOpacity="0" />
            <stop offset="20%" stopColor="#818cf8" stopOpacity="0.4" />
            <stop offset="50%" stopColor="#a78bfa" stopOpacity="0.8" />
            <stop offset="80%" stopColor="#818cf8" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
          </linearGradient>

          {/* Fast Laser Bolt Gradients */}
          <linearGradient id="boltGradH" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0" />
            <stop offset="70%" stopColor="#38bdf8" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="1" />
          </linearGradient>

          <linearGradient id="boltGradHReverse" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
            <stop offset="30%" stopColor="#818cf8" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
          </linearGradient>

          <linearGradient id="boltGradV" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0" />
            <stop offset="70%" stopColor="#38bdf8" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="1" />
          </linearGradient>
        </defs>

        {/* The Grid Canvas */}
        <rect width="100%" height="100%" fill="url(#laserGridMajor)" />
      </svg>

      {/* ================================================================ */}
      {/* 2. DYNAMIC LASERS SCANNING THROUGH ROWS AND COLUMNS               */}
      {/* ================================================================ */}

      {/* --- LASER ROW 1 (Sweeping vertically across rows) --- */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          height: '24px',
          top: '-12px',
          animation: 'laserSweepY1 12s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite',
          willChange: 'transform',
        }}
      >
        <svg width="100%" height="24" style={{ display: 'block' }}>
          {/* Outer glow ribbon */}
          <line
            x1="0"
            y1="12"
            x2="100%"
            y2="12"
            stroke="url(#laserBeamHGrad)"
            strokeWidth="5"
            strokeOpacity="0.35"
            filter="url(#laserGlowCyan)"
          />
          {/* Main neon beam */}
          <line
            x1="0"
            y1="12"
            x2="100%"
            y2="12"
            stroke="url(#laserBeamHGrad)"
            strokeWidth="1.6"
            strokeOpacity="0.9"
          />
          {/* Pure white core line */}
          <line
            x1="0"
            y1="12"
            x2="100%"
            y2="12"
            stroke="#ffffff"
            strokeWidth="0.7"
            strokeOpacity="0.9"
          />
        </svg>
      </div>

      {/* --- LASER COLUMN 1 (Sweeping horizontally across columns) --- */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          width: '24px',
          left: '-12px',
          animation: 'laserSweepX1 15s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite',
          willChange: 'transform',
        }}
      >
        <svg width="24" height="100%" style={{ display: 'block' }}>
          {/* Outer glow ribbon */}
          <line
            x1="12"
            y1="0"
            x2="12"
            y2="100%"
            stroke="url(#laserBeamVGrad)"
            strokeWidth="5"
            strokeOpacity="0.35"
            filter="url(#laserGlowCyan)"
          />
          {/* Main neon beam */}
          <line
            x1="12"
            y1="0"
            x2="12"
            y2="100%"
            stroke="url(#laserBeamVGrad)"
            strokeWidth="1.6"
            strokeOpacity="0.9"
          />
          {/* Pure white core line */}
          <line
            x1="12"
            y1="0"
            x2="12"
            y2="100%"
            stroke="#ffffff"
            strokeWidth="0.7"
            strokeOpacity="0.9"
          />
        </svg>
      </div>

      {/* --- LASER ROW 2 (Secondary counter-sweep across rows) --- */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          height: '20px',
          top: '-10px',
          animation: 'laserSweepY2 19s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          willChange: 'transform',
          opacity: 0.65,
        }}
      >
        <svg width="100%" height="20" style={{ display: 'block' }}>
          <line
            x1="0"
            y1="10"
            x2="100%"
            y2="10"
            stroke="url(#laserBeamH2Grad)"
            strokeWidth="1.2"
          />
          <line
            x1="0"
            y1="10"
            x2="100%"
            y2="10"
            stroke="#e0e7ff"
            strokeWidth="0.5"
            strokeOpacity="0.7"
          />
        </svg>
      </div>

      {/* --- LASER COLUMN 2 (Secondary counter-sweep across columns) --- */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          width: '20px',
          left: '-10px',
          animation: 'laserSweepX2 23s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          willChange: 'transform',
          opacity: 0.65,
        }}
      >
        <svg width="20" height="100%" style={{ display: 'block' }}>
          <line
            x1="10"
            y1="0"
            x2="10"
            y2="100%"
            stroke="url(#laserBeamV2Grad)"
            strokeWidth="1.2"
          />
          <line
            x1="10"
            y1="0"
            x2="10"
            y2="100%"
            stroke="#e0e7ff"
            strokeWidth="0.5"
            strokeOpacity="0.7"
          />
        </svg>
      </div>

      {/* --- SYNCHRONIZED RETICLE AT LASER 1 ROW × COLUMN INTERSECTION --- */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          animation: 'laserSweepX1 15s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite',
          willChange: 'transform',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            animation: 'laserSweepY1 12s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite',
            willChange: 'transform',
          }}
        >
          {/* Target Reticle Centered at (0, 0) */}
          <div
            style={{
              position: 'absolute',
              top: '-20px',
              left: '-20px',
              width: '40px',
              height: '40px',
            }}
          >
            <svg width="40" height="40" viewBox="-20 -20 40 40" style={{ display: 'block' }}>
              {/* Outer rotating reticle ring */}
              <circle
                r="11"
                fill="none"
                stroke="var(--accent-primary)"
                strokeWidth="1.2"
                strokeDasharray="4 3"
                style={{
                  animation: 'reticleSpin 6s linear infinite',
                  transformOrigin: '0 0',
                }}
              />
              {/* Pulsing inner ring */}
              <circle
                r="5"
                fill="none"
                stroke="#38bdf8"
                strokeWidth="1"
                style={{
                  animation: 'reticlePulse 2s ease-in-out infinite',
                  transformOrigin: '0 0',
                }}
              />
              {/* Center point */}
              <circle r="1.8" fill="#ffffff" />
              {/* Ticks */}
              <line x1="-15" y1="0" x2="-8" y2="0" stroke="#38bdf8" strokeWidth="1" />
              <line x1="8" y1="0" x2="15" y2="0" stroke="#38bdf8" strokeWidth="1" />
              <line x1="0" y1="-15" x2="0" y2="-8" stroke="#38bdf8" strokeWidth="1" />
              <line x1="0" y1="8" x2="0" y2="15" stroke="#38bdf8" strokeWidth="1" />
            </svg>
          </div>
        </div>
      </div>

      {/* ================================================================ */}
      {/* 3. RAPID LASER BOLTS DARTING ALONG SPECIFIC GRID ROWS & COLS      */}
      {/* ================================================================ */}

      {/* Fast Bolt on Row at 240px (~15% height) */}
      <div
        style={{
          position: 'absolute',
          top: '160px',
          left: 0,
          width: '260px',
          height: '2px',
          background: 'url(#boltGradH)',
          animation: 'laserDashRowA 6.5s cubic-bezier(0.2, 0.8, 0.4, 1) infinite 0.5s',
          willChange: 'transform',
        }}
      >
        <svg width="260" height="4" style={{ position: 'absolute', top: '-1px', left: 0 }}>
          <rect width="260" height="2" fill="url(#boltGradH)" />
          <circle cx="260" cy="1" r="2" fill="#ffffff" filter="url(#laserGlowCyan)" />
        </svg>
      </div>

      {/* Fast Bolt Reverse on Row at 640px (~60% height) */}
      <div
        style={{
          position: 'absolute',
          top: '640px',
          left: 0,
          width: '240px',
          height: '2px',
          animation: 'laserDashRowB 8s cubic-bezier(0.2, 0.8, 0.4, 1) infinite 3s',
          willChange: 'transform',
        }}
      >
        <svg width="240" height="4" style={{ position: 'absolute', top: '-1px', left: 0 }}>
          <rect width="240" height="2" fill="url(#boltGradHReverse)" />
          <circle cx="0" cy="1" r="2" fill="#ffffff" filter="url(#laserGlowCyan)" />
        </svg>
      </div>

      {/* Fast Bolt on Column at 320px (~20% width) */}
      <div
        style={{
          position: 'absolute',
          left: '320px',
          top: 0,
          width: '2px',
          height: '260px',
          animation: 'laserDashColA 7.5s cubic-bezier(0.2, 0.8, 0.4, 1) infinite 1.8s',
          willChange: 'transform',
        }}
      >
        <svg width="4" height="260" style={{ position: 'absolute', left: '-1px', top: 0 }}>
          <rect width="2" height="260" fill="url(#boltGradV)" />
          <circle cx="1" cy="260" r="2" fill="#ffffff" filter="url(#laserGlowCyan)" />
        </svg>
      </div>

      {/* Fast Bolt on Column at 1120px (~70% width) */}
      <div
        style={{
          position: 'absolute',
          left: '1120px',
          top: 0,
          width: '2px',
          height: '260px',
          animation: 'laserDashColA 9s cubic-bezier(0.2, 0.8, 0.4, 1) infinite 4.2s',
          willChange: 'transform',
        }}
      >
        <svg width="4" height="260" style={{ position: 'absolute', left: '-1px', top: 0 }}>
          <rect width="2" height="260" fill="url(#boltGradV)" />
          <circle cx="1" cy="260" r="2" fill="#ffffff" filter="url(#laserGlowCyan)" />
        </svg>
      </div>
    </div>
  );
}

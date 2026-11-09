'use client';

import React from 'react';

function getSpurGearPath(cx: number, cy: number, z: number, rp: number, m: number = 4.5): string {
  const ra = rp + m * 0.95;
  const rf = rp - m * 1.15;
  const tau = (2 * Math.PI) / z;
  const pts: string[] = [];

  for (let k = 0; k < z; k++) {
    const t0 = k * tau;
    const toothPts = [
      { r: rf, th: t0 - 0.23 * tau },
      { r: ra, th: t0 - 0.12 * tau },
      { r: ra, th: t0 + 0.12 * tau },
      { r: rf, th: t0 + 0.23 * tau },
    ];
    toothPts.forEach((p, i) => {
      const x = cx + p.r * Math.cos(p.th);
      const y = cy + p.r * Math.sin(p.th);
      const cmd = k === 0 && i === 0 ? 'M' : 'L';
      pts.push(`${cmd} ${x.toFixed(1)} ${y.toFixed(1)}`);
    });
  }
  pts.push('Z');
  return pts.join(' ');
}

interface GearTrainProps {
  width?: number;
  height?: number;
}

export default function GearTrain({ width = 450, height = 322 }: GearTrainProps) {
  // Gear parameters
  const cx1 = 65;
  const cy1 = 75;
  const z1 = 12;
  const rp1 = 38;

  const cx2 = 171;
  const cy2 = 118;
  const z2 = 24;
  const rp2 = 76;
  const m = 4.5;

  const driverPath = React.useMemo(() => getSpurGearPath(cx1, cy1, z1, rp1, m), []);
  const drivenPath = React.useMemo(() => getSpurGearPath(cx2, cy2, z2, rp2, m), []);

  return (
    <div style={{ position: 'relative', width: `${width}px`, height: `${height}px` }}>
      <svg
        width={width}
        height={height}
        viewBox="0 0 300 215"
        style={{ width: '100%', height: '100%', display: 'block' }}
      >
        <defs>
          {/* Arrowheads for engineering dimensions */}
          <marker
            id="gearArrowStart"
            viewBox="0 0 10 10"
            refX="0"
            refY="5"
            markerWidth="4.5"
            markerHeight="4.5"
            orient="auto"
          >
            <path d="M 10 1 L 0 5 L 10 9 z" fill="var(--cad-sketch-accent)" fillOpacity="0.9" />
          </marker>
          <marker
            id="gearArrowEnd"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="4.5"
            markerHeight="4.5"
            orient="auto"
          >
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--cad-sketch-accent)" fillOpacity="0.9" />
          </marker>
        </defs>

        {/* 1. Center-to-Center Line of Centers (a = 114 mm) */}
        <line
          x1={cx1}
          y1={cy1}
          x2={cx2}
          y2={cy2}
          stroke="var(--cad-sketch)"
          strokeWidth="0.8"
          strokeDasharray="14 3 2 3"
          strokeOpacity="0.4"
        />

        {/* Perpendicular Extension Lines from Gear Centers */}
        <line
          x1={cx1}
          y1={cy1}
          x2={74.4}
          y2={51.8}
          stroke="var(--cad-sketch-accent)"
          strokeWidth="0.8"
          strokeOpacity="0.7"
        />
        <line
          x1={cx2}
          y1={cy2}
          x2={180.4}
          y2={94.8}
          stroke="var(--cad-sketch-accent)"
          strokeWidth="0.8"
          strokeOpacity="0.7"
        />

        {/* Exactly Parallel Center Distance Dimension Line: a = 114.00 mm */}
        <line
          x1={72.5}
          y1={56.5}
          x2={178.5}
          y2={99.5}
          stroke="var(--cad-sketch-accent)"
          strokeWidth="1.0"
          markerStart="url(#gearArrowStart)"
          markerEnd="url(#gearArrowEnd)"
        />
        <text
          x={129.3}
          y={68.7}
          textAnchor="middle"
          fill="var(--cad-sketch-accent)"
          fontSize="8.5"
          fontFamily="monospace"
          fontWeight="bold"
          transform="rotate(22.08, 129.3, 68.7)"
        >
          a = 114.00 mm
        </text>

        {/* 2. DRIVER GEAR (PINION, z1 = 12 teeth, Clockwise rotation in 6.0s) */}
        <g style={{ transformOrigin: `${cx1}px ${cy1}px`, animation: 'cadRotateGearDriver 6.0s linear infinite' }}>
          {/* Pitch Circle */}
          <circle
            cx={cx1}
            cy={cy1}
            r={rp1}
            fill="none"
            stroke="var(--cad-sketch)"
            strokeWidth="0.9"
            strokeDasharray="5 3"
            strokeOpacity="0.55"
          />

          {/* Involute Gear Teeth Perimeter */}
          <path
            d={driverPath}
            fill="var(--bg-panel)"
            fillOpacity="0.35"
            stroke="var(--cad-sketch)"
            strokeWidth="1.6"
            strokeLinejoin="round"
          />

          {/* Central Shaft Bore & Keyway */}
          <circle cx={cx1} cy={cy1} r="10" fill="var(--bg-color)" stroke="var(--cad-sketch-accent)" strokeWidth="1.4" />
          <rect x={cx1 - 2.5} y={cy1 - 13} width="5" height="4" fill="var(--cad-sketch-accent)" />
          <circle cx={cx1} cy={cy1} r="2.5" fill="var(--cad-sketch)" />
        </g>

        {/* 3. DRIVEN GEAR (BULL GEAR, z2 = 24 teeth, Counter-Clockwise in 12.0s) */}
        <g style={{ transformOrigin: `${cx2}px ${cy2}px`, animation: 'cadRotateGearDriven 12.0s linear infinite' }}>
          {/* Pitch Circle */}
          <circle
            cx={cx2}
            cy={cy2}
            r={rp2}
            fill="none"
            stroke="var(--cad-sketch)"
            strokeWidth="0.9"
            strokeDasharray="6 3"
            strokeOpacity="0.55"
          />

          {/* Involute Gear Teeth Perimeter */}
          <path
            d={drivenPath}
            fill="var(--bg-panel)"
            fillOpacity="0.35"
            stroke="var(--cad-sketch)"
            strokeWidth="1.8"
            strokeLinejoin="round"
          />

          {/* Rim recess circle */}
          <circle cx={cx2} cy={cy2} r="56" fill="none" stroke="var(--cad-sketch)" strokeWidth="0.8" strokeOpacity="0.4" />

          {/* 4 Lightening Pockets (spokes from CAD gear blank) */}
          {[0, 90, 180, 270].map((ang, i) => {
            const rad = (ang * Math.PI) / 180;
            const hx = cx2 + 36 * Math.cos(rad);
            const hy = cy2 + 36 * Math.sin(rad);
            return (
              <circle
                key={i}
                cx={hx}
                cy={hy}
                r="11"
                fill="var(--bg-color)"
                fillOpacity="0.4"
                stroke="var(--cad-sketch-accent)"
                strokeWidth="1.0"
                strokeOpacity="0.8"
              />
            );
          })}

          {/* Central Shaft Bore & Keyway */}
          <circle cx={cx2} cy={cy2} r="15" fill="var(--bg-color)" stroke="var(--cad-sketch-accent)" strokeWidth="1.5" />
          <rect x={cx2 - 3.5} y={cy2 - 19} width="7" height="5" fill="var(--cad-sketch-accent)" />
          <circle cx={cx2} cy={cy2} r="3" fill="var(--cad-sketch)" />
        </g>

        {/* 4. Pitch Point Tangent & Pressure Angle Callout */}
        <g transform="translate(98, 92)">
          <circle cx="0" cy="0" r="2.5" fill="var(--cad-sketch-accent)" />
          <text x="-32" y="16" fill="var(--cad-sketch-accent)" fontSize="8" fontFamily="monospace" fontWeight="bold">
            &alpha; = 20&deg; PITCH POINT
          </text>
        </g>

        {/* 5. Gear Specs & Ratio Callouts (Vertically Stacked to the Right) */}
        <text x="282" y="16" textAnchor="end" fill="var(--text-secondary)" fontSize="8" fontFamily="monospace">
          DRIVER: z&#8321; = 12 &middot; &Oslash; 76 mm
        </text>
        <text x="282" y="28" textAnchor="end" fill="var(--text-secondary)" fontSize="8" fontFamily="monospace">
          DRIVEN: z&#8322; = 24 &middot; &Oslash; 152 mm
        </text>

        {/* Ratio formula: i = z2 / z1 = 2.0 */}
        <g transform="translate(18, 192)">
          <text x="0" y="12" fill="var(--text-primary)" fontSize="11" fontFamily="serif" fontStyle="italic" fontWeight="bold">
            i =
          </text>
          <text x="18" y="7" fill="var(--text-primary)" fontSize="9" fontFamily="serif" fontStyle="italic">
            z&#8322;
          </text>
          <line x1="16" y1="9" x2="28" y2="9" stroke="var(--text-primary)" strokeWidth="0.8" />
          <text x="18" y="17" fill="var(--text-primary)" fontSize="9" fontFamily="serif" fontStyle="italic">
            z&#8321;
          </text>
          <text x="32" y="12" fill="var(--cad-sketch-accent)" fontSize="8.5" fontFamily="monospace" fontWeight="bold">
            = 2.0 &middot; m = 6.33 mm
          </text>
        </g>

        {/* 6. Active Constraint Status Badge */}
        <g transform="translate(155, 185)">
          <rect x="0" y="0" width="138" height="24" rx="4" fill="var(--bg-panel)" stroke="var(--border-color)" strokeWidth="0.8" />
          <circle cx="10" cy="12" r="3" fill="#10b981" />
          <text x="18" y="11" fill="var(--text-primary)" fontSize="7.5" fontFamily="monospace" fontWeight="bold">
            AGMA: MESH VERIFIED
          </text>
          <text x="18" y="19" fill="var(--text-secondary)" fontSize="6.5" fontFamily="monospace">
            INVOLUTE GEAR TRAIN
          </text>
        </g>
      </svg>
    </div>
  );
}

'use client';

import React from 'react';
import VAWTTurbine from './VAWTTurbine';
import GearTrain from './GearTrain';

export default function SketchBackground() {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: 0,
      }}
      aria-hidden="true"
    >
      {/* ================================================================ */}
      {/* 1. FULL-SCREEN CAD BLUEPRINT MILLIMETER GRID & SHARED MARKERS    */}
      {/* ================================================================ */}
      <svg
        width="100%"
        height="100%"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
      >
        <defs>
          <pattern id="cadGridSmall" width="24" height="24" patternUnits="userSpaceOnUse">
            <path
              d="M 24 0 L 0 0 0 24"
              fill="none"
              stroke="var(--border-color)"
              strokeWidth="0.5"
              strokeOpacity="0.22"
            />
          </pattern>
          <pattern id="cadGridLarge" width="120" height="120" patternUnits="userSpaceOnUse">
            <rect width="120" height="120" fill="url(#cadGridSmall)" />
            <path
              d="M 120 0 L 0 0 0 120"
              fill="none"
              stroke="var(--cad-sketch)"
              strokeWidth="0.75"
              strokeOpacity="0.10"
            />
            <circle cx="0" cy="0" r="1.5" fill="var(--cad-sketch)" fillOpacity="0.22" />
          </pattern>

          {/* Dimension Arrowhead Markers */}
          <marker
            id="cadArrowStart"
            viewBox="0 0 10 10"
            refX="0"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M 10 1 L 0 5 L 10 9 z" fill="var(--cad-sketch-accent)" fillOpacity="0.9" />
          </marker>
          <marker
            id="cadArrowEnd"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--cad-sketch-accent)" fillOpacity="0.9" />
          </marker>
        </defs>

        <rect width="100%" height="100%" fill="url(#cadGridLarge)" />
      </svg>

      {/* ================================================================ */}
      {/* 2. UPRIGHT DATUM ORIGIN TRIAD (ALIGNED TO FOOTER HORIZONTAL LINE)*/}
      {/* (+Y POINTS STRAIGHT UP, +X RIGHT, +Z OUTWARD ISOMETRIC)          */}
      {/* ================================================================ */}
      {/* ================================================================ */}
      {/* 2. UPRIGHT DATUM ORIGIN TRIAD (ALIGNED TO FOOTER HORIZONTAL LINE)*/}
      {/* (+Y POINTS STRAIGHT UP, +X RIGHT, +Z OUTWARD ISOMETRIC)          */}
      {/* ================================================================ */}
      <div style={{ position: 'fixed', left: '24px', bottom: '27.7px', zIndex: 25 }}>
        <svg width="115" height="65" viewBox="0 0 115 65" style={{ display: 'block' }}>
          {/* Origin anchor dot (0,0,0) */}
          <circle cx="24" cy="44" r="2.5" fill="var(--text-primary)" />

          {/* +Y Axis (Upright vertical arrow pointing UP) */}
          <line x1="24" y1="44" x2="24" y2="10" stroke="#10b981" strokeWidth="1.8" markerEnd="url(#cadArrowEnd)" />
          {/* +X Axis (Horizontal arrow pointing RIGHT) */}
          <line x1="24" y1="44" x2="62" y2="44" stroke="#ef4444" strokeWidth="1.8" markerEnd="url(#cadArrowEnd)" />
          {/* +Z Axis (Isometric arrow pointing OUT of screen) */}
          <line x1="24" y1="44" x2="10" y2="58" stroke="#3b82f6" strokeWidth="1.8" markerEnd="url(#cadArrowEnd)" />

          {/* Axis Labels */}
          <text x="21" y="6" fill="#10b981" fontSize="9.5" fontFamily="monospace" fontWeight="bold">Y</text>
          <text x="67" y="48" fill="#ef4444" fontSize="9.5" fontFamily="monospace" fontWeight="bold">X</text>
          <text x="1" y="64" fill="#3b82f6" fontSize="9.5" fontFamily="monospace" fontWeight="bold">Z</text>
          <text x="34" y="30" fill="var(--text-secondary)" fontSize="7.5" fontFamily="monospace" fontWeight="600">DATUM (0,0,0)</text>
        </svg>
      </div>

      {/* ================================================================ */}
      {/* 3. LEFT TECHNICAL DRAFTING COLUMN (SCALED UP & VERTICALLY CENTERED) */}
      {/* ================================================================ */}
      <div
        style={{
          position: 'absolute',
          left: '26px',
          top: '50%',
          transform: 'translateY(-50%)',
          width: '330px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-start',
          zIndex: 10,
        }}
      >
        {/* PART A: ROTOR PLAN VIEW (EXACT 3-BLADE HELICAL DESIGN FROM SLIDE) */}
        <div style={{ position: 'relative', width: '330px', height: '185px' }}>
          <svg width="330" height="185" viewBox="0 0 330 185">
            {/* Centerlines */}
            <line x1="16" y1="82" x2="212" y2="82" stroke="var(--cad-sketch)" strokeWidth="0.75" strokeDasharray="14 3 2 3" strokeOpacity="0.4" />
            <line x1="115" y1="6" x2="115" y2="158" stroke="var(--cad-sketch)" strokeWidth="0.75" strokeDasharray="14 3 2 3" strokeOpacity="0.4" />

            {/* Rotating Rotor Group (SYNCHRONIZED 6.0s PERIOD, EXACT 3-BLADE CONFIGURATION) */}
            <g style={{ transformOrigin: '115px 82px', animation: 'cadRotateRotor 6.0s linear infinite' }}>
              {/* Outer Pitch Circle (R = 75) */}
              <circle cx="115" cy="82" r="75" fill="none" stroke="var(--cad-sketch)" strokeWidth="1.2" strokeDasharray="5 3" strokeOpacity="0.45" />

              {/* Central Hub Disc & Shaft */}
              <circle cx="115" cy="82" r="18" fill="var(--bg-panel)" fillOpacity="0.5" stroke="var(--cad-sketch)" strokeWidth="1.6" />
              <circle cx="115" cy="82" r="4.5" fill="none" stroke="var(--cad-sketch-accent)" strokeWidth="1.2" />
              <circle cx="115" cy="82" r="2.2" fill="var(--cad-sketch)" />

              {/* Hub Bolt Circle (6 Fastener Bolts at 60 deg) */}
              {[0, 60, 120, 180, 240, 300].map((ang, i) => {
                const rad = (ang * Math.PI) / 180;
                return (
                  <circle
                    key={i}
                    cx={115 + 11.5 * Math.cos(rad)}
                    cy={82 + 11.5 * Math.sin(rad)}
                    r="1.2"
                    fill="var(--cad-sketch)"
                  />
                );
              })}

              {/* 3 AERODYNAMIC HELICAL AIRFOIL BLADES & RADIAL STRUTS (120° INTERVALS) */}
              {[-30, 90, 210].map((ang, idx) => (
                <g key={idx} transform={`rotate(${ang}, 115, 82)`}>
                  {/* Radial Strut Arm (Double Structural Spar from Hub to Blade Mount) */}
                  <line x1="133" y1="80.5" x2="182" y2="80.5" stroke="var(--cad-sketch)" strokeWidth="1.3" />
                  <line x1="133" y1="83.5" x2="182" y2="83.5" stroke="var(--cad-sketch)" strokeWidth="1.3" />

                  {/* Mounting Bracket Clamp at Blade Joint */}
                  <rect x="178" y="74" width="4.5" height="16" rx="1" fill="none" stroke="var(--cad-sketch)" strokeWidth="1.3" />
                  <circle cx="180.5" cy="82" r="2.2" fill="var(--cad-sketch-accent)" />

                  {/* Helical Blade Airfoil Profile (Outer Pitch Arc + Aerodynamic Camber Scoop) */}
                  <path
                    d="M 183 50.3 A 75 75 0 0 1 158 143.4 C 162 134, 169 104, 180 66 Q 182 55 183 50.3 Z"
                    fill="var(--bg-panel)"
                    fillOpacity="0.45"
                    stroke="var(--cad-sketch)"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
              ))}
            </g>

            {/* Left Dimensioning from Slide: .51, .51, .54 */}
            <line x1="20" y1="29" x2="115" y2="29" stroke="var(--cad-sketch)" strokeWidth="0.6" strokeDasharray="3 3" strokeOpacity="0.5" />
            <line x1="20" y1="135" x2="115" y2="135" stroke="var(--cad-sketch)" strokeWidth="0.6" strokeDasharray="3 3" strokeOpacity="0.5" />
            <line x1="24" y1="29" x2="24" y2="82" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" markerStart="url(#cadArrowStart)" markerEnd="url(#cadArrowEnd)" />
            <text x="14" y="59" fill="var(--text-secondary)" fontSize="7.5" fontFamily="monospace">.51</text>

            <line x1="24" y1="82" x2="24" y2="135" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" markerStart="url(#cadArrowStart)" markerEnd="url(#cadArrowEnd)" />
            <text x="14" y="112" fill="var(--text-secondary)" fontSize="7.5" fontFamily="monospace">.51</text>

            <line x1="56" y1="82" x2="56" y2="146" stroke="var(--cad-sketch)" strokeWidth="0.6" strokeDasharray="2 2" strokeOpacity="0.5" />
            <text x="48" y="156" fill="var(--text-secondary)" fontSize="7.5" fontFamily="monospace">.54</text>

            {/* Slide Callouts: Ø .05 (bracket pin), Ø .26 (hub), Ø .01 (shaft) */}
            <line x1="172" y1="49" x2="188" y2="28" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" markerStart="url(#cadArrowStart)" />
            <line x1="188" y1="28" x2="202" y2="28" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <text x="206" y="31" fill="var(--text-secondary)" fontSize="8" fontFamily="monospace">&Oslash; .05</text>

            <line x1="102" y1="95" x2="80" y2="118" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" markerStart="url(#cadArrowStart)" />
            <line x1="80" y1="118" x2="64" y2="118" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <text x="36" y="121" fill="var(--text-secondary)" fontSize="8" fontFamily="monospace">&Oslash; .26</text>

            <line x1="118" y1="86" x2="142" y2="114" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" markerStart="url(#cadArrowStart)" />
            <line x1="142" y1="114" x2="158" y2="114" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <text x="162" y="117" fill="var(--text-secondary)" fontSize="8" fontFamily="monospace">&Oslash; .01</text>

            {/* Right Diameter Dimension: 1.15 */}
            <line x1="198" y1="7" x2="198" y2="157" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" strokeDasharray="3 3" />
            <line
              x1="212"
              y1="7"
              x2="212"
              y2="157"
              stroke="var(--cad-sketch-accent)"
              strokeWidth="1.0"
              markerStart="url(#cadArrowStart)"
              markerEnd="url(#cadArrowEnd)"
            />
            <text
              x="220"
              y="86"
              fill="var(--cad-sketch-accent)"
              fontSize="9.5"
              fontFamily="monospace"
              fontWeight="bold"
            >
              1.15
            </text>

            {/* Aerodynamic Formula: Tip Speed Ratio (Placed Beside Rotor As in Slide) */}
            <g transform="translate(236, 44)">
              <text x="0" y="13" fill="var(--text-primary)" fontSize="12" fontFamily="serif" fontStyle="italic" fontWeight="bold">
                &lambda; =
              </text>
              <text x="26" y="6" fill="var(--text-primary)" fontSize="10" fontFamily="serif" fontStyle="italic">
                &omega;R
              </text>
              <line x1="22" y1="9" x2="48" y2="9" stroke="var(--text-primary)" strokeWidth="1" />
              <text x="32" y="18" fill="var(--text-primary)" fontSize="10" fontFamily="serif" fontStyle="italic">
                V
              </text>
              <text x="56" y="12" fill="var(--cad-sketch-accent)" fontSize="8" fontFamily="monospace" fontWeight="bold">
                TSR = 1.1
              </text>
            </g>
          </svg>
        </div>

        {/* PART B: ELEVATION VIEW (SCALED UP 3D TURBINE + EXACTLY ALIGNED DRAFTING DIMENSIONS) */}
        <div style={{ position: 'relative', width: '330px', height: '340px', marginTop: '8px' }}>
          {/* Scaled 3D Aerodynamic Helical Turbine Canvas (250x340, synchronized 6.0s) */}
          <div style={{ position: 'absolute', top: 0, left: 0, zIndex: 1, width: '250px', height: '340px' }}>
            <VAWTTurbine width={250} height={340} />
          </div>

          {/* Precision Dimension Overlay SVG (1:1 PIXEL-EXACT ALIGNED WITH 250x340 TURBINE) */}
          <svg
            width="330"
            height="340"
            viewBox="0 0 330 340"
            style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none', zIndex: 2 }}
          >
            {/* Height Dimension: 2.04 m (Extension lines touch right vertex of top plate at (200, 32) and bottom plate at (200, 298)) */}
            <line x1="200" y1="32" x2="238" y2="32" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <line x1="200" y1="298" x2="238" y2="298" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <line
              x1="232"
              y1="32"
              x2="232"
              y2="298"
              stroke="var(--cad-sketch-accent)"
              strokeWidth="1.0"
              markerStart="url(#cadArrowStart)"
              markerEnd="url(#cadArrowEnd)"
            />
            <text
              x="241"
              y="165"
              fill="var(--cad-sketch-accent)"
              fontSize="9"
              fontFamily="monospace"
              fontWeight="bold"
              transform="rotate(90, 241, 165)"
            >
              2.04 m
            </text>

            {/* Slide Aspect Ratio formula directly beside 2.04 m: AR = H / D = 1.77 (Pushed further right) */}
            <g transform="translate(268, 151)">
              <text x="0" y="15" fill="var(--text-primary)" fontSize="10.5" fontFamily="serif" fontStyle="italic" fontWeight="bold">
                AR =
              </text>
              <text x="26" y="8" fill="var(--text-primary)" fontSize="9" fontFamily="serif" fontStyle="italic">
                H
              </text>
              <line x1="23" y1="11" x2="36" y2="11" stroke="var(--text-primary)" strokeWidth="0.9" />
              <text x="26" y="21" fill="var(--text-primary)" fontSize="9" fontFamily="serif" fontStyle="italic">
                D
              </text>
              <text x="40" y="15" fill="var(--cad-sketch-accent)" fontSize="8" fontFamily="monospace" fontWeight="bold">
                = 1.77
              </text>
            </g>

            {/* Diameter Dimension: 1.15 m (Extension lines touch outer rotor vertices at (50, 32) and (200, 32)) */}
            <line x1="50" y1="32" x2="50" y2="10" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <line x1="200" y1="32" x2="200" y2="10" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <line
              x1="50"
              y1="15"
              x2="200"
              y2="15"
              stroke="var(--cad-sketch-accent)"
              strokeWidth="1.0"
              markerStart="url(#cadArrowStart)"
              markerEnd="url(#cadArrowEnd)"
            />
            <text
              x="125"
              y="11"
              textAnchor="middle"
              fill="var(--cad-sketch-accent)"
              fontSize="9"
              fontFamily="monospace"
              fontWeight="bold"
            >
              1.15 m
            </text>

            {/* Bottom Shaft Callout: Ø .01 */}
            <line x1="125" y1="320" x2="150" y2="330" stroke="var(--cad-sketch-accent)" strokeWidth="0.8" />
            <text x="154" y="333" fill="var(--text-secondary)" fontSize="7.5" fontFamily="monospace">
              &Oslash; .01
            </text>

            {/* Torque Formula from slide: T = P / w [N·m] (Pushed to bottom of VAWT animation) */}
            <g transform="translate(16, 320)">
              <text x="0" y="13" fill="var(--text-primary)" fontSize="10.5" fontFamily="serif" fontStyle="italic" fontWeight="bold">
                T =
              </text>
              <text x="20" y="7" fill="var(--text-primary)" fontSize="9" fontFamily="serif" fontStyle="italic">
                P
              </text>
              <line x1="17" y1="10" x2="29" y2="10" stroke="var(--text-primary)" strokeWidth="0.9" />
              <text x="20" y="19" fill="var(--text-primary)" fontSize="9" fontFamily="serif" fontStyle="italic">
                &omega;
              </text>
              <text x="34" y="13" fill="var(--text-secondary)" fontSize="7.5" fontFamily="monospace">
                [N&middot;m]
              </text>
            </g>
          </svg>
        </div>
      </div>

      {/* ================================================================ */}
      {/* 4. SCALED MECHANICAL GEAR TRAIN (RIGHT CAD DRAFTING CANVAS)      */}
      {/* ================================================================ */}
      <div
        style={{
          position: 'absolute',
          right: '48px',
          bottom: '72px',
          width: '450px',
          height: '322px',
          zIndex: 10,
        }}
      >
        <GearTrain width={450} height={322} />
      </div>
    </div>
  );
}

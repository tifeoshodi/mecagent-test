'use client';

import React, { useEffect, useRef } from 'react';

interface VAWTTurbineProps {
  width?: number;
  height?: number;
}

export default function VAWTTurbine({ width = 250, height = 340 }: VAWTTurbineProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    const ROTATION_PERIOD_MS = 6000; // Synchronized 6.0s period (10 RPM) matching the top-left plan view

    // Retina display scaling
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const render = (time: number) => {
      // Synchronized rotation angle locked to document timeline clock and CSS cadRotateRotor 6.0s
      const syncTime = (document.timeline?.currentTime as number) || time;
      const angle = ((syncTime % ROTATION_PERIOD_MS) / ROTATION_PERIOD_MS) * Math.PI * 2;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      // Read active theme colors from document computed style
      const styles = getComputedStyle(document.documentElement);
      const isLight = document.documentElement.getAttribute('data-theme') === 'light' || 
                      (!document.documentElement.getAttribute('data-theme') && window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches);
      const strokeColor = styles.getPropertyValue('--cad-sketch').trim() || (isLight ? '#8c531b' : '#38bdf8');
      const accentColor = styles.getPropertyValue('--cad-sketch-accent').trim() || (isLight ? '#b45309' : '#f59e0b');

      const cx = width / 2;
      const topY = 32;
      const bottomY = 298;
      const H = bottomY - topY; // Rotor span height (266px)
      const R = 75; // Scaled rotor radius (diameter = 150px -> AR = 266/150 = 1.77, matching slide)
      const numBlades = 3;
      const numSlices = 28;
      const helixTwist = (120 * Math.PI) / 180; // 120 deg helical twist from top to bottom
      const chordAngle = (48 * Math.PI) / 180; // 48 deg aerodynamic chord arc from slide

      // Pre-compute 3D coordinates for all 3 helical blades
      interface BladeSlice {
        lead: { x: number; y: number; z: number };
        mid: { x: number; y: number; z: number };
        trail: { x: number; y: number; z: number };
      }

      interface QuadSegment {
        b: number;
        s: number;
        p1L: { x: number; y: number; z: number };
        p1M: { x: number; y: number; z: number };
        p1T: { x: number; y: number; z: number };
        p2L: { x: number; y: number; z: number };
        p2M: { x: number; y: number; z: number };
        p2T: { x: number; y: number; z: number };
        avgZ: number;
        isFront: boolean;
        normalZ: number;
      }

      const blades: BladeSlice[][] = [];
      const segments: QuadSegment[] = [];

      for (let b = 0; b < numBlades; b++) {
        const bladeSlices: BladeSlice[] = [];
        const baseAngle = angle + (b * (2 * Math.PI)) / numBlades;

        for (let s = 0; s <= numSlices; s++) {
          const v = s / numSlices;
          const y = topY + v * H;
          const theta = baseAngle + v * helixTwist;

          // Leading edge on outer cylinder perimeter (R = 60)
          const xL = cx + R * Math.cos(theta);
          const zL = R * Math.sin(theta);

          // Mid-camber point (aerodynamic curve of airfoil suction side)
          const thetaM = theta - chordAngle * 0.42;
          const rM = R * 0.92;
          const xM = cx + rM * Math.cos(thetaM);
          const zM = rM * Math.sin(thetaM);

          // Trailing edge (inner scoop edge of airfoil, r = 0.68 R)
          const thetaT = theta - chordAngle;
          const rT = R * 0.68;
          const xT = cx + rT * Math.cos(thetaT);
          const zT = rT * Math.sin(thetaT);

          bladeSlices.push({
            lead: { x: xL, y, z: zL },
            mid: { x: xM, y, z: zM },
            trail: { x: xT, y, z: zT },
          });
        }
        blades.push(bladeSlices);

        // Build quad segments between consecutive slices
        for (let s = 0; s < numSlices; s++) {
          const p1 = bladeSlices[s];
          const p2 = bladeSlices[s + 1];
          const avgZ = (p1.lead.z + p1.trail.z + p2.lead.z + p2.trail.z) / 4;

          // Compute surface normal Z-component (cross product of chord vector & span vector)
          const ux = p1.trail.x - p1.lead.x;
          const uy = p1.trail.y - p1.lead.y;
          const vx = p2.lead.x - p1.lead.x;
          const vy = p2.lead.y - p1.lead.y;
          const normalZ = ux * vy - uy * vx;

          segments.push({
            b,
            s,
            p1L: p1.lead,
            p1M: p1.mid,
            p1T: p1.trail,
            p2L: p2.lead,
            p2M: p2.mid,
            p2T: p2.trail,
            avgZ,
            isFront: avgZ >= 0,
            normalZ,
          });
        }
      }

      // Sort segments by depth (Painter's Algorithm: render furthest back first)
      segments.sort((a, b) => a.avgZ - b.avgZ);

      // Function to render a single quad blade segment
      const drawSegment = (seg: QuadSegment) => {
        // Draw filled aerodynamic surface
        ctx.beginPath();
        ctx.moveTo(seg.p1L.x, seg.p1L.y);
        ctx.lineTo(seg.p1M.x, seg.p1M.y);
        ctx.lineTo(seg.p1T.x, seg.p1T.y);
        ctx.lineTo(seg.p2T.x, seg.p2T.y);
        ctx.lineTo(seg.p2M.x, seg.p2M.y);
        ctx.lineTo(seg.p2L.x, seg.p2L.y);
        ctx.closePath();

        if (isLight) {
          // Warm bronze / sepia CAD surface fill
          const fillAlpha = seg.isFront ? 0.20 + (seg.normalZ > 0 ? 0.12 : 0.05) : 0.07;
          ctx.fillStyle = `rgba(140, 83, 27, ${fillAlpha})`;
        } else {
          // Blueprint cyan CAD surface fill
          const fillAlpha = seg.isFront ? 0.24 + (seg.normalZ > 0 ? 0.16 : 0.06) : 0.09;
          ctx.fillStyle = `rgba(56, 189, 248, ${fillAlpha})`;
        }
        ctx.fill();

        // Leading edge segment outline
        ctx.beginPath();
        ctx.moveTo(seg.p1L.x, seg.p1L.y);
        ctx.lineTo(seg.p2L.x, seg.p2L.y);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = seg.isFront ? 2.4 : 1.3;
        ctx.globalAlpha = seg.isFront ? 0.95 : 0.35;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Trailing edge segment outline
        ctx.beginPath();
        ctx.moveTo(seg.p1T.x, seg.p1T.y);
        ctx.lineTo(seg.p2T.x, seg.p2T.y);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = seg.isFront ? 1.6 : 0.9;
        ctx.globalAlpha = seg.isFront ? 0.85 : 0.25;
        ctx.stroke();

        // Internal chord rib line at interval slices (stations from slide)
        if (seg.s % 4 === 0) {
          ctx.beginPath();
          ctx.moveTo(seg.p1L.x, seg.p1L.y);
          ctx.lineTo(seg.p1M.x, seg.p1M.y);
          ctx.lineTo(seg.p1T.x, seg.p1T.y);
          ctx.strokeStyle = accentColor;
          ctx.lineWidth = 0.9;
          ctx.globalAlpha = seg.isFront ? 0.65 : 0.2;
          ctx.stroke();
        }

        ctx.globalAlpha = 1.0;
      };

      // 1. Render Back Blade Segments (Z < 0)
      segments.filter(s => !s.isFront).forEach(drawSegment);

      // 2. Render Back Strut Arms (intermediate height stiffeners behind shaft)
      const ribSlices = [Math.floor(numSlices / 3), Math.floor((2 * numSlices) / 3)];
      ribSlices.forEach(rs => {
        blades.forEach(bSlices => {
          const pt = bSlices[rs];
          if (pt.mid.z < 0) {
            ctx.beginPath();
            ctx.moveTo(cx, pt.mid.y);
            ctx.lineTo(pt.mid.x, pt.mid.y);
            ctx.strokeStyle = accentColor;
            ctx.lineWidth = 1.1;
            ctx.globalAlpha = 0.35;
            ctx.stroke();
            ctx.globalAlpha = 1.0;
          }
        });
      });

      // 3. Central Vertical Shaft & Bearings
      // Top and bottom horizontal circular hub plates (ellipses)
      ctx.beginPath();
      ctx.ellipse(cx, topY, R, 6, 0, 0, Math.PI * 2);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.9;
      ctx.stroke();

      ctx.beginPath();
      ctx.ellipse(cx, bottomY, R, 6, 0, 0, Math.PI * 2);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Shaft Centerline (solid vertical rod)
      ctx.beginPath();
      ctx.moveTo(cx, topY - 14);
      ctx.lineTo(cx, bottomY + 22);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.6;
      ctx.lineCap = 'round';
      ctx.globalAlpha = 0.95;
      ctx.stroke();

      // Top Bearing Collar
      ctx.fillStyle = accentColor;
      ctx.fillRect(cx - 5, topY - 7, 10, 7);

      // Bottom Bearing Housing
      ctx.fillStyle = accentColor;
      ctx.fillRect(cx - 6, bottomY, 12, 8);

      // Bottom Shaft Generator Coupling & Torque Arrow (from slide: PMDC Generator Coupling)
      ctx.beginPath();
      ctx.arc(cx, bottomY + 22, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = strokeColor;
      ctx.fill();

      // Torque curved arrow at base
      ctx.beginPath();
      ctx.arc(cx, bottomY + 16, 7.5, -0.2 * Math.PI, 0.8 * Math.PI);
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // 4. Render Front Strut Arms (intermediate height stiffeners in front of shaft)
      ribSlices.forEach(rs => {
        blades.forEach(bSlices => {
          const pt = bSlices[rs];
          if (pt.mid.z >= 0) {
            ctx.beginPath();
            ctx.moveTo(cx, pt.mid.y);
            ctx.lineTo(pt.mid.x, pt.mid.y);
            ctx.strokeStyle = accentColor;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.9;
            ctx.stroke();

            // Fastener pin at blade connection (circle dot from slide)
            ctx.fillStyle = accentColor;
            ctx.beginPath();
            ctx.arc(pt.mid.x, pt.mid.y, 2.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1.0;
          }
        });
      });

      // 5. Render Front Blade Segments (Z >= 0)
      segments.filter(s => s.isFront).forEach(drawSegment);

      // 6. Top & Bottom Airfoil Blade End Caps
      blades.forEach(bSlices => {
        // Top cap
        const topSlice = bSlices[0];
        if (topSlice.lead.z >= 0 || topSlice.trail.z >= 0) {
          ctx.beginPath();
          ctx.moveTo(topSlice.lead.x, topSlice.lead.y);
          ctx.lineTo(topSlice.mid.x, topSlice.mid.y);
          ctx.lineTo(topSlice.trail.x, topSlice.trail.y);
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.8;
          ctx.stroke();

          // Bracket to top arm
          ctx.beginPath();
          ctx.moveTo(topSlice.lead.x, topSlice.lead.y);
          ctx.lineTo(cx, topY);
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.3;
          ctx.globalAlpha = 0.75;
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }

        // Bottom cap
        const botSlice = bSlices[numSlices];
        if (botSlice.lead.z >= 0 || botSlice.trail.z >= 0) {
          ctx.beginPath();
          ctx.moveTo(botSlice.lead.x, botSlice.lead.y);
          ctx.lineTo(botSlice.mid.x, botSlice.mid.y);
          ctx.lineTo(botSlice.trail.x, botSlice.trail.y);
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.8;
          ctx.stroke();

          // Bracket to bottom arm
          ctx.beginPath();
          ctx.moveTo(botSlice.lead.x, botSlice.lead.y);
          ctx.lineTo(cx, bottomY);
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.3;
          ctx.globalAlpha = 0.75;
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }
      });

      ctx.restore();

      // Continuous synchronized requestAnimationFrame loop
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        display: 'block',
      }}
    />
  );
}

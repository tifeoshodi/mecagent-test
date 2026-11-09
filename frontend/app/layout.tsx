import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AIMDEA — AI Mechanical Design Engineer Assistant',
  description: 'Autonomous AI Mechanical Engineering & 3D CAD Synthesis Platform',
};

import ThemeProvider from '@/components/ThemeProvider';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}

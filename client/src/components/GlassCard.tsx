import { cn } from '@/lib/utils';
import { ReactNode, CSSProperties } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  gradient?: boolean;
  glow?: 'primary' | 'secondary' | 'success' | 'warning' | 'destructive';
  animate?: boolean;
  style?: CSSProperties;
}

export const GlassCard = ({ 
  children, 
  className, 
  gradient = false,
  glow,
  animate = false,
  style 
}: GlassCardProps) => {
  const glowClass = glow ? `glow-${glow}` : '';
  
  return (
    <div
      className={cn(
        'glass rounded-xl p-6',
        gradient && 'gradient-border',
        glowClass,
        animate && 'animate-fade-in-up',
        className
      )}
      style={style}
    >
      {children}
    </div>
  );
};

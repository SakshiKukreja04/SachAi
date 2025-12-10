import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

interface ConfidenceGaugeProps {
  score: number;
  label: 'authentic' | 'suspected' | 'deepfake';
  className?: string;
}

export const ConfidenceGauge = ({ score, label, className }: ConfidenceGaugeProps) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  const [circumference] = useState(2 * Math.PI * 90);
  
  useEffect(() => {
    const duration = 2000;
    const startTime = Date.now();
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setAnimatedScore(Math.round(score * eased));
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    animate();
  }, [score]);
  
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;
  
  const getColor = () => {
    switch (label) {
      case 'authentic': return 'stroke-success';
      case 'suspected': return 'stroke-warning';
      case 'deepfake': return 'stroke-destructive';
    }
  };
  
  const getGlow = () => {
    switch (label) {
      case 'authentic': return 'drop-shadow-[0_0_15px_hsl(142,76%,45%)]';
      case 'suspected': return 'drop-shadow-[0_0_15px_hsl(45,93%,58%)]';
      case 'deepfake': return 'drop-shadow-[0_0_15px_hsl(0,84%,60%)]';
    }
  };
  
  const getTextColor = () => {
    switch (label) {
      case 'authentic': return 'text-success';
      case 'suspected': return 'text-warning';
      case 'deepfake': return 'text-destructive';
    }
  };

  return (
    <div className={cn('relative flex flex-col items-center', className)}>
      <svg className={cn('transform -rotate-90 w-52 h-52', getGlow())} viewBox="0 0 200 200">
        <circle
          cx="100"
          cy="100"
          r="90"
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth="8"
        />
        <circle
          cx="100"
          cy="100"
          r="90"
          fill="none"
          className={cn('transition-all duration-100', getColor())}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={cn('text-5xl font-heading font-bold', getTextColor())}>
          {animatedScore}%
        </span>
        <span className="text-muted-foreground text-sm mt-1">Confidence</span>
      </div>
    </div>
  );
};

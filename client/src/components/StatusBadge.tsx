import { cn } from '@/lib/utils';
import { Shield, AlertTriangle, XOctagon } from 'lucide-react';

interface StatusBadgeProps {
  status: 'authentic' | 'suspected' | 'deepfake';
  className?: string;
}

export const StatusBadge = ({ status, className }: StatusBadgeProps) => {
  const config = {
    authentic: {
      icon: Shield,
      label: 'Authentic',
      bg: 'bg-success/20',
      text: 'text-success',
      border: 'border-success/50',
      glow: 'glow-success',
    },
    suspected: {
      icon: AlertTriangle,
      label: 'Suspected',
      bg: 'bg-warning/20',
      text: 'text-warning',
      border: 'border-warning/50',
      glow: 'glow-warning',
    },
    deepfake: {
      icon: XOctagon,
      label: 'Deepfake Detected',
      bg: 'bg-destructive/20',
      text: 'text-destructive',
      border: 'border-destructive/50',
      glow: 'glow-destructive',
    },
  };

  const { icon: Icon, label, bg, text, border, glow } = config[status];

  return (
    <div
      className={cn(
        'inline-flex items-center gap-2 px-4 py-2 rounded-full border',
        bg, text, border, glow,
        'animate-scale-in',
        className
      )}
    >
      <Icon className="w-5 h-5" />
      <span className="font-semibold">{label}</span>
    </div>
  );
};

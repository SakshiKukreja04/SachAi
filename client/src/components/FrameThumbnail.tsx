import { useState } from 'react';
import { cn } from '@/lib/utils';
import { X, ZoomIn } from 'lucide-react';

interface FrameThumbnailProps {
  src: string;
  timestamp: string;
  confidence: number;
  className?: string;
}

export const FrameThumbnail = ({ src, timestamp, confidence, className }: FrameThumbnailProps) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsModalOpen(true)}
        className={cn(
          'group relative rounded-lg overflow-hidden',
          'border border-destructive/30 hover:border-destructive/60',
          'transition-all duration-300 hover:scale-105',
          'focus:outline-none focus:ring-2 focus:ring-destructive',
          className
        )}
      >
        <img
          src={src}
          alt={`Suspicious frame at ${timestamp}`}
          className="w-full h-32 object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background/90 to-transparent" />
        <div className="absolute bottom-2 left-2 right-2">
          <p className="text-xs text-foreground font-mono">{timestamp}</p>
          <p className="text-xs text-destructive">{confidence}% suspicious</p>
        </div>
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-background/40">
          <ZoomIn className="w-6 h-6 text-foreground" />
        </div>
      </button>

      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-md animate-fade-in"
          onClick={() => setIsModalOpen(false)}
        >
          <div
            className="relative max-w-4xl w-full glass-strong rounded-xl overflow-hidden animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setIsModalOpen(false)}
              className="absolute top-4 right-4 z-10 w-10 h-10 rounded-full glass flex items-center justify-center hover:bg-white/20 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            <img
              src={src}
              alt={`Suspicious frame at ${timestamp}`}
              className="w-full h-auto"
            />
            <div className="p-4 border-t border-border">
              <p className="text-foreground font-mono">Timestamp: {timestamp}</p>
              <p className="text-destructive">Confidence: {confidence}% suspicious</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

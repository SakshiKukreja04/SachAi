import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Calendar, ArrowRight } from 'lucide-react';
import { StatusBadge } from './StatusBadge';
import { Button } from './ui/button';

interface HistoryCardProps {
  id: string;
  date: string;
  score: number;
  label: 'authentic' | 'suspected' | 'deepfake';
  thumbnail?: string;
  className?: string;
  delay?: number;
}

export const HistoryCard = ({ id, date, score, label, thumbnail, className, delay = 0 }: HistoryCardProps) => {
  return (
    <div
      className={cn(
        'glass rounded-xl p-4 transition-all duration-300 hover:scale-[1.02]',
        'opacity-0 animate-fade-in-up',
        className
      )}
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
    >
      <div className="flex gap-4">
        {thumbnail && (
          <div className="w-24 h-16 rounded-lg overflow-hidden flex-shrink-0 border border-border bg-muted">
            <img 
              src={thumbnail} 
              alt="Video thumbnail" 
              className="w-full h-full object-cover"
              onError={(e) => {
                // Hide image on error
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>
        )}
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Calendar className="w-4 h-4" />
            <span>{date}</span>
          </div>
          
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <StatusBadge status={label} className="text-xs py-1 px-2" />
              <span className="text-foreground font-mono">{score}%</span>
            </div>
            
            <Link to={`/results/${id}`}>
              <Button variant="ghost" size="sm" className="group">
                View Report
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

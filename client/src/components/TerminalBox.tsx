import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

interface TerminalBoxProps {
  text: string;
  className?: string;
  typingSpeed?: number;
}

export const TerminalBox = ({ text, className, typingSpeed = 30 }: TerminalBoxProps) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    setDisplayedText('');
    setIsTyping(true);
    let index = 0;
    
    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayedText(text.slice(0, index + 1));
        index++;
      } else {
        setIsTyping(false);
        clearInterval(interval);
      }
    }, typingSpeed);

    return () => clearInterval(interval);
  }, [text, typingSpeed]);

  return (
    <div
      className={cn(
        'glass rounded-lg p-4 font-mono text-sm',
        'bg-background/80 border border-primary/30',
        className
      )}
    >
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-border">
        <div className="w-3 h-3 rounded-full bg-destructive/70" />
        <div className="w-3 h-3 rounded-full bg-warning/70" />
        <div className="w-3 h-3 rounded-full bg-success/70" />
        <span className="text-muted-foreground text-xs ml-2">analysis-output.log</span>
      </div>
      <div className="text-primary/90 whitespace-pre-wrap">
        <span className="text-success">$</span>{' '}
        <span>{displayedText}</span>
        {isTyping && <span className="animate-pulse">▋</span>}
      </div>
    </div>
  );
};

import { cn } from '@/lib/utils';
import { Check, Loader2 } from 'lucide-react';

interface Step {
  id: string;
  label: string;
}

interface ProgressStepsProps {
  steps: Step[];
  currentStep: number;
  className?: string;
}

export const ProgressSteps = ({ steps, currentStep, className }: ProgressStepsProps) => {
  return (
    <div className={cn('flex flex-col gap-4', className)}>
      {steps.map((step, index) => {
        const isCompleted = index < currentStep;
        const isCurrent = index === currentStep;
        const isPending = index > currentStep;
        
        return (
          <div
            key={step.id}
            className={cn(
              'flex items-center gap-4 transition-all duration-500',
              isPending && 'opacity-40'
            )}
          >
            <div
              className={cn(
                'w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-500',
                isCompleted && 'bg-success border-success glow-success',
                isCurrent && 'border-primary bg-primary/20 animate-pulse-glow',
                isPending && 'border-muted bg-transparent'
              )}
            >
              {isCompleted ? (
                <Check className="w-5 h-5 text-success-foreground" />
              ) : isCurrent ? (
                <Loader2 className="w-5 h-5 text-primary animate-spin" />
              ) : (
                <span className="text-muted-foreground text-sm">{index + 1}</span>
              )}
            </div>
            
            <div className="flex-1">
              <p
                className={cn(
                  'font-medium transition-colors duration-300',
                  isCompleted && 'text-success',
                  isCurrent && 'text-primary',
                  isPending && 'text-muted-foreground'
                )}
              >
                {step.label}
              </p>
            </div>
            
            {index < steps.length - 1 && (
              <div className="absolute left-5 mt-14 w-0.5 h-4 bg-muted">
                <div
                  className={cn(
                    'w-full bg-success transition-all duration-500',
                    isCompleted ? 'h-full' : 'h-0'
                  )}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

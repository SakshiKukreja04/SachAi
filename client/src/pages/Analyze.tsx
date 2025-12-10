import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ParticleBackground } from '@/components/ParticleBackground';
import { Navbar } from '@/components/Navbar';
import { GlassCard } from '@/components/GlassCard';
import { ProgressSteps } from '@/components/ProgressSteps';
import { Shield } from 'lucide-react';

const steps = [
  { id: 'upload', label: 'Uploading video' },
  { id: 'preprocess', label: 'Preprocessing frames' },
  { id: 'analyze', label: 'AI analyzing content' },
  { id: 'report', label: 'Generating report' },
];

const Analyze = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Simulate analysis progress
    const stepDuration = 2000; // 2 seconds per step
    
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= steps.length - 1) {
          clearInterval(stepInterval);
          // Navigate to results after completion
          setTimeout(() => {
            navigate('/results/demo-123');
          }, 1000);
          return prev;
        }
        return prev + 1;
      });
    }, stepDuration);

    // Progress animation
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + 1;
      });
    }, 80);

    return () => {
      clearInterval(stepInterval);
      clearInterval(progressInterval);
    };
  }, [navigate]);

  return (
    <div className="min-h-screen bg-hero-gradient relative overflow-hidden">
      <ParticleBackground />
      <Navbar />
      
      <main className="relative z-10 pt-24 pb-16 min-h-screen flex items-center justify-center">
        <div className="container mx-auto px-4">
          <div className="max-w-xl mx-auto">
            <GlassCard gradient className="animate-fade-in-up">
              {/* Header */}
              <div className="text-center mb-8">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center mx-auto mb-4 animate-pulse-glow">
                  <Shield className="w-10 h-10 text-primary-foreground" />
                </div>
                <h1 className="font-heading text-2xl font-bold mb-2">Analyzing Video</h1>
                <p className="text-muted-foreground">Please wait while our AI processes your content</p>
              </div>

              {/* Progress Bar */}
              <div className="mb-8">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Progress</span>
                  <span className="text-primary font-mono">{progress}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary via-secondary to-primary bg-[length:200%_100%] animate-border-flow rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              {/* Steps */}
              <ProgressSteps steps={steps} currentStep={currentStep} />

              {/* Info */}
              <div className="mt-8 pt-6 border-t border-border text-center">
                <p className="text-muted-foreground text-sm">
                  Analyzing {location.state?.file?.name || location.state?.url || 'your video'}
                </p>
              </div>
            </GlassCard>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Analyze;

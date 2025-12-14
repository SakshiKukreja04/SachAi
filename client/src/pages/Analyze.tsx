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
    // If we have a jobId from the previous page, poll the backend status endpoint
    const jobId = (location.state as any)?.jobId;
    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

    if (!jobId) {
      // Fallback to simulated progress when no jobId
      const stepDuration = 2000; // 2 seconds per step
      const stepInterval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= steps.length - 1) {
            clearInterval(stepInterval);
            setTimeout(() => navigate('/results/demo-123'), 1000);
            return prev;
          }
          return prev + 1;
        });
      }, stepDuration);

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
    }

    let stopped = false;
    const poll = async () => {
      try {
        const resp = await fetch(`${API_URL}/api/status/${jobId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (stopped) return;
        setProgress(data.progress ?? 0);
        // Map progress to step (approx)
        if (data.progress >= 75) setCurrentStep(3);
        else if (data.progress >= 50) setCurrentStep(2);
        else if (data.progress >= 25) setCurrentStep(1);
        else setCurrentStep(0);

        if (data.status === 'done') {
          // Navigate to results with jobId
          navigate(`/results/${jobId}`, { state: { jobId } });
        } else if (data.status === 'failed') {
          alert(`Analysis failed: ${data.error || 'unknown error'}`);
        }
      } catch (e) {
        console.error('Status poll error', e);
      }
    };

    const interval = setInterval(poll, 2000);
    // initial poll
    poll();

    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [location.state, navigate]);

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

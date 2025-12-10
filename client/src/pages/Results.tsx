import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { ParticleBackground } from '@/components/ParticleBackground';
import { Navbar } from '@/components/Navbar';
import { GlassCard } from '@/components/GlassCard';
import { ConfidenceGauge } from '@/components/ConfidenceGauge';
import { StatusBadge } from '@/components/StatusBadge';
import { TerminalBox } from '@/components/TerminalBox';
import { FrameThumbnail } from '@/components/FrameThumbnail';
import { Button } from '@/components/ui/button';
import { Download, Share2, ThumbsUp, ThumbsDown } from 'lucide-react';
import { toast } from '@/hooks/use-toast';

// Demo data
const demoResult = {
  score: 87,
  label: 'deepfake' as const,
  explanation: `Analysis complete. Multiple indicators of synthetic manipulation detected:
  
> Face region inconsistencies: 94% confidence
> Temporal artifacts in frames 234-289
> Audio-visual sync anomaly at 00:12:34
> Unnatural eye blinking pattern detected
> Compression artifacts suggest re-encoding

Recommendation: This video shows strong signs of AI-generated manipulation. Exercise caution before sharing.`,
  suspiciousFrames: [
    { src: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=300&fit=crop', timestamp: '00:02:34', confidence: 94 },
    { src: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&h=300&fit=crop', timestamp: '00:05:12', confidence: 89 },
    { src: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=300&fit=crop', timestamp: '00:08:45', confidence: 86 },
    { src: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&h=300&fit=crop', timestamp: '00:12:33', confidence: 91 },
  ],
};

const Results = () => {
  const { id } = useParams();
  const [feedback, setFeedback] = useState<'helpful' | 'not-accurate' | null>(null);

  const handleSaveReport = () => {
    toast({
      title: 'Report Saved',
      description: 'Your analysis report has been downloaded.',
    });
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    toast({
      title: 'Link Copied',
      description: 'Report link copied to clipboard.',
    });
  };

  const handleFeedback = (type: 'helpful' | 'not-accurate') => {
    setFeedback(type);
    toast({
      title: 'Thank you!',
      description: 'Your feedback helps improve our AI.',
    });
  };

  return (
    <div className="min-h-screen bg-hero-gradient relative overflow-hidden">
      <ParticleBackground />
      <Navbar />
      
      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto space-y-8">
            {/* Header */}
            <div className="text-center mb-8 animate-fade-in">
              <h1 className="font-heading text-3xl md:text-4xl font-bold mb-2">Analysis Complete</h1>
              <p className="text-muted-foreground">Report ID: {id}</p>
            </div>

            {/* Main Results */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Score Card */}
              <GlassCard className="flex flex-col items-center justify-center py-8 animate-fade-in-up">
                <ConfidenceGauge score={demoResult.score} label={demoResult.label} />
                <div className="mt-6">
                  <StatusBadge status={demoResult.label} />
                </div>
              </GlassCard>

              {/* Actions Card */}
              <GlassCard className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <h3 className="font-heading font-semibold text-lg mb-4">Actions</h3>
                <div className="space-y-3">
                  <Button variant="default" className="w-full" onClick={handleSaveReport}>
                    <Download className="w-4 h-4" />
                    Save Report
                  </Button>
                  <Button variant="outline" className="w-full" onClick={handleShare}>
                    <Share2 className="w-4 h-4" />
                    Share Result
                  </Button>
                </div>

                <div className="mt-6 pt-6 border-t border-border">
                  <p className="text-sm text-muted-foreground mb-3">Was this analysis helpful?</p>
                  <div className="flex gap-3">
                    <Button
                      variant={feedback === 'helpful' ? 'success' : 'glass'}
                      size="sm"
                      onClick={() => handleFeedback('helpful')}
                      className="flex-1"
                    >
                      <ThumbsUp className="w-4 h-4" />
                      Helpful
                    </Button>
                    <Button
                      variant={feedback === 'not-accurate' ? 'destructive' : 'glass'}
                      size="sm"
                      onClick={() => handleFeedback('not-accurate')}
                      className="flex-1"
                    >
                      <ThumbsDown className="w-4 h-4" />
                      Not Accurate
                    </Button>
                  </div>
                </div>
              </GlassCard>
            </div>

            {/* Suspicious Frames */}
            <GlassCard className="animate-fade-in-up" style={{ animationDelay: '200ms' }}>
              <h3 className="font-heading font-semibold text-lg mb-4">Suspicious Frames</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {demoResult.suspiciousFrames.map((frame, index) => (
                  <FrameThumbnail
                    key={index}
                    src={frame.src}
                    timestamp={frame.timestamp}
                    confidence={frame.confidence}
                  />
                ))}
              </div>
            </GlassCard>

            {/* Analysis Explanation */}
            <GlassCard className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <h3 className="font-heading font-semibold text-lg mb-4">Analysis Details</h3>
              <TerminalBox text={demoResult.explanation} />
            </GlassCard>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Results;

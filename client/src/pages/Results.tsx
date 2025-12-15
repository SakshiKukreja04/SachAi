import { useEffect, useState } from 'react';
import { useParams, useLocation } from 'react-router-dom';
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

// Function to generate analysis summary from actual results
const generateAnalysisSummary = (result: any): string => {
  if (!result) {
    return `Analysis in progress...
    
> Waiting for model inference to complete
> Please refresh the page to see updated results`;
  }

  // Use final_prob if available (aggregated score), otherwise fall back to visual_prob
  const finalProb = result.final_prob ?? result.visual_prob ?? result.score ?? 0;
  const visualProb = result.visual_prob ?? result.score ?? 0;
  // Handle audio sync score - check for null/undefined, but allow 0 as valid value
  const audioSyncScore = (result.audio_sync_score !== null && result.audio_sync_score !== undefined) 
    ? result.audio_sync_score 
    : null;
  const scorePercent = Math.round(finalProb * 100);
  const numFrames = result.meta?.num_frames ?? result.visual_scores?.length ?? 0;
  const suspiciousFrames = result.suspiciousFrames || [];
  const visualScores = result.visual_scores || [];
  const explanations = result.explanations || [];

  // Use classification from server if available, otherwise compute
  // IMPORTANT: If visual_prob >= 0.6, it's always DEEPFAKE
  let classification: string = result.classification || 'UNKNOWN';
  let confidenceLevel: string = result.confidence_level || 'UNKNOWN';
  
  if (classification === 'UNKNOWN') {
    // Fallback classification logic: visual_prob >= 0.6 → DEEPFAKE
    // Otherwise use final_prob with thresholds: <0.3=REAL, 0.3-0.59=SUSPECTED, >=0.6=FAKE
    if (visualProb >= 0.6) {
      classification = 'DEEPFAKE';
      confidenceLevel = 'HIGH';
    } else if (finalProb < 0.3) {
      classification = 'AUTHENTIC';
      confidenceLevel = 'HIGH';
    } else if (finalProb < 0.6) {
      classification = 'SUSPECTED';
      confidenceLevel = 'MEDIUM';
    } else {
      classification = 'DEEPFAKE';
      confidenceLevel = 'HIGH';
    }
  }
  
  // Convert classification to UI label format (lowercase)
  const classificationLabel = classification === 'DEEPFAKE' ? 'deepfake' 
    : classification === 'SUSPECTED' ? 'suspected' 
    : 'authentic';

  // Calculate statistics
  const frameScores = visualScores.map((s: any) => s.score ?? 0).filter((s: number) => s > 0);
  const minScore = frameScores.length > 0 ? Math.min(...frameScores) : 0;
  const maxScore = frameScores.length > 0 ? Math.max(...frameScores) : 0;
  const meanScore = frameScores.length > 0 ? frameScores.reduce((a: number, b: number) => a + b, 0) / frameScores.length : 0;

  // Build summary
  let summary = `Analysis complete. Deepfake detection results:\n\n`;
  summary += `> Final probability (of being fake): ${scorePercent}% (${finalProb.toFixed(4)})\n`;
  summary += `>   Note: This is the probability the video is AI-generated, NOT model accuracy\n`;
  summary += `> Visual probability: ${Math.round(visualProb * 100)}% (${visualProb.toFixed(4)})\n`;
  
  if (audioSyncScore !== null && audioSyncScore !== undefined) {
    const audioSyncPercent = Math.round(audioSyncScore * 100);
    summary += `> Audio sync score: ${audioSyncPercent}% (${audioSyncScore.toFixed(4)})\n`;
    summary += `>   Note: This is the probability that audio and video are synchronized (higher = better sync, more authentic)\n`;
    summary += `>   ${audioSyncScore >= 0.6 ? 'Good' : audioSyncScore >= 0.3 ? 'Moderate' : 'Poor'} lip-sync quality\n`;
  } else {
    summary += `> Audio sync score: Not available\n`;
  }
  
  summary += `> Classification: ${classification} (${confidenceLevel} confidence)\n`;
  summary += `> Frames analyzed: ${numFrames}\n`;
  
  if (frameScores.length > 0) {
    summary += `> Score range: ${Math.round(minScore * 100)}% - ${Math.round(maxScore * 100)}%\n`;
    summary += `> Average frame score: ${Math.round(meanScore * 100)}%\n`;
  }
  
  // Add explanations if available
  if (explanations.length > 0) {
    summary += `\n> Detailed Analysis:\n`;
    explanations.forEach((exp: string) => {
      summary += `  - ${exp}\n`;
    });
  }

  if (suspiciousFrames.length > 0) {
    summary += `\n> Top suspicious frames detected: ${suspiciousFrames.length}\n`;
    suspiciousFrames.slice(0, 5).forEach((frame: any, idx: number) => {
      const frameScore = frame.score ?? frame.confidence ?? 0;
      const framePercent = Math.round(frameScore * 100);
      const fileName = frame.file || frame.filename || `Frame ${idx + 1}`;
      summary += `  - ${fileName}: ${framePercent}% confidence`;
      if (frame.rank) {
        summary += ` (Rank #${frame.rank})`;
      }
      summary += `\n`;
    });
  }

  // Add recommendation based on final probability (fake_ratio thresholds)
  summary += `\nRecommendation: `;
  if (finalProb < 0.3) {
    summary += `The video appears to be AUTHENTIC (${scorePercent}% fake frame ratio). Low likelihood of AI manipulation.`;
  } else if (finalProb < 0.6) {
    summary += `SUSPECTED manipulation detected (${scorePercent}% fake frame ratio). Review the suspicious frames and audio sync issues.`;
    if (audioSyncScore !== null && audioSyncScore < 0.6) {
      summary += ` Audio-visual sync mismatches detected, which may indicate deepfake manipulation.`;
    }
  } else {
    summary += `This video shows strong signs of AI-generated manipulation (${scorePercent}% fake frame ratio). Exercise caution before sharing or using this content.`;
    if (audioSyncScore !== null && audioSyncScore < 0.6) {
      summary += ` Significant audio-visual sync mismatches support this classification.`;
    }
  }

  if (result.meta?.model) {
    summary += `\n\nModel: ${result.meta.model}`;
    if (result.meta.checkpoint_loaded) {
      summary += ` (Trained model loaded)`;
    } else {
      summary += ` (Untrained - results may be unreliable)`;
    }
  }

  return summary;
};

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
  const location = useLocation();
  const [feedback, setFeedback] = useState<'helpful' | 'not-accurate' | null>(null);
  const [result, setResult] = useState<any | null>(null);

  const handleSaveReport = async () => {
    if (!id) {
      toast({
        title: 'Error',
        description: 'No report ID available.',
        variant: 'destructive',
      });
      return;
    }

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
      const response = await fetch(`${API_URL}/api/history/${id}/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to save report');
      }

      toast({
        title: 'Report Saved',
        description: 'Your analysis report has been saved to history.',
      });
    } catch (error) {
      console.error('Error saving report:', error);
      toast({
        title: 'Error',
        description: 'Failed to save report. Please try again.',
        variant: 'destructive',
      });
    }
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

  useEffect(() => {
    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
    const jobId = (location.state as any)?.jobId || id;
    if (!jobId) return;

    const fetchStatus = async () => {
      try {
        const resp = await fetch(`${API_URL}/api/status/${jobId}`);
        if (!resp.ok) {
          console.error(`Status fetch failed: ${resp.status} ${resp.statusText}`);
          return;
        }
        const data = await resp.json();
        const resultData = data.result || null;
        setResult(resultData);
        
        // Log final score for debugging - handle both visual_prob and legacy score field
        const score = resultData?.visual_prob ?? resultData?.score;
        if (score !== undefined && score !== null) {
          let classification = 'UNKNOWN';
          let confidenceLevel = 'UNKNOWN';
          // Updated thresholds: Using fake_ratio approach
          // fake_ratio: <0.3=REAL, 0.3-0.59=SUSPECTED, >=0.6=FAKE
          if (score >= 0.6) {
            classification = 'DEEPFAKE';
            confidenceLevel = 'HIGH';
          } else if (score >= 0.3) {
            classification = 'SUSPECTED';
            confidenceLevel = 'MEDIUM';
          } else {
            classification = 'AUTHENTIC';
            confidenceLevel = 'HIGH';
          }
          
          console.log('\n' + '='.repeat(60));
          console.log(`FINAL SCORE (Frontend):`);
          console.log(`  visual_prob (fake_ratio) = ${score.toFixed(4)} (${(score * 100).toFixed(2)}%)`);
          console.log(`  Classification: ${classification} (Confidence: ${confidenceLevel})`);
          console.log(`  Thresholds (fake_ratio): >=0.6=FAKE, 0.3-0.59=SUSPECTED, <0.3=REAL`);
          console.log(`  Job ID: ${jobId}`);
          console.log(`  Displayed as: ${Math.round(score * 100)}%`);
          console.log(`  Suspicious frames: ${resultData.suspiciousFrames?.length || 0}`);
          if (resultData.suspiciousFrames?.length > 0) {
            console.log(`  Frame scores:`, resultData.suspiciousFrames.map((f: any) => 
              `${f.file || f.filename || 'unknown'}: ${((f.score ?? 0) * 100).toFixed(2)}%`
            ));
          }
          console.log('='.repeat(60) + '\n');
        }
      } catch (e) {
        console.error('Failed to fetch result', e);
      }
    };

    fetchStatus();
  }, [id, location.state]);

  // Compute classification for UI display (use the same logic as generateAnalysisSummary)
  const finalProb = result ? (result.final_prob ?? result.visual_prob ?? result.score ?? 0) : 0;
  const visualProb = result ? (result.visual_prob ?? result.score ?? 0) : 0;
  let classification: string = result?.classification || 'UNKNOWN';
  let confidenceLevel: string = result?.confidence_level || 'UNKNOWN';
  
  if (classification === 'UNKNOWN') {
    // Fallback classification logic: visual_prob >= 0.6 → DEEPFAKE
    // Otherwise use final_prob with thresholds: <0.3=REAL, 0.3-0.59=SUSPECTED, >=0.6=FAKE
    if (visualProb >= 0.6) {
      classification = 'DEEPFAKE';
      confidenceLevel = 'HIGH';
    } else if (finalProb < 0.3) {
      classification = 'AUTHENTIC';
      confidenceLevel = 'HIGH';
    } else if (finalProb < 0.6) {
      classification = 'SUSPECTED';
      confidenceLevel = 'MEDIUM';
    } else {
      classification = 'DEEPFAKE';
      confidenceLevel = 'HIGH';
    }
  }
  
  // Convert classification to UI label format (lowercase)
  const classificationLabel = classification === 'DEEPFAKE' ? 'deepfake' 
    : classification === 'SUSPECTED' ? 'suspected' 
    : 'authentic';

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
                <ConfidenceGauge
                  score={result ? Math.round(((result.final_prob ?? result.visual_prob ?? result.score ?? 0) * 100)) : demoResult.score}
                  label={
                    result
                      ? classificationLabel
                      : demoResult.label
                  }
                />
                <div className="mt-6">
                  <StatusBadge
                    status={
                      result
                        ? classificationLabel
                        : demoResult.label
                    }
                  />
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
              <h3 className="font-heading font-semibold text-lg mb-6">Suspicious Frames</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {(result?.suspiciousFrames ?? demoResult.suspiciousFrames).map((frame: any, index: number) => {
                    const jobId = (location.state as any)?.jobId || id;
                    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
                    // frame.file is expected from backend (e.g. 'face_frame_00006_1.jpg')
                    const fileName = frame.file || frame.filename || frame.src || '';
                    // prefer DB-backed image id when available
                    const imgSrc = frame.id ? `${API_URL}/images/${frame.id}` : (jobId && frame.file ? `${API_URL}/api/tmp/${jobId}/faces/${fileName}` : (frame.src || ''));
                    const frameScore = frame.score ?? frame.confidence ?? 0;
                    const scorePercent = Math.round(frameScore * 100);

                    return (
                      <div key={index} className="flex flex-col border rounded-lg overflow-hidden bg-card/50 hover:bg-card/70 transition-colors">
                        {/* Large Image */}
                        <div className="w-full aspect-video bg-muted relative overflow-hidden">
                          <img 
                            src={imgSrc} 
                            alt={fileName} 
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iIzMzMzMzMyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTgiIGZpbGw9IiM2NjY2NjYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+';
                            }}
                          />
                          {/* Score Badge Overlay */}
                          <div className="absolute top-2 right-2 bg-background/90 backdrop-blur-sm px-3 py-1.5 rounded-full border">
                            <span className="text-lg font-bold font-mono">
                              {frameScore !== undefined && frameScore !== null 
                                ? `${scorePercent}%` 
                                : frame.confidence !== undefined && frame.confidence !== null
                                ? `${Math.round(frame.confidence * 100)}%`
                                : 'N/A'}
                            </span>
                          </div>
                        </div>
                        {/* Frame Info */}
                        <div className="p-4">
                          <div className="font-medium text-sm mb-1 truncate" title={fileName}>
                            {fileName}
                          </div>
                          {frame.timestamp && (
                            <div className="text-xs text-muted-foreground">{frame.timestamp}</div>
                          )}
                          {frame.rank && (
                            <div className="text-xs text-muted-foreground mt-1">
                              Rank: #{frame.rank}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
            </GlassCard>

            {/* Analysis Explanation */}
            <GlassCard className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <h3 className="font-heading font-semibold text-lg mb-4">Analysis Details</h3>
              <TerminalBox text={result ? generateAnalysisSummary(result) : demoResult.explanation} />
            </GlassCard>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Results;

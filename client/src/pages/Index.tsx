import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { ParticleBackground } from '@/components/ParticleBackground';
import { Navbar } from '@/components/Navbar';
import { UploadZone } from '@/components/UploadZone';
import { Shield, Zap, Lock, Eye } from 'lucide-react';

const Index = () => {
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(false);

  const handleUpload = async (file: File | null, url: string | null) => {
    setIsLoading(true);
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

      if (file) {
        const form = new FormData();
        form.append('video', file);

        const resp = await fetch(`${API_URL}/api/analyze`, {
          method: 'POST',
          body: form,
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err?.error || `Upload failed: ${resp.status}`);
        }
        const data = await resp.json();
        navigate('/analyze', { state: { jobId: data.jobId, file } });
      } else if (url) {
        const resp = await fetch(`${API_URL}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ videoUrl: url }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err?.error || `Request failed: ${resp.status}`);
        }
        const data = await resp.json();
        navigate('/analyze', { state: { jobId: data.jobId, url } });
      }
    } catch (e) {
      console.error('Upload error', e);
      alert(`Upload error: ${(e as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const features = [
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Get results in seconds with our optimized AI pipeline',
    },
    {
      icon: Shield,
      title: 'Military-Grade AI',
      description: 'Trained on millions of deepfake samples for maximum accuracy',
    },
    {
      icon: Lock,
      title: 'Privacy First',
      description: 'Videos are processed locally and never stored permanently',
    },
    {
      icon: Eye,
      title: 'Frame Analysis',
      description: 'Deep inspection of every frame for subtle manipulations',
    },
  ];

  return (
    <div className="min-h-screen bg-hero-gradient relative overflow-hidden">
      <ParticleBackground />
      <Navbar />
      
      <main className="relative z-10 pt-24 pb-16">
        {/* Hero Section */}
        <section key="hero-section" className="container mx-auto px-4 py-16 md:py-24">
          <div className="text-center max-w-4xl mx-auto mb-12 md:mb-16">
            <h1 className="font-heading text-4xl md:text-6xl lg:text-7xl font-bold mb-6 animate-fade-in">
              <span className="text-gradient">Seeing is no longer believing.</span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8 animate-fade-in-up" style={{ animationDelay: '150ms' }}>
              Verify videos with SachAI — the most advanced deepfake detection platform.
            </p>
          </div>

          {/* Upload Zone */}
          <div className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <UploadZone onUpload={handleUpload} isLoading={isLoading} />
          </div>
        </section>

        {/* Features Section */}
        <section key="features-section" className="container mx-auto px-4 py-16">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className="glass rounded-xl p-6 text-center hover:scale-105 transition-transform duration-300 opacity-0 animate-fade-in-up"
                style={{ animationDelay: `${400 + index * 100}ms`, animationFillMode: 'forwards' }}
              >
                <div className="w-14 h-14 rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center mx-auto mb-4">
                  <feature.icon className="w-7 h-7 text-primary" />
                </div>
                <h3 className="font-heading font-semibold text-lg mb-2">{feature.title}</h3>
                <p className="text-muted-foreground text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Privacy Badge */}
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50">
          <div className="glass px-4 py-2 rounded-full flex items-center gap-2 text-sm">
            <Lock className="w-4 h-4 text-success" />
            <span className="text-muted-foreground">
              Videos are processed temporarily and automatically deleted
            </span>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;

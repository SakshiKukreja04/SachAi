import { ParticleBackground } from '@/components/ParticleBackground';
import { Navbar } from '@/components/Navbar';
import { HistoryCard } from '@/components/HistoryCard';
import { GlassCard } from '@/components/GlassCard';
import { History as HistoryIcon, Search, RefreshCw } from 'lucide-react';
import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface HistoryItem {
  id: string;
  date: string;
  score: number;
  label: 'authentic' | 'suspected' | 'deepfake';
  thumbnail?: string | null;
  videoUrl?: string;
  createdAt: string;
}

const History = () => {
  const [search, setSearch] = useState('');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
      const response = await fetch(`${API_URL}/api/history`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch history');
      }
      
      const data = await response.json();
      setHistory(data);
    } catch (err) {
      console.error('Error fetching history:', err);
      setError('Failed to load analysis history. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const filteredHistory = history.filter((item) => {
    const searchLower = search.toLowerCase();
    return (
      item.date.toLowerCase().includes(searchLower) ||
      item.label.toLowerCase().includes(searchLower) ||
      item.id.toLowerCase().includes(searchLower) ||
      (item.videoUrl && item.videoUrl.toLowerCase().includes(searchLower))
    );
  });

  return (
    <div className="min-h-screen bg-hero-gradient relative overflow-hidden">
      <ParticleBackground />
      <Navbar />
      
      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto">
            {/* Header */}
            <div className="text-center mb-8 animate-fade-in">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center mx-auto mb-4">
                <HistoryIcon className="w-8 h-8 text-primary" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold mb-2">Analysis History</h1>
              <p className="text-muted-foreground">View your previous video analyses</p>
            </div>

            {/* Search and Refresh */}
            <div className="flex gap-4 mb-6 animate-fade-in-up">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search by date, status, or ID..."
                  className={cn(
                    'w-full h-12 pl-12 pr-4 rounded-lg',
                    'glass border-white/20',
                    'text-foreground placeholder:text-muted-foreground',
                    'focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent',
                    'transition-all duration-300'
                  )}
                />
              </div>
              <Button
                variant="outline"
                onClick={fetchHistory}
                disabled={loading}
                className="h-12 px-4"
              >
                <RefreshCw className={cn('w-4 h-4', loading && 'animate-spin')} />
              </Button>
            </div>

            {/* Loading State */}
            {loading && (
              <GlassCard className="text-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
                <p className="text-muted-foreground">Loading analysis history...</p>
              </GlassCard>
            )}

            {/* Error State */}
            {error && !loading && (
              <GlassCard className="text-center py-12">
                <p className="text-destructive mb-4">{error}</p>
                <Button onClick={fetchHistory} variant="outline">
                  Try Again
                </Button>
              </GlassCard>
            )}

            {/* History List */}
            {!loading && !error && (
              <>
                {filteredHistory.length > 0 ? (
                  <div className="space-y-4">
                    {filteredHistory.map((item, index) => (
                      <HistoryCard
                        key={item.id}
                        id={item.id}
                        date={item.date}
                        score={item.score}
                        label={item.label}
                        thumbnail={item.thumbnail || undefined}
                        delay={index * 100}
                      />
                    ))}
                  </div>
                ) : (
                  <GlassCard className="text-center py-12">
                    {history.length === 0 ? (
                      <>
                        <p className="text-muted-foreground mb-2">No analysis history yet.</p>
                        <p className="text-sm text-muted-foreground">
                          Start analyzing videos to see your history here.
                        </p>
                      </>
                    ) : (
                      <p className="text-muted-foreground">No analyses found matching your search.</p>
                    )}
                  </GlassCard>
                )}
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default History;
